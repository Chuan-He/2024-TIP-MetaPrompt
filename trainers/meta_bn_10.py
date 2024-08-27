import os.path as osp
import random

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.optim as optim
import math
import learn2learn as l2l
from . import maml

from copy import deepcopy

class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, input_feat):
        
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        
        return final_feat.squeeze(-1).squeeze(-1)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class VisionEncoder(nn.Module):
    def __init__(self, cfg, clip_model): #, image_weight
        super().__init__()
        visual = clip_model.visual  # CLIP's visual encoder
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.layers = len(self.transformer)
        self.n_pro = cfg.TRAINER.META.N_PRO
        self.layer_p = cfg.TRAINER.META.LAYERS

    def forward(self, x, ctx_v):
        x = torch.cat([x, ctx_v[:, 0, :, :]], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        for i in range(self.layers):
            if 1 <= i < self.layer_p:
                ctx = ctx_v[:, i].permute(1, 0, 2)
                prefix = x[:-self.n_pro, :, :]
                x = torch.cat([prefix, ctx], dim=0)
            x = self.transformer[i](x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        n_pro = cfg.TRAINER.META.N_PRO
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.visual.ln_pre.weight.shape[0]
        self.visual = clip_model.visual
        self.conv1 = self.visual.conv1
        self.class_embedding = self.visual.class_embedding
        self.positional_embedding = self.visual.positional_embedding
        self.layers = len(self.visual.transformer.resblocks)
        self.layer_p = cfg.TRAINER.META.LAYERS

        ctx_vectors = torch.empty(self.layer_p, n_pro, ctx_dim, dtype=self.dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        
    def forward(self, x):
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(len(x), -1, -1, -1)

        x = self.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.positional_embedding.type(self.dtype)
        
        return x, ctx


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.layers = len(clip_model.transformer.resblocks)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.layers = len(self.transformer)
        self.n_ctx = cfg.TRAINER.META.N_CTX
        self.layer_p = cfg.TRAINER.META.LAYERS

    def forward(self, prompts, tokenized_prompts, ctx_t):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.layers):
            if 1 <= i < self.layer_p:
                ctx = ctx_t[:, i].permute(1, 0, 2)
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx:, :, :]
                x = torch.cat([prefix, ctx, suffix], dim=0) 
            x = self.transformer[i](x)
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.META.N_CTX
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = len(clip_model.transformer.resblocks)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.layer_p = cfg.TRAINER.META.LAYERS

        # use given words to initialize context vectors
        ctx_init = 'a photo of a'
        ctx_init = ctx_init.replace("_", " ")
        # prompt = clip.tokenize(ctx_init).to('cuda:0')
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(prompt).type(dtype)
        # ctx_vectors_0 = embedding[0, 1: 1 + n_ctx, :]
        prompt_prefix = ctx_init

        ctx_vectors = torch.empty(self.layer_p, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        #ctx_vectors[0,:,:] = ctx_vectors_0

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # self.ctx = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512)) for _ in range(self.layer_p)])
        # for single_para in self.ctx:
            #nn.init.normal_(single_para, std=0.02)
        
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        #self.token_prefix = embedding[:, :1, :]
        #self.token_suffix = embedding[:, 1 + n_ctx :, :]
        self.token_prefix = embedding[:math.ceil(self.n_cls / 2), :1, :]  # SOS
        self.token_suffix = embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :]  # CLS, EOS

        self.token_prefix2 = embedding[math.ceil(self.n_cls / 2):, :1, :]  # SOS
        self.token_suffix2 = embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :]  # CLS, EOS

        # self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

        # self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
        # self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS



    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            
        prefix = self.token_prefix
        suffix = self.token_suffix
        prefix = torch.cat([prefix, self.token_prefix2], dim=0)
        suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = torch.cat([prefix, ctx[:, 0], suffix], dim=1)
        return prompts, ctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        for p in clip_model.parameters():
            p.requires_grad = False

        t = {
            "OxfordPets": "a photo of a {}, a type of pet.",
            "OxfordFlowers": "a photo of a {}, a type of flower.",
            "FGVCAircraft": "a photo of a {}, a type of aircraft.",
            "DescribableTextures": "a photo of a {}, a type of texture.",
            "EuroSAT": "a centered satellite photo of {}.",
            "StanfordCars": "a photo of a {}.",
            "Food101": "a photo of {}, a type of food.",
            "SUN397": "a photo of a {}.",
            "Caltech101": "a photo of a {}.",
            "UCF101": "a photo of a person doing {}.",
            "ImageNet": "a photo of a {}.",
            "ImageNetSketch": "a photo of a {}.",
            "ImageNetV2": "a photo of a {}.",
            "ImageNetA": "a photo of a {}.",
            "ImageNetR": "a photo of a {}.",
            'PACS':'a photo of a {}',
            'OfficeHomeFS':'a photo of a {}',
            'PACSFS':'a photo of a {}',
            'VLCSFS':'a photo of a {}',
            'DomainNetFS':'a photo of a {}',
            'TERRAFS':'a photo of a {}'
        }

        if 'ImageNet' not in cfg.DATASET.NAME:
            templates = [t[cfg.DATASET.NAME]]
        else:
            templates = [
                'itap of a {}',
                'a bad photo of the {}',
                'a origami {}',
                'a photo of the large {}',
                'a {} in a video game',
                'art of the {}',
                'a photo of the small {}',
                'a photo of a {}'
            ]

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.vision_prompt_learner = VisionPromptLearner(cfg, clip_model)
        self.image_encoder = VisionEncoder(cfg, clip_model)
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.model = clip_model
        self.classname = classnames

        # self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 512)
        # self.VPT_image_trans = self.VPT_image_trans.cuda()

        with torch.no_grad():
            # zeroshot_weights = []
            # for classname in classnames:
            #     texts = [template.format(classname) for template in templates]
            #     texts = clip.tokenize(texts).cuda()
            #     class_embeddings = clip_model.encode_text(texts) 
            #     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            #     class_embedding = class_embeddings.mean(dim=0)
            #     class_embedding /= class_embedding.norm()
            #     zeroshot_weights.append(class_embedding)
            # self.text_features_zs = torch.stack(zeroshot_weights, dim=0).cuda()
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in templates:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))
            self.text_features_zs = torch.cat(all_teacher_features, dim=1).mean(dim=1)

    def forward_mix(self, image, label, image_sup, label_sup, mix_ids, lam):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        lam = lam.type(self.dtype)

        prompts, ctx_t = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts, ctx_t)
        x, ctx_v = self.vision_prompt_learner(image.type(self.dtype))
        image_features = self.image_encoder(x, ctx_v.type(self.dtype))
 
        b_size = image.size(0)
        x_sup, sup_ctx_v = self.vision_prompt_learner(image_sup.type(self.dtype))
        image_features_sup = self.image_encoder(x_sup, sup_ctx_v.type(self.dtype))
        image_features = lam.view(b_size, 1)*image_features + (1-lam).view(b_size,1)*image_features_sup[mix_ids]
        label_b = label_sup[mix_ids]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()

        # Now calculate the frozen pre-trained features
        fixed_embeddings = self.text_features_zs  # precomputed pre-trained frozen textual features
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            zero_shot_features = self.model.encode_image(image.type(self.dtype))
            zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
            # Compute pre-trained frozen visual features
            zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
        
        ce_loss = (lam*F.cross_entropy(logits, label, reduce=False)+(1-lam)*F.cross_entropy(logits, label_b, reduce=False)).mean()
    
        # Now calculate L_SCL_logits
        kl_loss = F.kl_div(
            F.log_softmax(logits / 1, dim=1),
            F.log_softmax(zero_shot_logits / 1, dim=1),
            reduction='sum',
            log_target=True
        ) * (1 * 1) / logits.numel()

        reg_text = F.l1_loss(text_features, fixed_embeddings,
                                    reduction='mean')
        reg_image = F.l1_loss(image_features, zero_shot_features,
                                    reduction='mean')

        return ce_loss, kl_loss, reg_image, reg_text
        
    def forward(self, image, label_idx = None):
        logit_scale = self.logit_scale.exp()

        with torch.no_grad():
            if label_idx != None:
                text_features_zs = self.text_features_zs[label_idx, :]
            else:
                text_features_zs = self.text_features_zs
            text_features_zs = text_features_zs / text_features_zs.norm(dim=-1, keepdim=True)
            image_features_zs = self.model.encode_image(image.type(self.dtype))
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
            zero_shot_logits = logit_scale * image_features_zs @ text_features_zs.t()


        prompts, ctx_t = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        if label_idx != None:
            prompts = prompts[label_idx]
            ctx_t = ctx_t[label_idx]
            tokenized_prompts = tokenized_prompts[label_idx]
        text_features = self.text_encoder(prompts, tokenized_prompts, ctx_t.half())

        x, ctx_v = self.vision_prompt_learner(image)
        image_features = self.image_encoder(x, ctx_v.half())
        #image_features = self.VPT_image_trans(image_features.float()).half()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_ce = logit_scale * (image_features @ text_features.t())

        kl_loss = F.kl_div(
            F.log_softmax(logits_ce / 1, dim=1),
            F.log_softmax(zero_shot_logits / 1, dim=1),
            reduction='sum',
            log_target=True
        ) * (1 * 1) / logits_ce.numel()

        reg_text = F.l1_loss(text_features, text_features_zs,
                                    reduction='mean')
        reg_image = F.l1_loss(image_features, image_features_zs,
                                    reduction='mean')
        
        if self.training:
            return logits_ce, kl_loss, reg_image, reg_text
        else:
            return logits_ce

class MAMLFewShotClassifier(nn.Module):
    def __init__(self, device):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()

        self.device = device
        #self.lr_t = nn.ParameterList([nn.Parameter(torch.ones(1) * init_lr) for _ in range(3)]).to(device=device)
        #self.lr_i = nn.ParameterList([nn.Parameter(torch.ones(1) * init_lr) for _ in range(3)]).to(device=device)
        ctx_dim = 512
        self.reg_t = nn.Sequential(nn.Linear(ctx_dim, ctx_dim),nn.ReLU(),nn.Linear(ctx_dim, ctx_dim)).to(device=self.device)
        vis_dim = 768
        self.reg_i = nn.Sequential(nn.Linear(vis_dim, vis_dim),nn.ReLU(),nn.Linear(vis_dim, vis_dim)).to(device=self.device)

    def update_grads(self, params, grads_list):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        a_grad, b_grad = grads_list

        layers, n_ctx, ctx_dim = params.size()

        b_grad_norm = b_grad / torch.linalg.norm(b_grad)
        a_grad_norm = a_grad / torch.linalg.norm(a_grad)
        # aa = phi(a_grad_norm.float()).half()
        # params.grad = a_grad + torch.dot(aa.flatten(), b_grad_norm.flatten()) * b_grad
    
        if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
            params.grad = a_grad - 0.5 * torch.dot(a_grad.flatten(), b_grad_norm.flatten()) * b_grad_norm
        else:
            params.grad = a_grad

    # def prograd_backward_and_update(
    #     self, loss_a, loss_b, lambda_=1, names=None
    # ):
    #     # loss_b not increase is okay
    #     # loss_a has to decline
    #     self.model_zero_grad(names)
    #     # get name of the model parameters
    #     names = self.get_model_names(names)
    #     # backward loss_a
    #     self.detect_anomaly(loss_b)
    #     loss_b.backward(retain_graph=True)
    #     # normalize gradient
    #     b_grads = []
    #     for name in names:
    #         for p in self._models[name].parameters():
    #             b_grads.append(p.grad.clone())

    #     # optimizer don't step
    #     for name in names:
    #         self._optims[name].zero_grad()

    #     # backward loss_a
    #     self.detect_anomaly(loss_a)
    #     loss_a.backward()
    #     for name in names:
    #         for p, b_grad in zip(self._models[name].parameters(), b_grads):
    #             # calculate cosine distance
    #             b_grad_norm = b_grad / torch.linalg.norm(b_grad)
    #             a_grad = p.grad.clone()
    #             a_grad_norm = a_grad / torch.linalg.norm(a_grad)

    #             if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
    #                 p.grad = a_grad - lambda_ * torch.dot(
    #                     a_grad.flatten(), b_grad_norm.flatten()
    #                 ) * b_grad_norm

    #     # optimizer
    #     for name in names:
    #         self._optims[name].step()

@TRAINER_REGISTRY.register()
class Meta_B2N(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.META.PREC in ["fp16", "fp32", "amp"]


    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg).cuda()
        
        if cfg.TRAINER.META.PREC == "fp32" or cfg.TRAINER.META.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        #self.meta_step = cfg.TRAINER.META.META_STEP
        self.num_layers = cfg.TRAINER.META.LAYERS
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.adapt_lr = cfg.TRAINER.META.ADAPT_LR
        self.lr_ratio = cfg.TRAINER.META.LR_RATIO

        self.fast_adaptation = False
        
        print("Building custom CLIP")
        print(self.num_layers)
        self.model = CustomCLIP(cfg, classnames, clip_model)
        # self.vnet = maml.VNet(cfg, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        
        # for name, param in self.vnet.named_parameters():
        #     if param.requires_grad:
        #         enabled.add(name)          
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # self.vnet.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        # self.optim_vnet = build_optimizer(self.vnet, cfg.OPTIM_META)
        # self.sched_vnet = build_lr_scheduler(self.optim_vnet, cfg.OPTIM_META)
        self.register_model("model", self.model, self.optim, self.sched)
        #self.register_model("meta_model", self.meta_model, self.meta_optim, self.meta_sched)
    
    def gradient_update(self, loss_ce, loss_kl, reg_image, reg_text):

        ctx = self.model.prompt_learner.ctx
        vis_ctx = self.model.vision_prompt_learner.ctx

        grad_ce_t = autograd.grad(loss_ce, ctx, retain_graph=True, create_graph=False)[0].clone()
        grad_kl_t = autograd.grad(loss_kl, ctx, retain_graph=True, create_graph=False)[0].clone()
        grad_ce_i = autograd.grad(loss_ce, vis_ctx, retain_graph=True, create_graph=False)[0].clone()
        grad_kl_i = autograd.grad(loss_kl, vis_ctx, retain_graph=True, create_graph=False)[0].clone()
        grad_reg_t = autograd.grad(reg_text, ctx, retain_graph=True, create_graph=False)[0].clone()
        grad_reg_i = autograd.grad(reg_image, vis_ctx, retain_graph=True, create_graph=False)[0].clone()

        grad_ce_t_norm = grad_ce_t / torch.linalg.norm(grad_ce_t)
        grad_kl_t_norm = grad_kl_t / torch.linalg.norm(grad_kl_t)
        grad_ce_i_norm = grad_ce_i / torch.linalg.norm(grad_ce_i)
        grad_kl_i_norm = grad_kl_i / torch.linalg.norm(grad_kl_i)
        grad_reg_t_norm = grad_reg_t / torch.linalg.norm(grad_reg_t)
        grad_reg_i_norm = grad_reg_i / torch.linalg.norm(grad_reg_i)
    
        angle_t = torch.dot(grad_ce_t_norm.flatten(), grad_reg_t_norm.flatten())
        angle_i = torch.dot(grad_ce_i_norm.flatten(), grad_reg_i_norm.flatten())
        if angle_t > 0:
            ctx.grad = grad_ce_t
        else:
            ctx.grad = grad_reg_t
        
        if angle_i > 0:
            vis_ctx.grad = grad_ce_i
        else:
            vis_ctx.grad = grad_ce_i + grad_reg_i
            
        #ctx.grad = grad_ce_t - 1.0 * torch.dot(grad_ce_t.flatten(), grad_reg_t_norm.flatten()) * grad_reg_t_norm

        # if angle_t < 0:
        #     ctx.grad = grad_ce_t - 1.0 * torch.dot(grad_ce_t.flatten(), grad_kl_t_norm.flatten()) * grad_kl_t_norm
        # else:
        #     ctx.grad = grad_ce_t
        # if angle_i < 0:
        #     vis_ctx.grad = grad_ce_i - 1.0 * torch.dot(grad_ce_i.flatten(), grad_kl_i_norm.flatten()) * grad_kl_i_norm
        # else:
        #     vis_ctx.grad = grad_ce_i


    def forward_backward(self, batch):
        #image, image2, label = self.parse_batch_train_2(batch)
        image, label = self.parse_batch_train(batch)

        #lr=self.get_current_lr()
        # logits, kl_loss, reg_image, reg_text = self.model(image)
        # loss_ce = F.cross_entropy(logits, label)
        # loss = loss_ce + reg_image + reg_text

        # self.model_backward_and_update(loss)
        logits, kl_loss, reg_image, reg_text = self.model(image)
        loss_ce = F.cross_entropy(logits, label)

        self.optim.zero_grad()
        #loss_ce.backward()
        self.gradient_update(loss_ce, kl_loss, reg_image, reg_text)
        self.optim.step()

        loss = torch.tensor(0.0)
        unique_label = torch.unique(label)
        maml_ = maml.MAML(self.model, lr=self.adapt_lr, first_order=self.fast_adaptation)
        if len(unique_label) != 1:
            qry_l = unique_label[torch.randperm(len(unique_label))][0]
            qry_ids = torch.where(label==qry_l)[0]
            sup_ids = torch.where(label!=qry_l)[0]
            x_sup, y_sup = image[sup_ids], label[sup_ids]
            x_qry, y_qry = image[qry_ids], label[qry_ids]

            b_size = x_qry.size(0)
            lam = torch.distributions.beta.Beta(0.5, 0.5).sample((b_size,)).to(image.device)
            mix_ids = torch.randint(x_sup.size(0), (x_qry.size(0),))
            task_model = maml_.clone(allow_nograd=True)

            # adaptation_logits_ce, kl_loss, adaptation_rag_image, adaptation_rag_text = self.model(x_sup)
            # adaptation_loss_ce = F.cross_entropy(adaptation_logits_ce, y_sup)
            # task_model.adapt(adaptation_loss_ce, kl_loss, adaptation_rag_image, adaptation_rag_text, allow_nograd=True, grad_func=None, allow_unused=True)
            loss_ce, kl_loss, rag_image, rag_text = task_model.forward_mix(x_qry, y_qry, x_sup, y_sup, mix_ids, lam=lam)
            #task_model.adapt(loss_ce, kl_loss, rag_image, rag_text, allow_nograd=True, grad_func=None, allow_unused=True)
            #loss = loss2_ce + rag_image + rag_text
            self.optim.zero_grad()
            #self.gradient_update(loss_ce, kl_loss, rag_image, rag_text)
            loss_ce.backward()
            self.optim.step()
        # loss = torch.tensor(0.0)
        # unique_label = torch.unique(label)
        # maml_ = maml.MAML(self.model, lr=self.adapt_lr, first_order=self.fast_adaptation)
        # if len(unique_label) != 1:
        #     qry_l = unique_label[torch.randperm(len(unique_label))][0]
        #     qry_ids = torch.where(label==qry_l)[0]
        #     sup_ids = torch.where(label!=qry_l)[0]
        #     x_sup, y_sup = image[sup_ids], label[sup_ids]
        #     x_qry, y_qry = image[qry_ids], label[qry_ids]

        #     b_size = x_qry.size(0)
        #     lam = torch.distributions.beta.Beta(0.5, 0.5).sample((b_size,)).to(image.device)
        #     mix_ids = torch.randint(x_sup.size(0), (x_qry.size(0),))

        #     task_model = maml_.clone(allow_nograd=True)
        #     adaptation_logits_ce, kl_loss, adaptation_rag_image, adaptation_rag_text = task_model(x_sup)
        #     adaptation_loss_ce = F.cross_entropy(adaptation_logits_ce, y_sup)
            
        #     task_model.adapt(adaptation_loss_ce, kl_loss, adaptation_rag_image, adaptation_rag_text, allow_nograd=True, grad_func=None, allow_unused=True)
        #     loss_ce, kl_loss, rag_image, rag_text = task_model.forward_mix(x_qry, y_qry, x_sup, y_sup, mix_ids, lam=lam) 

        #     #loss = loss2_ce + rag_image + rag_text
        #     loss = loss_ce
        #     self.optim.zero_grad()
        #     #self.gradient_update(loss2_ce, kl_loss, rag_image, rag_text)
        #     loss.backward()
        #     self.optim.step()

        loss_summary = {
            "loss": loss_ce.item(),
            #"acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label
    
    def parse_batch_train_2(self, batch):
        input = batch["img"]
        image1, image2 = input[0], input[1]
        label = batch["label"]
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        label = label.to(self.device)
        return image1, image2, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix2" in state_dict:
                del state_dict["token_prefix2"]
            if "token_suffix2" in state_dict:
                del state_dict["token_suffix2"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
