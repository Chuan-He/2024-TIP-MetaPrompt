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

from copy import deepcopy



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
        # self.ctx = nn.ParameterList([nn.Parameter(torch.empty(n_pro, ctx_dim)) for _ in range(self.layer_p)])
        # for single_para in self.ctx:
        #     nn.init.normal_(single_para, std=0.02)

        
    def forward(self, x, ctx=None):
        if ctx == None:
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
        # ctx = []
        # for vector in ctx_t:
        #     ctx.append(vector)
        # ctx = torch.stack(ctx)
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
        prompt = clip.tokenize(ctx_init).to('cuda:0')
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors_0 = embedding[0, 1: 1 + n_ctx, :]
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

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + n_ctx :, :]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self, ctx = None):
        if ctx == None:
            ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx[:, 0], suffix], dim=1)
        return prompts, ctx


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        for p in clip_model.parameters():
            p.requires_grad = False

        # t = {
        #     'StanfordCars':'a photo of a {}',
        #     'Caltech101':'a photo of a {}',
        #     'OxfordFlowers':'a photo of a {}, a type of flower',
        #     'EuroSAT':'a centered satellite photo of {}',
        #     'DescribableTextures':'{} texture',
        #     'Food101':'a photo of a {}, a type of food',
        #     'FGVCAircraft':'a photo of a {}, a type of aircraft',
        #     'OxfordPets':'a photo of a {}, a type of pet',
        #     'UCF101':'a photo of a person doing {}',
        #     'SUN397':'a photo of a {}',
        #     'PACS':'a photo of a {}',
        #     'OfficeHomeFS':'a photo of a {}',
        #     'PACSFS':'a photo of a {}',
        #     'VLCSFS':'a photo of a {}',
        #     'DomainNetFS':'a photo of a {}',
        #     'TERRAFS':'a photo of a {}'
        # }

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
            # templates = [
            #     'itap of a {}',
            #     'a bad photo of the {}',
            #     'a origami {}',
            #     'a photo of the large {}',
            #     'a {} in a video game',
            #     'art of the {}',
            #     'a photo of the small {}',
            #     'a photo of a {}'
            # ]
            templates = [
                "a bad photo of a {}.",
                "a photo of many {}.",
                "a sculpture of a {}.",
                "a photo of the hard to see {}.",
                "a low resolution photo of the {}.",
                "a rendering of a {}.",
                "graffiti of a {}.",
                "a bad photo of the {}.",
                "a cropped photo of the {}.",
                "a tattoo of a {}.",
                "the embroidered {}.",
                "a photo of a hard to see {}.",
                "a bright photo of a {}.",
                "a photo of a clean {}.",
                "a photo of a dirty {}.",
                "a dark photo of the {}.",
                "a drawing of a {}.",
                "a photo of my {}.",
                "the plastic {}.",
                "a photo of the cool {}.",
                "a close-up photo of a {}.",
                "a black and white photo of the {}.",
                "a painting of the {}.",
                "a painting of a {}.",
                "a pixelated photo of the {}.",
                "a sculpture of the {}.",
                "a bright photo of the {}.",
                "a cropped photo of a {}.",
                "a plastic {}.",
                "a photo of the dirty {}.",
                "a jpeg corrupted photo of a {}.",
                "a blurry photo of the {}.",
                "a photo of the {}.",
                "a good photo of the {}.",
                "a rendering of the {}.",
                "a {} in a video game.",
                "a photo of one {}.",
                "a doodle of a {}.",
                "a close-up photo of the {}.",
                "a photo of a {}.",
                "the origami {}.",
                "the {} in a video game.",
                "a sketch of a {}.",
                "a doodle of the {}.",
                "a origami {}.",
                "a low resolution photo of a {}.",
                "the toy {}.",
                "a rendition of the {}.",
                "a photo of the clean {}.",
                "a photo of a large {}.",
                "a rendition of a {}.",
                "a photo of a nice {}.",
                "a photo of a weird {}.",
                "a blurry photo of a {}.",
                "a cartoon {}.",
                "art of a {}.",
                "a sketch of the {}.",
                "a embroidered {}.",
                "a pixelated photo of a {}.",
                "itap of the {}.",
                "a jpeg corrupted photo of the {}.",
                "a good photo of a {}.",
                "a plushie {}.",
                "a photo of the nice {}.",
                "a photo of the small {}.",
                "a photo of the weird {}.",
                "the cartoon {}.",
                "art of the {}.",
                "a drawing of the {}.",
                "a photo of the large {}.",
                "a black and white photo of a {}.",
                "the plushie {}.",
                "a dark photo of a {}.",
                "itap of a {}.",
                "graffiti of the {}.",
                "a toy {}.",
                "itap of my {}.",
                "a photo of a cool {}.",
                "a photo of a small {}.",
                "a tattoo of the {}.",
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

        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts) 
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            self.text_features_zs = torch.stack(zeroshot_weights, dim=1).cuda()
        
        

    def forward(self, image, label_idx = None, ctx = None, vis_ctx = None):
        if label_idx != None:
            text_features_zs = self.text_features_zs[:, label_idx]
        else:
            text_features_zs = self.text_features_zs

        image_features_zs = self.model.encode_image(image.type(self.dtype))
        image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)

        x, ctx_v = self.vision_prompt_learner(image, vis_ctx)
        image_features = self.image_encoder(x, ctx_v)
        prompts, ctx_t = self.prompt_learner(ctx)
        
        tokenized_prompts = self.tokenized_prompts
        if label_idx != None:
            prompts = prompts[label_idx]
            ctx_t = ctx_t[label_idx]
            tokenized_prompts = tokenized_prompts[label_idx]

        text_features = self.text_encoder(prompts, tokenized_prompts, ctx_t.half())

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
        #logits_i = logit_scale * (image_features @ text_features_zs.t())
        #logits_t = logit_scale * (image_features_zs @ text_features.t())
        logits = logit_scale * (image_features @ text_features.t())
        #logits = 0.5*(logits_i + logits_t)
        rag_text = F.l1_loss(text_features, text_features_zs.t().cuda(),
                                    reduction='mean') * 25
        rag_image = F.l1_loss(image_features, image_features_zs.cuda(),
                                    reduction='mean') * 10
        if self.training:
            return logits, rag_image, rag_text
        else:
            return logits

class  LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, total_num_inner_loop_steps, layers):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(LSLRGradientDescentLearningRule, self).__init__()

        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.layers = layers

        # self.alpha_list = nn.ParameterList()
        # self.beta_list = nn.ParameterList()
        # # self.weight_list = nn.ParameterList()

        # for i in range(layers):
            
        #     self.beta_list.append(nn.Parameter(
        #         data=torch.ones(self.total_num_inner_loop_steps),
        #         requires_grad=True))

        #     # per-step per-layer meta-learnable learning rate bias term (for more stable training and better performance by 2~3%)
        #     self.alpha_list.append(nn.Parameter(
        #         data=torch.ones(self.total_num_inner_loop_steps),
        #         requires_grad=True))
 
    def update_params(self, ctx, grads_wrt_params_a, grads_wrt_params_b, generated_alpha, generated_beta, lr, num_step):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """

        new_ctx = [] 
        for i in range(self.layers):
            new_ctx.append(ctx[i,:,:] - generated_beta[i] * lr * grads_wrt_params_b[i,:,:] - \
                                             generated_alpha[i] * lr * grads_wrt_params_a[i,:,:])
        new_ctx = torch.stack(new_ctx)
        return new_ctx
   

class MAMLFewShotClassifier(nn.Module):
    def __init__(self, cfg, model, classnames, device, num_layers, meta_step):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()

        self.model = model
        self.classnames = classnames
        self.meta_step = meta_step
        self.layers = num_layers
        self.device = device

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device, total_num_inner_loop_steps=cfg.TRAINER.META.META_STEP, layers=num_layers)
        # self.attenuator = nn.Sequential(
        #         nn.Linear(num_layers, num_layers),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(num_layers, num_layers),
        #         nn.Sigmoid()
        #     ).to(device=self.device)
        
        input_dim = num_layers*2
        self.regularizer_t = nn.Linear(input_dim, input_dim).to(device=self.device)
        self.regularizer_i = nn.Linear(input_dim, input_dim).to(device=self.device)

        # self.regularizer_t = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(input_dim, input_dim)
        # ).to(device=self.device).half()

        # self.regularizer_i = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(input_dim, input_dim)
        # ).to(device=self.device).half()

        
    def forward(self, image, label, lr):

        label_lst = label.tolist()
        label_set = list(range(len(self.classnames)))
        random.shuffle(label_set)

        N = 2
        l = len(label_set)
        m = (l-1) // N + 1
        cls_li = [label_set[i*m:(i+1)*m] for i in range(N)]
        relabeler = [{y: y_new for y_new, y in enumerate(cls_li[i])} for i in range(N)]
        index = [[j for j, l in enumerate(label_lst) if l in cls_li[i]] for i in range(N)]
        label_li = [torch.tensor([relabeler[i][j] for j in label[index[i]].tolist()]).cuda() for i in range(N)]
        image_li = [image[index[i]].cuda() for i in range(N)]

        total_losses = []
        # text_prompt = deepcopy(self.model.prompt_learner.ctx)
        # vision_prompt = deepcopy(self.model.vision_prompt_learner.ctx)

        for i in range(N):

            image = torch.cat([image_li[j] for j in range(N) if j != i])
            label = torch.cat([label_li[j] for j in range(N) if j != i])
            cls = [item for j in range(N) if j != i for item in cls_li[j]]
            
            if len(cls_li[i]) < 2 or len(image_li[i]) == 0 or len(image) == 0:
                continue

            task_losses = []
            ctx = self.model.prompt_learner.ctx.clone()
            vis_ctx = self.model.vision_prompt_learner.ctx.clone()
            for k in range(self.meta_step):

                logits_mix_support, reg_image, reg_text = self.model(image, cls, ctx, vis_ctx)
                support_loss_mix = F.cross_entropy(logits_mix_support, label)

                grads_wrt_ctx = autograd.grad(support_loss_mix, ctx, retain_graph=True)[0]
                reg_wrt_ctx = autograd.grad(reg_text, ctx, retain_graph=True, create_graph=True)[0]

                per_step_task_embedding = []
                for l in range(self.layers):
                    per_step_task_embedding.append(grads_wrt_ctx[l,:,:].mean().float())
                for l in range(self.layers):
                    per_step_task_embedding.append(reg_wrt_ctx[l,:,:].mean().float())
                per_step_task_embedding = torch.stack(per_step_task_embedding)
                generated_params = self.regularizer_t(per_step_task_embedding)
                generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=self.layers)

                ctx = self.inner_loop_optimizer.update_params(ctx=ctx,
                                            grads_wrt_params_a=grads_wrt_ctx,
                                            grads_wrt_params_b=reg_wrt_ctx,
                                            generated_alpha=generated_alpha,
                                            generated_beta=generated_beta,
                                            lr = lr,
                                            num_step=k)

                grads_wrt_vis = autograd.grad(support_loss_mix, vis_ctx, retain_graph=True)[0]
                reg_wrt_vis = autograd.grad(reg_image, vis_ctx, retain_graph=True)[0]

                per_step_task_embedding = []
                for l in range(self.layers):
                    per_step_task_embedding.append(grads_wrt_vis[l,:,:].mean().float())
                for l in range(self.layers):
                    per_step_task_embedding.append(reg_wrt_vis[l,:,:].mean().float())
                per_step_task_embedding = torch.stack(per_step_task_embedding)
                generated_params = self.regularizer_i(per_step_task_embedding)
                generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=self.layers)

                vis_ctx = self.inner_loop_optimizer.update_params(ctx=vis_ctx,
                                            grads_wrt_params_a=grads_wrt_vis,
                                            grads_wrt_params_b=reg_wrt_vis,
                                            generated_alpha=generated_alpha,
                                            generated_beta=generated_beta,
                                            lr = lr,
                                            num_step=k)
                if k == self.meta_step - 1:
                    logits_mix_query, reg_image, reg_text = self.model(image_li[i], cls_li[i], ctx, vis_ctx)
                    query_loss = F.cross_entropy(logits_mix_query, label_li[i])
                    task_losses.append(query_loss)
            
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
        loss = torch.mean(torch.stack(total_losses))
    
        return loss


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

        self.meta_step = cfg.TRAINER.META.META_STEP
        self.num_layers = cfg.TRAINER.META.LAYERS
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.meta_lr = cfg.OPTIM_META.LR
        self.min_lr = cfg.OPTIM_META.WARMUP_MIN_LR
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.meta_model = MAMLFewShotClassifier(cfg, self.model, classnames, self.device, self.num_layers, self.meta_step)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        
        for name, param in self.meta_model.named_parameters():
            if param.requires_grad:
                enabled.add(name)          
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.meta_model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.meta_optim = build_optimizer(model=None, param_groups=[
            {'params':self.model.prompt_learner.parameters()},
            {'params':self.model.vision_prompt_learner.parameters()},
            {'params': self.meta_model.regularizer_t.parameters()},
            {'params':self.meta_model.regularizer_i.parameters()},
        ],optim_cfg=cfg.OPTIM)
        self.meta_sched = build_lr_scheduler(self.meta_optim, cfg.OPTIM)
        # self.meta_optim = optim.Adam([
        #     {'params':self.model.prompt_learner.parameters()},
        #     {'params':self.model.vision_prompt_learner.parameters()},
        #     {'params': self.meta_model.regularizer_t.parameters()},
        #     {'params': self.meta_model.regularizer_i.parameters()},
        # ], lr=0.001, amsgrad=False)
        # self.meta_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.meta_optim, T_max=self.total_epochs, eta_min=self.min_lr)
        # self.meta_optim = build_optimizer(self.meta_model, cfg.OPTIM)
        # self.meta_sched = build_lr_scheduler(self.meta_optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        #self.register_model("meta_model", self.meta_model, self.meta_optim, self.meta_sched)
        # self.scaler = GradScaler() if cfg.TRAINER.META.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1: 
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        logits_ce, reg_image, reg_text = self.model(image)
        loss_ce = F.cross_entropy(logits_ce, label)
        # loss_i = F.cross_entropy(logits_i, label)
        # loss_t = F.cross_entropy(logits_t, label)
        loss = loss_ce + reg_image + reg_text

        self.model_backward_and_update(loss)

        meta_loss = self.meta_model(image, label, lr=self.get_current_lr())
        
        self.meta_optim.zero_grad()
        meta_loss.backward()
        self.meta_optim.step()

        loss_summary = {
            "loss": loss_ce.item(),
            "acc": compute_accuracy(logits_ce, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

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

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            self._models[name].load_state_dict(state_dict, strict=False)



@TRAINER_REGISTRY.register()
class Meta_DG(TrainerX):

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

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.meta_model = LSLRGradientDescentLearningRule(device=self.device,total_num_inner_loop_steps=5)
        self.num_layers = self.model.prompt_learner.layer_p
        input_dim = self.num_layers*2
        self.regularizer_i = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim)
        ).to(device=self.device)
        self.regularizer_t = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim)
        ).to(device=self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.meta_model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.meta_optim = build_optimizer(self.meta_model, cfg.OPTIM_META)
        self.meta_sched = build_lr_scheduler(self.meta_optim, cfg.OPTIM_META)
        self.register_model("model", self.model, self.optim, self.sched)
        self.meta_step = cfg.TRAINER.META.META_STEP

        self.scaler = GradScaler() if cfg.TRAINER.META.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1: 
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)

        logits_mix, logits_i, logits_t = self.model(image)
        loss = F.cross_entropy(logits_mix, label)
        loss_i = F.cross_entropy(logits_i, label)
        loss_t = F.cross_entropy(logits_t, label)
        loss = loss + loss_i + loss_t
        self.model_backward_and_update(loss)

        dd = domain.tolist()
        N = len(set(dd))
        
        train_dm = [list(range(N)) for _ in range(N)]
        test_dm = [[train_dm[i].pop(i)] for i in range(N)]

        index_train = [[i for i, l in enumerate(dd) if l in train_dm[j]] for j in range(N)]
        index_test = [[i for i, l in enumerate(dd) if l in test_dm[j]] for j in range(N)]

        label_train = [label[index_train[i]] for i in range(N)]
        label_test = [label[index_test[i]] for i in range(N)]

        image_train = [image[index_train[i]] for i in range(N)]
        image_test = [image[index_test[i]] for i in range(N)]

        #self.model.prompt_learner.ctx.requires_grad_(False)
        text_prompt = deepcopy(self.model.prompt_learner.ctx)
        vision_prompt = deepcopy(self.model.vision_prompt_learner.ctx)

        loss_total = []
        for i in range(N): #N tasks
            ctx = self.model.prompt_learner.ctx
            vis_ctx = self.model.vision_prompt_learner.ctx
            if image_train[i].shape[0] == 0 or image_test[i].shape[0] == 0:
                continue
            
            logits_mix, logits_i, logits_t = self.model(image_train[i])
            support_loss_mix = F.cross_entropy(logits_mix, label_train[i])
            support_loss_i = F.cross_entropy(logits_i, label_train[i])
            support_loss_t = F.cross_entropy(logits_t, label_train[i])
            #self.meta_optim.zero_grad()
            support_grads_mix = autograd.grad(support_loss_mix, ctx, create_graph=False)[0]
            support_grads_i = autograd.grad(support_loss_i, ctx, create_graph=False)[0]
            per_step_task_embedding = []
            # for v in list(ctx):
            #     per_step_task_embedding.append(v.mean())
            for g in range(len(support_grads_mix)):
                per_step_task_embedding.append(support_grads_mix[g].mean())
            for g in range(len(support_grads_i)):
                per_step_task_embedding.append(support_grads_i[g].mean())
            per_step_task_embedding = torch.stack(per_step_task_embedding)
            generated_params = self.regularizer_i(per_step_task_embedding)
            generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=self.num_layers)

            generated_alpha_params = []
            generated_beta_params = []
            for g in range(len(generated_alpha)):
                generated_alpha_params.append(generated_alpha[g])
                generated_beta_params.append(generated_beta[g])

            new_ctx = self.meta_model.update_params(weights_list=ctx,
                                                    grads_wrt_params=support_grads_mix,
                                                    grads_wrt_params1=support_grads_i,
                                                    generated_alpha_params=generated_alpha_params,
                                                    generated_beta_params=generated_beta_params,
                                                    num_step=i)
            support_grads_mix = autograd.grad(support_loss_mix, ctx, create_graph=False)[0]
            support_grads_t = autograd.grad(support_loss_t, ctx, create_graph=False)[0]
            per_step_task_embedding = []
            for g in range(len(support_grads_mix)):
                per_step_task_embedding.append(support_grads_mix[g].mean())
            for g in range(len(support_grads_t)):
                per_step_task_embedding.append(support_grads_t[g].mean())
            per_step_task_embedding = torch.stack(per_step_task_embedding)

            generated_params = self.regularizer_t(per_step_task_embedding)
            generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=self.num_layers)

            generated_alpha_params = list(generated_alpha)
            generated_beta_params = list(generated_beta)

            new_vis_ctx = self.meta_model.update_params(weights_list=vis_ctx,
                                                    grads_wrt_params=support_grads_mix,
                                                    grads_wrt_params1=support_grads_t,
                                                    generated_alpha_params=generated_alpha_params,
                                                    generated_beta_params=generated_beta_params,
                                                    num_step=i)
            #self.model_backward_and_update(loss)
            self.model.prompt_learner.ctx.data = new_ctx
            self.model.vision_prompt_learner.ctx.data = new_vis_ctx
            logits_mix, logits_i, logits_t = self.model(image_test[i])
            query_loss = F.cross_entropy(logits_mix, label_test[i])
            loss_total.append(query_loss)
            # query_grads_t = autograd.grad(query_loss, new_ctx, create_graph=False)[0]
            # query_grads_i = autograd.grad(query_loss, new_vis_ctx, create_graph=False)[0]
            # grad_t.append(query_grads_t)
            # grad_i.append(query_grads_i)
            self.model.prompt_learner.ctx.data = text_prompt
            self.model.vision_prompt_learner.ctx.data = vision_prompt
            
        # for grad in grad_i:
        #     self.model.vision_prompt_learner.ctx.data -= self.meta_step*self.get_current_lr()*grad
        # for grad in grad_t:
        #     self.model.prompt_learner.ctx.data -= self.meta_step*self.get_current_lr()*grad
        loss = loss_total
        self.optim.zero_grad()
        self.meta_optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.meta_optim.step()
        
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits_mix, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        domain = batch['domain'].to(self.device)
        return input, label, domain

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

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
