#!/usr/bin/env python3

import traceback
from torch.autograd import grad
import torch.nn as nn
import torch
from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module

## Original ###
class VNet(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        vision_ctx_dim = clip_model.visual.conv1.weight.size(0)*cfg.TRAINER.META.N_PRO
        text_ctx_dim = clip_model.ln_final.weight.shape[0]*cfg.TRAINER.META.N_CTX

        self.linear_vision_gamma = nn.Sequential(nn.Linear(vision_ctx_dim*2, vision_ctx_dim//8, bias=False),nn.Linear(vision_ctx_dim//8, vision_ctx_dim//8, bias=False)).type(self.dtype)
        self.linear_text_gamma = nn.Sequential(nn.Linear(text_ctx_dim*2, text_ctx_dim//8, bias=False),nn.Linear(text_ctx_dim//8, text_ctx_dim//8, bias=False)).type(self.dtype)
        
        
    def forward(self, gradients1, gradients2, gradients3, param):
            
        d_0, d_1, d_2 = gradients1.size()
        if gradients1.shape[-1] == 512:
            linear_gamma = self.linear_text_gamma
        else:
            linear_gamma = self.linear_vision_gamma
        if gradients2 == None:
            input_gradients = torch.cat((gradients1, gradients3), 0)
            gamma_t = torch.sigmoid(linear_gamma(input_gradients.reshape(d_0, 1,-1))).repeat_interleave(8,-1).reshape(d_0, d_1, d_2)
            changed_gradients = gamma_t*(gradients3)*2
            changed_gradients = gradients1 + changed_gradients
            
        elif gradients3 == None:
            input_gradients = torch.cat((gradients1, gradients2), 0)
            gamma_t = torch.sigmoid(linear_gamma(input_gradients.reshape(d_0, 1,-1))).repeat_interleave(8,-1).reshape(d_0, d_1, d_2)
            changed_gradients = gamma_t*(gradients2)*2
            changed_gradients = gradients1 + changed_gradients
        else:
            raise NotImplemented
        
        # beta_t = torch.sigmoid(linear_beta(gradients.reshape(1, -1))).reshape(d_1, d_2)

        return changed_gradients

def maml_update(model, lr, grads=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

    **Description**

    Performs a MAML update on model using grads and lr.
    The function re-routes the Python object, thus avoiding in-place
    operations.

    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.

    **Arguments**

    * **model** (Module) - The model to update.
    * **lr** (float) - The learning rate used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.

    **Example**
    ~~~python
    maml = l2l.algorithms.MAML(Model(), lr=0.1)
    model = maml.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    maml_update(model, lr=0.1, grads)
    ~~~
    """
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    return update_module(model)


class MAML(BaseLearner):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

    **Description**

    High-level implementation of *Model-Agnostic Meta-Learning*.

    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt()`
    methods.

    For the first-order version of MAML (i.e. FOMAML), set the `first_order` flag to `True`
    upon initialization.

    **Arguments**

    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Fast adaptation learning rate.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order
        approximation of MAML. (FOMAML)
    * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to `allow_nograd`.
    * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
        parameters that have `requires_grad = False`.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."

    **Example**

    ~~~python
    linear = l2l.algorithms.MAML(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(self,
                 model,
                 lr,
                 first_order=False,
                 allow_unused=None,
                 allow_nograd=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self,
              loss_ce,
              loss_kl,
              reg_image,
              reg_text,
              first_order=None,
              allow_unused=None,
              allow_nograd=None,
              grad_func=None):
        """
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        # Compute relevant gradients
        #diff_params = [p for p in self.module.parameters() if p.requires_grad]
        # for p in self.module.parameters():
        #     if p.requires_grad is False:
        #         p.grad = None

        ctx = self.module.prompt_learner.ctx
        vis_ctx = self.module.vision_prompt_learner.ctx

        grad_ce_t = grad(loss_ce, ctx, retain_graph=True, create_graph=False)[0]
        grad_kl_t = grad(loss_kl, ctx, retain_graph=True, create_graph=True)[0]
        grad_ce_i = grad(loss_ce, vis_ctx, retain_graph=True, create_graph=False)[0]
        grad_kl_i = grad(loss_kl, vis_ctx, retain_graph=True, create_graph=False)[0]
        grad_reg_t = grad(reg_text, ctx, retain_graph=True, create_graph=False)[0].clone()
        grad_reg_i = grad(reg_image, vis_ctx, retain_graph=True, create_graph=False)[0].clone()

        grad_ce_t_norm = grad_ce_t / torch.linalg.norm(grad_ce_t)
        grad_kl_t_norm = grad_kl_t / torch.linalg.norm(grad_kl_t)
        grad_ce_i_norm = grad_ce_i / torch.linalg.norm(grad_ce_i)
        grad_kl_i_norm = grad_kl_i / torch.linalg.norm(grad_kl_i)
        grad_reg_t_norm = grad_reg_t / torch.linalg.norm(grad_reg_t)
        grad_reg_i_norm = grad_reg_i / torch.linalg.norm(grad_reg_i)
        
        angle_t = torch.dot(grad_ce_t_norm.flatten(), grad_reg_t_norm.flatten())
        angle_i = torch.dot(grad_ce_i_norm.flatten(), grad_reg_i_norm.flatten())
        if angle_t > 0:
            ctx_grad = grad_ce_t
        else:
            ctx_grad = grad_reg_t
        
        if angle_i > 0:
            vis_ctx_grad = grad_ce_i
        else:
            vis_ctx_grad = grad_ce_i + grad_reg_i

        gradients = []
        # grad_counter = 0
        # Handles gradients for non-differentiable parameters
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                if "vision_prompt_learner.ctx" in name:
                    gradient = vis_ctx_grad
                elif "prompt_learner.ctx" in name:
                    gradient = ctx_grad
                else:
                    gradient = None
            else:
                gradient = None
            gradients.append(gradient)

        self.module = maml_update(self.module, self.lr, gradients)
    
    def adapt_2(self,
              loss1,
              loss2,
              first_order=None,
              allow_unused=None,
              allow_nograd=None,
              grad_func=None):
        """
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]

            grad_params1 = grad(loss1,
                               diff_params,
                               retain_graph=True,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            grad_params2 = grad(loss2,
                               diff_params,
                               retain_graph=True,
                               create_graph=second_order,
                               allow_unused=allow_unused)

            gradients = []
            grad_counter = 0
            # Handles gradients for non-differentiable parameters
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    gradient1 = grad_params1[grad_counter]
                    gradient2 = grad_params2[grad_counter]

                    if grad_func:
                        gradient = grad_func.forward_2(gradient1, gradient2, name)
    
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss,
                                 self.module.parameters(),
                                 retain_graph=second_order,
                                 create_graph=second_order,
                                 allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        # Update the module
        self.module = maml_update(self.module, self.lr, gradients)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Returns a `MAML`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.

        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().

        **Arguments**

        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)