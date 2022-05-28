# Modified from https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
import copy
import warnings
import math
from copy import deepcopy

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler

from domainbed.ur_networks import URResNet


class AveragedModel(Module):
    def filter(self, model):
        if isinstance(model, AveragedModel):
            # prevent nested averagedmodel
            model = model.module

        if hasattr(model, "get_forward_model"):
            model = model.get_forward_model()
            # URERM models use URNetwork, which manages features internally.
            for m in model.modules():
                if isinstance(m, URResNet):
                    m.clear_features()

        return model

    def __init__(self, model, device=None, avg_fn=None, rm_optimizer=False):
        super(AveragedModel, self).__init__()
        self.start_step = -1
        self.end_step = -1
        model = self.filter(model)
        self.module = deepcopy(model)
        self.module.zero_grad(set_to_none=True)
        if rm_optimizer:
            for k, v in vars(self.module).items():
                if isinstance(v, torch.optim.Optimizer):
                    setattr(self.module, k, None)
                    #  print(f"{k} -> {getattr(self.module, k)}")
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        #  return self.predict(*args, **kwargs)
        return self.module(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.module.predict(*args, **kwargs)

    @property
    def network(self):
        return self.module.network

    def update_parameters(self, model, step=None, start_step=None, end_step=None):
        model = self.filter(model)
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1

        if step is not None:
            if start_step is None:
                start_step = step
            if end_step is None:
                end_step = step

        if start_step is not None:
            if self.n_averaged == 1:
                self.start_step = start_step

        if end_step is not None:
            self.end_step = end_step

    def clone(self):
        clone = copy.deepcopy(self.module)
        clone.optimizer = clone.new_optimizer(clone.network.parameters())
        return clone


@torch.no_grad()
def update_bn(iterator, model, n_steps, device='cuda'):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    #  for input in loader:
    for i in range(n_steps):
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(iterator)
        x = torch.cat([
            dic["x"] for dic in batches_dictlist
        ])
        x = x.to(device)

        model(x)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class SWALR(_LRScheduler):
    r"""Anneals the learning rate in each parameter group to a fixed value.
    This learning rate scheduler is meant to be used with Stochastic Weight 
    Averaging (SWA) method (see `torch.optim.swa_utils.AveragedModel`).
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer
        swa_lrs (float or list): the learning rate value for all param groups
            together or separately for each group.
        annealing_epochs (int): number of epochs in the annealing phase 
            (default: 10)
        annealing_strategy (str): "cos" or "linear"; specifies the annealing 
            strategy: "cos" for cosine annealing, "linear" for linear annealing
            (default: "cos")
        last_epoch (int): the index of the last epoch (default: 'cos')
    The :class:`SWALR` scheduler is can be used together with other
    schedulers to switch to a constant learning rate late in the training 
    as in the example below.
    Example:
        >>> loader, optimizer, model = ...
        >>> lr_lambda = lambda epoch: 0.9
        >>> scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
        >>>        lr_lambda=lr_lambda)
        >>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, 
        >>>        anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05)
        >>> swa_start = 160
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    """
    def __init__(self, optimizer, swa_lr, anneal_epochs=10, anneal_strategy='cos', last_epoch=-1):
        swa_lrs = self._format_param(optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups):
            group['swa_lr'] = swa_lr
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', "
                             "instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._cosine_anneal
        elif anneal_strategy == 'linear':
            self.anneal_func = self._linear_anneal
        if not isinstance(anneal_epochs, int) or anneal_epochs < 1:
            raise ValueError("anneal_epochs must be a positive integer, got {}".format(
                             anneal_epochs)) 
        self.anneal_epochs = anneal_epochs

        super(SWALR, self).__init__(optimizer, last_epoch)

    @staticmethod
    def _format_param(optimizer, swa_lrs):
        if isinstance(swa_lrs, (list, tuple)):
            if len(swa_lrs) != len(optimizer.param_groups):
                raise ValueError("swa_lr must have the same length as "
                                 "optimizer.param_groups: swa_lr has {}, "
                                 "optimizer.param_groups has {}".format(
                                     len(swa_lrs), len(optimizer.param_groups)))
            return swa_lrs
        else:
            return [swa_lrs] * len(optimizer.param_groups)

    @staticmethod
    def _linear_anneal(t):
        return t

    @staticmethod
    def _cosine_anneal(t):
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        step = self._step_count - 1
        prev_t = max(0, min(1, (step - 1) / self.anneal_epochs))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [self._get_initial_lr(group['lr'], group['swa_lr'], prev_alpha)
                    for group in self.optimizer.param_groups]
        t = max(0, min(1, step / self.anneal_epochs))
        alpha = self.anneal_func(t)
        return [group['swa_lr'] * alpha + lr * (1 - alpha) 
                for group, lr in zip(self.optimizer.param_groups, prev_lrs)]
