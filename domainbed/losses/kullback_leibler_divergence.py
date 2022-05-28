import torch
import torch.nn.functional as F

from losses.importance_loss import importance

def kl_divergence(x, num_experts):
    p = (importance(x) + 1e-10) / x.shape[0]
    q = torch.full(size=(num_experts, ), fill_value= 1.0 / num_experts, device=p.device)
    divergence = torch.sum(p * torch.log(p / q))
    return divergence
