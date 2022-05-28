import torch

def variance_loss(x):
    mean_weightings = x.mean(dim=1).mean(dim=0)
    std = torch.std(x, dim=1)
    coefficients_of_variation = std / mean_weightings
    var_loss = -(coefficients_of_variation.sum(dim=0) / x.shape[0])
    return var_loss
