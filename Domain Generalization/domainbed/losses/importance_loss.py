def importance(x):
    return x.sum(dim=0)


def squared_coefficient_of_variation(x):
    x = x.float()
    cv_squared = x.var() / (x.mean()**2 + 1e-10)
    return cv_squared

# Maximum loss = num_experts
def importance_loss(x):
    imp = importance(x)
    cv_squared = squared_coefficient_of_variation(imp)
    return cv_squared