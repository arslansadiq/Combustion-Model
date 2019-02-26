import torch.nn.functional as F


def mean_squared_error(output, target):
    return F.mse_loss(output, target)
