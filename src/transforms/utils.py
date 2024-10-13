import torch
from torch import nn


class LogStable(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input_tensor: torch.Tensor):
        return torch.log(input_tensor + self.eps)


class Normalize0d(nn.Module):
    """
    Normalize tensor to have zero mean and std equal to one
    over all dimesions
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, input_tensor: torch.Tensor):
        return (input_tensor - self.mean) / self.std
