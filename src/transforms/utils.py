import torch

class LogStable:
    def __init__(self, eps=1e-8):
        self.eps = eps
    
    def __call__(self, input_tensor: torch.Tensor):
        return torch.log(input_tensor + self.eps)