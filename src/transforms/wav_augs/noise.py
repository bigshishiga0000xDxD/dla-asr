from torch import distributions, nn, Tensor


class GaussianNoiser(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.noiser = distributions.Normal(0, std)

    def __call__(self, data: Tensor):
        return data + self.noiser.sample(data.size())
