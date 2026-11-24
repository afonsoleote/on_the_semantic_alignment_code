import torch

from torch import nn


class PermuteLayer(nn.Module):
    def __init__(self, permutation):
        super(PermuteLayer, self).__init__()
        self.permutation = permutation

    def forward(self, x):
        return torch.permute(x, self.permutation)
