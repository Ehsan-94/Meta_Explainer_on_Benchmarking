import torch
import torch.nn as nn
import torch_geometric
class IdenticalPooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch=None):
        return x