from torch_geometric.utils import dropout
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch_geometric
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
import sys 
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_add_pool


import torch
import torch.nn as nn
import torch.nn.functional as F


class GIN_MLPs(nn.Module):
    def __init__(self, num_slp_layers, mlp_input_dim, mlp_hid_dim, mlp_act_fun, bias,):
        super().__init__()
        for name, value in (("num_slp_layers", num_slp_layers), ("mlp_input_dim", mlp_input_dim), ("mlp_hid_dim", mlp_hid_dim),):
            if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer.")
        activations = {"ReLu": F.relu, "eLu": F.elu, "tanh": torch.tanh,}
        if mlp_act_fun not in activations:
            raise ValueError(f"Unsupported activation: {mlp_act_fun}")
        if not isinstance(bias, bool):
            raise ValueError("bias must be True or False.")

        self.mlp_input_dim = mlp_input_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.num_slp_layers = num_slp_layers
        self.bias = bias
        self.mlp_act_fun = activations[mlp_act_fun]

        self.gin_mlp_layers = nn.ModuleList()
        self.gin_batch_normalization = nn.ModuleList()

        for index in range(num_slp_layers):
            input_dim = mlp_input_dim if index == 0 else mlp_hid_dim
            self.gin_mlp_layers.append(nn.Linear(input_dim, mlp_hid_dim, bias=bias))
            self.gin_batch_normalization.append(nn.BatchNorm1d(mlp_hid_dim))

    def forward(self, h):
        if h.size(0) == 0:
            raise ValueError("GIN MLP received no nodes.")
        if h.ndim != 2 or h.size(1) != self.mlp_input_dim:
            raise ValueError(f"Expected node features shaped [num_nodes, "f"{self.mlp_input_dim}].")
        if self.training and h.size(0) < 2:
            raise ValueError("Training BatchNorm requires at least two nodes per batch.")
        for linear, batch_norm in zip(self.gin_mlp_layers, self.gin_batch_normalization,):
            h = self.mlp_act_fun(batch_norm(linear(h)))

        return h
        
#mlp_example = GIN_MLPs(num_slp_layers=2, mlp_input_dim=7, mlp_hid_dim=7, mlp_output_dim=7, mlp_act_fun="ReLu", bias=True)
#x = torch.rand(20,7)
#y = mlp_example(x)
#print(y.size())