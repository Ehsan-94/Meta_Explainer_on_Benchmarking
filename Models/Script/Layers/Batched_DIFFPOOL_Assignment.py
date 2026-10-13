import sys 
py_path = '/data/cs.aau.dk/ey33jw/Explainability_Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import (Batched_GraphSage_Layer as batched_graphsage_layer)


class Batched_DiffPool_Assignment_Layer(nn.Module):
    def __init__(self, input_dim, assignment_num_block_layers, hid_dim, assigned_dim, bias, normalize_graphsage, dropout,
                 aggregation, concat, act_fun, use_batch_norm=False):
        super(Batched_DiffPool_Assignment_Layer, self).__init__()
        self.input_dim = input_dim
        self.assignment_num_block_layers = assignment_num_block_layers
        self.hid_dim = hid_dim
        self.assigned_dim = assigned_dim
        self.assignment_prediction_layer_input_dim = assigned_dim
        #########################################################.  General Parameters
        self.bias = bias
        self.normalize_graphsage = normalize_graphsage
        self.dropout = dropout
        self.aggregation = aggregation
        activations = {"ReLu": F.relu, "eLu": F.elu, "tanh": torch.tanh,}
        if act_fun not in activations:
            raise ValueError(f"Unsupported activation: {act_fun}")

        self.act_fun = activations[act_fun]
        self.concat=concat

        self.diffPool_assignment = nn.ModuleList()

        self.diffPool_assignment.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.input_dim, output_dim=self.hid_dim,
                                                                                            bias=self.bias, normalize_graphsage=self.normalize_graphsage,
                                                                                            dropout=self.dropout, aggregation=self.aggregation, concat=self.concat,
                                                                                            use_batch_norm=use_batch_norm))

        for i in range(assignment_num_block_layers):
            self.diffPool_assignment.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.hid_dim, output_dim=self.hid_dim,
                                                                                                bias=self.bias, normalize_graphsage=self.normalize_graphsage,
                                                                                                dropout=self.dropout, aggregation=self.aggregation, concat=self.concat,
                                                                                                use_batch_norm=use_batch_norm))

        self.diffPool_assignment.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.hid_dim, output_dim=self.assigned_dim,
                                                                                            bias=self.bias, normalize_graphsage=self.normalize_graphsage,
                                                                                            dropout=self.dropout, aggregation=self.aggregation, concat=self.concat,
                                                                                            use_batch_norm=use_batch_norm))

    def forward(self, input_tensor, tilda_adjacency_matrix, node_mask=None):
        for layer in self.diffPool_assignment[:-1]:
            input_tensor = self.act_fun(layer(input_tensor, tilda_adjacency_matrix, node_mask=node_mask,))
        assignment_logits = self.diffPool_assignment[-1](input_tensor, tilda_adjacency_matrix, node_mask=node_mask,)

        return F.softmax(assignment_logits, dim=-1)
