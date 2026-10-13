import sys 
py_path = '/data/cs.aau.dk/ey33jw/Explainability_Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
import torch
import torch.nn as nn
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import (Batched_DIFFPOOL_Embedding as batched_diffpool_embedding,
                                                                 Batched_DIFFPOOL_Assignment as batched_diffpool_assignment,)


class Batched_DiffPool_Layer(nn.Module):
    def __init__(self, embedding_input_dim, embedding_num_block_layers, embedding_hid_dim, embedding_output_dim, assignment_input_dim,
                 assignment_num_block_layers, assignment_hid_dim, assignment_output_dim, concat, bias,
                 dropout_rate, normalize_graphsage, aggregation, act_fun, use_batch_norm=False):
        super(Batched_DiffPool_Layer, self).__init__()
        ################################################.  General Parameters
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.aggregation = aggregation
        self.concat = concat
        self.normalize_graphsage = normalize_graphsage
        self.act_fun=act_fun

        ################################################.  Embedding and Assignment

        self.embedding_input_dim = embedding_input_dim
        self.embedding_num_block_layers = embedding_num_block_layers
        self.embedding_hid_dim = embedding_hid_dim
        self.embedding_output_dim = embedding_output_dim

        self.assignment_input_dim = assignment_input_dim
        self.assignment_num_block_layers = assignment_num_block_layers
        self.assignment_hid_dim = assignment_hid_dim
        self.assignment_output_dim = assignment_output_dim      # new number of nodes

        ################################################.  Embedding
        self.diffpool_embedding = batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer(input_dim=self.embedding_input_dim,
                                                                                              embedding_num_block_layers=self.embedding_num_block_layers,
                                                                                              hid_dim=self.embedding_hid_dim,
                                                                                              embedded_dim=self.embedding_output_dim,
                                                                                              concat=self.concat,bias=self.bias,
                                                                                              normalize_graphsage=self.normalize_graphsage,
                                                                                              dropout=self.dropout_rate,
                                                                                              aggregation=self.aggregation, act_fun=act_fun,
                                                                                              use_batch_norm=use_batch_norm)

        ################################################.  Assignment
        self.diffpool_assignment = batched_diffpool_assignment.Batched_DiffPool_Assignment_Layer(input_dim=self.assignment_input_dim,
                                                                                                 assignment_num_block_layers=self.assignment_num_block_layers,
                                                                                                 hid_dim=self.assignment_hid_dim,
                                                                                                 assigned_dim=self.assignment_output_dim,
                                                                                                 concat=self.concat, bias=self.bias,
                                                                                                 normalize_graphsage=self.normalize_graphsage,
                                                                                                 dropout=self.dropout_rate,
                                                                                                 aggregation=self.aggregation, act_fun=act_fun,
                                                                                                 use_batch_norm=use_batch_norm)
    def forward(self, features, adjacecny, node_mask):
        embedding_output = self.diffpool_embedding(features, adjacecny, node_mask=node_mask)
        assignment_output = self.diffpool_assignment(features, adjacecny, node_mask=node_mask)
        return embedding_output, assignment_output