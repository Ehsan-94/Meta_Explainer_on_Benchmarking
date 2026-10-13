from torch_geometric.utils import dropout
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils.convert import to_scipy_sparse_matrix
#from torch_geometric.utils.train_test_split_edges import torch_geometric
import torch_geometric
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
import sys
import torch_geometric.nn as gnn
from scipy.sparse import csr_matrix
from torch_geometric.datasets import TUDataset
py_path = 'Models/Script/Layers/'
sys.path.insert(0,py_path)
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import Batched_GraphSage_Layer as batched_graphsage_layer
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import Batched_DIFFPOOL_Assignment as batched_diffpool_assignment
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import Batched_DIFFPOOL_Embedding as batched_diffpool_embedding
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import Batched_DIFFPOOL_Layer as batched_diffpool_layer


class GlobalMeanPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)
################################################################################
class IdenticalPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

################################################################################
class DIFFPOOL_Model(nn.Module):
    def __init__(self, embedding_input_dim, embedding_num_block_layers, embedding_hid_dim, new_feature_size, assignment_input_dim,
                 assignment_num_block_layers, assignment_hid_dim, max_number_of_nodes, concat_neighborhood, prediction_hid_layers,
                 num_classes, weight_initializer_index, bias, dropout_rate, normalize_graphsage, aggregation, act_fun,
                 concat_diffpools_outputs, num_pooling, pooling, use_batch_norm, pooling_ratio):

        super(DIFFPOOL_Model, self).__init__()
        positive_integers = {"embedding_input_dim": embedding_input_dim, "embedding_hid_dim": embedding_hid_dim, "new_feature_size": new_feature_size,
                             "assignment_input_dim": assignment_input_dim, "assignment_hid_dim": assignment_hid_dim,
                             "max_number_of_nodes": max_number_of_nodes, "num_classes": num_classes, "num_pooling": num_pooling,}

        for name, value in positive_integers.items():
            if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in (("embedding_num_block_layers", embedding_num_block_layers), ("assignment_num_block_layers", assignment_num_block_layers)):
            if (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer.")
        if embedding_input_dim != assignment_input_dim:
            raise ValueError("Embedding and assignment networks receive the same features; their input dimensions must match.")
        if aggregation not in ("mean", "sum"):
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        if act_fun not in ("ReLu", "eLu", "tanh"):
            raise ValueError(f"Unsupported activation: {act_fun}")
        if pooling not in ("mean", "sum", "max"):
            raise ValueError(f"Unsupported readout: {pooling}")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("Dropout must be between 0 and 1.")
        if weight_initializer_index not in (1, 2, 3):
            raise ValueError("Initializer must be 1, 2, or 3.")
        if any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in prediction_hid_layers):
            raise ValueError("Prediction hidden widths must be positive integers.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.aggregation = aggregation
        self.act_fun = act_fun
        self.weight_initializer_index = weight_initializer_index
        self.concat_diffpools_outputs = concat_diffpools_outputs
        self.num_pooling = num_pooling
        self.concat = concat_neighborhood
        self.pooling = pooling

        self.embedding_input_dim = embedding_input_dim
        self.embedding_num_block_layers = embedding_num_block_layers
        self.embedding_hid_dim = embedding_hid_dim
        self.embedding_output_dim = new_feature_size
        self.normalize_graphsage = normalize_graphsage

        self.assignment_input_dim = assignment_input_dim
        self.assignment_num_block_layers = assignment_num_block_layers
        self.assignment_hid_dim = assignment_hid_dim
        self.max_number_of_nodes = max_number_of_nodes
        if pooling_ratio is None:
            pooling_ratio = 0.10 if num_pooling == 1 else 0.25
        if not 0 < pooling_ratio <= 1:
            raise ValueError("pooling_ratio must be in (0, 1].")
        self.pooling_ratio = float(pooling_ratio)
        self.assignment_output_dim = max(1, int(self.max_number_of_nodes * self.pooling_ratio))

        prediction_input_dim_sum = 0

        ###################################################.    DiffPool Layers
        diffpool_layers = []
        for i in range(self.num_pooling):
            a_new_layer = batched_diffpool_layer.Batched_DiffPool_Layer(embedding_input_dim=self.embedding_input_dim,
                                                                        embedding_num_block_layers=self.embedding_num_block_layers,
                                                                        embedding_hid_dim=self.embedding_hid_dim,
                                                                        embedding_output_dim=self.embedding_output_dim,
                                                                        assignment_input_dim=self.assignment_input_dim,
                                                                        assignment_num_block_layers=self.assignment_num_block_layers,
                                                                        assignment_hid_dim=self.assignment_hid_dim,
                                                                        assignment_output_dim=self.assignment_output_dim,
                                                                        concat=self.concat, bias=self.bias, dropout_rate=self.dropout_rate,
                                                                        normalize_graphsage=self.normalize_graphsage,
                                                                        aggregation=self.aggregation, act_fun=self.act_fun,
                                                                        use_batch_norm=use_batch_norm).to(self.device)
            diffpool_layers.append(a_new_layer)

            self.assignment_output_dim = max(1, int(self.assignment_output_dim * self.pooling_ratio))
            self.embedding_input_dim = self.embedding_output_dim
            self.assignment_input_dim = self.embedding_output_dim
            prediction_input_dim_sum = prediction_input_dim_sum + self.embedding_output_dim
        self.diffpool_layers = nn.Sequential(*diffpool_layers).to(self.device)

        ###################################################.    Last Extra Embedding

        self.last_extra_embedding = batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer(input_dim=self.embedding_output_dim,
                                                                                                embedding_num_block_layers=self.embedding_num_block_layers,
                                                                                                hid_dim=self.embedding_hid_dim, concat=self.concat,
                                                                                                embedded_dim=self.embedding_output_dim, bias=self.bias,
                                                                                                normalize_graphsage=self.normalize_graphsage, dropout=self.dropout_rate,
                                                                                                aggregation=self.aggregation,
                                                                                                use_batch_norm=use_batch_norm,
                                                                                                act_fun=self.act_fun,).to(self.device)
        prediction_input_dim_sum = prediction_input_dim_sum + self.embedding_output_dim

        ###################################################.    Predictions
        self.prediction_input_dim = self.embedding_output_dim
        self.prediction_hid_layers = prediction_hid_layers
        self.num_classes = num_classes

        prediction_layers = []
        if len(self.prediction_hid_layers) == 0:
            if self.concat_diffpools_outputs:
                a_new_layer = nn.Linear(prediction_input_dim_sum, self.num_classes, bias=self.bias,).to(self.device)
                prediction_layers.append(a_new_layer)
                self.prediction_model = nn.Sequential(*prediction_layers).to(self.device)
            else:
                a_new_layer = nn.Linear(self.prediction_input_dim, self.num_classes, bias=self.bias,).to(self.device)
                prediction_layers.append(a_new_layer)
                self.prediction_model = nn.Sequential(*prediction_layers).to(self.device)
        else:
            if self.concat_diffpools_outputs:
                predict_input_dim = prediction_input_dim_sum
                for i in range(len(self.prediction_hid_layers)):
                    a_new_layer = nn.Linear(predict_input_dim, prediction_hid_layers[i], bias=self.bias, ).to(self.device)
                    prediction_layers.append(a_new_layer)
                    predict_input_dim = prediction_hid_layers[i]
                a_new_layer = nn.Linear(predict_input_dim, self.num_classes, bias=self.bias,).to(self.device)
                prediction_layers.append(a_new_layer)
                self.prediction_model = nn.Sequential(*prediction_layers).to(self.device)
            else:
                predict_input_dim = self.prediction_input_dim
                for i in range(len(self.prediction_hid_layers)):
                    a_new_layer = nn.Linear(predict_input_dim, prediction_hid_layers[i], bias=self.bias, ).to(self.device)
                    prediction_layers.append(a_new_layer)
                    predict_input_dim = prediction_hid_layers[i]
                a_new_layer = nn.Linear(predict_input_dim, self.num_classes, bias=self.bias, ).to(self.device)
                prediction_layers.append(a_new_layer)
                self.prediction_model = nn.Sequential(*prediction_layers).to(self.device)

        if act_fun == 'ReLu':
            self.act_fun = F.relu
            # print('ReLu is Selected.')
        elif act_fun == 'eLu':
            self.act_fun = nn.functional.elu
            # print('eLu is Selected.')
        elif act_fun == 'tanh':
            self.act_fun = torch.tanh
            # print('tanh is Selected.')
        self.act_fun_softmax = F.softmax

        mean = 0
        std = 0.1
        self.initialize_weights(weight_initializer_index, bias, mean, std)

    def initialize_weights(self, weight_initializer_index, bias, mean, std):
        # 1. Xavier Normal_.  2. Kaiming Normal_.  3. Uniform (0,0.1std)
        if weight_initializer_index not in (1, 2, 3):
            raise ValueError("Initializer must be 1, 2, or 3.")
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    if weight_initializer_index == 1:
                        nn.init.xavier_normal_(module.weight)
                    elif weight_initializer_index == 2:
                        nn.init.kaiming_normal_(module.weight, nonlinearity="relu",)
                    else:
                        nn.init.normal_(module.weight, mean=mean, std=std)

                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                elif isinstance(module, nn.BatchNorm1d):
                    module.reset_parameters()
        # if weight_initializer_index == 1:                                             #.      1. Xavier Normal_.
        #     for i, module in enumerate(model.children()):
        #         #print(i, module)
        #         if isinstance(module, torch.nn.Sequential):
        #             for j, module_sub in enumerate(module):
        #                 #print("j: ",j,module_sub)
        #                 if isinstance(module_sub, batched_diffpool_layer.Batched_DiffPool_Layer):
        #                     #print(module_sub)
        #                     for party in module_sub.children():
        #                         if isinstance(party, batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer):
        #                             for diff_embd_party in party.children():
        #                                 #print("diff_embd_party: ")#, diff_embd_party)
        #                                 if isinstance(diff_embd_party, torch.nn.ModuleList):
        #                                     #print("moduel list for diffpool embed")
        #                                     for diff_embed_party_in_modulelist in diff_embd_party.children():
        #                                         if isinstance(diff_embed_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                                             torch.nn.init.xavier_normal_(diff_embed_party_in_modulelist.learnable_weights.weight)
        #                                             #torch.nn.init.zeros_(diff_embed_party_in_modulelist.learnable_weights.weight)
        #                                             #print("diff_embed_party_in_modulelist.learnable_weights.weight: ",diff_embed_party_in_modulelist.learnable_weights.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_embed_party_in_modulelist.learnable_weights.bias)
        #                                                 #print(diff_embed_party_in_modulelist.learnable_weights.bias)
        #                         elif isinstance(party, batched_diffpool_assignment.Batched_DiffPool_Assignment_Layer):
        #                             for diff_assign_party in party.children():
        #                                 #print("diff_embd_party: ")#, diff_embd_party)
        #                                 if isinstance(diff_assign_party, torch.nn.ModuleList):
        #                                     #print("moduel list for diffpool embed")
        #                                     for diff_assign_party_in_modulelist in diff_assign_party.children():
        #                                         if isinstance(diff_assign_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                                             torch.nn.init.xavier_normal_(diff_assign_party_in_modulelist.learnable_weights.weight)
        #                                             #torch.nn.init.zeros_(diff_assign_party_in_modulelist.learnable_weights.weight)
        #                                             #print("diff_assign_party_in_modulelist.learnable_weights.weight: ",diff_assign_party_in_modulelist.learnable_weights.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_assign_party_in_modulelist.learnable_weights.bias)
        #                                                 #print(diff_assign_party_in_modulelist.learnable_weights.bias)
        #                                 elif isinstance(diff_assign_party, torch.nn.Sequential):
        #                                     for diff_assign_party_in_sequential in diff_assign_party:
        #                                         if isinstance(diff_assign_party_in_sequential, torch.nn.Linear):
        #                                             torch.nn.init.xavier_normal_(diff_assign_party_in_sequential.weight)
        #                                             #torch.nn.init.zeros_(diff_assign_party_in_sequential.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_assign_party_in_sequential.bias)
        #                                                 #print(diff_assign_party_in_sequential.bias)
        #                 elif isinstance(module_sub, torch.nn.Linear):
        #                     #print("linear final prediction layers")
        #                     torch.nn.init.xavier_normal_(module_sub.weight)
        #                     if bias:
        #                         torch.nn.init.zeros_(module_sub.bias)
        #                         #print(module_sub.bias)
        #         elif isinstance(module, batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer):
        #             for embd_party in module.children():
        #                 #print("embd_party: ")#, embd_party)
        #                 if isinstance(embd_party, torch.nn.ModuleList):
        #                     for embed_party_in_modulelist in embd_party.children():
        #                         if isinstance(embed_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                             torch.nn.init.xavier_normal_(embed_party_in_modulelist.learnable_weights.weight)
        #                             #torch.nn.init.zeros_(embed_party_in_modulelist.learnable_weights.weight)
        #                             #print("embed_party_in_modulelist.learnable_weights.weight: ",embed_party_in_modulelist.learnable_weights.weight)
        #                             if bias:
        #                                 torch.nn.init.zeros_(embed_party_in_modulelist.learnable_weights.bias)
        #                                 #print(embed_party_in_modulelist.learnable_weights.bias)
        #
        # if weight_initializer_index == 2:                                             #.      1. Kaiming Normal_.
        #     for i, module in enumerate(model.children()):
        #         #print(i, module)
        #         if isinstance(module, torch.nn.Sequential):
        #             for j, module_sub in enumerate(module):
        #                 #print("j: ",j,module_sub)
        #                 if isinstance(module_sub, batched_diffpool_layer.Batched_DiffPool_Layer):
        #                     #print(module_sub)
        #                     for party in module_sub.children():
        #                         if isinstance(party, batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer):
        #                             for diff_embd_party in party.children():
        #                                 #print("diff_embd_party: ")#, diff_embd_party)
        #                                 if isinstance(diff_embd_party, torch.nn.ModuleList):
        #                                     #print("moduel list for diffpool embed")
        #                                     for diff_embed_party_in_modulelist in diff_embd_party.children():
        #                                         if isinstance(diff_embed_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                                             torch.nn.init.kaiming_normal_(diff_embed_party_in_modulelist.learnable_weights.weight)
        #                                             #torch.nn.init.zeros_(diff_embed_party_in_modulelist.learnable_weights.weight)
        #                                             #print("diff_embed_party_in_modulelist.learnable_weights.weight: ",diff_embed_party_in_modulelist.learnable_weights.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_embed_party_in_modulelist.learnable_weights.bias)
        #                                                 #print(diff_embed_party_in_modulelist.learnable_weights.bias)
        #                         elif isinstance(party, batched_diffpool_assignment.Batched_DiffPool_Assignment_Layer):
        #                             for diff_assign_party in party.children():
        #                                 #print("diff_embd_party: ")#, diff_embd_party)
        #                                 if isinstance(diff_assign_party, torch.nn.ModuleList):
        #                                     #print("moduel list for diffpool embed")
        #                                     for diff_assign_party_in_modulelist in diff_assign_party.children():
        #                                         if isinstance(diff_assign_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                                             torch.nn.init.kaiming_normal_(diff_assign_party_in_modulelist.learnable_weights.weight)
        #                                             #torch.nn.init.zeros_(diff_assign_party_in_modulelist.learnable_weights.weight)
        #                                             #print("diff_assign_party_in_modulelist.learnable_weights.weight: ",diff_assign_party_in_modulelist.learnable_weights.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_assign_party_in_modulelist.learnable_weights.bias)
        #                                                 #print(diff_assign_party_in_modulelist.learnable_weights.bias)
        #                                 elif isinstance(diff_assign_party, torch.nn.Sequential):
        #                                     for diff_assign_party_in_sequential in diff_assign_party:
        #                                         if isinstance(diff_assign_party_in_sequential, torch.nn.Linear):
        #                                             torch.nn.init.kaiming_normal_(diff_assign_party_in_sequential.weight)
        #                                             #torch.nn.init.zeros_(diff_assign_party_in_sequential.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_assign_party_in_sequential.bias)
        #                                                 #print(diff_assign_party_in_sequential.bias)
        #                 elif isinstance(module_sub, torch.nn.Linear):
        #                     #print("linear final prediction layers")
        #                     torch.nn.init.kaiming_normal_(module_sub.weight)
        #                     if bias:
        #                         torch.nn.init.zeros_(module_sub.bias)
        #                         #print(module_sub.bias)
        #         elif isinstance(module, batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer):
        #             for embd_party in module.children():
        #                 #print("embd_party: ")#, embd_party)
        #                 if isinstance(embd_party, torch.nn.ModuleList):
        #                     for embed_party_in_modulelist in embd_party.children():
        #                         if isinstance(embed_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                             torch.nn.init.kaiming_normal_(embed_party_in_modulelist.learnable_weights.weight)
        #                             #torch.nn.init.zeros_(embed_party_in_modulelist.learnable_weights.weight)
        #                             #print("embed_party_in_modulelist.learnable_weights.weight: ",embed_party_in_modulelist.learnable_weights.weight)
        #                             if bias:
        #                                 torch.nn.init.zeros_(embed_party_in_modulelist.learnable_weights.bias)
        #                                 #print(embed_party_in_modulelist.learnable_weights.bias)
        #
        # if weight_initializer_index == 3:                                             #.      3. Uniform (0,0.1std)
        #     for i, module in enumerate(model.children()):
        #         #print(i, module)
        #         if isinstance(module, torch.nn.Sequential):
        #             for j, module_sub in enumerate(module):
        #                 #print("j: ",j,module_sub)
        #                 if isinstance(module_sub, batched_diffpool_layer.Batched_DiffPool_Layer):
        #                     #print(module_sub)
        #                     for party in module_sub.children():
        #                         if isinstance(party, batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer):
        #                             for diff_embd_party in party.children():
        #                                 #print("diff_embd_party: ")#, diff_embd_party)
        #                                 if isinstance(diff_embd_party, torch.nn.ModuleList):
        #                                     #print("moduel list for diffpool embed")
        #                                     for diff_embed_party_in_modulelist in diff_embd_party.children():
        #                                         if isinstance(diff_embed_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                                             torch.nn.init.normal_(diff_embed_party_in_modulelist.learnable_weights.weight, mean=mean, std=std)
        #                                             #torch.nn.init.zeros_(diff_embed_party_in_modulelist.learnable_weights.weight)
        #                                             #print("diff_embed_party_in_modulelist.learnable_weights.weight: ",diff_embed_party_in_modulelist.learnable_weights.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_embed_party_in_modulelist.learnable_weights.bias)
        #                                                 #print(diff_embed_party_in_modulelist.learnable_weights.bias)
        #                         elif isinstance(party, batched_diffpool_assignment.Batched_DiffPool_Assignment_Layer):
        #                             for diff_assign_party in party.children():
        #                                 #print("diff_embd_party: ")#, diff_embd_party)
        #                                 if isinstance(diff_assign_party, torch.nn.ModuleList):
        #                                     #print("moduel list for diffpool embed")
        #                                     for diff_assign_party_in_modulelist in diff_assign_party.children():
        #                                         if isinstance(diff_assign_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                                             torch.nn.init.normal_(diff_assign_party_in_modulelist.learnable_weights.weight, mean=mean, std=std)
        #                                             #torch.nn.init.zeros_(diff_assign_party_in_modulelist.learnable_weights.weight)
        #                                             #print("diff_assign_party_in_modulelist.learnable_weights.weight: ",diff_assign_party_in_modulelist.learnable_weights.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_assign_party_in_modulelist.learnable_weights.bias)
        #                                                 #print(diff_assign_party_in_modulelist.learnable_weights.bias)
        #                                 elif isinstance(diff_assign_party, torch.nn.Sequential):
        #                                     for diff_assign_party_in_sequential in diff_assign_party:
        #                                         if isinstance(diff_assign_party_in_sequential, torch.nn.Linear):
        #                                             torch.nn.init.normal_(diff_assign_party_in_sequential.weight, mean=mean, std=std)
        #                                             #torch.nn.init.zeros_(diff_assign_party_in_sequential.weight)
        #                                             if bias:
        #                                                 torch.nn.init.zeros_(diff_assign_party_in_sequential.bias)
        #                                                 #print(diff_assign_party_in_sequential.bias)
        #                 elif isinstance(module_sub, torch.nn.Linear):
        #                     #print("linear final prediction layers")
        #                     torch.nn.init.normal_(module_sub.weight, mean=mean, std=std)
        #                     if bias:
        #                         torch.nn.init.zeros_(module_sub.bias)
        #                         #print(module_sub.bias)
        #         elif isinstance(module, batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer):
        #             for embd_party in module.children():
        #                 #print("embd_party: ")#, embd_party)
        #                 if isinstance(embd_party, torch.nn.ModuleList):
        #                     for embed_party_in_modulelist in embd_party.children():
        #                         if isinstance(embed_party_in_modulelist, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
        #                             torch.nn.init.normal_(embed_party_in_modulelist.learnable_weights.weight, mean=mean, std=std)
        #                             #torch.nn.init.zeros_(embed_party_in_modulelist.learnable_weights.weight)
        #                             #print("embed_party_in_modulelist.learnable_weights.weight: ",embed_party_in_modulelist.learnable_weights.weight)
        #                             if bias:
        #                                 torch.nn.init.zeros_(embed_party_in_modulelist.learnable_weights.bias)
        #                                 #print(embed_party_in_modulelist.learnable_weights.bias)

    def pad_adjacency_tensor(self, adjacency_tensor, pad, value):
        if adjacency_tensor.layout != torch.strided:
            adjacency_tensor = adjacency_tensor.to_dense()
        return F.pad(adjacency_tensor, pad, mode="constant", value=value,)

    def masked_readout(self, embeddings, node_mask):
        valid_nodes = node_mask.unsqueeze(-1)
        if self.pooling == "sum":
            return embeddings.masked_fill(~valid_nodes, 0).sum(dim=1)
        if self.pooling == "mean":
            total = embeddings.masked_fill(~valid_nodes, 0).sum(dim=1)
            count = node_mask.sum(dim=1, keepdim=True).to(embeddings.dtype)
            if (count == 0).any().item():
                raise ValueError("Cannot read out an empty graph.")
            return total / count
        if self.pooling == "max":
            if (~node_mask.any(dim=1)).any().item():
                raise ValueError("Cannot read out an empty graph.")
            return embeddings.masked_fill(~valid_nodes, float("-inf")).max(dim=1).values
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def auxiliary_losses(self, adjacency, assignments, node_mask):
        valid_nodes = node_mask.to(dtype=torch.bool)
        node_counts = valid_nodes.sum(dim=1)
        if (node_counts == 0).any().item():
            raise ValueError("Cannot calculate losses for an empty graph.")

        valid_pairs = (valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1))
        reconstructed_adj = torch.bmm(assignments, assignments.transpose(1, 2),)
        residual = (adjacency - reconstructed_adj).masked_fill(~valid_pairs, 0)
        link_loss = torch.linalg.vector_norm(residual.flatten(start_dim=1), ord=2, dim=1,).mean()

        safe_assignments = assignments.clamp_min(torch.finfo(assignments.dtype).tiny)
        node_entropy = -(assignments * safe_assignments.log()).sum(dim=-1)
        node_entropy = node_entropy.masked_fill(~valid_nodes, 0)
        entropy_loss = (node_entropy.sum(dim=1) / node_counts.to(assignments.dtype)).mean()

        return link_loss, entropy_loss

    def computational_matricess(self, batched_graphs, edge_mask):
        node_features = batched_graphs.x
        edge_index = batched_graphs.edge_index
        batch_tensor = batched_graphs.batch
        if edge_mask is not None:
            if edge_mask.ndim != 1 or edge_mask.numel() != edge_index.size(1):
                raise ValueError("Expected one mask value per edge.")
            edge_mask = edge_mask.to(device=node_features.device, dtype=node_features.dtype,)
            if not torch.isfinite(edge_mask).all().item():
                raise ValueError("Edge mask contains nonfinite values.")
            if (edge_mask < 0).any().item():
                raise ValueError("Negative adjacency weights are unsupported.")

        if batch_tensor is not None:
            batch_tensor = batch_tensor.to(edge_index.device)
            batch_size = batch_tensor.max().item() + 1
            unique_graph_indices, counts = batch_tensor.unique(return_counts=True)
        else:
            batch_size = 1
            batch_tensor = torch.zeros(node_features.size(0), dtype=torch.long).to(edge_index.device)
            unique_graph_indices, counts = batch_tensor.unique(return_counts=True)
        if not torch.isfinite(node_features).all().item():
            raise ValueError("Node features contain nonfinite values.")
        if edge_index.numel() > 0:
            if (edge_index.min().item() < 0 or edge_index.max().item() >= node_features.size(0)):
                raise ValueError("Invalid node index in edge_index.")
        if not torch.equal(batch_tensor[edge_index[0]], batch_tensor[edge_index[1]],):
            raise ValueError("An edge connects different graphs in the batch.")

        max_graph_size = counts.max().item()

        adj_3d_list = []
        graph_3d_list = []

        for graph_index in unique_graph_indices:
            one_graph_node_indices = (batch_tensor == graph_index).nonzero(as_tuple=True)[0]
            node_map = {node.item(): idx for idx, node in enumerate(one_graph_node_indices)}
            edge_index_intersection = (batch_tensor[edge_index[0]] == graph_index) & (batch_tensor[edge_index[1]] == graph_index)
            local_edge_index = edge_index[:, edge_index_intersection].clone()

            local_edge_index[0] = torch.tensor([node_map[n.item()] for n in local_edge_index[0]], dtype=torch.long, device=edge_index.device)
            local_edge_index[1] = torch.tensor([node_map[n.item()] for n in local_edge_index[1]], dtype=torch.long, device=edge_index.device)
            num_nodes = one_graph_node_indices.size(0)
            nonloop_edges = local_edge_index[0] != local_edge_index[1]
            graph_edges = local_edge_index[:, nonloop_edges]

            if edge_mask is None:
                edge_values = torch.ones(graph_edges.size(1), device=node_features.device, dtype=node_features.dtype, )
            else:
                edge_values = edge_mask[edge_index_intersection][nonloop_edges]
            adj_matrix = torch.sparse_coo_tensor(graph_edges, edge_values, (num_nodes, num_nodes), )

            identity_indices = torch.arange(num_nodes, device=adj_matrix.device)
            identity_values = torch.ones(num_nodes, device=node_features.device, dtype=node_features.dtype,)
            identity_sparse = torch.sparse_coo_tensor(torch.stack([identity_indices, identity_indices]), identity_values, (num_nodes, num_nodes))

            tilda_adj_matrix = adj_matrix + identity_sparse
            padding_offset = max_graph_size - num_nodes
            padded_tilda_adj_matrix = self.pad_adjacency_tensor(tilda_adj_matrix,(0, padding_offset, 0, padding_offset),value=0).unsqueeze(0)
            adj_3d_list.append(padded_tilda_adj_matrix)

            one_graph_node_features = node_features[batch_tensor == graph_index]
            one_graph_node_features_3d = F.pad(one_graph_node_features, (0, 0, 0, padding_offset), value=0).unsqueeze(0)
            graph_3d_list.append(one_graph_node_features_3d)

        adjacency_batch = torch.cat(adj_3d_list, dim=0)
        new_feat_batch = torch.cat(graph_3d_list, dim=0)
        # counts follows the same graph order as the padded tensors.
        node_mask = (torch.arange(max_graph_size, device=node_features.device, ).unsqueeze(0) < counts.to(node_features.device).unsqueeze(1))
        return adjacency_batch.to_dense(), new_feat_batch, node_mask

    def forward(self, batched_graphs, edge_mask=None, return_aux_losses=False):
        parameter = next(self.parameters())
        batched_graphs = batched_graphs.clone().to(parameter.device)
        batched_graphs.x = batched_graphs.x.to(dtype=parameter.dtype)
        adjacecny, features, node_mask = self.computational_matricess(batched_graphs, edge_mask, )
        concatination_list_of_poolings = []
        link_losses = []
        entropy_losses = []

        for i in range(self.num_pooling):
            embedding_output, assignment_output = (self.diffpool_layers[i](features, adjacecny, node_mask=node_mask))
            valid_nodes = node_mask.unsqueeze(-1)
            embedding_output = embedding_output.masked_fill(~valid_nodes, 0)
            assignment_output = assignment_output.masked_fill(~valid_nodes, 0)
            if return_aux_losses:
                link_loss, entropy_loss = self.auxiliary_losses(adjacecny, assignment_output, node_mask,)
                link_losses.append(link_loss)
                entropy_losses.append(entropy_loss)
            embedding_output_pooled = self.masked_readout(embedding_output, node_mask,)
            concatination_list_of_poolings.append(embedding_output_pooled)
            assignment_transposed = assignment_output.transpose(1, 2)
            features = torch.bmm(assignment_transposed, embedding_output, )
            adjacecny = torch.bmm(torch.bmm(assignment_transposed, adjacecny), assignment_output,)
            node_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device,)

        extra_embed = self.last_extra_embedding(features, adjacecny, node_mask=node_mask)
        extra_embed_pooled = self.masked_readout(extra_embed, node_mask,)
        concatination_list_of_poolings.append(extra_embed_pooled)
        if self.concat_diffpools_outputs:
            output = torch.cat(concatination_list_of_poolings, dim=1,)
        else:
            output = extra_embed_pooled
        prediction_output = output
        for i in range(len(self.prediction_hid_layers)):
            prediction_output = self.act_fun(self.prediction_model[i](prediction_output))
        prediction_output = self.prediction_model[-1](prediction_output)
        prediction_output_soft = F.softmax(prediction_output, dim=-1, )
        outputs = (concatination_list_of_poolings, prediction_output, prediction_output_soft,)
        if not return_aux_losses:
            return outputs
        zero = prediction_output.new_zeros(())
        auxiliary = {"link_loss": (torch.stack(link_losses).sum() if link_losses else zero),
                     "entropy_loss": (torch.stack(entropy_losses).sum() if entropy_losses else zero), }

        return (*outputs, auxiliary)
        
        
        
#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#sum=0
#length_list = []
#for i in range(len(dataset)):
#    sum = sum + len(dataset[i].x)
#    length_list.append(len(dataset[i].x))

#print("max number of nodes in graphs: ", max(length_list))
#batch_size = 32
#new_num_nodes = 10#int(length_list[0] / batch_size)
#print("new_num_nodes: ", new_num_nodes)

#node_feat_size = len(dataset[0].x[0])

#hid_dim = 64

#diffpool_model_example = DIFFPOOL_Model(embedding_input_dim=7, embedding_num_block_layers=1, embedding_hid_dim=10, embedding_output_dim=7, 
#                                        assignment_num_block_layers=1, assignment_hid_dim=10, assignment_output_dim=7, assignment_pred_hid_layers=[10], 
#                                        prediction_hid_layers=[50, 50, 50], num_classes=2, Weight_Initializer=3, bias=True, 
#                                        dropout_rate=0, normalize_embedding=True, normalize_assignment=True, aggregation="mean", act_fun="ReLu", 
#                                        concat_outputs=False, num_pooling=1)

#batched_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#for batched_graph in batched_dataset:
#    x, edge_index, batch, y = batched_graph.x, batched_graph.edge_index, batched_graph.batch, batched_graph.y
#    concatination_list_of_poolings, prediction_output = diffpool_model_example(batched_graph, None)
#    print("Final Output: ", prediction_output)
#    break
