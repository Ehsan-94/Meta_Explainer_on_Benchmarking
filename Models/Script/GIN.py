import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch_geometric
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import dropout
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
import sys 
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_add_pool
from scipy.sparse import csr_matrix
py_path = '/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/Models/Script/Layers/'
sys.path.insert(0, py_path)
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import GIN_MLP_Layers as gin_mlp_layers


class GlobalSUMPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return torch_geometric.nn.global_add_pool(x, batch)
################################################################################
class IdenticalPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

################################################################################
class GIN_Model(nn.Module):
    def __init__(self, num_mlp_layers, bias, num_slp_layers, mlp_input_dim, mlp_hid_dim, mlp_output_dim, mlp_act_fun,
                 dropout_rate, weight_initializer_index, joint_embeddings, train_eps=True,):
        super(GIN_Model, self).__init__()
        for name, value in (("num_mlp_layers", num_mlp_layers), ("num_slp_layers", num_slp_layers), ("mlp_input_dim", mlp_input_dim),
                            ("mlp_hid_dim", mlp_hid_dim), ("mlp_output_dim", mlp_output_dim),):
            if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer.")
        if mlp_act_fun not in ("ReLu", "eLu", "tanh"):
            raise ValueError(f"Unsupported activation: {mlp_act_fun}")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("Dropout must be between 0 and 1.")
        if not isinstance(bias, bool):
            raise ValueError("bias must be True or False.")
        if not isinstance(joint_embeddings, bool):
            raise ValueError("joint_embeddings must be True or False.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp_input_dim = mlp_input_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.mlp_output_dim = mlp_output_dim
        self.num_slp_layers = num_slp_layers
        self.mlp_act_fun = mlp_act_fun
        self.lin_act_fun = mlp_act_fun
        self.dropout_rate = dropout_rate
        self.weight_initializer_index = weight_initializer_index

        self.num_mlp_layers = num_mlp_layers
        self.bias = bias
        self.joint_embeddings = joint_embeddings

        if not isinstance(train_eps, bool):
            raise ValueError("train_eps must be True or False.")
        self.train_eps = train_eps
        initial_eps = torch.zeros(self.num_mlp_layers)
        if self.train_eps:
            self.eps = nn.Parameter(initial_eps)
        else:
            self.register_buffer("eps", initial_eps)
        self.gin_mlp_layers = nn.ModuleList()#.to(self.device)
        self.global_summing = GlobalSUMPool()#.to(self.device)

        readout_dim = (self.mlp_input_dim + self.num_mlp_layers * self.mlp_hid_dim if self.joint_embeddings else self.mlp_hid_dim)
        self.lin1 = nn.Linear(in_features=readout_dim, out_features=self.mlp_hid_dim, bias=self.bias, )
        self.lin2 = nn.Linear(in_features=self.mlp_hid_dim, out_features=self.mlp_output_dim, bias=self.bias, )

        self.dorpout = nn.Dropout(p=dropout_rate)#.to(self.device)
        self.act_fun_softmax = F.softmax

        for i in range(self.num_mlp_layers):
            layer_input_dim = (self.mlp_input_dim if i == 0 else self.mlp_hid_dim)
            self.gin_mlp_layers.append(
                gin_mlp_layers.GIN_MLPs(num_slp_layers=self.num_slp_layers, mlp_input_dim=layer_input_dim, mlp_hid_dim=self.mlp_hid_dim,
                                        mlp_act_fun=self.mlp_act_fun, bias=self.bias, ))

        if self.lin_act_fun == 'ReLu':
            self.lin_act_fun = F.relu
            #print('ReLu is Selected.')
        elif self.lin_act_fun == 'eLu':
            self.lin_act_fun = nn.functional.elu
            #print('eLu is Selected.')
        elif self.lin_act_fun == 'tanh':
            self.lin_act_fun = torch.tanh
            #print('tanh is Selected.')

        mean = 0
        std = 0.1
        self.initialize_weights(self.weight_initializer_index, bias, mean, std)
        self.to(self.device)

    def initialize_weights(self, weight_initializer_index, bias, mean=0.0, std=0.1):
        if (not isinstance(weight_initializer_index, int) or isinstance(weight_initializer_index, bool) or weight_initializer_index not in (1, 2, 3)):
            raise ValueError("Initializer must be integer 1, 2, or 3.")
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    if weight_initializer_index == 1:
                        nn.init.xavier_normal_(module.weight)
                    elif weight_initializer_index == 2:
                        nn.init.kaiming_normal_(module.weight, nonlinearity="relu",)
                    else:
                        nn.init.normal_(module.weight, mean=mean, std=std,)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm1d):
                    module.reset_parameters()
    # def initialize_weights(model, weight_initializer_index, bias, mean, std):
    #     # 1. Xavier Normal_.  2. Kaiming Normal_.  3. Uniform (0,0.1std)
    #     if weight_initializer_index == 1:                                             #.      1. Xavier Normal_.
    #         for i, modules in enumerate(model.children()):
    #             if isinstance(modules, torch.nn.ModuleList):
    #                 for module in modules:
    #                     if isinstance(module, gin_mlp_layers.GIN_MLPs):
    #                         for final_module in module.children():
    #                             if isinstance(final_module, torch.nn.ModuleList):
    #                                 for layers in final_module:
    #                                     if isinstance(layers, torch.nn.Linear):
    #                                         torch.nn.init.xavier_normal_(layers.weight)
    #                                         #layers.weight.data.zero_()
    #                                         #print(layers.weight)
    #                                         if bias:
    #                                             layers.bias.data.zero_()
    #                                             #print("ok")
    #                                     elif isinstance(layers, torch.nn.BatchNorm1d):
    #                                         #print("ok")
    #                                         pass
    #             elif isinstance(modules, torch.nn.Linear):
    #                 #print("predict layer")
    #                 #modules.weight.data.zero_()
    #                 #print(modules.weight)
    #                 torch.nn.init.xavier_normal_(modules.weight)
    #             elif isinstance(modules, GlobalSUMPool):
    #                 #print("GlobalSUMPool")
    #                 #print(modules)
    #                 pass
    #             elif isinstance(modules, nn.Dropout):
    #                 #print("Dropout")
    #                 #print(modules)
    #                 pass
    #             else:
    #                 pass
    #
    #     if weight_initializer_index == 2:                                             #.      2. Kaiming Normal_.
    #         for i, modules in enumerate(model.children()):
    #             if isinstance(modules, torch.nn.ModuleList):
    #                 for module in modules:
    #                     if isinstance(module, gin_mlp_layers.GIN_MLPs):
    #                         for final_module in module.children():
    #                             if isinstance(final_module, torch.nn.ModuleList):
    #                                 for layers in final_module:
    #                                     if isinstance(layers, torch.nn.Linear):
    #                                         torch.nn.init.kaiming_normal_(layers.weight)
    #                                         #layers.weight.data.zero_()
    #                                         #print(layers.weight)
    #                                         if bias:
    #                                             layers.bias.data.zero_()
    #                                             #print("ok")
    #                                     elif isinstance(layers, torch.nn.BatchNorm1d):
    #                                         #print("ok")
    #                                         pass
    #             elif isinstance(modules, torch.nn.Linear):
    #                 #print("predict layer")
    #                 #modules.weight.data.zero_()
    #                 #print(modules.weight)
    #                 torch.nn.init.kaiming_normal_(modules.weight)
    #             elif isinstance(modules, GlobalSUMPool):
    #                 #print("GlobalSUMPool")
    #                 #print(modules)
    #                 pass
    #             elif isinstance(modules, nn.Dropout):
    #                 #print("Dropout")
    #                 #print(modules)
    #                 pass
    #             else:
    #                 pass
    #
    #     if weight_initializer_index == 3:                                             #.      3. Uniform (0,0.1std)
    #         for i, modules in enumerate(model.children()):
    #             if isinstance(modules, torch.nn.ModuleList):
    #                 for module in modules:
    #                     if isinstance(module, gin_mlp_layers.GIN_MLPs):
    #                         for final_module in module.children():
    #                             if isinstance(final_module, torch.nn.ModuleList):
    #                                 for layers in final_module:
    #                                     if isinstance(layers, torch.nn.Linear):
    #                                         torch.nn.init.normal_(layers.weight, mean=mean, std=std)
    #                                         #layers.weight.data.zero_()
    #                                         #print(layers.weight)
    #                                         if bias:
    #                                             layers.bias.data.zero_()
    #                                             #print("ok")
    #                                     elif isinstance(layers, torch.nn.BatchNorm1d):
    #                                         #print("ok")
    #                                         pass
    #             elif isinstance(modules, torch.nn.Linear):
    #                 #print("predict layer")
    #                 #modules.weight.data.zero_()
    #                 #print(modules.weight)
    #                 torch.nn.init.normal_(modules.weight, mean=mean, std=std)
    #             elif isinstance(modules, GlobalSUMPool):
    #                 #print("GlobalSUMPool")
    #                 #print(modules)
    #                 pass
    #             elif isinstance(modules, nn.Dropout):
    #                 #print("Dropout")
    #                 #print(modules)
    #                 pass
    #             else:
    #                 pass


    def gin_neighborhood_aggregation(self, h, batched_graphs, edge_mask):
        edge_index = batched_graphs.edge_index
        num_nodes = h.size(0)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must have dtype torch.long.")
        edge_index = edge_index.to(h.device)
        if edge_index.numel() > 0:
            if (edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes):
                raise ValueError("edge_index contains an invalid node index.")
        batch = batched_graphs.batch
        if batch is not None:
            batch = batch.to(h.device)
            if batch.ndim != 1 or batch.numel() != num_nodes:
                raise ValueError("Expected one graph index per node.")
            if not torch.equal(batch[edge_index[0]], batch[edge_index[1]],):
                raise ValueError("An edge connects different graphs in the batch.")
        if edge_mask is None:
            edge_values = torch.ones(edge_index.size(1), dtype=h.dtype, device=h.device, )
        else:
            if edge_mask.ndim != 1 or edge_mask.numel() != edge_index.size(1):
                raise ValueError("Expected one mask value per input edge.")
            edge_values = edge_mask.to(dtype=h.dtype, device=h.device, )

            if not torch.isfinite(edge_values).all().item():
                raise ValueError("Edge mask contains nonfinite values.")
        nonloop_edges = edge_index[0] != edge_index[1]
        neighbor_edges = edge_index[:, nonloop_edges]
        neighbor_values = edge_values[nonloop_edges]
        adjacency_indices = neighbor_edges.flip(0)
        adjacency = torch.sparse_coo_tensor(adjacency_indices, neighbor_values, size=(num_nodes, num_nodes), dtype=h.dtype,
                                            device=h.device, ).coalesce()
        pooled = torch.sparse.mm(adjacency, h)

        return adjacency, pooled

    def gin_layer_process_eps(self, h, layer, batched_graphs, edge_mask):
        _, pooled = self.gin_neighborhood_aggregation(h, batched_graphs, edge_mask)
        pooled = pooled + (1 + self.eps[layer]) * h

        return self.gin_mlp_layers[layer](pooled)

    def merging_process(self, one_mlp, graph_sizes):
        new = []
        start = 0
        for j in range(len(graph_sizes)):
            end = start + graph_sizes[j]
            new.append(one_mlp[start:end])
            start = end
        return new

    def reshape_mlps_outputs(self, mlps_output_embeds, graph_sizes):
        merged_mlps_output_embeds = []
        for i in range(len(graph_sizes)):
            merged_mlps_output_embeds.append([])

        for i in range(len(mlps_output_embeds)):
            for j in range(len(mlps_output_embeds[i])):
                merged_mlps_output_embeds[j].extend(mlps_output_embeds[i][j])
        return merged_mlps_output_embeds


    def forward(self, batched_graphs, edge_mask):
        parameter = next(self.parameters())
        batched_graphs = batched_graphs.clone().to(parameter.device)
        batched_graphs.x = batched_graphs.x.to(dtype=parameter.dtype)

        if batched_graphs.batch is not None:
            graph_sizes = [len(batched_graphs[i].x) for i in range(len(batched_graphs))]
        else:
            graph_sizes = [len(batched_graphs.x)]
        mlps_output_embeds_pooled = []
        mlps_output_embeds = []

        hid_rep = batched_graphs.x
        mlps_output_embeds.append(hid_rep)
        for layer in range(self.num_mlp_layers):
            hid_rep = self.gin_layer_process_eps(hid_rep, layer, batched_graphs, edge_mask,)
            mlps_output_embeds.append(hid_rep)

        if self.joint_embeddings:
            pooled_layers = [self.global_summing(rep, batched_graphs.batch) for rep in mlps_output_embeds]
            mlp_outputs_globalSUMpooled = torch.cat(pooled_layers, dim=-1,)
        else:
            mlp_outputs_globalSUMpooled = self.global_summing(hid_rep, batched_graphs.batch,)

        lin1_output = self.lin1(mlp_outputs_globalSUMpooled)
        lin1_output = self.lin_act_fun(lin1_output)

        lin1_output_dropouted = self.dorpout(lin1_output)

        lin2_output = self.lin2(lin1_output_dropouted)
        lin2_output_softmaxed = self.act_fun_softmax(lin2_output, dim=-1)

        return mlps_output_embeds, mlp_outputs_globalSUMpooled, lin1_output, lin1_output_dropouted, lin2_output, lin2_output_softmaxed




#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#batch_size = 20
#node_feat_size = len(dataset[0].x[0])
#batched_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#gin_model_example = GIN_Model(num_mlp_layers=4, bias=True, num_slp_layers=2, mlp_input_dim=node_feat_size, mlp_hid_dim=7, mlp_output_dim=7, #mlp_act_fun="ReLu", num_classes=2, dropout_rate=0.5, Weight_Initializer=3)
#print(gin_model_example)

#for batched_graphs in batched_dataset:
#    #x, edge_index, batch, y = batched_graphs.x, batched_graphs.edge_index, batched_graphs.batch, batched_graphs.y
#    mlps_output_embeds, mlps_output_embeds_stacked, mlp_outputs_globalSUMpooled, merged_mlps_output_embeds_reshaped, lin1_output, #lin1_output_dropouted, lin2_output, lin2_output_softmaxed = gin_model_example(batched_graphs)
#    print("lin2_output_softmaxed: ", lin2_output_softmaxed.size())
#    print(lin2_output_softmaxed)
#    break