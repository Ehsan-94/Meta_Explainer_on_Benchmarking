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
from torch_geometric.datasets import TUDataset
from scipy.sparse import csr_matrix
py_path = '/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/Models/Script/Layers/'
sys.path.insert(0,py_path)
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import GlobalAveragePooling as global_avg_pooling
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import IdenticalPooling as identical_pooling
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import DGCNN_Layer as dgcnn_layer
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import DGCNN_GNN_Layers as dgcnn_gnn_layers
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import DGCNN_SortPooling_Layer as sortpooling_layer
from Meta_Explainer_on_Benchmarking.Models.Script.Layers import DGCNN_MLP as dgcnn_mlp


class DGCNN_Model(nn.Module):
    '''
        DGCNN Layers using Sparse Adjacency Matrix Multiplication
    '''
    def __init__(self, GNN_layers, mlp_act_fun, dgcnn_act_fun, mlp_dropout_rate, weight_initializer_index, bias, num_classes, dgcnn_k,
                 node_feat_size, hid_channels, conv1d_kernels, ffn_layer_size, strides):

        super(DGCNN_Model, self).__init__()
        if not GNN_layers or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in GNN_layers):
            raise ValueError("GNN_layers must contain positive integer widths.")
        for name, value in (
        ("node_feat_size", node_feat_size), ("num_classes", num_classes), ("dgcnn_k", dgcnn_k), ("ffn_layer_size", ffn_layer_size),):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if dgcnn_k < 10:
            raise ValueError("The updated Conv1d architecture requires k >= 10.")
        if len(hid_channels) != 2 or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in hid_channels):
            raise ValueError("hid_channels must contain two positive integers.")
        if dgcnn_act_fun not in ("ReLu", "eLu", "tanh"):
            raise ValueError(f"Unsupported graph activation: {dgcnn_act_fun}")
        if mlp_act_fun != "ReLu":
            raise ValueError(f"Unsupported MLP activation: {mlp_act_fun}")
        if not 0 <= mlp_dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1.")
        self.GNN_layers = GNN_layers
        self.last_gnn_layer_dim = sum(GNN_layers)
        self.num_GNN_layers = len(GNN_layers)
        self.mlp_dropout_rate = mlp_dropout_rate
        self.bias = bias
        self.weight_initializer_index = weight_initializer_index
        self.dgcnn_k = dgcnn_k
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes
        self.hid_channels = hid_channels
        self.conv1d_kernels = conv1d_kernels
        self.ffn_layer_size = ffn_layer_size
        self.strides = strides

        self.gnn_layers = dgcnn_gnn_layers.DGCNN_GNN_Layers(GNN_layers=self.GNN_layers,
                                                            node_feat_size=self.node_feat_size,
                                                            bias=self.bias, dgcnn_act_fun=dgcnn_act_fun)

        self.sort_pool = sortpooling_layer.SortPooling_for_BMM(self.dgcnn_k)

        self.classic_conv = dgcnn_mlp.DGCNN_MLP(num_class=self.num_classes, last_gnn_layer_dim=self.last_gnn_layer_dim,
                                                mlp_act_fun=mlp_act_fun, dropout_rate=self.mlp_dropout_rate,
                                                hid_channels=self.hid_channels, conv1d_kernels=self.conv1d_kernels,
                                                dgcnn_k=self.dgcnn_k, ffn_layer_size=self.ffn_layer_size,
                                                bias=self.bias, strides=self.strides)
        if dgcnn_act_fun == 'ReLu':
            self.dgcnn_act_fun = F.relu
            # print('ReLu is Selected.')
        elif dgcnn_act_fun == 'eLu':
            self.dgcnn_act_fun = nn.functional.elu
            # print('eLu is Selected.')
        elif dgcnn_act_fun == 'tanh':
            self.dgcnn_act_fun = torch.tanh
            # print('tanh is Selected.')

        mean = 0
        std = 0.1
        self.initialize_weights(weight_initializer_index, bias, mean, std)

    def initialize_weights(self, weight_initializer_index, bias, mean=0.0, std=0.1):
        # 1: Xavier Normal
        # 2: Kaiming Normal
        # 3: Normal(mean, std)
        if weight_initializer_index not in (1, 2, 3):
            raise ValueError("Weight_Initializer must be 1 (Xavier), 2 (Kaiming), or 3 (Normal).")

        with torch.no_grad():
            for module in self.modules():
                if not isinstance(module, (nn.Linear, nn.Conv1d)):
                    continue
                if weight_initializer_index == 1:
                    nn.init.xavier_normal_(module.weight)
                elif weight_initializer_index == 2:
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu",)
                else:
                    nn.init.normal_(module.weight, mean=mean, std=std,)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # def initialize_weights(model, Weight_Initializer, bias, mean, std):
    #     # 1. Xavier Normal_.  2. Kaiming Normal_.  3. Uniform (0,0.1std)
    #     if weight_initializer_index == 1:                                             #.      1. Xavier Normal_.
    #         for i,layers in enumerate(model.children()):
    #             if isinstance(layers, dgcnn_gnn_layers.DGCNN_GNN_Layers):
    #                 for j, layer in enumerate(layers.modules()):
    #                     if isinstance(layer, nn.Linear):
    #                         torch.nn.init.xavier_normal_(layer.weight.data)
    #                         if bias:
    #                             layer.bias.data.zero_()
    #                     if isinstance(layer, dgcnn_layer.DGCNN_Layer):
    #                         torch.nn.init.xavier_normal_(layer.conv_params.weight)
    #                         if bias:
    #                             layer.conv_params.bias.data.zero_()
    #                     else:
    #                         pass
    #             if isinstance(layers, torch.nn.Linear):
    #                 torch.nn.init.xavier_normal_(layers.weight)
    #                 if bias:
    #                     layers.bias.data.zero_()
    #             if isinstance(layers, (dgcnn_mlp.DGCNN_MLP)):
    #                 torch.nn.init.xavier_normal_(layers.conv1d_1.weight)
    #                 torch.nn.init.xavier_normal_(layers.conv1d_2.weight)
    #                 torch.nn.init.xavier_normal_(layers.linear1.weight)
    #                 torch.nn.init.xavier_normal_(layers.linear2.weight)
    #
    #             elif isinstance(layers, (global_avg_pooling.GlobalAveragePooling)):
    #                 pass
    #             elif isinstance(layers, (identical_pooling.IdenticalPooling)):
    #                 pass
    #
    #     if weight_initializer_index == 2:                                             #.      2. Kaiming Normal_.
    #         for i,layers in enumerate(model.children()):
    #             if isinstance(layers, dgcnn_gnn_layers.DGCNN_GNN_Layers):
    #                 for j, layer in enumerate(layers.modules()):
    #                     if isinstance(layer, nn.Linear):
    #                         torch.nn.init.kaiming_normal_(layer.weight.data)
    #                         if bias:
    #                             layer.bias.data.zero_()
    #                     if isinstance(layer, dgcnn_layer.DGCNN_Layer):
    #                         torch.nn.init.kaiming_normal_(layer.conv_params.weight)
    #                         if bias:
    #                             layer.conv_params.bias.data.zero_()
    #                     else:
    #                         pass
    #             if isinstance(layers, torch.nn.Linear):
    #                 torch.nn.init.kaiming_normal_(layers.weight)
    #                 if bias:
    #                     layers.bias.data.zero_()
    #             if isinstance(layers, (dgcnn_mlp.DGCNN_MLP)):
    #                 torch.nn.init.kaiming_normal_(layers.conv1d_1.weight)
    #                 torch.nn.init.kaiming_normal_(layers.conv1d_2.weight)
    #                 torch.nn.init.kaiming_normal_(layers.linear1.weight)
    #                 torch.nn.init.kaiming_normal_(layers.linear2.weight)
    #
    #             elif isinstance(layers, (global_avg_pooling.GlobalAveragePooling)):
    #                 pass
    #             elif isinstance(layers, (identical_pooling.IdenticalPooling)):
    #                 pass
    #
    #     if weight_initializer_index == 3:                                             #.      3. Uniform (0,0.1std)
    #         for i,layers in enumerate(model.children()):
    #             if isinstance(layers, dgcnn_gnn_layers.DGCNN_GNN_Layers):
    #                 for j, layer in enumerate(layers.modules()):
    #                     #print("here2")
    #                     if isinstance(layer, nn.Linear):
    #                         torch.nn.init.normal_(layer.weight.data, mean, std)
    #                         if bias:
    #                             layer.bias.data.zero_()
    #                     if isinstance(layer, dgcnn_layer.DGCNN_Layer):
    #                         torch.nn.init.normal_(layer.conv_params.weight, mean, std)
    #                         if bias:
    #                             layer.conv_params.bias.data.zero_()
    #                     else:
    #                         pass
    #             if isinstance(layers, torch.nn.Linear):
    #                 torch.nn.init.normal_(layers.weight, mean, std)
    #                 if bias:
    #                     layers.bias.data.zero_()
    #             if isinstance(layers, (dgcnn_mlp.DGCNN_MLP)):
    #                 torch.nn.init.normal_(layers.conv1d_1.weight, mean, std)
    #                 torch.nn.init.normal_(layers.conv1d_2.weight, mean, std)
    #                 torch.nn.init.normal_(layers.linear1.weight, mean, std)
    #                 torch.nn.init.normal_(layers.linear2.weight, mean, std)
    #             elif isinstance(layers, (global_avg_pooling.GlobalAveragePooling)):
    #                 pass
    #             elif isinstance(layers, (identical_pooling.IdenticalPooling)):
    #                 pass


    def forward(self, graph, edge_mask):

        if graph.batch is not None:
            graph_sizes = [len(graph[i].x) for i in range(len(graph))]
        else:
            graph_sizes = [len(graph.x)]

        Output_of_GNN_Layers = self.gnn_layers(graph, edge_mask)

        if Output_of_GNN_Layers.requires_grad:
            Output_of_GNN_Layers.retain_grad()

        sortpooled_embedings = self.sort_pool(output_of_dgcnn_layer=Output_of_GNN_Layers, graph_sizes=graph_sizes,)

        if sortpooled_embedings.requires_grad:
            sortpooled_embedings.retain_grad()

        output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, ffn_1, dropout_ffn_1, ffn_2, softmaxed_ffn_2 = self.classic_conv(sortpooled_embedings=sortpooled_embedings, graph_sizes=graph_sizes)

        return Output_of_GNN_Layers, sortpooled_embedings, output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, ffn_1, dropout_ffn_1, ffn_2, softmaxed_ffn_2