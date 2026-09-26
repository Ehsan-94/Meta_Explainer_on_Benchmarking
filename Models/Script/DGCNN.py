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
    def __init__(self, GNN_layers, mlp_act_fun, dgcnn_act_fun, mlp_dropout_rate, Weight_Initializer, Bias, num_classes, dgcnn_k,
                 node_feat_size, hid_channels, conv1d_kernels, ffn_layer_size, strides):

        super(DGCNN_Model, self).__init__()
        self.GNN_layers = GNN_layers
        self.last_gnn_layer_dim = GNN_layers[-1]
        self.num_GNN_layers = len(GNN_layers)
        self.mlp_dropout_rate = mlp_dropout_rate
        self.Bias = Bias
        self.Weight_Initializer = Weight_Initializer
        self.dgcnn_k = dgcnn_k
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes
        self.hid_channels = hid_channels
        self.conv1d_kernels = conv1d_kernels
        self.ffn_layer_size = ffn_layer_size
        self.strides = strides

        self.gnn_layers = dgcnn_gnn_layers.DGCNN_GNN_Layers(GNN_layers=self.GNN_layers,
                                                            node_feat_size=self.node_feat_size,
                                                            Bias=self.Bias, dgcnn_act_fun=dgcnn_act_fun)

        self.sort_pool = sortpooling_layer.SortPooling_for_BMM(self.dgcnn_k)

        self.classic_conv = dgcnn_mlp.DGCNN_MLP(num_class=self.num_classes, last_gnn_layer_dim=self.last_gnn_layer_dim,
                                                mlp_act_fun=mlp_act_fun, dropout_rate=self.mlp_dropout_rate,
                                                hid_channels=self.hid_channels, conv1d_kernels=self.conv1d_kernels,
                                                dgcnn_k=self.dgcnn_k, ffn_layer_size=self.ffn_layer_size,
                                                Bias=self.Bias, strides=self.strides)
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
        self.initialize_weights(Weight_Initializer, Bias, mean, std)


    def initialize_weights(model, Weight_Initializer, Bias, mean, std):
        # 1. Xavier Normal_.  2. Kaiming Normal_.  3. Uniform (0,0.1std)
        if Weight_Initializer == 1:                                             #.      1. Xavier Normal_.
            for i,layers in enumerate(model.children()):
                if isinstance(layers, dgcnn_gnn_layers.DGCNN_GNN_Layers):
                    for j, layer in enumerate(layers.modules()):
                        if isinstance(layer, nn.Linear):
                            torch.nn.init.xavier_normal_(layer.weight.data)
                            if Bias:
                                layer.bias.data.zero_()
                        if isinstance(layer, dgcnn_layer.DGCNN_Layer):
                            torch.nn.init.xavier_normal_(layer.conv_params.weight)
                            if Bias:
                                layer.conv_params.bias.data.zero_()
                        else:
                            pass
                if isinstance(layers, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(layers.weight)
                    if Bias:
                        layers.bias.data.zero_()
                if isinstance(layers, (dgcnn_mlp.DGCNN_MLP)):
                    torch.nn.init.xavier_normal_(layers.conv1d_1.weight)
                    torch.nn.init.xavier_normal_(layers.conv1d_2.weight)
                    torch.nn.init.xavier_normal_(layers.linear1.weight)
                    torch.nn.init.xavier_normal_(layers.linear2.weight)

                elif isinstance(layers, (global_avg_pooling.GlobalAveragePooling)):
                    pass
                elif isinstance(layers, (identical_pooling.IdenticalPooling)):
                    pass

        if Weight_Initializer == 2:                                             #.      2. Kaiming Normal_.
            for i,layers in enumerate(model.children()):
                if isinstance(layers, dgcnn_gnn_layers.DGCNN_GNN_Layers):
                    for j, layer in enumerate(layers.modules()):
                        if isinstance(layer, nn.Linear):
                            torch.nn.init.kaiming_normal_(layer.weight.data)
                            if Bias:
                                layer.bias.data.zero_()
                        if isinstance(layer, dgcnn_layer.DGCNN_Layer):
                            torch.nn.init.kaiming_normal_(layer.conv_params.weight)
                            if Bias:
                                layer.conv_params.bias.data.zero_()
                        else:
                            pass
                if isinstance(layers, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(layers.weight)
                    if Bias:
                        layers.bias.data.zero_()
                if isinstance(layers, (dgcnn_mlp.DGCNN_MLP)):
                    torch.nn.init.kaiming_normal_(layers.conv1d_1.weight)
                    torch.nn.init.kaiming_normal_(layers.conv1d_2.weight)
                    torch.nn.init.kaiming_normal_(layers.linear1.weight)
                    torch.nn.init.kaiming_normal_(layers.linear2.weight)

                elif isinstance(layers, (global_avg_pooling.GlobalAveragePooling)):
                    pass
                elif isinstance(layers, (identical_pooling.IdenticalPooling)):
                    pass

        if Weight_Initializer == 3:                                             #.      3. Uniform (0,0.1std)
            for i,layers in enumerate(model.children()):
                if isinstance(layers, dgcnn_gnn_layers.DGCNN_GNN_Layers):
                    for j, layer in enumerate(layers.modules()):
                        #print("here2")
                        if isinstance(layer, nn.Linear):
                            torch.nn.init.normal_(layer.weight.data, mean, std)
                            if Bias:
                                layer.bias.data.zero_()
                        if isinstance(layer, dgcnn_layer.DGCNN_Layer):
                            torch.nn.init.normal_(layer.conv_params.weight, mean, std)
                            if Bias:
                                layer.conv_params.bias.data.zero_()
                        else:
                            pass
                if isinstance(layers, torch.nn.Linear):
                    torch.nn.init.normal_(layers.weight, mean, std)
                    if Bias:
                        layers.bias.data.zero_()
                if isinstance(layers, (dgcnn_mlp.DGCNN_MLP)):
                    torch.nn.init.normal_(layers.conv1d_1.weight, mean, std)
                    torch.nn.init.normal_(layers.conv1d_2.weight, mean, std)
                    torch.nn.init.normal_(layers.linear1.weight, mean, std)
                    torch.nn.init.normal_(layers.linear2.weight, mean, std)
                elif isinstance(layers, (global_avg_pooling.GlobalAveragePooling)):
                    pass
                elif isinstance(layers, (identical_pooling.IdenticalPooling)):
                    pass


    def forward(self, graph, edge_mask):

        if graph.batch is not None:
            graph_sizes = [len(graph[i].x) for i in range(len(graph))]
        else:
            graph_sizes = [len(graph.x)]

        Output_of_GNN_Layers = self.gnn_layers(graph, edge_mask)

        if Output_of_GNN_Layers.requires_grad:
            Output_of_GNN_Layers.retain_grad()

        sortpooled_embedings = self.sort_pool(output_of_dgcnn_layer=Output_of_GNN_Layers)

        if sortpooled_embedings.requires_grad:
            sortpooled_embedings.retain_grad()

        output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, ffn_1, dropout_ffn_1, ffn_2, softmaxed_ffn_2 = self.classic_conv(sortpooled_embedings=sortpooled_embedings, graph_sizes=graph_sizes)

        return Output_of_GNN_Layers, sortpooled_embedings, output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, ffn_1, dropout_ffn_1, ffn_2, softmaxed_ffn_2