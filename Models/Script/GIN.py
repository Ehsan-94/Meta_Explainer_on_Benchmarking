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
    def __init__(self, num_mlp_layers, Bias, num_slp_layers, mlp_input_dim, mlp_hid_dim, mlp_output_dim, mlp_act_fun,
                 dropout_rate, Weight_Initializer, joint_embeddings):
        super(GIN_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlp_input_dim = mlp_input_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.mlp_output_dim = mlp_output_dim
        self.num_slp_layers = num_slp_layers
        self.mlp_act_fun = mlp_act_fun
        self.lin_act_fun = mlp_act_fun
        self.dropout_rate = dropout_rate
        self.Weight_Initializer = Weight_Initializer

        self.num_mlp_layers = num_mlp_layers
        self.Bias = Bias
        self.joint_embeddings = joint_embeddings

        self.eps = nn.Parameter(torch.zeros(self.num_mlp_layers))#.to(self.device)   # important not to be included.
        self.gin_mlp_layers = nn.ModuleList()#.to(self.device)
        self.global_summing = GlobalSUMPool()#.to(self.device)

        self.the_first_layer = nn.Linear(in_features=self.mlp_input_dim, out_features=self.mlp_hid_dim, bias=self.Bias)#.to(self.device)
        #if self.joint_embeddings:
        #    self.lin1 = nn.Linear(in_features=self.mlp_hid_dim * (self.num_mlp_layers + 1), out_features=self.mlp_hid_dim * (self.num_mlp_layers + 1))
        #    self.lin2 = nn.Linear(in_features=self.mlp_hid_dim * (self.num_mlp_layers + 1), out_features=self.mlp_output_dim)
        #else:
        self.lin1 = nn.Linear(in_features=self.mlp_hid_dim, out_features=self.mlp_hid_dim)#.to(self.device)
        self.lin2 = nn.Linear(in_features=self.mlp_hid_dim, out_features=self.mlp_output_dim)#.to(self.device)
        self.dorpout = nn.Dropout(p=dropout_rate)#.to(self.device)
        self.act_fun_softmax = F.softmax

        for i in range(self.num_mlp_layers):
            a_new_layer = gin_mlp_layers.GIN_MLPs(num_slp_layers=self.num_slp_layers, mlp_input_dim=self.mlp_hid_dim, mlp_hid_dim=self.mlp_hid_dim, mlp_act_fun=self.mlp_act_fun, Bias=self.Bias)#.to(self.device)
            self.gin_mlp_layers.append(a_new_layer)

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
        self.initialize_weights(self.Weight_Initializer, Bias, mean, std)
        self.to(self.device)

    def initialize_weights(model, Weight_Initializer, Bias, mean, std):
        # 1. Xavier Normal_.  2. Kaiming Normal_.  3. Uniform (0,0.1std)
        if Weight_Initializer == 1:                                             #.      1. Xavier Normal_.
            for i, modules in enumerate(model.children()):
                if isinstance(modules, torch.nn.ModuleList):
                    for module in modules:
                        if isinstance(module, gin_mlp_layers.GIN_MLPs):
                            for final_module in module.children():
                                if isinstance(final_module, torch.nn.ModuleList):
                                    for layers in final_module:
                                        if isinstance(layers, torch.nn.Linear):
                                            torch.nn.init.xavier_normal_(layers.weight)
                                            #layers.weight.data.zero_()
                                            #print(layers.weight)
                                            if Bias:
                                                layers.bias.data.zero_()
                                                #print("ok")
                                        elif isinstance(layers, torch.nn.BatchNorm1d):
                                            #print("ok")
                                            pass
                elif isinstance(modules, torch.nn.Linear):
                    #print("predict layer")
                    #modules.weight.data.zero_()
                    #print(modules.weight)
                    torch.nn.init.xavier_normal_(modules.weight)
                elif isinstance(modules, GlobalSUMPool):
                    #print("GlobalSUMPool")
                    #print(modules)
                    pass
                elif isinstance(modules, nn.Dropout):
                    #print("Dropout")
                    #print(modules)
                    pass
                else:
                    pass

        if Weight_Initializer == 2:                                             #.      2. Kaiming Normal_.
            for i, modules in enumerate(model.children()):
                if isinstance(modules, torch.nn.ModuleList):
                    for module in modules:
                        if isinstance(module, gin_mlp_layers.GIN_MLPs):
                            for final_module in module.children():
                                if isinstance(final_module, torch.nn.ModuleList):
                                    for layers in final_module:
                                        if isinstance(layers, torch.nn.Linear):
                                            torch.nn.init.kaiming_normal_(layers.weight)
                                            #layers.weight.data.zero_()
                                            #print(layers.weight)
                                            if Bias:
                                                layers.bias.data.zero_()
                                                #print("ok")
                                        elif isinstance(layers, torch.nn.BatchNorm1d):
                                            #print("ok")
                                            pass
                elif isinstance(modules, torch.nn.Linear):
                    #print("predict layer")
                    #modules.weight.data.zero_()
                    #print(modules.weight)
                    torch.nn.init.kaiming_normal_(modules.weight)
                elif isinstance(modules, GlobalSUMPool):
                    #print("GlobalSUMPool")
                    #print(modules)
                    pass
                elif isinstance(modules, nn.Dropout):
                    #print("Dropout")
                    #print(modules)
                    pass
                else:
                    pass

        if Weight_Initializer == 3:                                             #.      3. Uniform (0,0.1std)
            for i, modules in enumerate(model.children()):
                if isinstance(modules, torch.nn.ModuleList):
                    for module in modules:
                        if isinstance(module, gin_mlp_layers.GIN_MLPs):
                            for final_module in module.children():
                                if isinstance(final_module, torch.nn.ModuleList):
                                    for layers in final_module:
                                        if isinstance(layers, torch.nn.Linear):
                                            torch.nn.init.normal_(layers.weight, mean=mean, std=std)
                                            #layers.weight.data.zero_()
                                            #print(layers.weight)
                                            if Bias:
                                                layers.bias.data.zero_()
                                                #print("ok")
                                        elif isinstance(layers, torch.nn.BatchNorm1d):
                                            #print("ok")
                                            pass
                elif isinstance(modules, torch.nn.Linear):
                    #print("predict layer")
                    #modules.weight.data.zero_()
                    #print(modules.weight)
                    torch.nn.init.normal_(modules.weight, mean=mean, std=std)
                elif isinstance(modules, GlobalSUMPool):
                    #print("GlobalSUMPool")
                    #print(modules)
                    pass
                elif isinstance(modules, nn.Dropout):
                    #print("Dropout")
                    #print(modules)
                    pass
                else:
                    pass


    def gin_neighborhood_aggregation(self, h, batched_graphs, edge_mask):
        num_nodes = batched_graphs.x.size(0)
        #joint_tilda_adjacency_matrix = torch.tensor(to_scipy_sparse_matrix(batched_graphs.edge_index).todense())# + torch.eye(len(torch.tensor(to_scipy_sparse_matrix(batched_graphs.edge_index).todense())))
        if edge_mask == None:
            # joint_tilda_adjacency_matrix = torch.tensor(to_scipy_sparse_matrix(batched_graphs.edge_index).todense())
            joint_tilda_adjacency_matrix = torch.sparse_coo_tensor(batched_graphs.edge_index, torch.ones(batched_graphs.edge_index.shape[1], dtype=torch.float32, device=batched_graphs.edge_index.device), (num_nodes, num_nodes))
        else:
            # edge_mask = edge_mask.to(self.device)
            # joint_tilda_adjacency_matrix = torch.tensor(csr_matrix((np.array(edge_mask), (np.array(batched_graphs.edge_index[0]), np.array(batched_graphs.edge_index[1])))).todense())
            joint_tilda_adjacency_matrix = torch.sparse_coo_tensor(batched_graphs.edge_index, edge_mask, (num_nodes, num_nodes))
        joint_tilda_adjacency_matrix = joint_tilda_adjacency_matrix.type(torch.float32).to(self.device)
        if batched_graphs.batch == None:
            batch_size = 1
        else:
            batch_size = batched_graphs.num_graphs

        pooled = torch.mm(joint_tilda_adjacency_matrix, h)


        return joint_tilda_adjacency_matrix, pooled


    def gin_layer_process_eps(self, h, layer, batched_graphs, edge_mask):

        joint_tilda_adjacency_matrix, pooled = self.gin_neighborhood_aggregation(h, batched_graphs, edge_mask)
        
        pooled = pooled + (1 + self.eps[layer])*h
        pooled = pooled.to(self.device)
        pooled_rep = self.gin_mlp_layers[layer](pooled)


        return pooled_rep

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
        batched_graphs = batched_graphs.to(self.device)
        if batched_graphs.batch is not None:
            graph_sizes = [len(batched_graphs[i].x) for i in range(len(batched_graphs))]
        else:
            graph_sizes = [len(batched_graphs.x)]
        if edge_mask is not None:
            edge_mask = edge_mask.to(self.device)

        mlps_output_embeds = []
        mlps_output_embeds_pooled = []
        hid_rep = self.the_first_layer(batched_graphs.x).to(self.device)
        mlps_output_embeds.append(hid_rep)

        for layer in range(self.num_mlp_layers):
            hid_rep = self.gin_layer_process_eps(hid_rep, layer, batched_graphs, edge_mask).to(self.device)
            mlps_output_embeds.append(hid_rep)
        
        mlp_outputs_globalSUMpooled = 0
        if self.joint_embeddings:
            for mlp_output in mlps_output_embeds:
                mlp_outputs_globalSUMpooled += self.global_summing(mlp_output, batched_graphs.batch)
        else:
            mlp_outputs_globalSUMpooled = self.global_summing(hid_rep, batched_graphs.batch)

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
#gin_model_example = GIN_Model(num_mlp_layers=4, Bias=True, num_slp_layers=2, mlp_input_dim=node_feat_size, mlp_hid_dim=7, mlp_output_dim=7, #mlp_act_fun="ReLu", num_classes=2, dropout_rate=0.5, Weight_Initializer=3)
#print(gin_model_example)

#for batched_graphs in batched_dataset:
#    #x, edge_index, batch, y = batched_graphs.x, batched_graphs.edge_index, batched_graphs.batch, batched_graphs.y
#    mlps_output_embeds, mlps_output_embeds_stacked, mlp_outputs_globalSUMpooled, merged_mlps_output_embeds_reshaped, lin1_output, #lin1_output_dropouted, lin2_output, lin2_output_softmaxed = gin_model_example(batched_graphs)
#    print("lin2_output_softmaxed: ", lin2_output_softmaxed.size())
#    print(lin2_output_softmaxed)
#    break