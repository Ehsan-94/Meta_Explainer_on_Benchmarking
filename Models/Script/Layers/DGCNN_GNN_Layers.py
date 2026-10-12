import torch
import torch.nn as nn
import torch.nn.functional as F
import DGCNN_Layer as dgcnn_layer



class DGCNN_GNN_Layers(nn.Module):
    """
        Padding happens based on max size in a batch.
    """
    def __init__(self, GNN_layers, node_feat_size, bias, dgcnn_act_fun):
        super(DGCNN_GNN_Layers, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GNN_layers = GNN_layers
        self.num_GNN_layers = len(GNN_layers)
        self.node_feat_size = node_feat_size
        self.output_dim = GNN_layers[-1]
        self.bias = bias

        self.gnn_layers = []

        for i in range(self.num_GNN_layers):
            if self.num_GNN_layers == 1:
                self.gnn_layers.append(dgcnn_layer.DGCNN_Layer(input_dim=self.node_feat_size, latent_dim=self.output_dim, bias=self.bias))
            elif self.num_GNN_layers > 1:
                if i == 0:
                    self.gnn_layers.append(dgcnn_layer.DGCNN_Layer(input_dim=self.node_feat_size, latent_dim=self.GNN_layers[i], bias=self.bias))
                elif 0 < i < self.num_GNN_layers-1:
                    self.gnn_layers.append(dgcnn_layer.DGCNN_Layer(input_dim=self.GNN_layers[i-1], latent_dim=self.GNN_layers[i], bias=self.bias))
                elif i == self.num_GNN_layers-1:
                    self.gnn_layers.append(dgcnn_layer.DGCNN_Layer(input_dim=self.GNN_layers[i-1], latent_dim=self.output_dim, bias=self.bias))
            else:
                print("please enter layer config")
        self.gnn_layers = nn.Sequential(*self.gnn_layers)

        if dgcnn_act_fun == 'ReLu':
            self.dgcnn_act_fun = F.relu
            # print('ReLu is Selected.')
        elif dgcnn_act_fun == 'eLu':
            self.dgcnn_act_fun = nn.functional.elu
            # print('eLu is Selected.')
        elif dgcnn_act_fun == 'tanh':
            self.dgcnn_act_fun = torch.tanh
            # print('tanh is Selected.')

    def pad_sparse_tensor(self, sparse_tensor, pad, value):
        dense_tensor = sparse_tensor.to_dense()
        padded_dense_tensor = F.pad(dense_tensor, pad, mode="constant", value=value,)

        return padded_dense_tensor.to_sparse()

    def compute_degree_matrix(self, sparse_adj_matrix):
        degree_vector = torch.sparse.sum(sparse_adj_matrix, dim=1).to_dense()
        num_nodes = degree_vector.size(0)
        indices = torch.arange(num_nodes, device=sparse_adj_matrix.device)
        degree_matrix = torch.sparse_coo_tensor(torch.stack([indices, indices]),
                                                degree_vector,
                                                (num_nodes, num_nodes),
                                                device=sparse_adj_matrix.device)
        return degree_matrix

    def compute_reciprocal_sparse_degree_matrix(self, degree_matrix):
        indices = degree_matrix._indices()
        values = degree_matrix._values()
        reciprocal_values = torch.reciprocal(values)
        reciprocal_degree_matrix = torch.sparse_coo_tensor(indices, reciprocal_values, degree_matrix.size(), device=degree_matrix.device)
        return reciprocal_degree_matrix

    def computational_matrices(self, batched_graphs, edge_mask=None):
        node_features = batched_graphs.x
        edge_index = batched_graphs.edge_index
        batch_tensor = batched_graphs.batch

        if node_features.ndim != 2 or node_features.size(0) == 0:
            raise ValueError("Expected a nonempty node-feature matrix.")
        if not node_features.is_floating_point():
            raise ValueError("Node features must use a floating-point dtype.")
        if not torch.isfinite(node_features).all().item():
            raise ValueError("Node features contain nonfinite values.")
        if node_features.size(1) != self.node_feat_size:
            raise ValueError("Node-feature width does not match the model.")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must have dtype torch.long.")
        if edge_index.device != node_features.device:
            raise ValueError("Features and edges must be on the same device.")
        if edge_index.numel() > 0:
            if (edge_index.min().item() < 0 or edge_index.max().item() >= node_features.size(0)):
                raise ValueError("edge_index contains an invalid node index.")
        if edge_mask is not None:
            if edge_mask.ndim != 1 or edge_mask.numel() != edge_index.size(1):
                raise ValueError("Expected one mask value per edge.")
            edge_mask = edge_mask.to(device=node_features.device, dtype=node_features.dtype,)
            if not torch.isfinite(edge_mask).all().item():
                raise ValueError("Edge mask contains nonfinite values.")
            if (edge_mask < 0).any().item():
                raise ValueError("Negative edge weights are unsupported.")

        if batch_tensor is not None:
            batch_tensor = batch_tensor.to(edge_index.device)
            batch_size = batch_tensor.max().item() + 1
            unique_graph_indices, counts = batch_tensor.unique(return_counts=True)
        else:
            batch_size = 1
            batch_tensor = torch.zeros(node_features.size(0), dtype=torch.long).to(edge_index.device)
            unique_graph_indices, counts = batch_tensor.unique(return_counts=True)
        if not torch.equal(batch_tensor[edge_index[0]], batch_tensor[edge_index[1]],):
            raise ValueError("An edge connects different graphs in the batch.")

        max_graph_size = counts.max().item()

        adj_3d_list = []
        graph_3d_list = []
        degree_3d_list = []

        for graph_index in unique_graph_indices:
            one_graph_node_indices = (batch_tensor == graph_index).nonzero(as_tuple=True)[0]
            node_map = {node.item(): idx for idx, node in enumerate(one_graph_node_indices)}
            edge_index_intersection = (batch_tensor[edge_index[0]] == graph_index) & (
                        batch_tensor[edge_index[1]] == graph_index)
            local_edge_index = edge_index[:, edge_index_intersection].clone()

            local_edge_index[0] = torch.tensor([node_map[n.item()] for n in local_edge_index[0]], dtype=torch.long,
                                               device=edge_index.device)
            local_edge_index[1] = torch.tensor([node_map[n.item()] for n in local_edge_index[1]], dtype=torch.long,
                                               device=edge_index.device)
            num_nodes = one_graph_node_indices.size(0)
            nonloop_edges = local_edge_index[0] != local_edge_index[1]
            graph_edges = local_edge_index[:, nonloop_edges]
            if edge_mask is None:
                edge_values = torch.ones(graph_edges.size(1), device=node_features.device, dtype=node_features.dtype,)
            else:
                edge_values = edge_mask[edge_index_intersection][nonloop_edges]
            adj_matrix = torch.sparse_coo_tensor(graph_edges, edge_values,(num_nodes, num_nodes),)
            identity_indices = torch.arange(num_nodes, device=node_features.device)
            identity_sparse = torch.sparse_coo_tensor(
                torch.stack([identity_indices, identity_indices]),
                torch.ones(num_nodes, device=node_features.device, dtype=node_features.dtype, ),(num_nodes, num_nodes),)

            tilda_adj_matrix = adj_matrix + identity_sparse
            padding_offset = max_graph_size - num_nodes

            padded_tilda_adj_matrix = self.pad_sparse_tensor(tilda_adj_matrix, (0, padding_offset, 0, padding_offset),
                                                             value=0).unsqueeze(0)
            adj_3d_list.append(padded_tilda_adj_matrix)

            one_graph_node_features = node_features[batch_tensor == graph_index]
            one_graph_node_features_3d = F.pad(one_graph_node_features, (0, 0, 0, padding_offset), value=0).unsqueeze(0)
            graph_3d_list.append(one_graph_node_features_3d)

            degree = tilda_adj_matrix.to_dense().sum(dim=1)
            if not torch.isfinite(degree).all().item():
                raise ValueError("Graph degrees contain nonfinite values.")
            if (degree <= 0).any().item():
                raise ValueError("DGCNN normalization requires positive graph degrees.")

            inverse_degree = degree.reciprocal()
            reciprocal_degree_matrix = torch.diag(inverse_degree)
            padded_reciprocal_tilda_degree_matrix = F.pad(reciprocal_degree_matrix, (0, padding_offset, 0, padding_offset), value=0,).unsqueeze(0)
            degree_3d_list.append(padded_reciprocal_tilda_degree_matrix)

        adjacency_batch = torch.cat(adj_3d_list, dim=0)
        padded_reciprocal_degree = torch.cat(degree_3d_list, dim=0)
        new_feat_batch = torch.cat(graph_3d_list, dim=0).to_dense()

        return adjacency_batch, padded_reciprocal_degree, new_feat_batch

    def forward(self, graph, edge_mask):
        x, edge_index, batch, y = graph.x, graph.edge_index, graph.batch, graph.y

        if batch is not None:
            graph_sizes = [len(graph[i].x) for i in range(len(graph))]
        else:
            graph_sizes = [len(graph.x)]

        Output_of_GNN_Layers = []
        new_adjacecny, reciprocal_tilda_degree_matrix, x = self.computational_matrices(graph, edge_mask)

        for i in range(self.num_GNN_layers):
            x = self.gnn_layers[i](x, new_adjacecny, reciprocal_tilda_degree_matrix,)
            x = self.dgcnn_act_fun(x)
            Output_of_GNN_Layers.append(x)

        return torch.cat(Output_of_GNN_Layers, dim=-1)