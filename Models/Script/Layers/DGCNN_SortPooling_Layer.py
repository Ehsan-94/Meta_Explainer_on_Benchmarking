import torch
import torch.nn as nn
import torch.nn.functional as F


class SortPooling_for_BMM(nn.Module):
    def __init__(self, sort_pooling_k):
        super().__init__()
        if sort_pooling_k <= 0:
            raise ValueError("sort_pooling_k must be positive.")
        self.sort_pooling_k = sort_pooling_k
    def forward(self, output_of_dgcnn_layer, graph_sizes):
        batch_size, max_nodes, _ = output_of_dgcnn_layer.shape
        if len(graph_sizes) != batch_size:
            raise ValueError("Expected one graph size per batch item.")
        pooled_graphs = []
        for graph_index, size in enumerate(graph_sizes):
            size = int(size)
            if size <= 0 or size > max_nodes:
                raise ValueError(f"Invalid node count for graph {graph_index}: {size}")

            real_nodes = output_of_dgcnn_layer[graph_index, :size, :]
            sorted_indices = torch.arange(real_nodes.size(0), device=real_nodes.device,)
            for channel in range(real_nodes.size(1)):
                order = torch.argsort(real_nodes[sorted_indices, channel], descending=True, stable=True,)
                sorted_indices = sorted_indices[order]
            selected_nodes = real_nodes[sorted_indices[:self.sort_pooling_k]]
            missing_nodes = self.sort_pooling_k - selected_nodes.size(0)
            if missing_nodes > 0:
                selected_nodes = F.pad(selected_nodes, (0, 0, 0, missing_nodes), value=0,)
            pooled_graphs.append(selected_nodes)

        return torch.stack(pooled_graphs, dim=0)