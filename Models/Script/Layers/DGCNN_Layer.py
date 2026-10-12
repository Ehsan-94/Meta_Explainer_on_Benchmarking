import torch
import torch.nn as nn

class DGCNN_Layer(nn.Module):
    """
    A single DGCNN Layer
    """
    def __init__(self, input_dim, latent_dim, bias):

        super(DGCNN_Layer, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv_params = nn.Linear(input_dim, latent_dim, bias=bias)

    def forward(self, input_tensor, tilda_adjacency_matrix, reciprocal_tilda_degree_matrix):

        adjacency_matrix_multiplied = torch.bmm(tilda_adjacency_matrix, input_tensor)       # Y = A~ * X
        node_linear = self.conv_params(adjacency_matrix_multiplied)     # Y = Y * W
        normalized_linear = torch.bmm(reciprocal_tilda_degree_matrix, node_linear)      # Y = D^-1 * Y

        return normalized_linear