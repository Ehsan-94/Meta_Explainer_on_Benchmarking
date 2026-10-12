import torch
import torch.nn as nn
class GCN_Layer(nn.Module):
    """
        A single GCN Layer, using propagation matrix defined by Kipf et al. in GCN
    """
    def __init__(self, input_dim, latent_dim, bias):

        super(GCN_Layer, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.bias = bias
        self.conv_params = nn.Linear(input_dim, latent_dim, bias=self.bias)


    def forward(self, input_tensor, tilda_adjacency_matrix, padded_reciprocal_sqrt_degree):
        d_a = torch.bmm(tilda_adjacency_matrix, padded_reciprocal_sqrt_degree)
        d_a_d = torch.bmm(padded_reciprocal_sqrt_degree, d_a)
        d_a_d_x = torch.bmm(d_a_d, input_tensor)
        d_a_d_x_w = self.conv_params(d_a_d_x)

        return d_a_d_x_w