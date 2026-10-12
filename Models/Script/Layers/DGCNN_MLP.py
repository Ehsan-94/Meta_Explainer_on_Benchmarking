from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNN_MLP(nn.Module):
    def __init__(self, num_class, last_gnn_layer_dim, mlp_act_fun, dropout_rate, hid_channels, conv1d_kernels, dgcnn_k,
                 ffn_layer_size, bias, strides):
        super(DGCNN_MLP, self).__init__()
        for name, values in (("conv1d_kernels", conv1d_kernels), ("strides", strides),):
            if len(values) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
                raise ValueError(f"{name} must contain two positive integers.")
        if list(conv1d_kernels) != [last_gnn_layer_dim, 5]:
            raise ValueError("Expected kernels [concatenated_feature_width, 5].")
        if list(strides) != [last_gnn_layer_dim, 1]:
            raise ValueError("Expected strides [concatenated_feature_width, 1].")
        self.conv1d_kernels = list(conv1d_kernels)
        self.strides = list(strides)
        self.dgcnn_k = dgcnn_k
        self.last_gnn_layer_dim = last_gnn_layer_dim
        self.hid_channels = hid_channels
        self.padding = 0
        self.ffn_layer_size = ffn_layer_size
        self.num_class = num_class
        self.bias = bias

        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=self.hid_channels[0], kernel_size=self.conv1d_kernels[0],
                                  stride=self.strides[0], padding=self.padding, bias=self.bias)

        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1d_2 = nn.Conv1d(in_channels=self.hid_channels[0], out_channels=self.hid_channels[1],
                                  kernel_size=self.conv1d_kernels[1], stride=self.strides[1], bias=self.bias)

        input_length = self.dgcnn_k * self.last_gnn_layer_dim
        conv1_length = (input_length + 2 * self.padding - self.conv1d_kernels[0]) // self.strides[0] + 1
        if conv1_length < 2:
            raise ValueError("First convolution output is too short for max pooling.")

        pool_length = (conv1_length - 2) // 2 + 1
        conv2_length = (pool_length - self.conv1d_kernels[1]) // self.strides[1] + 1

        if conv2_length <= 0:
            raise ValueError(f"dgcnn_k={self.dgcnn_k} is too small. " "This architecture requires k >= 10.")

        self.linear1 = nn.Linear(self.hid_channels[1] * conv2_length, self.ffn_layer_size, bias=self.bias,)
        self.linear2 = nn.Linear(self.ffn_layer_size, self.num_class, bias=self.bias)

        if mlp_act_fun == 'ReLu':
            self.mlp_act_fun = F.relu
        self.soft_fun = F.softmax
        self.dropout_linear1 = nn.Dropout(p=dropout_rate)


    def forward(self, sortpooled_embedings, graph_sizes):

        to_conv1d_1 = sortpooled_embedings.view((-1, 1, self.dgcnn_k * self.last_gnn_layer_dim)) # self.node_feat_size should change to the last GNN layer dimension

        conv1d_1_res = self.conv1d_1(to_conv1d_1)
        output_conv1d_1 = self.mlp_act_fun(conv1d_1_res)

        maxpooled_output_conv1d_1 = self.maxpool1d(output_conv1d_1)

        conv1d_2_res = self.conv1d_2(maxpooled_output_conv1d_1)
        output_conv1d_2 = self.mlp_act_fun(conv1d_2_res)

        all_but_last_two_dims = output_conv1d_2.size()[:-2]

        to_dense = output_conv1d_2.view(*all_but_last_two_dims, 1, -1)
        to_dense = torch.squeeze(to_dense, 1)

        ffn_1 = self.linear1(to_dense)
        ffn_1 = self.mlp_act_fun(ffn_1)

        dropout_ffn_1 = self.dropout_linear1(ffn_1)

        ffn_2 = self.linear2(dropout_ffn_1)
        softmaxed_ffn_2 = self.soft_fun(ffn_2, dim=1)

        return output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, ffn_1, dropout_ffn_1, ffn_2, softmaxed_ffn_2