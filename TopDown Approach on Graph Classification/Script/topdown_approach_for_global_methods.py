# -*- coding: utf-8 -*-
"""Evaluation of Global Methods.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/139tX8O2IrjHqD7ZCk7Rg2OmVjr7hgTV0

## ***Adaptations on the Evaluation of Global Methods by Instance-based Metrics***


> Moduled: Accpeting the four GNNs (GCN+GAP, DGCNN, DIFFPOOL, and GIN)


---

The algorithm:
1. find average size of explanation in each class by GraphMask in terms of number of edges.
2. generate explanations for that count of edges in XGNN and GNNInterpreter.
3. check existance of important edges from each class in each sample.


"
"""

import os
import torch
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from math import sqrt
import math
from torch_geometric.datasets import TUDataset
import torch as th
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from sklearn import metrics
from scipy.spatial.distance import hamming
import statistics
import pandas
from time import perf_counter
from IPython.core.display import deepcopy
from torch_geometric.nn import MessagePassing
import copy
from torch.nn import ReLU, Sequential
from torch import sigmoid
from itertools import chain
from time import perf_counter
from torch_geometric.data import Data, Batch, Dataset
from functools import partial
from torch_geometric.utils import to_networkx
from torch_geometric.utils import remove_self_loops
from typing import Callable, Union, Optional
#from torch_geometric.utils.num_nodes import maybe_num_nodes
import networkx as nx
from typing import List, Tuple, Dict
from collections import Counter
import statistics
import tqdm
import csv
from statistics import mean
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch_geometric.nn as gnn



class global_explanation_and_samples_intersection:
    def __init__(self, explanation, input_graph):
        self.explanation = explanation
        self.input_graph = input_graph

    def pair_source_and_target_node_features(self, edges, node_feats):
        paired_feat = {}
        for (source_index, target_index) in edges:
            paired_feat[(source_index, target_index)] = node_feats[source_index] + node_feats[target_index]
        return paired_feat

    def find_common_edges(self, ex_graph_node_features, ex_graph_edge_index, input_graph_node_features, input_graph_edge_index):

        ex_graph_edges = [(source1_index, target1_index) for source1_index, target1_index in ex_graph_edge_index.T.tolist()]
        input_graph_edges = [(source2_index, target2_index) for source2_index, target2_index in input_graph_edge_index.T.tolist()]

        ex_graph_paired_node_features = self.pair_source_and_target_node_features(ex_graph_edges, ex_graph_node_features)
        input_graph_paired_node_features = self.pair_source_and_target_node_features(input_graph_edges, input_graph_node_features)

        common_edges = {}
        for (inp_graph_source_index, inp_graph_target_index), inp_graph_paired_feat in input_graph_paired_node_features.items():
            presence = 0
            for (ex_graph_source_index, ex_graph_target_index), ex_graph_paired_feat in ex_graph_paired_node_features.items():
                if inp_graph_paired_feat == ex_graph_paired_feat:
                    presence = 1
            if presence == 1:
                common_edges[(inp_graph_source_index, inp_graph_target_index)] = 1
            else:
                common_edges[(inp_graph_source_index, inp_graph_target_index)] = 0

        return common_edges
    def __call__(self):
        return self.find_common_edges(self.explanation.x.tolist(), self.explanation.edge_index, self.input_graph.x.tolist(), self.input_graph.edge_index)




#common_edges_finder = global_explanation_and_samples_intersection(explanation=mutag_dataset[75], input_graph=mutag_dataset[75])
#intersection_edges = common_edges_finder()

#for key, value in intersection_edges.items():
#    print("key: ", key, "   value: ", value)

#sizes = []
#for graph in mutag_dataset:
#    sizes.append(len(graph.x))

#print(sizes.index(max(sizes)))
#print(sizes.index(min(sizes)))