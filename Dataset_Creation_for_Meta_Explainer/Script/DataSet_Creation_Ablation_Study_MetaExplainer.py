import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from jedi.inference.finder import filter_name
from torch_geometric.nn import GCNConv, global_add_pool
# from torch_geometric.data import DataLoader
import argparse
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from math import sqrt
from statistics import mean
import torch_geometric
from torch_geometric.datasets import TUDataset
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from sklearn import metrics
from scipy.spatial.distance import hamming
import statistics
import pandas
import csv
from time import perf_counter
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch_geometric.nn as gnn
from torch.autograd import graph
from typing import Any, Dict, Optional, Union
from IPython.core.display import deepcopy
from torch_geometric.nn import MessagePassing
import copy
from importlib import reload
import pickle
import random
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
from torch_geometric.data import Data, Batch, Dataset

Datasets_Name = ["MUTAG", "NCI1", "ENZYMES", "Graph-SST5", "PROTEINS", "IsCyclic"]
GNN_Models = ["GCN", "DGCNN", "DIFFPOOL", "GIN"]
GNN_Evaluations = ["AUCROC", "AUCPR", "ACC", "PREDICTION_TIME"]
Explainers = ["GNNExplainer", "SubgraphX", "PGMExplainer", "CF2", "PGExplainer", "GraphMask", "XGNN", "GNNInterpreter"]
Explainer_Evaluation = ["Fidelity+", "Fidelity-", "Contrastivity", "Sparsity", "Stability", "Explanation_Time"]


GNNs_Stats = \
    {
        "MUTAG":
            {
                "GCN": {"AUC-ROC": 0.822, "AUC-PR": 0.872, "Accuracy": 0.856, "Running Time [sec]": 0.078},
                "DGCNN": {"AUC-ROC": 0.864, "AUC-PR": 0.893, "Accuracy": 0.872, "Running Time [sec]": 0.105},
                "DIFFPOOL": {"AUC-ROC": 0.84, "AUC-PR": 0.882, "Accuracy": 0.867, "Running Time [sec]": 0.039},
                "GIN": {"AUC-ROC": 0.903, "AUC-PR": 0.916, "Accuracy": 0.899, "Running Time [sec]": 0.041}
            },
        "NCI1":
            {
                "GCN": {"AUC-ROC": 0.72, "AUC-PR": 0.8, "Accuracy": 0.689, "Running Time [sec]": 0.56},
                "DGCNN": {"AUC-ROC": 0.722, "AUC-PR": 0.797, "Accuracy": 0.732, "Running Time [sec]": 2.4},
                "DIFFPOOL": {"AUC-ROC": 0.84, "AUC-PR": 0.818, "Accuracy": 0.769, "Running Time [sec]": 1.17},
                "GIN": {"AUC-ROC": 0.83, "AUC-PR": 0.914, "Accuracy": 0.842, "Running Time [sec]": 0.1}
            },
        "ENZYMES":
            {
                "GCN": {"AUC-ROC": 0.82, "AUC-PR": 0.634, "Accuracy": 0.376, "Running Time [sec]": 0.12},
                "DGCNN": {"AUC-ROC": 0.929, "AUC-PR": 0.867, "Accuracy": 0.716, "Running Time [sec]": 0.55},
                "DIFFPOOL": {"AUC-ROC": 0.816, "AUC-PR": 0.717, "Accuracy": 0.608, "Running Time [sec]": 0.39},
                "GIN": {"AUC-ROC": 0.813, "AUC-PR": 0.765, "Accuracy": 0.578, "Running Time [sec]": 0.3}
            },
        "Graph-SST5":
            {
                "GCN": {"AUC-ROC": 0.683, "AUC-PR": 0.59, "Accuracy": 0.426, "Running Time [sec]": 0.65},
                "DGCNN": {"AUC-ROC": 0.669, "AUC-PR": 0.6, "Accuracy": 0.418, "Running Time [sec]": 2.6},
                "DIFFPOOL": {"AUC-ROC": 0.69, "AUC-PR": 0.65, "Accuracy": 0.445, "Running Time [sec]": 1.74},
                "GIN": {"AUC-ROC": 0.702, "AUC-PR": 0.638, "Accuracy": 0.433, "Running Time [sec]": 0.76}
            },
        "PROTEINS":
            {
                "GCN": {"AUC-ROC": 0.628, "AUC-PR": 0.819, "Accuracy": 0.692, "Running Time [sec]": 0.23},
                "DGCNN": {"AUC-ROC": 0.714, "AUC-PR": 0.844, "Accuracy": 0.755, "Running Time [sec]": 1.46},
                "DIFFPOOL": {"AUC-ROC": 0.742, "AUC-PR": 0.859, "Accuracy": 0.767, "Running Time [sec]": 1.09},
                "GIN": {"AUC-ROC": 0.732, "AUC-PR": 0.858, "Accuracy": 0.749, "Running Time [sec]": 0.67}
            },
        "IsCyclic":
            {
                "GCN": {"AUC-ROC": 0.921, "AUC-PR": 0.941, "Accuracy": 0.92, "Running Time [sec]": 0.122},
                "DGCNN": {"AUC-ROC": 0.916, "AUC-PR": 0.938, "Accuracy": 0.91, "Running Time [sec]": 0.137},
                "DIFFPOOL": {"AUC-ROC": 0.971, "AUC-PR": 0.978, "Accuracy": 0.97, "Running Time [sec]": 0.093},
                "GIN": {"AUC-ROC": 1.0, "AUC-PR": 1.0, "Accuracy": 1.0, "Running Time [sec]": 0.062}
            }
    }

# Example of accessing data:
# print(data['IsCyclic']['GIN']['AUC-ROC'])  # Output: 1.0
Fidelity_plus = {
    "MUTAG": {
        "GCN": {
            "GNNExplainer": 0.053,
            "SubgraphX": 0.0104,
            "PGMExplainer": 0.021,
            "CF2": 0.101,
            "PGExplainer": 0.012,
            "GraphMask": 0.009,
            "XGNN": 0.012,
            "GNNInterpreter": 0.023
        },
        "DGCNN": {
            "GNNExplainer": 0.031,
            "SubgraphX": 0.005,
            "PGMExplainer": 0.077,
            "CF2": 0.109,
            "PGExplainer": 0.079,
            "GraphMask": 0.015,
            "XGNN": 0.031,
            "GNNInterpreter": 0.017
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.009,
            "SubgraphX": 0.287,
            "PGMExplainer": 0.073,
            "CF2": 0.074,
            "PGExplainer": 0.061,
            "GraphMask": 0.061,
            "XGNN": 0.065,
            "GNNInterpreter": 0.073
        },
        "GIN": {
            "GNNExplainer": 0.165,
            "SubgraphX": 0.146,
            "PGMExplainer": 0.236,
            "CF2": 0.126,
            "PGExplainer": 0.253,
            "GraphMask": 0.064,
            "XGNN": 0.0297,
            "GNNInterpreter": 0.119
        }
    },
    "NCI1": {
        "GCN": {
            "GNNExplainer": 0.104,
            "SubgraphX": 0.117,
            "PGMExplainer": 0.109,
            "CF2": 0.063,
            "PGExplainer": 0.063,
            "GraphMask": 0.0627,
            "XGNN": 0,
            "GNNInterpreter": 0.063
        },
        "DGCNN": {
            "GNNExplainer": 0.177,
            "SubgraphX": 0.161,
            "PGMExplainer": 0.137,
            "CF2": 0.231,
            "PGExplainer": 0.231,
            "GraphMask": 0.231,
            "XGNN": 0,
            "GNNInterpreter": 0.232
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.199,
            "SubgraphX": 0.139,
            "PGMExplainer": 0.126,
            "CF2": 0.198,
            "PGExplainer": 0.198,
            "GraphMask": 0.198,
            "XGNN": 0,
            "GNNInterpreter": 0.199
        },
        "GIN": {
            "GNNExplainer": 0.276,
            "SubgraphX": 0.281,
            "PGMExplainer": 0.292,
            "CF2": 0.089,
            "PGExplainer": 0.304,
            "GraphMask": 0.089,
            "XGNN": 0,
            "GNNInterpreter": 0.311
        }
    },
    "ENZYMES": {
        "GCN": {
            "GNNExplainer": 0.134,
            "SubgraphX": 0.101,
            "PGMExplainer": 0.0078,
            "CF2": 0.001,
            "PGExplainer": 0.002,
            "GraphMask": 0.0011,
            "XGNN": 0,
            "GNNInterpreter": 0.038
        },
        "DGCNN": {
            "GNNExplainer": 0.148,
            "SubgraphX": 0.236,
            "PGMExplainer": 0.154,
            "CF2": 0.136,
            "PGExplainer": 0.136,
            "GraphMask": 0.138,
            "XGNN": 0,
            "GNNInterpreter": 0.136
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.139,
            "SubgraphX": 0.203,
            "PGMExplainer": 0.134,
            "CF2": 0.134,
            "PGExplainer": 0.134,
            "GraphMask": 0.136,
            "XGNN": 0,
            "GNNInterpreter": 0.134
        },
        "GIN": {
            "GNNExplainer": 0.162,
            "SubgraphX": 0.162,
            "PGMExplainer": 0.177,
            "CF2": -0.021,
            "PGExplainer": 0.197,
            "GraphMask": 0.0126,
            "XGNN": 0,
            "GNNInterpreter": 0.096
        }
    },
    "Graph-SST5": {
        "GCN": {
            "GNNExplainer": 0.116,
            "SubgraphX": 0,
            "PGMExplainer": 0.022,
            "CF2": 0.0012,
            "PGExplainer": 0.016,
            "GraphMask": -0.018,
            "XGNN": 0,
            "GNNInterpreter": 0.069
        },
        "DGCNN": {
            "GNNExplainer": 0.039,
            "SubgraphX": 0,
            "PGMExplainer": 0.116,
            "CF2": 0.011,
            "PGExplainer": 0.064,
            "GraphMask": -0.012,
            "XGNN": 0,
            "GNNInterpreter": 0.218
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.049,
            "SubgraphX": 0,
            "PGMExplainer": 0.063,
            "CF2": 0.021,
            "PGExplainer": 0.0202,
            "GraphMask": 0.0202,
            "XGNN": 0,
            "GNNInterpreter": 0.0263
        },
        "GIN": {
            "GNNExplainer": 0.071,
            "SubgraphX": 0,
            "PGMExplainer": 0.047,
            "CF2": 0.0158,
            "PGExplainer": 0.0939,
            "GraphMask": 0.054,
            "XGNN": 0,
            "GNNInterpreter": 0.296
        }
    },
    "PROTEINS": {
        "GCN": {
            "GNNExplainer": 0.061,
            "SubgraphX": 0.0201,
            "PGMExplainer": 0.019,
            "CF2": 0.017,
            "PGExplainer": 0.009,
            "GraphMask": 0.064,
            "XGNN": 0,
            "GNNInterpreter": 0.047
        },
        "DGCNN": {
            "GNNExplainer": 0.075,
            "SubgraphX": 0.097,
            "PGMExplainer": 0.101,
            "CF2": 0.061,
            "PGExplainer": 0.0601,
            "GraphMask": 0.0601,
            "XGNN": 0,
            "GNNInterpreter": 0.067
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.144,
            "SubgraphX": 0.0935,
            "PGMExplainer": 0.071,
            "CF2": 0.078,
            "PGExplainer": 0.079,
            "GraphMask": 0.051,
            "XGNN": 0,
            "GNNInterpreter": 0.0786
        },
        "GIN": {
            "GNNExplainer": 0.181,
            "SubgraphX": 0.1016,
            "PGMExplainer": 0.189,
            "CF2": 0.013,
            "PGExplainer": 0.237,
            "GraphMask": 0.0172,
            "XGNN": 0,
            "GNNInterpreter": 0.155
        }
    },
    "IsCyclic": {
        "GCN": {
            "GNNExplainer": 0.013,
            "SubgraphX": 0.012,
            "PGMExplainer": 0.027,
            "CF2": 0.076,
            "PGExplainer": 0.023,
            "GraphMask": 0.0012,
            "XGNN": 0.033,
            "GNNInterpreter": 0.029
        },
        "DGCNN": {
            "GNNExplainer": 0.104,
            "SubgraphX": 0.0009,
            "PGMExplainer": 0.119,
            "CF2": 0.229,
            "PGExplainer": 0.229,
            "GraphMask": 0.229,
            "XGNN": 0.229,
            "GNNInterpreter": 0.229
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.013,
            "SubgraphX": 0.0132,
            "PGMExplainer": 0.079,
            "CF2": 0.017,
            "PGExplainer": 0.043,
            "GraphMask": 0.067,
            "XGNN": 0.073,
            "GNNInterpreter": 0.051
        },
        "GIN": {
            "GNNExplainer": 0.029,
            "SubgraphX": 0.072,
            "PGMExplainer": 0.106,
            "CF2": 0.291,
            "PGExplainer": 0.479,
            "GraphMask": 0.291,
            "XGNN": 0.033,
            "GNNInterpreter": 0.479
        }
    }
}

# Example access:
# print(data['MUTAG']['GCN']['GNNExplainer'])  # Output: 0.053

Fidelity_minus = {
    "MUTAG": {
        "GCN": {
            "GNNExplainer": 0.04,
            "SubgraphX": 0,
            "PGMExplainer": 0.008,
            "CF2": -0.001,
            "PGExplainer": -0.005,
            "GraphMask": -0.003,
            "XGNN": 0.004,
            "GNNInterpreter": -0.046
        },
        "DGCNN": {
            "GNNExplainer": -0.02,
            "SubgraphX": 0.001,
            "PGMExplainer": -0.029,
            "CF2": 0,
            "PGExplainer": 0.013,
            "GraphMask": -0.002,
            "XGNN": 0.0067,
            "GNNInterpreter": 0.011
        },
        "DIFFPOOL": {
            "GNNExplainer": -0.058,
            "SubgraphX": -0.059,
            "PGMExplainer": -0.082,
            "CF2": -0.017,
            "PGExplainer": -0.044,
            "GraphMask": -0.044,
            "XGNN": -0.0446,
            "GNNInterpreter": -0.044
        },
        "GIN": {
            "GNNExplainer": -0.028,
            "SubgraphX": 0.069,
            "PGMExplainer": -0.007,
            "CF2": -0.035,
            "PGExplainer": 0.021,
            "GraphMask": -0.013,
            "XGNN": 0.008,
            "GNNInterpreter": 0.108
        }
    },
    "NCI1": {
        "GCN": {
            "GNNExplainer": 0.096,
            "SubgraphX": -0.09,
            "PGMExplainer": 0.05,
            "CF2": 0.003,
            "PGExplainer": 0.056,
            "GraphMask": 0.047,
            "XGNN": 0,
            "GNNInterpreter": 0.006
        },
        "DGCNN": {
            "GNNExplainer": 0.104,
            "SubgraphX": 0.015,
            "PGMExplainer": 0.083,
            "CF2": 0.108,
            "PGExplainer": 0.181,
            "GraphMask": 0.203,
            "XGNN": 0,
            "GNNInterpreter": 0.027
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.105,
            "SubgraphX": 0.018,
            "PGMExplainer": 0.021,
            "CF2": 0.007,
            "PGExplainer": 0.068,
            "GraphMask": 0.133,
            "XGNN": 0,
            "GNNInterpreter": -0.021
        },
        "GIN": {
            "GNNExplainer": 0.186,
            "SubgraphX": 0.0238,
            "PGMExplainer": 0.021,
            "CF2": -0.001,
            "PGExplainer": 0.185,
            "GraphMask": 0.068,
            "XGNN": 0,
            "GNNInterpreter": 0.033
        }
    },
    "ENZYMES": {
        "GCN": {
            "GNNExplainer": 0.032,
            "SubgraphX": 0.06,
            "PGMExplainer": -0.0106,
            "CF2": -0.0001,
            "PGExplainer": -0.001,
            "GraphMask": -0.001,
            "XGNN": 0,
            "GNNInterpreter": -0.0012
        },
        "DGCNN": {
            "GNNExplainer": 0.106,
            "SubgraphX": 0.026,
            "PGMExplainer": 0.096,
            "CF2": 0.014,
            "PGExplainer": 0.088,
            "GraphMask": 0.135,
            "XGNN": 0,
            "GNNInterpreter": 0.002
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.117,
            "SubgraphX": 0.029,
            "PGMExplainer": 0.049,
            "CF2": -0.001,
            "PGExplainer": 0.008,
            "GraphMask": 0.043,
            "XGNN": 0,
            "GNNInterpreter": 0.008
        },
        "GIN": {
            "GNNExplainer": 0.145,
            "SubgraphX": 0.087,
            "PGMExplainer": 0.116,
            "CF2": -0.0006,
            "PGExplainer": 0.154,
            "GraphMask": -0.007,
            "XGNN": 0,
            "GNNInterpreter": 0.008
        }
    },
    "Graph-SST5": {
        "GCN": {
            "GNNExplainer": -0.018,
            "SubgraphX": 0,
            "PGMExplainer": 0.005,
            "CF2": -0.0003,
            "PGExplainer": -0.011,
            "GraphMask": -0.014,
            "XGNN": 0,
            "GNNInterpreter": 0.011
        },
        "DGCNN": {
            "GNNExplainer": 0.005,
            "SubgraphX": 0,
            "PGMExplainer": 0.029,
            "CF2": -0.0002,
            "PGExplainer": 0.016,
            "GraphMask": -0.016,
            "XGNN": 0,
            "GNNInterpreter": 0.017
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.008,
            "SubgraphX": 0,
            "PGMExplainer": -0.027,
            "CF2": 0.003,
            "PGExplainer": 0.0069,
            "GraphMask": -0.0108,
            "XGNN": 0,
            "GNNInterpreter": 0.0021
        },
        "GIN": {
            "GNNExplainer": 0.003,
            "SubgraphX": 0,
            "PGMExplainer": 0.013,
            "CF2": -0.0006,
            "PGExplainer": 0.0205,
            "GraphMask": 0.096,
            "XGNN": 0,
            "GNNInterpreter": 0.037
        }
    },
    "PROTEINS": {
        "GCN": {
            "GNNExplainer": 0.037,
            "SubgraphX": 0.0071,
            "PGMExplainer": 0.011,
            "CF2": -0.002,
            "PGExplainer": -0.001,
            "GraphMask": -0.013,
            "XGNN": 0,
            "GNNInterpreter": 0.0041
        },
        "DGCNN": {
            "GNNExplainer": 0.031,
            "SubgraphX": 0.0096,
            "PGMExplainer": 0.009,
            "CF2": -0.006,
            "PGExplainer": 0.002,
            "GraphMask": 0.028,
            "XGNN": 0,
            "GNNInterpreter": 0.0103
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.052,
            "SubgraphX": 0.0075,
            "PGMExplainer": 0.227,
            "CF2": 0.019,
            "PGExplainer": 0.003,
            "GraphMask": -0.041,
            "XGNN": 0,
            "GNNInterpreter": 0.0013
        },
        "GIN": {
            "GNNExplainer": 0.109,
            "SubgraphX": 0.0085,
            "PGMExplainer": 0.0085,
            "CF2": -0.081,
            "PGExplainer": 0.0219,
            "GraphMask": -0.022,
            "XGNN": 0,
            "GNNInterpreter": 0
        }
    },
    "IsCyclic": {
        "GCN": {
            "GNNExplainer": 0.004,
            "SubgraphX": -0.0208,
            "PGMExplainer": 0.003,
            "CF2": -0.008,
            "PGExplainer": 0.002,
            "GraphMask": -0.007,
            "XGNN": 0.009,
            "GNNInterpreter": -0.0019
        },
        "DGCNN": {
            "GNNExplainer": -0.114,
            "SubgraphX": 0.00002,
            "PGMExplainer": 0.062,
            "CF2": -0.119,
            "PGExplainer": -0.104,
            "GraphMask": -0.171,
            "XGNN": 0.125,
            "GNNInterpreter": -0.078
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.004,
            "SubgraphX": -0.009,
            "PGMExplainer": -0.051,
            "CF2": -0.003,
            "PGExplainer": 0.009,
            "GraphMask": -0.032,
            "XGNN": 0.002,
            "GNNInterpreter": -0.019
        },
        "GIN": {
            "GNNExplainer": -0.002,
            "SubgraphX": 0.011,
            "PGMExplainer": 0.0019,
            "CF2": 0.145,
            "PGExplainer": 0.166,
            "GraphMask": 0.093,
            "XGNN": -0.208,
            "GNNInterpreter": 0.033
        }
    }
}

# Example access:
# print(data['MUTAG']['GCN']['GNNExplainer'])  # Output: 0.04

Contrastivity = {
    "MUTAG": {
        "GCN": {
            "GNNExplainer": 0.642,
            "SubgraphX": 0.5749,
            "PGMExplainer": 0.147,
            "CF2": 0.509,
            "PGExplainer": 0.5002,
            "GraphMask": 0.489,
            "XGNN": 0.6902,
            "GNNInterpreter": 0.22
        },
        "DGCNN": {
            "GNNExplainer": 0.576,
            "SubgraphX": 0.351,
            "PGMExplainer": 0.139,
            "CF2": 0.521,
            "PGExplainer": 0.504,
            "GraphMask": 0.477,
            "XGNN": 0.634,
            "GNNInterpreter": 0.191
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.501,
            "SubgraphX": 0.849,
            "PGMExplainer": 0.117,
            "CF2": 0.492,
            "PGExplainer": 0.493,
            "GraphMask": 0.753,
            "XGNN": 0.831,
            "GNNInterpreter": 0.303
        },
        "GIN": {
            "GNNExplainer": 0.637,
            "SubgraphX": 0.568,
            "PGMExplainer": 0.136,
            "CF2": 0.47,
            "PGExplainer": 0.508,
            "GraphMask": 0.528,
            "XGNN": 0.162,
            "GNNInterpreter": 0.302
        }
    },
    "NCI1": {
        "GCN": {
            "GNNExplainer": 0.575,
            "SubgraphX": 0.878,
            "PGMExplainer": 0.178,
            "CF2": 0.498,
            "PGExplainer": 0.502,
            "GraphMask": 0.5019,
            "XGNN": 0,
            "GNNInterpreter": 0.087
        },
        "DGCNN": {
            "GNNExplainer": 0.498,
            "SubgraphX": 0.666,
            "PGMExplainer": 0.137,
            "CF2": 0.501,
            "PGExplainer": 0.5003,
            "GraphMask": 0.492,
            "XGNN": 0,
            "GNNInterpreter": 0.036
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.554,
            "SubgraphX": 0.857,
            "PGMExplainer": 0.131,
            "CF2": 0.498,
            "PGExplainer": 0.503,
            "GraphMask": 0.306,
            "XGNN": 0,
            "GNNInterpreter": 0.084
        },
        "GIN": {
            "GNNExplainer": 0.578,
            "SubgraphX": 0.671,
            "PGMExplainer": 0.135,
            "CF2": 0.496,
            "PGExplainer": 0.502,
            "GraphMask": 0.4969,
            "XGNN": 0,
            "GNNInterpreter": 0.0366
        }
    },
    "ENZYMES": {
        "GCN": {
            "GNNExplainer": 0.512,
            "SubgraphX": 0.466,
            "PGMExplainer": 0.124,
            "CF2": 0.5002,
            "PGExplainer": 0.498,
            "GraphMask": 0.4599,
            "XGNN": 0,
            "GNNInterpreter": 0.476
        },
        "DGCNN": {
            "GNNExplainer": 0.488,
            "SubgraphX": 0.469,
            "PGMExplainer": 0.134,
            "CF2": 0.498,
            "PGExplainer": 0.499,
            "GraphMask": 0.499,
            "XGNN": 0,
            "GNNInterpreter": 0.234
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.506,
            "SubgraphX": 0.456,
            "PGMExplainer": 0.131,
            "CF2": 0.499,
            "PGExplainer": 0.5003,
            "GraphMask": 0.424,
            "XGNN": 0,
            "GNNInterpreter": 0.053
        },
        "GIN": {
            "GNNExplainer": 0.497,
            "SubgraphX": 0.433,
            "PGMExplainer": 0.129,
            "CF2": 0.5006,
            "PGExplainer": 0.501,
            "GraphMask": 0.4964,
            "XGNN": 0,
            "GNNInterpreter": 0.228
        }
    },
    "Graph-SST5": {
        "GCN": {
            "GNNExplainer": 0.499,
            "SubgraphX": 0,
            "PGMExplainer": 0.074,
            "CF2": 0.499,
            "PGExplainer": 0.499,
            "GraphMask": 0.494,
            "XGNN": 0,
            "GNNInterpreter": 0.113
        },
        "DGCNN": {
            "GNNExplainer": 0.493,
            "SubgraphX": 0,
            "PGMExplainer": 0.073,
            "CF2": 0.5007,
            "PGExplainer": 0.499,
            "GraphMask": 0.486,
            "XGNN": 0,
            "GNNInterpreter": 0.183
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.5,
            "SubgraphX": 0,
            "PGMExplainer": 0.074,
            "CF2": 0.4989,
            "PGExplainer": 0.5003,
            "GraphMask": 0.487,
            "XGNN": 0,
            "GNNInterpreter": 0.059
        },
        "GIN": {
            "GNNExplainer": 0.493,
            "SubgraphX": 0,
            "PGMExplainer": 0.0737,
            "CF2": 0.5001,
            "PGExplainer": 0.5007,
            "GraphMask": 0.471,
            "XGNN": 0,
            "GNNInterpreter": 0.108
        }
    },
    "PROTEINS": {
        "GCN": {
            "GNNExplainer": 0.463,
            "SubgraphX": 0.759,
            "PGMExplainer": 0.11,
            "CF2": 0.5009,
            "PGExplainer": 0.498,
            "GraphMask": 0.491,
            "XGNN": 0,
            "GNNInterpreter": 0.184
        },
        "DGCNN": {
            "GNNExplainer": 0.498,
            "SubgraphX": 0.6203,
            "PGMExplainer": 0.099,
            "CF2": 0.509,
            "PGExplainer": 0.498,
            "GraphMask": 0.494,
            "XGNN": 0,
            "GNNInterpreter": 0.069
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.442,
            "SubgraphX": 0.805,
            "PGMExplainer": 0.095,
            "CF2": 0.505,
            "PGExplainer": 0.5015,
            "GraphMask": 0.2902,
            "XGNN": 0,
            "GNNInterpreter": 0.041
        },
        "GIN": {
            "GNNExplainer": 0.434,
            "SubgraphX": 0.5608,
            "PGMExplainer": 0.103,
            "CF2": 0.499,
            "PGExplainer": 0.5008,
            "GraphMask": 0.509,
            "XGNN": 0,
            "GNNInterpreter": 0.421
        }
    },
    "IsCyclic": {
        "GCN": {
            "GNNExplainer": 0.489,
            "SubgraphX": 0.828,
            "PGMExplainer": 0.114,
            "CF2": 0.511,
            "PGExplainer": 0.503,
            "GraphMask": 0.523,
            "XGNN": 0.168,
            "GNNInterpreter": 0.321
        },
        "DGCNN": {
            "GNNExplainer": 0.496,
            "SubgraphX": 0.005,
            "PGMExplainer": 0.085,
            "CF2": 0.524,
            "PGExplainer": 0.493,
            "GraphMask": 0.465,
            "XGNN": 0.174,
            "GNNInterpreter": 0.589
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.488,
            "SubgraphX": 0.017,
            "PGMExplainer": 0.083,
            "CF2": 0.458,
            "PGExplainer": 0.498,
            "GraphMask": 0.5177,
            "XGNN": 0.196,
            "GNNInterpreter": 0.272
        },
        "GIN": {
            "GNNExplainer": 0.526,
            "SubgraphX": 0.244,
            "PGMExplainer": 0.0918,
            "CF2": 0.497,
            "PGExplainer": 0.514,
            "GraphMask": 0.482,
            "XGNN": 0.153,
            "GNNInterpreter": 0.355
        }
    }
}

# Example access:
# print(data['MUTAG']['GCN']['GNNExplainer'])  # Output: 0.642

Sparsity = {
    "MUTAG": {
        "GCN": {
            "GNNExplainer": 0.519,
            "SubgraphX": 0.6856,
            "PGMExplainer": 0.681,
            "CF2": 0.36,
            "PGExplainer": 0.237,
            "GraphMask": 0.855,
            "XGNN": 0.654,
            "GNNInterpreter": 0.861
        },
        "DGCNN": {
            "GNNExplainer": 0.515,
            "SubgraphX": 0.842,
            "PGMExplainer": 0.681,
            "CF2": 0.371,
            "PGExplainer": 0.484,
            "GraphMask": 0.704,
            "XGNN": 0.827,
            "GNNInterpreter": 0.894
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.499,
            "SubgraphX": 0.727,
            "PGMExplainer": 0.681,
            "CF2": 0.382,
            "PGExplainer": 0.522,
            "GraphMask": 0.528,
            "XGNN": 0.584,
            "GNNInterpreter": 0.841
        },
        "GIN": {
            "GNNExplainer": 0.467,
            "SubgraphX": 0.517,
            "PGMExplainer": 0.681,
            "CF2": 0.356,
            "PGExplainer": 0.5427,
            "GraphMask": 0.862,
            "XGNN": 0.918,
            "GNNInterpreter": 0.834
        }
    },
    "NCI1": {
        "GCN": {
            "GNNExplainer": 0.504,
            "SubgraphX": 0.561,
            "PGMExplainer": 0.805,
            "CF2": 0.387,
            "PGExplainer": 0.479,
            "GraphMask": 0.784,
            "XGNN": 0,
            "GNNInterpreter": 0.781
        },
        "DGCNN": {
            "GNNExplainer": 0.494,
            "SubgraphX": 0.558,
            "PGMExplainer": 0.805,
            "CF2": 0.388,
            "PGExplainer": 0.494,
            "GraphMask": 0.707,
            "XGNN": 0,
            "GNNInterpreter": 0.799
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.487,
            "SubgraphX": 0.571,
            "PGMExplainer": 0.806,
            "CF2": 0.383,
            "PGExplainer": 0.493,
            "GraphMask": 0.849,
            "XGNN": 0,
            "GNNInterpreter": 0.801
        },
        "GIN": {
            "GNNExplainer": 0.495,
            "SubgraphX": 0.5902,
            "PGMExplainer": 0.806,
            "CF2": 0.382,
            "PGExplainer": 0.536,
            "GraphMask": 0.6694,
            "XGNN": 0,
            "GNNInterpreter": 0.843
        }
    },
    "ENZYMES": {
        "GCN": {
            "GNNExplainer": 0.509,
            "SubgraphX": 0.627,
            "PGMExplainer": 0.818,
            "CF2": 0.364,
            "PGExplainer": 0.452,
            "GraphMask": 0.8995,
            "XGNN": 0,
            "GNNInterpreter": 0.671
        },
        "DGCNN": {
            "GNNExplainer": 0.518,
            "SubgraphX": 0.532,
            "PGMExplainer": 0.819,
            "CF2": 0.374,
            "PGExplainer": 0.493,
            "GraphMask": 0.908,
            "XGNN": 0,
            "GNNInterpreter": 0.785
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.514,
            "SubgraphX": 0.604,
            "PGMExplainer": 0.809,
            "CF2": 0.3609,
            "PGExplainer": 0.4901,
            "GraphMask": 0.601,
            "XGNN": 0,
            "GNNInterpreter": 0.869
        },
        "GIN": {
            "GNNExplainer": 0.501,
            "SubgraphX": 0.374,
            "PGMExplainer": 0.817,
            "CF2": 0.358,
            "PGExplainer": 0.486,
            "GraphMask": 0.919,
            "XGNN": 0,
            "GNNInterpreter": 0.885
        }
    },
    "Graph-SST5": {
        "GCN": {
            "GNNExplainer": 0.535,
            "SubgraphX": 0,
            "PGMExplainer": 0.677,
            "CF2": 0.386,
            "PGExplainer": 0.502,
            "GraphMask": 0.663,
            "XGNN": 0,
            "GNNInterpreter": 0.754
        },
        "DGCNN": {
            "GNNExplainer": 0.502,
            "SubgraphX": 0,
            "PGMExplainer": 0.676,
            "CF2": 0.389,
            "PGExplainer": 0.486,
            "GraphMask": 0.711,
            "XGNN": 0,
            "GNNInterpreter": 0.807
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.503,
            "SubgraphX": 0,
            "PGMExplainer": 0.677,
            "CF2": 0.387,
            "PGExplainer": 0.5067,
            "GraphMask": 0.671,
            "XGNN": 0,
            "GNNInterpreter": 0.843
        },
        "GIN": {
            "GNNExplainer": 0.503,
            "SubgraphX": 0,
            "PGMExplainer": 0.676,
            "CF2": 0.389,
            "PGExplainer": 0.461,
            "GraphMask": 0.757,
            "XGNN": 0,
            "GNNInterpreter": 0.803
        }
    },
    "PROTEINS": {
        "GCN": {
            "GNNExplainer": 0.548,
            "SubgraphX": 0.568,
            "PGMExplainer": 0.755,
            "CF2": 0.357,
            "PGExplainer": 0.581,
            "GraphMask": 0.651,
            "XGNN": 0,
            "GNNInterpreter": 0.873
        },
        "DGCNN": {
            "GNNExplainer": 0.562,
            "SubgraphX": 0.496,
            "PGMExplainer": 0.754,
            "CF2": 0.374,
            "PGExplainer": 0.467,
            "GraphMask": 0.687,
            "XGNN": 0,
            "GNNInterpreter": 0.901
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.552,
            "SubgraphX": 0.589,
            "PGMExplainer": 0.755,
            "CF2": 0.353,
            "PGExplainer": 0.5306,
            "GraphMask": 0.766,
            "XGNN": 0,
            "GNNInterpreter": 0.903
        },
        "GIN": {
            "GNNExplainer": 0.555,
            "SubgraphX": 0.5283,
            "PGMExplainer": 0.754,
            "CF2": 0.367,
            "PGExplainer": 0.485,
            "GraphMask": 0.741,
            "XGNN": 0,
            "GNNInterpreter": 0.894
        }
    },
    "IsCyclic": {
        "GCN": {
            "GNNExplainer": 0.474,
            "SubgraphX": 0.585,
            "PGMExplainer": 0.643,
            "CF2": 0.396,
            "PGExplainer": 0.436,
            "GraphMask": 0.868,
            "XGNN": 0.866,
            "GNNInterpreter": 0.839
        },
        "DGCNN": {
            "GNNExplainer": 0.49,
            "SubgraphX": 0.961,
            "PGMExplainer": 0.643,
            "CF2": 0.364,
            "PGExplainer": 0.481,
            "GraphMask": 0.7904,
            "XGNN": 0.897,
            "GNNInterpreter": 0.445
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.551,
            "SubgraphX": 0.899,
            "PGMExplainer": 0.644,
            "CF2": 0.391,
            "PGExplainer": 0.4958,
            "GraphMask": 0.6496,
            "XGNN": 0.851,
            "GNNInterpreter": 0.4079
        },
        "GIN": {
            "GNNExplainer": 0.506,
            "SubgraphX": 0.874,
            "PGMExplainer": 0.649,
            "CF2": 0.382,
            "PGExplainer": 0.556,
            "GraphMask": 0.873,
            "XGNN": 0.923,
            "GNNInterpreter": 0.799
        }
    }
}

# Example access:
# print(data['MUTAG']['GCN']['GNNExplainer'])  # Output: 0.519


Stability = {
    "MUTAG": {
        "GCN": {
            "GNNExplainer": 0.063,
            "SubgraphX": 0.102,
            "PGMExplainer": 0.137,
            "CF2": 0.117,
            "PGExplainer": 0.1187,
            "GraphMask": 0.222,
            "XGNN": 0.048,
            "GNNInterpreter": 0.088
        },
        "DGCNN": {
            "GNNExplainer": 0.193,
            "SubgraphX": 0.147,
            "PGMExplainer": 0.084,
            "CF2": 0.099,
            "PGExplainer": 0.112,
            "GraphMask": 0.092,
            "XGNN": 0.866,
            "GNNInterpreter": 0.074
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.061,
            "SubgraphX": 0.0259,
            "PGMExplainer": 0.029,
            "CF2": 0.106,
            "PGExplainer": 0.103,
            "GraphMask": 0.0456,
            "XGNN": 0.125,
            "GNNInterpreter": 0.147
        },
        "GIN": {
            "GNNExplainer": 0.077,
            "SubgraphX": 0.101,
            "PGMExplainer": 0.137,
            "CF2": 0.18,
            "PGExplainer": 0.5427,
            "GraphMask": 0.132,
            "XGNN": 0.0701,
            "GNNInterpreter": 0.111
        }
    },
    "NCI1": {
        "GCN": {
            "GNNExplainer": 0.045,
            "SubgraphX": 0.061,
            "PGMExplainer": 0.066,
            "CF2": 0.011,
            "PGExplainer": 0.0023,
            "GraphMask": 0.013,
            "XGNN": 0,
            "GNNInterpreter": 0.0032
        },
        "DGCNN": {
            "GNNExplainer": 0.045,
            "SubgraphX": 0.099,
            "PGMExplainer": 0.058,
            "CF2": 0.011,
            "PGExplainer": 0.0126,
            "GraphMask": 0.016,
            "XGNN": 0,
            "GNNInterpreter": 0.065
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.044,
            "SubgraphX": 0.048,
            "PGMExplainer": 0.049,
            "CF2": 0.011,
            "PGExplainer": 0.012,
            "GraphMask": 0.008,
            "XGNN": 0,
            "GNNInterpreter": 0.033
        },
        "GIN": {
            "GNNExplainer": 0.046,
            "SubgraphX": 0.08903,
            "PGMExplainer": 0.034,
            "CF2": 0.0125,
            "PGExplainer": 0.097,
            "GraphMask": 0.0126,
            "XGNN": 0,
            "GNNInterpreter": 0.0101
        }
    },
    "ENZYMES": {
        "GCN": {
            "GNNExplainer": 0.011,
            "SubgraphX": 0.029,
            "PGMExplainer": 0.0808,
            "CF2": 0.021,
            "PGExplainer": 0.021,
            "GraphMask": 0.013,
            "XGNN": 0,
            "GNNInterpreter": 0.121
        },
        "DGCNN": {
            "GNNExplainer": 0.012,
            "SubgraphX": 0.054,
            "PGMExplainer": 0.0866,
            "CF2": 0.021,
            "PGExplainer": 0.0212,
            "GraphMask": 0.019,
            "XGNN": 0,
            "GNNInterpreter": 0.033
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.013,
            "SubgraphX": 0.065,
            "PGMExplainer": 0.0615,
            "CF2": 0.022,
            "PGExplainer": 0.0213,
            "GraphMask": 0.153,
            "XGNN": 0,
            "GNNInterpreter": 0.048
        },
        "GIN": {
            "GNNExplainer": 0.011,
            "SubgraphX": 0.094,
            "PGMExplainer": 0.077,
            "CF2": 0.0223,
            "PGExplainer": 0.023,
            "GraphMask": 0.0074,
            "XGNN": 0,
            "GNNInterpreter": 0.021
        }
    },
    "Graph-SST5": {
        "GCN": {
            "GNNExplainer": 0.012,
            "SubgraphX": 0,
            "PGMExplainer": 0.011,
            "CF2": 0.0019,
            "PGExplainer": 0.0018,
            "GraphMask": 0.289,
            "XGNN": 0,
            "GNNInterpreter": 0.069
        },
        "DGCNN": {
            "GNNExplainer": 0.011,
            "SubgraphX": 0,
            "PGMExplainer": 0.0081,
            "CF2": 0.0043,
            "PGExplainer": 0.002,
            "GraphMask": 0.302,
            "XGNN": 0,
            "GNNInterpreter": 0.052
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.011,
            "SubgraphX": 0,
            "PGMExplainer": 0.0101,
            "CF2": 0.004,
            "PGExplainer": 0.013,
            "GraphMask": 0.028,
            "XGNN": 0,
            "GNNInterpreter": 0.035
        },
        "GIN": {
            "GNNExplainer": 0.012,
            "SubgraphX": 0,
            "PGMExplainer": 0.0101,
            "CF2": 0.002,
            "PGExplainer": 0.0045,
            "GraphMask": 0.312,
            "XGNN": 0,
            "GNNInterpreter": 0.019
        }
    },
    "PROTEINS": {
        "GCN": {
            "GNNExplainer": 0.007,
            "SubgraphX": 0.0318,
            "PGMExplainer": 0.024,
            "CF2": 0.023,
            "PGExplainer": 0.013,
            "GraphMask": 0.007,
            "XGNN": 0,
            "GNNInterpreter": 0.108
        },
        "DGCNN": {
            "GNNExplainer": 0.008,
            "SubgraphX": 0.0534,
            "PGMExplainer": 0.019,
            "CF2": 0.014,
            "PGExplainer": 0.0151,
            "GraphMask": 0.0064,
            "XGNN": 0,
            "GNNInterpreter": 0.022
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.006,
            "SubgraphX": 0.04805,
            "PGMExplainer": 0.025,
            "CF2": 0.013,
            "PGExplainer": 0.0153,
            "GraphMask": 0.095,
            "XGNN": 0,
            "GNNInterpreter": 0.029
        },
        "GIN": {
            "GNNExplainer": 0.007,
            "SubgraphX": 0.107,
            "PGMExplainer": 0.021,
            "CF2": 0.015,
            "PGExplainer": 0.0185,
            "GraphMask": 0.0108,
            "XGNN": 0,
            "GNNInterpreter": 0.0131
        }
    },
    "IsCyclic": {
        "GCN": {
            "GNNExplainer": 0.084,
            "SubgraphX": 0.0137,
            "PGMExplainer": 0.099,
            "CF2": 0.032,
            "PGExplainer": 0.0407,
            "GraphMask": 0.024,
            "XGNN": 0.0242,
            "GNNInterpreter": 0.104
        },
        "DGCNN": {
            "GNNExplainer": 0.062,
            "SubgraphX": 0.013,
            "PGMExplainer": 0.062,
            "CF2": 0.047,
            "PGExplainer": 0.028,
            "GraphMask": 0.0335,
            "XGNN": 0.064,
            "GNNInterpreter": 0.124
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.092,
            "SubgraphX": 0.067,
            "PGMExplainer": 0.084,
            "CF2": 0.008,
            "PGExplainer": 0.0357,
            "GraphMask": 0.0325,
            "XGNN": 0.031,
            "GNNInterpreter": 0.101
        },
        "GIN": {
            "GNNExplainer": 0.057,
            "SubgraphX": 0.0017,
            "PGMExplainer": 0.058,
            "CF2": 0.019,
            "PGExplainer": 0.051,
            "GraphMask": 0.0272,
            "XGNN": 0.045,
            "GNNInterpreter": 0.127
        }
    }
}

# Example access:
# print(data['MUTAG']['GCN']['GNNExplainer'])  # Output: 0.063

Explanation_Time = {
    "MUTAG": {
        "GCN": {
            "GNNExplainer": 0.015,
            "SubgraphX": 1866.49,
            "PGMExplainer": 10.6,
            "CF2": 0.179,
            "PGExplainer": 0.067,
            "GraphMask": 0.016,
            "XGNN": 0.2415,
            "GNNInterpreter": 0.003
        },
        "DGCNN": {
            "GNNExplainer": 0.02,
            "SubgraphX": 1039.83,
            "PGMExplainer": 33.2,
            "CF2": 0.346,
            "PGExplainer": 0.1185,
            "GraphMask": 0.0299,
            "XGNN": 0.356,
            "GNNInterpreter": 0.011
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.014,
            "SubgraphX": 3660.67,
            "PGMExplainer": 9,
            "CF2": 0.075,
            "PGExplainer": 0.012,
            "GraphMask": 0.0245,
            "XGNN": 0.162,
            "GNNInterpreter": 0.003
        },
        "GIN": {
            "GNNExplainer": 0.012,
            "SubgraphX": 296.015,
            "PGMExplainer": 8.6,
            "CF2": 0.095,
            "PGExplainer": 0.097,
            "GraphMask": 0.0378,
            "XGNN": 0.178,
            "GNNInterpreter": 0.004
        }
    },
    "NCI1": {
        "GCN": {
            "GNNExplainer": 0.048,
            "SubgraphX": 4749.14,
            "PGMExplainer": 61.4,
            "CF2": 0.227,
            "PGExplainer": 0.044,
            "GraphMask": 0.306,
            "XGNN": 0,
            "GNNInterpreter": 0.0024
        },
        "DGCNN": {
            "GNNExplainer": 0.105,
            "SubgraphX": 5371.19,
            "PGMExplainer": 94.6,
            "CF2": 0.431,
            "PGExplainer": 0.074,
            "GraphMask": 0.455,
            "XGNN": 0,
            "GNNInterpreter": 0.006
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.027,
            "SubgraphX": 4189.81,
            "PGMExplainer": 40.8,
            "CF2": 0.088,
            "PGExplainer": 0.026,
            "GraphMask": 0.276,
            "XGNN": 0,
            "GNNInterpreter": 0.0014
        },
        "GIN": {
            "GNNExplainer": 0.031,
            "SubgraphX": 912.015,
            "PGMExplainer": 102.6,
            "CF2": 0.107,
            "PGExplainer": 0.0289,
            "GraphMask": 0.311,
            "XGNN": 0,
            "GNNInterpreter": 0.002
        }
    },
    "ENZYMES": {
        "GCN": {
            "GNNExplainer": 0.161,
            "SubgraphX": 14722.35,
            "PGMExplainer": 2281.2,
            "CF2": 0.6303,
            "PGExplainer": 0.144,
            "GraphMask": 2.498,
            "XGNN": 0,
            "GNNInterpreter": 0.004
        },
        "DGCNN": {
            "GNNExplainer": 0.276,
            "SubgraphX": 15311.24,
            "PGMExplainer": 2210.4,
            "CF2": 1.282,
            "PGExplainer": 0.215,
            "GraphMask": 2.795,
            "XGNN": 0,
            "GNNInterpreter": 0.0098
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.093,
            "SubgraphX": 14469.12,
            "PGMExplainer": 3526.6,
            "CF2": 0.263,
            "PGExplainer": 0.0388,
            "GraphMask": 1.598,
            "XGNN": 0,
            "GNNInterpreter": 0.0043
        },
        "GIN": {
            "GNNExplainer": 0.133,
            "SubgraphX": 8744.201,
            "PGMExplainer": 1850.8,
            "CF2": 0.336,
            "PGExplainer": 0.029,
            "GraphMask": 2.507,
            "XGNN": 0,
            "GNNInterpreter": 0.004
        }
    },
    "Graph-SST5": {
        "GCN": {
            "GNNExplainer": 0.096,
            "SubgraphX": 20000,
            "PGMExplainer": 192.2,
            "CF2": 0.319,
            "PGExplainer": 0.198,
            "GraphMask": 0.732,
            "XGNN": 0,
            "GNNInterpreter": 0.009
        },
        "DGCNN": {
            "GNNExplainer": 0.135,
            "SubgraphX": 20000,
            "PGMExplainer": 287.54,
            "CF2": 0.409,
            "PGExplainer": 0.337,
            "GraphMask": 0.644,
            "XGNN": 0,
            "GNNInterpreter": 0.013
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.064,
            "SubgraphX": 20000,
            "PGMExplainer": 129.2,
            "CF2": 0.201,
            "PGExplainer": 0.084,
            "GraphMask": 0.247,
            "XGNN": 0,
            "GNNInterpreter": 0.00061
        },
        "GIN": {
            "GNNExplainer": 0.072,
            "SubgraphX": 20000,
            "PGMExplainer": 130.8,
            "CF2": 0.225,
            "PGExplainer": 0.124,
            "GraphMask": 1.03,
            "XGNN": 0,
            "GNNInterpreter": 0.005
        }
    },
    "PROTEINS": {
        "GCN": {
            "GNNExplainer": 0.121,
            "SubgraphX": 8330.32,
            "PGMExplainer": 336.2,
            "CF2": 0.119,
            "PGExplainer": 0.112,
            "GraphMask": 1.497,
            "XGNN": 0,
            "GNNInterpreter": 0.015
        },
        "DGCNN": {
            "GNNExplainer": 0.149,
            "SubgraphX": 8387.362,
            "PGMExplainer": 295,
            "CF2": 0.171,
            "PGExplainer": 0.192,
            "GraphMask": 1.102,
            "XGNN": 0,
            "GNNInterpreter": 0.0024
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.102,
            "SubgraphX": 7972.64,
            "PGMExplainer": 200.8,
            "CF2": 0.076,
            "PGExplainer": 0.056,
            "GraphMask": 0.997,
            "XGNN": 0,
            "GNNInterpreter": 0.0083
        },
        "GIN": {
            "GNNExplainer": 0.102,
            "SubgraphX": 1914.22,
            "PGMExplainer": 169.8,
            "CF2": 0.074,
            "PGExplainer": 0.061,
            "GraphMask": 1.023,
            "XGNN": 0,
            "GNNInterpreter": 0.0017
        }
    },
    "IsCyclic": {
        "GCN": {
            "GNNExplainer": 0.037,
            "SubgraphX": 5828.294,
            "PGMExplainer": 148.6,
            "CF2": 0.1502,
            "PGExplainer": 0.065,
            "GraphMask": 1.187,
            "XGNN": 0.151,
            "GNNInterpreter": 0.003
        },
        "DGCNN": {
            "GNNExplainer": 0.127,
            "SubgraphX": 3854.88,
            "PGMExplainer": 225.2,
            "CF2": 0.261,
            "PGExplainer": 0.113,
            "GraphMask": 1.335,
            "XGNN": 0.1902,
            "GNNInterpreter": 0.0109
        },
        "DIFFPOOL": {
            "GNNExplainer": 0.053,
            "SubgraphX": 3854.884,
            "PGMExplainer": 73.4,
            "CF2": 0.081,
            "PGExplainer": 0.0101,
            "GraphMask": 1.142,
            "XGNN": 0.118,
            "GNNInterpreter": 0.0033
        },
        "GIN": {
            "GNNExplainer": 0.058,
            "SubgraphX": 324.888,
            "PGMExplainer": 75.8,
            "CF2": 0.089,
            "PGExplainer": 0.013,
            "GraphMask": 1.1838,
            "XGNN": 0.105,
            "GNNInterpreter": 0.0053
        }
    }
}


# Example access:
# print(data['MUTAG']['GCN']['GNNExplainer'])  # Output: 0.015
def sort_explainers(data, descending):
    sort_dict = {}
    score_dict = {}
    for dataset, gnn_models in data.items():
        sort_dict[dataset] = {}
        score_dict[dataset] = {}
        for gnn_model, explainers in gnn_models.items():
            sorted_explainers = dict(sorted(explainers.items(), key=lambda item: item[1], reverse=descending))
            sort_dict[dataset][gnn_model] = sorted_explainers
            score_dict[dataset][gnn_model] = {}
            for rank, (key, value) in enumerate(sort_dict[dataset][gnn_model].items(), start=1):
                new_value = 1 - ((rank - 1) / 8)
                score_dict[dataset][gnn_model][key] = new_value
    return score_dict, sort_dict

example_dict = {
    "MUTAG": {
        "GCN": {
            "GNNExplainer": 0.058,
            "SubgraphX": 324.888,
            "PGMExplainer": 75.8,
            "CF2": 0.089,
            "PGExplainer": 0.013,
            "GraphMask": 1.1838,
            "XGNN": 0.105,
            "GNNInterpreter": 0.0053

        },

    },

}
import itertools
Fidelity_plus_score, Fidelity_plus_sorted = sort_explainers(Fidelity_plus, descending=True)
Fidelity_minus_score, Fidelity_minus_sorted = sort_explainers(Fidelity_minus, descending=False)
Contrastivity_score, Contrastivity_sorted = sort_explainers(Contrastivity, descending=True)
Sparsity_score, Sparsity_sorted = sort_explainers(Sparsity, descending=True)
Stability_score, Stability_sorted = sort_explainers(Stability, descending=True)
Explanation_Time_score, Explanation_Time_sorted = sort_explainers(Explanation_Time, descending=False)

# example_dict_score, example_dict_sorted = sort_explainers(example_dict, descending=True)
# for dataset_name, gnn_model in example_dict_score.items():
#     print(dataset_name)
#     print("     ", gnn_model.keys())
#     for explainer, score in gnn_model.items():
#         print(explainer, score)

weight_levels = [1, 2, 3, 4]
num_metrics = 6
ablation_index = 5
all_cases = []
for values in itertools.product(weight_levels, repeat=num_metrics-1):
    weights = list(values)
    weights.insert(ablation_index, 0)
    all_cases.append(tuple(weights))
experiment_title = "Ablation_Study"
weights_distribution = "_GridSearch_Weights_"
file_names = experiment_title + weights_distribution + '_'.join(str(i) for i in weight_levels) + "_ablated_index_" + str(ablation_index)

stats_dict = {"Fidelity+": Fidelity_plus_score, "Fidelity-": Fidelity_minus_score, "Contrastivity": Contrastivity_score,
              "Sparsity": Sparsity_score, "Stability": Stability_score, "ExplanationTime": Explanation_Time_score}
Label_Dict = {}
for dataset_name in Datasets_Name:
    Label_Dict[dataset_name] = {}
    for gnn_name in GNN_Models:

        Label_Dict[dataset_name][gnn_name] = {}
        for weights in all_cases:
            Label_Dict[dataset_name][gnn_name][weights] = {}
            for explainer_name in Explainers:

                Final_Score = (weights[0]*stats_dict["Fidelity+"][dataset_name][gnn_name][explainer_name] +
                               weights[1]*stats_dict["Fidelity-"][dataset_name][gnn_name][explainer_name] +
                               weights[2]*stats_dict["Contrastivity"][dataset_name][gnn_name][explainer_name] +
                               weights[3]*stats_dict["Sparsity"][dataset_name][gnn_name][explainer_name] +
                               weights[4]*stats_dict["Stability"][dataset_name][gnn_name][explainer_name] +
                               weights[5]*stats_dict["ExplanationTime"][dataset_name][gnn_name][explainer_name])
                Label_Dict[dataset_name][gnn_name][weights][explainer_name] = Final_Score

print("num_datasets * num_models * possible worlds= ", len(Label_Dict.keys()) * len(Label_Dict["MUTAG"].keys()) * len(Label_Dict["MUTAG"]["DGCNN"].keys()))


directory = ("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Representation_Learning/Experimental Results/" +
             "dataset_name_2_representation.pkl")
with open(directory, 'rb') as file:
    dataset_representation = pickle.load(file)
# for key, value in dataset_representation.items():
# print(key, value)

structured_dict = {}
for weights in all_cases:
    for dataset_name in Datasets_Name:
        structured_dict[dataset_name] = {}
        for gnn_name in GNN_Models:
            structured_dict[dataset_name][gnn_name] = {}
for weights in all_cases:
    for dataset_name in Datasets_Name:
        for gnn_name in GNN_Models:
            structured_dict[dataset_name][gnn_name][weights] = {}
            for explainer_name in Explainers:
                structured_dict[dataset_name][gnn_name][weights][explainer_name] = {}
                structured_dict[dataset_name][gnn_name][weights][explainer_name]["features"] = [

                    weights[0]*stats_dict["Fidelity+"][dataset_name][gnn_name][explainer_name],
                    weights[1]*stats_dict["Fidelity-"][dataset_name][gnn_name][explainer_name],
                    weights[2]*stats_dict["Contrastivity"][dataset_name][gnn_name][explainer_name],
                    weights[3]*stats_dict["Sparsity"][dataset_name][gnn_name][explainer_name],
                    weights[4]*stats_dict["Stability"][dataset_name][gnn_name][explainer_name],
                    weights[5]*stats_dict["ExplanationTime"][dataset_name][gnn_name][explainer_name]
                ]
                # structured_dict[dataset_name][gnn_name][weights]["label"] = max(Label_Dict[dataset_name][gnn_name][weights], key=Label_Dict[dataset_name][gnn_name][weights].get)

# print(structured_dict.keys())
# print(structured_dict["MUTAG"].keys())
# print(structured_dict["MUTAG"]["GCN"].keys())
# print(structured_dict["MUTAG"]["GCN"][(1, 1, 1, 1, 1, 1)].keys())
# print(structured_dict["MUTAG"]["GCN"][(1, 1, 1, 1, 1, 1)]["GNNExplainer"]["features"])
# print(structured_dict["MUTAG"]["GCN"][(1, 1, 1, 1, 1, 1)])



Data = {}
for dataset_name in structured_dict.keys():
    Data[dataset_name] = {}
    for gnn_name in structured_dict[dataset_name].keys():
        Data[dataset_name][gnn_name] = {}

        for weights in all_cases:
            temp = []
            Data[dataset_name][gnn_name][weights] = {}
            temp.extend(dataset_representation[dataset_name].tolist())
            temp.extend([GNNs_Stats[dataset_name][gnn_name]["AUC-ROC"],
                         GNNs_Stats[dataset_name][gnn_name]["AUC-PR"],
                         GNNs_Stats[dataset_name][gnn_name]["Accuracy"],
                         GNNs_Stats[dataset_name][gnn_name]["Running Time [sec]"]])
            for explainer_name in structured_dict[dataset_name][gnn_name][weights].keys():

                # print(structured_dict[dataset_name][gnn_name][weights][explainer_name]["features"])
                temp.extend(structured_dict[dataset_name][gnn_name][weights][explainer_name]["features"])
            Data[dataset_name][gnn_name][weights]["features"] = temp
            Data[dataset_name][gnn_name][weights]["label"] = max(Label_Dict[dataset_name][gnn_name][weights], key=Label_Dict[dataset_name][gnn_name][weights].get)

# print(Data.keys())
# print(Data["MUTAG"].keys())
# print(Data["MUTAG"]["GCN"].keys())
# print(Data["MUTAG"]["GCN"][(1, 1, 1, 1, 1, 1)].keys())
# print(Data["MUTAG"]["GCN"][(1, 1, 1, 1, 1, 1)]["GNNExplainer"]["features"])
# print(Data["MUTAG"]["GCN"][(1, 1, 1, 1, 1, 1)]["label"])
# print(Data["MUTAG"]["GCN"][(1, 1, 1, 1, 1, 1)])
X_data = []
Y_data = []
Y_index = []

for dataset_name in structured_dict.keys():
    for gnn_name in structured_dict[dataset_name].keys():
        for weights in all_cases:
            X_data.append(Data[dataset_name][gnn_name][weights]["features"])
            Y_index.append(Explainers.index(Data[dataset_name][gnn_name][weights]["label"]))

Y_data = np.eye(8)[Y_index]
# print(X_data)
# print(Y_data)

X_data = np.array(X_data)

min_vals = X_data.min(axis=0)
max_vals = X_data.max(axis=0)
ranges = max_vals - min_vals
ranges[ranges == 0] = 1
normalized_data = (X_data - min_vals) / ranges
normalized_data_list = normalized_data.tolist()

X = []
Y = []
for lst in X_data:
    lst_np = np.array(lst)
    X.append(torch.from_numpy(lst_np))
for lbl in Y_data:
    lbl_np = np.array(lbl)
    Y.append(torch.from_numpy(lbl_np))
directory_x = ("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Creation_for_MetaExplainer/Experimental Results/" +
               "X_" + file_names + ".pt")
directory_y = ("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Creation_for_MetaExplainer/Experimental Results/" +
               "Y_" + file_names + ".pt")
torch.save(X, directory_x)
torch.save(Y, directory_y)