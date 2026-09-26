import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
import random
from importlib import reload
import pickle
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
from torch_geometric.data import Data, Batch, Dataset


class load_my_dataset:
    def __init__(self, dataset_name, BATCH_SIZE):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.BATCH_SIZE = BATCH_SIZE
    def __call__(self):
        if self.dataset_name == "Graph-SST5":
            from dig.xgraph.dataset import SentiGraphDataset
            py_path = '/data/cs.aau.dk/ey33jw/'
            os.chdir(py_path)
            current_directory = os.getcwd()
            entire_dataset = SentiGraphDataset(root='./Datasets_for_Explainability_Methods/', name='Graph-SST5')
            py_path = '/data/cs.aau.dk/ey33jw/Explainability_Methods/'
            os.chdir(py_path)
            # current_directory = os.getcwd()
        elif self.dataset_name == "IsCyclic":
            with open("/data/cs.aau.dk/ey33jw/Datasets_for_Explainability_Methods/IsCyclic/iscyclic_graphs.pkl",
                      'rb') as f:
                entire_dataset = pickle.load(f)
        else:
            entire_dataset = TUDataset(root='data/TUDataset', name=self.dataset_name)

        labels = []
        for graph in entire_dataset:
            labels.append(int(graph.y))
        y_true = np.array(labels)
        unique_classes = np.unique(y_true)
        num_classes = len(unique_classes)
        # print("num_classes: ", num_classes)

        df = pandas.read_csv("/data/cs.aau.dk/ey33jw/Datasets_for_Explainability_Methods/" +
                             "Train and Test Indexes on Graph Classification/Experimental Results/train_test_indexes_" +
                             str(self.dataset_name) + ".csv")

        read_training_list_indexes__ = df['Train Indexes']
        read_test_list_indexes__ = df['Test Indexes']
        read_test_list_indexes__ = read_test_list_indexes__.dropna()
        read_test_list_indexes = []
        read_training_list_indexes = []
        for element in read_test_list_indexes__:
            read_test_list_indexes.append(int(element))
        for element in read_training_list_indexes__:
            read_training_list_indexes.append(int(element))

        # print(read_training_list_indexes)
        # print(read_test_list_indexes)

        train_dataset = []
        test_dataset = []
        for index in read_training_list_indexes:
            train_dataset.append(entire_dataset[index])
        for index in read_test_list_indexes:
            test_dataset.append(entire_dataset[index])

        #########################################################################################                    BUG
        if self.dataset_name == "ENZYMES":
            del test_dataset[13]
        if self.dataset_name == "NCI1":
            del test_dataset[751]
        # if self.dataset_name == "Graph-SST5":
        #     del test_dataset[411]

        # print(f'Number of training graphs: {len(train_dataset)}')
        # print(f'Number of test graphs: {len(test_dataset)}')
        #
        # print(train_dataset[0])
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        # print(train_dataloader.batch_size)
        # batch = next(iter(train_dataloader))
        # print(batch.y)
        # print(len(train_dataloader))

        return train_dataset, test_dataset, train_dataloader, test_dataloader, num_classes, entire_dataset


class shared_graph_autoencoder(nn.Module):
    def __init__(self, dataset_feature_dims, shared_input_dim, hidden_channels, latent_size):
        super(shared_graph_autoencoder, self).__init__()

        self.input_adapters = nn.ModuleDict({
            dataset_name: nn.Linear(
                feature_dim,
                shared_input_dim
            )
            for dataset_name, feature_dim
            in dataset_feature_dims.items()
        })

        self.input_norm = nn.LayerNorm(shared_input_dim)

        self.conv1 = GCNConv(shared_input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.graph_norm = nn.LayerNorm(hidden_channels)
        self.fc = nn.Linear(hidden_channels, latent_size)

        self.decoder = nn.Linear(latent_size, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, shared_input_dim)

        self.output_adapters = nn.ModuleDict({
            dataset_name: nn.Linear(
                shared_input_dim,
                feature_dim
            )
            for dataset_name, feature_dim
            in dataset_feature_dims.items()
        })

    def encode(self, x, edge_index, batch, dataset_name):
        x = self.input_adapters[dataset_name](x)
        x = self.input_norm(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.graph_norm(x)
        latent_data = self.fc(x)
        return latent_data

    def decode(self, z, edge_index, batch, dataset_name):
        z = F.relu(self.decoder(z))
        z = z[batch]
        z = self.conv3(z, edge_index)
        reconstructed_x = self.output_adapters[dataset_name](z)
        return reconstructed_x

    def forward(self, x, edge_index, batch, dataset_name):
        latent_representation = self.encode(x, edge_index, batch, dataset_name)
        reconstructed_x = self.decode(latent_representation, edge_index, batch, dataset_name)
        return (reconstructed_x, latent_representation)


class shared_graph_autoencoder_trainer:
    def __init__(self, shared_graph_ae, datasets_dict, batch_size, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_ae = shared_graph_ae.to(self.device)
        self.optimizer = optim.Adam(self.graph_ae.parameters(),lr=lr)

        self.data_loaders = {
            dataset_name: DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for dataset_name, dataset in datasets_dict.items()
        }
    def train(self, epochs):
        for epoch in range(epochs):

            self.graph_ae.train()

            total_loss = 0.0
            total_updates = 0

            dataset_losses = {}
            batches_per_dataset = min(
                len(loader)
                for loader in self.data_loaders.values()
            )

            dataset_names = list(self.data_loaders.keys())

            random.shuffle(dataset_names)
            for dataset_name in dataset_names:

                data_loader = self.data_loaders[dataset_name]

                data_iterator = iter(data_loader)

                dataset_total_loss = 0.0
                dataset_updates = 0

                for _ in range(batches_per_dataset):

                    data = next(data_iterator)
                    data = data.to(self.device)

                    self.optimizer.zero_grad()

                    reconstructed_x, latent_representation = (
                        self.graph_ae(data.x, data.edge_index, data.batch, dataset_name))

                    loss = F.l1_loss(reconstructed_x, data.x)
                    loss.backward()
                    self.optimizer.step()
                    dataset_total_loss += (loss.item())
                    dataset_updates += 1

                    total_loss += (loss.item())
                    total_updates += 1

                dataset_losses[dataset_name] = (dataset_total_loss / dataset_updates)

            avg_loss = (total_loss / total_updates)
            print(
                f"Epoch {epoch + 1:04d} | "
                f"Average Loss: {avg_loss:.6f} | "
                f"Batches per Dataset: "
                f"{batches_per_dataset}"
            )
            for (dataset_name, dataset_loss) in dataset_losses.items():
                print(
                    f"    "
                    f"{dataset_name:12s}: "
                    f"{dataset_loss:.6f}"
                )

    def compute_dataset_representation(self, dataset_name, dataset, batch_size):
        self.graph_ae.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        graph_representations = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                latent_data = self.graph_ae.encode(data.x, data.edge_index, data.batch, dataset_name)
                graph_representations.append(latent_data.cpu())
        stacked_representations = torch.cat(graph_representations, dim=0)
        dataset_representation = torch.mean(stacked_representations, dim=0)

        return dataset_representation



# ============================================================
# Meta-Explainer HyperParameters
# ============================================================
gae_epoch = 2000
batch_size = 32
classifier_weight_decay = 1e-6
ae_lr = 0.001
dataset_representation_size = 32
dataset_name_2_representation = {}
datasets_name = ["MUTAG", "NCI1", "ENZYMES", "Graph-SST5", "PROTEINS", "IsCyclic"]


# ============================================================
# Load all datasets
# ============================================================
datasets_dict = {}
dataset_feature_dims = {}

for dataset_name in datasets_name:
    loading_dataset = load_my_dataset(dataset_name=dataset_name, BATCH_SIZE=batch_size)
    (train_dataset, test_dataset, train_dataloader, test_dataloader, num_classes, entire_dataset) = loading_dataset()
    datasets_dict[dataset_name] = entire_dataset
    dataset_feature_dims[dataset_name] = (entire_dataset[0].x.size(-1))

print("Dataset feature dimensions:")
print(dataset_feature_dims)

# ============================================================
# Ensure Dataset Scales
# ============================================================
for dataset_name, dataset in datasets_dict.items():
    all_x = torch.cat([graph.x for graph in dataset], dim=0)

    print(
        dataset_name,
        "min:", all_x.min().item(),
        "max:", all_x.max().item(),
        "mean:", all_x.mean().item(),
        "std:", all_x.std().item()
    )


# ============================================================
# Graph AutoEncoder
# ============================================================
shared_input_dim = 64
hidden_channels = 64

shared_graph_ae = shared_graph_autoencoder(dataset_feature_dims=dataset_feature_dims, shared_input_dim=shared_input_dim,
                                           hidden_channels=hidden_channels, latent_size=dataset_representation_size)


# ============================================================
# Graph AutoEncoder Train
# ============================================================
trainer = shared_graph_autoencoder_trainer(shared_graph_ae=shared_graph_ae, datasets_dict=datasets_dict,
                                           batch_size=batch_size, lr=ae_lr)
print("Device:", trainer.device)
print("Model device:", next(trainer.graph_ae.parameters()).device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

trainer.train(epochs=gae_epoch)


# ============================================================
# Dataset Representation
# ============================================================
for dataset_name in datasets_name:
    dataset_representation = (
        trainer.compute_dataset_representation(dataset_name=dataset_name, dataset=datasets_dict[dataset_name],
                                               batch_size=batch_size))
    print(f"Representation of {dataset_name}:")
    print(dataset_representation)
    dataset_name_2_representation[dataset_name] = dataset_representation

directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Representation_Learning/"
    "Experimental Results/"
    "dataset_name_2_representation.pkl"
)

with open(directory, "wb") as write_file:
    pickle.dump(dataset_name_2_representation, write_file)

with open(directory, 'rb') as read_file:
    loaded_dict = pickle.load(read_file)

print("Dictionary loaded from file:")
print(loaded_dict)