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
import itertools
import random
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
from torch_geometric.data import Data, Batch, Dataset
from collections import Counter, defaultdict

# ============================================================
# Load Meta-Explainer constants
# ============================================================
constants_file = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
    "Constants/"
    "Meta_Explainer_Constants.pkl"
)
with open(constants_file, "rb") as f:
    constants = pickle.load(f)

Datasets_Name = constants["Datasets_Name"]
GNN_Models = constants["GNN_Models"]
GNN_Evaluations_Metrics = constants["GNN_Evaluations_Metrics"]
Explainers = constants["Explainers"]
Explainer_Evaluation_Metrics = constants["Explainer_Evaluation_Metrics"]

expected_feature_size_for_meta_explainer = constants["expected_feature_size_for_meta_explainer"]

num_explainers = constants["num_explainers"]

Datasets_Stats = constants["Datasets_Stats"]
Dataset_Stats_Features = constants["Dataset_Stats_Features"]
GNNs_Evaluation_Stats = constants["GNNs_Evaluation_Stats"]

Fidelity_plus = constants["Fidelity_plus"]
Fidelity_minus = constants["Fidelity_minus"]
Contrastivity = constants["Contrastivity"]
Sparsity = constants["Sparsity"]
Stability = constants["Stability"]
Explanation_RunTime = constants["Explanation_RunTime"]

random_state = 42
number_of_continuous_weight_vectors = 4000
number_of_metrics = 6



np.random.seed(random_state)
torch.manual_seed(random_state)


# ============================================================
# Directories
# ============================================================
base_dataset_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
)

continuous_output_directory = base_dataset_results_directory + "Continuous_Weights/"
os.makedirs(continuous_output_directory, exist_ok=True)


# ============================================================
# Load GNN parameter counts
# ============================================================
gnn_parameter_count_file = base_dataset_results_directory + "Constants/" + "GNN_Parameter_Counts.pkl"
with open(gnn_parameter_count_file, "rb") as file:
    GNN_Parameter_Counts = pickle.load(file)


# ============================================================
# Load dataset latent representations
# ============================================================
dataset_latent_features_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Representation_Learning/"
    "Experimental Results/"
)
dataset_latent_representation_file = dataset_latent_features_directory + "dataset_name_2_representation.pkl"
with open(dataset_latent_representation_file, "rb") as file:
    Dataset_Latent_Representations = pickle.load(file)


# ============================================================
# Feature Helpers
# ============================================================
def get_dataset_latent_features(dataset_name):
    latent = Dataset_Latent_Representations[dataset_name]
    if isinstance(latent, torch.Tensor):
        latent = (latent.detach().cpu().numpy())
    latent = np.asarray(latent, dtype=np.float32).reshape(-1)
    assert len(latent) == 32, (
        f"{dataset_name}: "
        f"expected 32 latent features, "
        f"found {len(latent)}"
    )
    return latent


def get_dataset_stat_features(dataset_name):
    stats = Datasets_Stats[dataset_name]
    return np.array(
        [
            stats["Num_Graphs"],
            stats["Avg_Num_Nodes"],
            stats["Avg_Num_Edges"],
            stats["Num_Node_Features"],
            stats["Num_Classes"]
        ],
        dtype=np.float32
    )


def get_gnn_evaluation_features(dataset_name, gnn_name):
    stats = GNNs_Evaluation_Stats[dataset_name][gnn_name]
    return np.array(
        [
            stats["AUC-ROC"],
            stats["AUC-PR"],
            stats["Accuracy"],
            stats["Running Time [sec]"]
        ],
        dtype=np.float32
    )


def get_gnn_parameter_feature(dataset_name, gnn_name):
    parameter_count = (GNN_Parameter_Counts[dataset_name][gnn_name])
    return np.array([parameter_count], dtype=np.float32)


# ============================================================
# label generation, same procedure used for grid-search
# ============================================================
def sort_explainers(data, descending):
    sort_dict = {}
    score_dict = {}
    for dataset, gnn_models in data.items():
        sort_dict[dataset] = {}
        score_dict[dataset] = {}
        for gnn_model, explainers in gnn_models.items():
            sorted_explainers = dict(
                sorted(explainers.items(), key=lambda item: item[1], reverse=descending))
            sort_dict[dataset][gnn_model] = sorted_explainers
            score_dict[dataset][gnn_model] = {}
            for rank, (explainer_name, value) in enumerate(sorted_explainers.items(), start=1):
                new_value = (1 - ((rank - 1) / num_explainers))
                score_dict[dataset][gnn_model][explainer_name] = new_value

    return (score_dict, sort_dict)


# ============================================================
# Rank-score dictionaries
# ============================================================
(
    Fidelity_plus_score,
    Fidelity_plus_sorted
) = sort_explainers(Fidelity_plus, descending=True)

(
    Fidelity_minus_score,
    Fidelity_minus_sorted
) = sort_explainers(Fidelity_minus, descending=False)

(
    Contrastivity_score,
    Contrastivity_sorted
) = sort_explainers(Contrastivity, descending=True)

(
    Sparsity_score,
    Sparsity_sorted
) = sort_explainers(Sparsity, descending=True)

(
    Stability_score,
    Stability_sorted
) = sort_explainers(Stability, descending=True)

(
    Explanation_RunTime_score,
    Explanation_RunTime_sorted
) = sort_explainers(Explanation_RunTime, descending=False)


Explainer_Score_Dicts = {
    "Fidelity+": Fidelity_plus_score,
    "Fidelity-": Fidelity_minus_score,
    "Contrastivity": Contrastivity_score,
    "Sparsity": Sparsity_score,
    "Stability": Stability_score,
    "Explanation_RunTime": Explanation_RunTime_score
}


# ============================================================
# Generate continuous preference vectors
# ============================================================
def generate_continuous_weights(number_of_weight_vectors, number_of_metrics, random_state):
    rng = np.random.default_rng(random_state)
    continuous_weights = rng.dirichlet(alpha=np.ones(number_of_metrics), size=number_of_weight_vectors)
    return continuous_weights


# ============================================================
# Calculate label
# ============================================================
def find_best_explainer_weighted_average(dataset_name, gnn_name, normalized_weights):
    final_scores = {}
    for explainer in Explainers:
        metric_scores = np.array(
            [Explainer_Score_Dicts[metric_name][dataset_name][gnn_name][explainer]
                for metric_name in Explainer_Evaluation_Metrics], dtype=np.float64)
        final_score = np.sum(normalized_weights * metric_scores)
        final_scores[explainer] = final_score
    best_explainer = max(final_scores, key=final_scores.get)
    best_explainer_index = (Explainers.index(best_explainer))
    return (best_explainer_index, best_explainer, final_scores)


# ============================================================
# Create one 48-D feature vector
# ============================================================
def create_meta_explainer_features(dataset_name, gnn_name, normalized_weights):
    temp = []
    temp.extend(get_dataset_latent_features(dataset_name).tolist())
    temp.extend(get_dataset_stat_features(dataset_name).tolist())
    temp.extend(get_gnn_evaluation_features(dataset_name, gnn_name).tolist())
    temp.extend(get_gnn_parameter_feature(dataset_name, gnn_name).tolist())
    temp.extend(normalized_weights.tolist())
    assert (len(temp) == expected_feature_size_for_meta_explainer), (
        f"Expected "
        f"{expected_feature_size_for_meta_explainer} "
        f"features, got {len(temp)}"
    )

    return temp


# ============================================================
# Generate ONE fixed continuous weight set
# ============================================================
continuous_weights = generate_continuous_weights(number_of_weight_vectors=number_of_continuous_weight_vectors,
                                                 number_of_metrics=number_of_metrics, random_state=random_state)
print("\nContinuous weight matrix shape:", continuous_weights.shape)
print("First continuous preference vector:")
print(continuous_weights[0])
print("Sum:", continuous_weights[0].sum())


# ============================================================
# Continuous Weight Sanity Checks
# ============================================================
assert np.all(continuous_weights >= 0)
assert np.allclose(continuous_weights.sum(axis=1), 1.0)
print("Continuous-weight sanity checks passed.")





# ============================================================
# Verify continuous weights are not exact grid-search weights
# ============================================================
grid_weight_levels = [1, 2, 3, 4]
raw_grid_cases = list(itertools.product(grid_weight_levels, repeat=number_of_metrics))
grid_normalized_weights = set()
for weights in raw_grid_cases:
    weights_array = np.array(weights, dtype=np.float64)
    normalized = (weights_array / weights_array.sum())
    key = tuple(np.round(normalized, decimals=12))
    grid_normalized_weights.add(key)
continuous_grid_matches = 0
for weights in continuous_weights:
    key = tuple(np.round(weights, decimals=12))
    if key in grid_normalized_weights:
        continuous_grid_matches += 1
print(
    "Continuous vectors matching "
    "grid configurations:",
    continuous_grid_matches
)

assert continuous_grid_matches == 0, (
    "At least one continuous preference vector "
    "matches a grid-search preference vector."
)
# ============================================================
# Create Structured Continuous Dataset
# ============================================================
Continuous_Meta_Explainer_Data = {}
for dataset_name in Datasets_Name:
    Continuous_Meta_Explainer_Data[dataset_name] = {}
    for gnn_name in GNN_Models:
        Continuous_Meta_Explainer_Data[dataset_name][gnn_name] = {}
        for (weight_index, normalized_weights) in enumerate(continuous_weights):
            features = (create_meta_explainer_features(dataset_name=dataset_name, gnn_name=gnn_name,
                                                       normalized_weights=normalized_weights))
            (
                best_explainer_index,
                best_explainer,
                final_scores
            ) = find_best_explainer_weighted_average(dataset_name, gnn_name, normalized_weights)
            Continuous_Meta_Explainer_Data[dataset_name][gnn_name][weight_index] = {
                "features": features,
                "weights": normalized_weights.tolist(),
                "label": best_explainer,
                "label_index": best_explainer_index,
                "final_scores": final_scores
            }


# ============================================================
# Final Sanity Checks
# ============================================================
for dataset_name in Datasets_Name:
    for gnn_name in GNN_Models:
        for (weight_index, sample) in Continuous_Meta_Explainer_Data[dataset_name][gnn_name].items():
            assert (len(sample["features"]) == expected_feature_size_for_meta_explainer)
            assert (Explainers[sample["label_index"]] == sample["label"])
            assert (max(sample["final_scores"], key=sample["final_scores"].get) == sample["label"])
            weights = sample["features"][-6:]
            assert np.isclose(sum(weights),1.0)
            assert np.allclose(weights, sample["weights"])
            assert np.all(np.isfinite(np.array(sample["features"], dtype=np.float64)))
print(
    "All continuous Meta-Explainer "
    "samples passed sanity checks."
)


# ============================================================
# Convert Structured Dictionary to X and Y
# ============================================================
X = []
Y = []
for dataset_name in Datasets_Name:
    for gnn_name in GNN_Models:
        for weight_index in (
                Continuous_Meta_Explainer_Data[dataset_name][gnn_name].keys()):
            sample = (Continuous_Meta_Explainer_Data[dataset_name][gnn_name][weight_index])
            feature_array = np.array(sample["features"], dtype=np.float32)
            X.append(torch.from_numpy(feature_array))
            Y.append(torch.tensor(sample["label_index"], dtype=torch.long))

# ============================================================
# Tensor Information
# ============================================================
X_tensor = torch.stack(X)
Y_tensor = torch.stack(Y)
print("\nX shape:", X_tensor.shape)
print("Y shape:", Y_tensor.shape)
print("Number of samples:", len(X))
print("Number of features:", X_tensor.shape[1])
print("Unique classes:", torch.unique(Y_tensor))


# ============================================================
# Expected number of samples
#
# 4000 preference vectors
# x 6 datasets
# x 4 GNNs
# ============================================================
expected_samples = (number_of_continuous_weight_vectors * len(Datasets_Name) * len(GNN_Models))
assert (len(X) == expected_samples)
print("Expected number of samples:", expected_samples)


# ============================================================
# Class Distribution
# ============================================================
counts = Counter(Y_tensor.tolist())
total = len(Y_tensor)
print("\nContinuous-test class distribution:")

for class_index in range(len(Explainers)):
    count = counts.get(class_index, 0)
    print(
        f"  "
        f"{class_index} "
        f"({Explainers[class_index]}): "
        f"{count} "
        f"({count / total:.2%})"
    )


# ============================================================
# Save
# ============================================================
x_file = continuous_output_directory + "X_ContinuousWeights_" + str(
    number_of_continuous_weight_vectors) + "_WeightVectors.pt"
y_file = continuous_output_directory + "Y_ContinuousWeights_" + str(
    number_of_continuous_weight_vectors) + "_WeightVectors.pt"
structured_data_file = continuous_output_directory + "Meta_Explainer_Data_ContinuousWeights_" + str(
    number_of_continuous_weight_vectors) + "_WeightVectors.pkl"
continuous_weights_file = continuous_output_directory + "Continuous_Preference_Weights_" + str(
    number_of_continuous_weight_vectors) + "_WeightVectors.pkl"
torch.save(X, x_file)
torch.save(Y, y_file)


with open(structured_data_file, "wb") as file:
    pickle.dump(Continuous_Meta_Explainer_Data, file)
with open(continuous_weights_file, "wb") as file:
    pickle.dump(continuous_weights, file)
print("\nSaved:")
print("X:", x_file)
print("Y:", y_file)
print("Structured Data:", structured_data_file)
print("Continuous Weights:", continuous_weights_file)