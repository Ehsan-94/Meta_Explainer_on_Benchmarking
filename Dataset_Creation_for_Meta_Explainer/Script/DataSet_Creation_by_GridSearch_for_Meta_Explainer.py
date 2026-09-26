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
from importlib import reload
import pickle
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
from torch_geometric.data import Data, Batch, Dataset




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

expected_feature_size_for_meta_explainer = (
    constants["expected_feature_size_for_meta_explainer"]
)

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


# ============================================================
# Load parameter-count dictionary
# ============================================================
gnn_parameter_count_file = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
    "Constants/"
    "GNN_Parameter_Counts.pkl"
)

with open(gnn_parameter_count_file, "rb") as f:
    GNN_Parameter_Counts = pickle.load(f)


# ============================================================
# Load dataset latent representations
# ============================================================
dataset_latent_representation_file = (
    "/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/"
    "Dataset_Representation_Learning/"
    "Experimental Results/"
    "dataset_name_2_representation.pkl"
)

with open(dataset_latent_representation_file, "rb") as f:
    Dataset_Latent_Representations = pickle.load(f)


# ============================================================
# Sort Explainers
# ============================================================
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
                new_value = 1 - ((rank - 1) / num_explainers)
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

# example_dict_score, example_dict_sorted = sort_explainers(example_dict, descending=True)
# for dataset_name, gnn_model in example_dict_score.items():
#     print(dataset_name)
#     print("     ", gnn_model.keys())
#     for explainer, score in gnn_model.items():
#         print(explainer, score)

import itertools
Fidelity_plus_score, Fidelity_plus_sorted = sort_explainers(Fidelity_plus, descending=True)
Fidelity_minus_score, Fidelity_minus_sorted = sort_explainers(Fidelity_minus, descending=False)
Contrastivity_score, Contrastivity_sorted = sort_explainers(Contrastivity, descending=True)
Sparsity_score, Sparsity_sorted = sort_explainers(Sparsity, descending=True)
Stability_score, Stability_sorted = sort_explainers(Stability, descending=True)
Explanation_RunTime_score, Explanation_RunTime_sorted = sort_explainers(Explanation_RunTime, descending=False)

Explainer_Score_Dicts = {"Fidelity+": Fidelity_plus_score, "Fidelity-": Fidelity_minus_score,
                         "Contrastivity": Contrastivity_score, "Sparsity": Sparsity_score,
                         "Stability": Stability_score, "Explanation_RunTime": Explanation_RunTime_score}


# ============================================================
# Normalize weight vectors and remove proportional duplicates
# ============================================================
def normalize_and_remove_duplicate_weights(raw_weight_cases, decimals=12):
    unique_normalized_weights = {}
    for weights in raw_weight_cases:
        weights_array = np.array(weights, dtype=np.float64)
        normalized_weights = (weights_array / weights_array.sum())
        key = tuple(np.round(normalized_weights, decimals=decimals))
        if key not in unique_normalized_weights:
            unique_normalized_weights[key] = normalized_weights
    return list(unique_normalized_weights.values())


# ============================================================
# Finalizers
# ============================================================
def get_dataset_latent_features(dataset_name):
    latent = Dataset_Latent_Representations[dataset_name]
    if isinstance(latent, torch.Tensor):
        latent = latent.detach().cpu().numpy()
    latent = np.asarray(latent, dtype=np.float32).reshape(-1)
    assert len(latent) == 32, (
        f"{dataset_name}: expected 32 latent features, "
        f"found {len(latent)}")
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
        ], dtype=np.float32)


def get_gnn_evaluation_features(dataset_name, gnn_name):
    stats = GNNs_Evaluation_Stats[dataset_name][gnn_name]
    return np.array(
        [
            stats["AUC-ROC"],
            stats["AUC-PR"],
            stats["Accuracy"],
            stats["Running Time [sec]"]
        ], dtype=np.float32)


def get_gnn_parameter_feature(dataset_name, gnn_name):
    parameter_count = (
        GNN_Parameter_Counts
        [dataset_name]
        [gnn_name]
    )

    return np.array([parameter_count], dtype=np.float32)


# ============================================================
# Calculate the label for one
# dataset-GNN-weight configuration
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
    best_explainer_index = Explainers.index(best_explainer)
    return (best_explainer_index, best_explainer, final_scores)


# ============================================================
# Weight Level Configurations
# ============================================================
weight_levels = {
    "[1, 2]": [1, 2],
    "[1, 2, 3]": [1, 2, 3],
    "[1, 2, 3, 4]": [1, 2, 3, 4]
}
num_explainer_metrics = 6


# ============================================================
# Output Directory
# ============================================================
base_dataset_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
)
grid_search_output_directory = (
        base_dataset_results_directory
        + "GridSearch_Weights/"
)
os.makedirs(grid_search_output_directory, exist_ok=True)


# ============================================================
# Create datasets for all weight-level configurations
# ============================================================
for weight_level_name, current_weight_levels in weight_levels.items():
    print("\n" + "=" * 100)
    print("Creating Meta-Explainer dataset for weight levels:", weight_level_name)
    print("=" * 100)
    raw_weight_cases = list(itertools.product(current_weight_levels, repeat=num_explainer_metrics))
    all_weight_cases = (normalize_and_remove_duplicate_weights(raw_weight_cases))
    print("Raw weight combinations:", len(raw_weight_cases))
    print("Unique normalized weight combinations:", len(all_weight_cases))
    Meta_Explainer_Data = {}
    for dataset_name in Datasets_Name:
        Meta_Explainer_Data[dataset_name] = {}
        for gnn_name in GNN_Models:
            Meta_Explainer_Data[dataset_name][gnn_name] = {}
            for normalized_weights in all_weight_cases:
                weights_key = tuple(normalized_weights.tolist())
                temp = []
                temp.extend(get_dataset_latent_features(dataset_name).tolist())
                temp.extend(get_dataset_stat_features(dataset_name).tolist())
                temp.extend(get_gnn_evaluation_features(dataset_name, gnn_name).tolist())
                temp.extend(get_gnn_parameter_feature(dataset_name, gnn_name).tolist())
                temp.extend(normalized_weights.tolist())
                (
                    best_explainer_index,
                    best_explainer,
                    final_scores
                ) = find_best_explainer_weighted_average(dataset_name, gnn_name, normalized_weights)
                Meta_Explainer_Data[dataset_name][gnn_name][weights_key] = {
                    "features": temp,
                    "label": best_explainer,
                    "label_index": best_explainer_index,
                    "final_scores": final_scores
                }

    for dataset_name in Datasets_Name:
        for gnn_name in GNN_Models:
            for weights_key, sample in (Meta_Explainer_Data[dataset_name][gnn_name].items()):
                assert (len(sample["features"]) == expected_feature_size_for_meta_explainer)
                assert (Explainers[sample["label_index"]] == sample["label"])
                assert (max(sample["final_scores"], key=sample["final_scores"].get) == sample["label"])
                weights = sample["features"][-6:]
                assert np.isclose(sum(weights),1.0)
    print("All Meta-Explainer samples passed sanity checks.")
    X = []
    Y = []
    for dataset_name in Datasets_Name:
        for gnn_name in GNN_Models:
            for weights_key in (Meta_Explainer_Data[dataset_name][gnn_name].keys()):
                sample = (Meta_Explainer_Data[dataset_name][gnn_name][weights_key])
                feature_array = np.array(sample["features"], dtype=np.float32)
                X.append(torch.from_numpy(feature_array))
                label_index = sample["label_index"]
                Y.append(torch.tensor(label_index, dtype=torch.long))

    X_tensor = torch.stack(X)
    Y_tensor = torch.stack(Y)
    print("X shape:", X_tensor.shape)
    print("Y shape:", Y_tensor.shape)
    print("Number of samples:", len(X))
    print("Number of features:", X_tensor.shape[1])
    print("Unique classes:", torch.unique(Y_tensor))

    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)
    x_file = grid_search_output_directory + "X_GridSearch_Weights_" + weight_file_name + ".pt"
    y_file = grid_search_output_directory + "Y_GridSearch_Weights_" + weight_file_name + ".pt"
    structured_data_file = (
                grid_search_output_directory + "Meta_Explainer_Data_GridSearch_Weights_" + weight_file_name + ".pkl")
    torch.save(X, x_file)
    torch.save(Y, y_file)
    with open(structured_data_file, "wb") as file:
        pickle.dump(Meta_Explainer_Data, file)

    print("\nSaved:")
    print("X:", x_file)
    print("Y:", y_file)
    print("Structured Data:", structured_data_file)
# ============================================================
# Create Meta-Explainer Data Dictionary
# ============================================================

# print(Meta_Explainer_Data.keys())
# print(Meta_Explainer_Data["MUTAG"].keys())
# # print(Meta_Explainer_Data["MUTAG"]["GCN"].keys())
# print(Meta_Explainer_Data["MUTAG"]["GCN"][(0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666)].keys())
# print(Meta_Explainer_Data["MUTAG"]["GCN"][(0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666)]['features'])
# print(Meta_Explainer_Data["MUTAG"]["GCN"][(0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666)]['label'])
# print(Meta_Explainer_Data["MUTAG"]["GCN"][(0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666)]['label_index'])
# print(Meta_Explainer_Data["MUTAG"]["GCN"][(0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666)]['final_scores'])