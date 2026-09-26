import os
import sys
import argparse
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
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
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
from collections import Counter, defaultdict
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict, Counter








device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

Datasets_Name = ["MUTAG", "NCI1", "ENZYMES", "Graph-SST5", "PROTEINS", "IsCyclic"]
Explainers = ["GNNExplainer", "SubgraphX", "PGMExplainer", "CF2", "PGExplainer", "GraphMask", "XGNN", "GNNInterpreter"]

weight_levels = {
    "[1, 2]": [1, 2],
    "[1, 2, 3]": [1, 2, 3],
    "[1, 2, 3, 4]": [1, 2, 3, 4]
}

hidden_size = 64
num_epochs = 500
batch_size = 32
test_size = 0.2
random_state = 42
PRINT_CLASS_DISTRIBUTION = False


# ============================================================
# Directories
# ============================================================
dataset_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
    "GridSearch_Weights/"
)

base_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Train_and_Evaluate_Meta_Explainer/"
    "Experimental Results/"
)

models_directory = base_results_directory + "Meta_Explainer_Models_Trained_on_GridSearch_Weights/"
results_directory = base_results_directory + "TestResults_of_Meta_Explainer_on_GridSearch_Weights/"

os.makedirs(models_directory, exist_ok=True)
os.makedirs(results_directory, exist_ok=True)


# ============================================================
# Meta-Explainer
# ============================================================
class MetaExplainer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MetaExplainer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# Scale Features
# ============================================================

def scale_features(X_train, X_test):

    """
    Standardize the first 42 features:

        0:32   Dataset latent representation
        32:37  Dataset statistics
        37:41  GNN evaluation statistics
        41:42  GNN parameter count

    Final six features are normalized user weights and are therefore left unchanged.
    Scaler is fitted ONLY on training data.
    """
    X_train_np = X_train.cpu().numpy().copy()
    X_test_np = X_test.cpu().numpy().copy()
    scaler = StandardScaler()
    X_train_np[:, :42] = scaler.fit_transform(X_train_np[:, :42])
    X_test_np[:, :42] = scaler.transform(X_test_np[:, :42])
    X_train_scaled = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_np, dtype=torch.float32)

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


# ============================================================
# Train and Evaluate
# ============================================================
def train_and_evaluate(train_loader, test_loader, input_size, hidden_size, output_size, num_epochs):
    meta_explainer = MetaExplainer(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta_explainer.parameters(), lr=0.001)

    # ========================================================
    # Training
    # ========================================================
    for epoch in range(num_epochs):
        meta_explainer.train()
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = meta_explainer(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

    # ========================================================
    # Evaluation
    # ========================================================
    meta_explainer.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_test_predictions = []
    all_test_labels = []
    with torch.no_grad():
        for test_X, test_Y in test_loader:
            test_X = test_X.to(device)
            test_Y = test_Y.to(device, dtype=torch.long)
            test_outputs = meta_explainer(test_X)
            loss = criterion(test_outputs, test_Y)
            test_loss += loss.item()
            test_predicted = torch.argmax(test_outputs, dim=1)
            test_correct += (test_predicted == test_Y).sum().item()
            test_total += (test_Y.size(0))
            all_test_predictions.extend(test_predicted.cpu().numpy())
            all_test_labels.extend(test_Y.cpu().numpy())

    avg_test_loss = (test_loss / len(test_loader))
    test_accuracy = (test_correct / test_total)

    # ========================================================
    # Classification Metrics
    # ========================================================
    precision_macro = precision_score(all_test_labels, all_test_predictions, average="macro", zero_division=0)
    recall_macro = recall_score(all_test_labels, all_test_predictions, average="macro", zero_division=0)
    f1_macro = f1_score(all_test_labels, all_test_predictions, average="macro", zero_division=0)
    precision_weighted = precision_score(all_test_labels, all_test_predictions, average="weighted", zero_division=0)
    recall_weighted = recall_score(all_test_labels, all_test_predictions, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_test_labels, all_test_predictions, average="weighted", zero_division=0)
    meta_explainer_metrics = {
        "test_loss": avg_test_loss,
        "accuracy": test_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }
    return meta_explainer, meta_explainer_metrics

# ============================================================
# Create nested stratified subsets from the 80% training pool
# ============================================================
def create_nested_stratified_indices(Y_train_pool, training_scenarios, random_state):
    """
    Creates nested, approximately stratified training subsets.
    T20 subset T40 subset T60 subset T80
    """
    generator = torch.Generator()
    generator.manual_seed(random_state)
    class_indices = {}
    unique_classes = torch.unique(Y_train_pool)
    # --------------------------------------------------------
    # Shuffle samples independently inside each class ONCE
    # --------------------------------------------------------

    for class_index in unique_classes:
        indices = torch.where(Y_train_pool == class_index)[0]
        permutation = torch.randperm(len(indices), generator=generator)
        class_indices[int(class_index.item())] = indices[permutation]
    nested_indices = {}
    for (full_dataset_fraction, pool_fraction) in training_scenarios.items():
        selected_indices = []
        for class_index in sorted(class_indices.keys()):
            indices = class_indices[class_index]
            if pool_fraction == 1.0:
                num_samples = len(indices)
            else:
                num_samples = int(len(indices) * pool_fraction)
                num_samples = max(1, num_samples)
            selected_indices.extend(indices[:num_samples].tolist())
        # Shuffle the selected subset itself.
        selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        subset_permutation = torch.randperm(len(selected_indices), generator=generator)
        selected_indices = selected_indices[subset_permutation]
        nested_indices[full_dataset_fraction] = selected_indices
    return nested_indices


# ============================================================
# Train for Every Weight-Level Configuration
# ============================================================
training_scenarios = {
    0.2: 0.25,
    0.4: 0.50,
    0.6: 0.75,
    0.8: 1.00
}
for (weight_level_name, current_weight_levels) in weight_levels.items():
    # print("\n" + "=" * 100)
    # print("Training Meta-Explainer for weight levels:", weight_level_name)
    # print("=" * 100)

    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)
    file_names = ("GridSearch_Weights_" + weight_file_name)
    directory_x = (dataset_directory + "X_" + file_names + ".pt")
    directory_y = (dataset_directory + "Y_" + file_names + ".pt")

    X_list = torch.load(directory_x)
    Y_list = torch.load(directory_y)
    X_data = torch.stack(X_list).float()
    Y_data = torch.stack(Y_list).long()
    weights_sum = X_data[:, -6:].sum(dim=1)

    assert torch.allclose(weights_sum, torch.ones_like(weights_sum),
                          atol=1e-6), "Some preference-weight vectors do not sum to 1."

    if Y_data.dim() > 1 and Y_data.size(1) > 1:
        Y_data = torch.argmax(Y_data, dim=1).long()
    # print("X shape:", X_data.shape)
    # print("Y shape:", Y_data.shape)

    if PRINT_CLASS_DISTRIBUTION:
        counts = Counter(Y_data.tolist())
        total = len(Y_data)
        # print("\nOverall class distribution:")
        for class_index in range(len(Explainers)):
            count = counts.get(class_index, 0)
            print(
                f"  "
                f"{class_index} "
                f"({Explainers[class_index]}): "
                f"{count} "
                f"({count / total:.2%})"
            )
    output_size = len(Explainers)
    results = defaultdict(list)
    results["explainer_order"] = Explainers.copy()
    results["weight_levels"] = current_weight_levels.copy()
    # ========================================================
    # ONE fixed 80% training pool + 20% test set
    # This test set stays identical for the four
    # training-size scenarios.
    # ========================================================
    (
        X_train_pool,
        X_test_raw,
        Y_train_pool,
        Y_test
    ) = train_test_split(X_data, Y_data, test_size=test_size, random_state=random_state, shuffle=True,
                         stratify=Y_data)
    print("\nFixed train/test split created.")
    print("Training pool size:", len(X_train_pool))
    print("Fixed test-set size:", len(X_test_raw))
    print("Fixed test percentage:", f"{len(X_test_raw) / len(X_data):.2%}")
    # ========================================================
    # Fixed Test Distribution
    # ========================================================
    if PRINT_CLASS_DISTRIBUTION:
        counts_test = Counter(Y_test.tolist())
        total_test = len(Y_test)
        print("\nFixed test-set distribution:")
        for class_index in range(len(Explainers)):
            count = counts_test.get(class_index, 0)
            print(
                f"  "
                f"{class_index} "
                f"({Explainers[class_index]}): "
                f"{count} "
                f"({count / total_test:.2%})"
            )
    # ========================================================
    # Create nested + stratified training subsets
    # ========================================================
    nested_training_indices = (create_nested_stratified_indices(Y_train_pool=Y_train_pool,
                                                                training_scenarios=training_scenarios,
                                                                random_state=random_state))
    # ========================================================
    # Verify nesting
    # ========================================================
    scenario_keys = list(training_scenarios.keys())
    for i in range(len(scenario_keys) - 1):
        smaller = set(nested_training_indices[scenario_keys[i]].tolist())
        larger = set(nested_training_indices[scenario_keys[i + 1]].tolist())
        assert smaller.issubset(larger), (
            f"Training subsets are not nested: "
            f"{scenario_keys[i]} -> "
            f"{scenario_keys[i + 1]}"
        )
    print("\nNested training-subset check passed:")
    print("T20 subset T40 subset T60 subset T80")


    # ========================================================
    # Training-size Scenarios
    # ========================================================
    for train_size in training_scenarios.keys():
        # print("\n" + "-" * 100)
        # print(
        #     f"Training with approximately "
        #     f"{int(train_size * 100)}% "
        #     f"of the complete dataset..."
        # )
        # print("-" * 100)


        # ====================================================
        # Select nested training subset
        # ====================================================
        train_indices = (nested_training_indices[train_size])
        X_train_raw = X_train_pool[train_indices]
        Y_train = Y_train_pool[train_indices]
        # print("Training samples:", len(X_train_raw))
        # print("Fixed test samples:", len(X_test_raw))
        print(
            "Actual training percentage:",
            f"{len(X_train_raw) / len(X_data):.2%}"
        )

        # ====================================================
        # Training Distribution
        # ====================================================
        if PRINT_CLASS_DISTRIBUTION:
            counts_train = Counter(Y_train.tolist())
            total_train = len(Y_train)
            print("Training-set distribution:")
            for class_index in range(len(Explainers)):
                count = counts_train.get(class_index, 0)
                print(
                    f"  "
                    f"{class_index} "
                    f"({Explainers[class_index]}): "
                    f"{count} "
                    f"({count / total_train:.2%})"
                )

        # ====================================================
        # Scale AFTER subset selection
        # The scaler is fitted ONLY on this training subset.
        # X_test_raw remains unchanged and is transformed
        # ====================================================
        (X_train, X_test, scaler) = scale_features(X_train_raw, X_test_raw)

        # ====================================================
        # DataLoaders
        # ====================================================
        train_dataset = (torch.utils.data.TensorDataset(X_train, Y_train))
        train_loader = (torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        test_dataset = (torch.utils.data.TensorDataset(X_test, Y_test))
        test_loader = (torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

        # ====================================================
        # Train + Evaluate
        # ====================================================
        meta_explainer, metrics = train_and_evaluate(train_loader=train_loader, test_loader=test_loader,
                                                     input_size=X_train.shape[1], hidden_size=hidden_size,
                                                     output_size=output_size, num_epochs=num_epochs)

        # ====================================================
        # Store Results
        # ====================================================
        results["train_size"].append(train_size * 100)
        results["actual_train_percentage"].append(100 * len(X_train_raw) / len(X_data))
        results["num_train_samples"].append(len(X_train_raw))
        results["num_test_samples"].append(len(X_test_raw))
        for (metric_name, value) in metrics.items():
            results[metric_name].append(value)

        # ====================================================
        # Print Results
        # ====================================================
        print(
            f"Test Loss: "
            f"{metrics['test_loss']:.4f}"
        )
        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )
        print(
            f"Precision (macro): "
            f"{metrics['precision_macro']:.4f}"
        )
        print(
            f"Recall (macro): "
            f"{metrics['recall_macro']:.4f}"
        )
        print(
            f"F1 (macro): "
            f"{metrics['f1_macro']:.4f}"
        )
        print(
            f"Precision (weighted): "
            f"{metrics['precision_weighted']:.4f}"
        )
        print(
            f"Recall (weighted): "
            f"{metrics['recall_weighted']:.4f}"
        )
        print(
            f"F1 (weighted): "
            f"{metrics['f1_weighted']:.4f}"
        )
        # ====================================================
        # Save trained model
        # ====================================================
        train_size_name = str(int(train_size * 100))
        model_file = (
                    models_directory + "Meta_Explainer_Model_" + file_names + "_Train_" + train_size_name + "Percent.pt")
        torch.save(meta_explainer.state_dict(), model_file)
        print("Saved model:", model_file)
        scaler_file = (
                    models_directory + "Meta_Explainer_Scaler_" + file_names + "_Train_" + train_size_name + "Percent.pkl")
        with open(scaler_file, "wb") as file:
            pickle.dump(scaler, file)
        print("Saved scaler:", scaler_file)
    # ========================================================
    # Save Results
    # ========================================================
    results_file = results_directory + "Meta_Explainer_Results_" + file_names + ".pkl"
    with open(results_file, "wb") as file:
        pickle.dump(dict(results), file)
    print("\nSaved results:")
    # print(results_file)