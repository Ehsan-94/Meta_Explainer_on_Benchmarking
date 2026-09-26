import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Summary of the Corrected WorkflowData:
# 1. Each tree gets a random mix of data rows (with shared samples).
# 2. Attributes: Each tree gets a random subset of columns/features (not just one).
# 3. The Tree: Each tree builds a complete flowchart based on those columns.
# 4. Voting: All flowcharts make a prediction, and the majority vote wins.

# ============================================================
# Load Constants
# ============================================================

constants_file = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
    "Constants/"
    "Meta_Explainer_Constants.pkl"
)

with open(constants_file, "rb") as file:
    constants = pickle.load(file)


Datasets_Name = constants["Datasets_Name"]
GNN_Models = constants["GNN_Models"]
Explainers = constants["Explainers"]

num_classes = len(Explainers)


# ============================================================
# Weight Configurations
# ============================================================

weight_levels = {
    "[1, 2]": [1, 2],
    "[1, 2, 3]": [1, 2, 3],
    "[1, 2, 3, 4]": [1, 2, 3, 4]
}


# ============================================================
# Grid Training Scenarios
#
# Must exactly match the Grid training script.
# ============================================================
training_scenarios = {
    0.2: 0.25,
    0.4: 0.50,
    0.6: 0.75,
    0.8: 1.00
}

test_size = 0.2
random_state = 42
number_of_continuous_weight_vectors = 4000


# ============================================================
# Directories
# ============================================================

base_dataset_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
)


grid_dataset_directory = base_dataset_directory + "GridSearch_Weights/"
continuous_dataset_directory = base_dataset_directory + "Continuous_Weights/"
lodo_dataset_directory = base_dataset_directory + "LODO_GridSearch_Weights/"
logo_dataset_directory = base_dataset_directory + "LOGO_GridSearch_Weights/"
lodo_logo_dataset_directory = base_dataset_directory + "LODO_LOGO_GridSearch_Weights/"

base_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Train_and_Evaluate_Meta_Explainer/"
    "Experimental Results/"
)

grid_results_directory = (
        base_results_directory
        + "TestResults_of_Random_Forest_on_GridSearch_Weights/"
)

continuous_results_directory = (
        base_results_directory
        + "TestResults_of_Random_Forest_on_Continuous_Weights/"
)

continuous_results_file = (
        continuous_results_directory
        + "Random_Forest_GridSearch_Models_"
        + "Continuous_Weights_Evaluation_Results.pkl"
)

lodo_results_file = (
        base_results_directory
        + "TestResults_of_Random_Forest_LODO/"
        + "Random_Forest_LODO_"
        + "GridSearch_Weights_"
        + "Evaluation_Results.pkl"
)

logo_results_file = (
        base_results_directory
        + "TestResults_of_Random_Forest_LOGO/"
        + "Random_Forest_LOGO_"
        + "GridSearch_Weights_"
        + "Evaluation_Results.pkl"
)

lodo_logo_results_file = (
        base_results_directory
        + "TestResults_of_Random_Forest_LODO_LOGO/"
        + "Random_Forest_LODO_LOGO_"
        + "GridSearch_Weights_"
        + "Evaluation_Results.pkl"
)

# ============================================================
# Generalization Diagnostics Output
# ============================================================
generalization_diagnostics_directory = base_results_directory + "Generalization_Diagnostics/"
analysis_output_directory = generalization_diagnostics_directory + "Random_Forest/"
os.makedirs(analysis_output_directory, exist_ok=True)


# ============================================================
# Load Saved Random Forest Results
# ============================================================

with open(lodo_results_file, "rb") as file:
    LODO_results = pickle.load(file)

with open(logo_results_file, "rb") as file:
    LOGO_results = pickle.load(file)

with open(lodo_logo_results_file, "rb") as file:
    LODO_LOGO_results = pickle.load(file)

with open(continuous_results_file, "rb") as file:
    Continuous_results = pickle.load(file)

continuous_y_file = (
        continuous_dataset_directory
        + "Y_ContinuousWeights_"
        + str(number_of_continuous_weight_vectors)
        + "_WeightVectors.pt"
)
Y_continuous = torch.stack(torch.load(continuous_y_file)).long()


# ============================================================
# Helper: Load Tensor Lists
# ============================================================

def load_xy(x_train_file, y_train_file, x_test_file, y_test_file):
    X_train = torch.stack(torch.load(x_train_file)).float()
    Y_train = torch.stack(torch.load(y_train_file)).long()
    X_test = torch.stack(torch.load(x_test_file)).float()
    Y_test = torch.stack(torch.load(y_test_file)).long()
    return (X_train, Y_train, X_test, Y_test)

# ============================================================
# Helper: Class Distribution
# ============================================================
def class_distribution(Y):
    counts = np.bincount(Y.cpu().numpy(), minlength=num_classes)
    distribution = counts / counts.sum()
    return (counts, distribution)

# ============================================================
# Label Distribution Shift
#
# Total Variation Distance:
#
# 0 = identical distributions
# 1 = completely different distributions
# ============================================================
def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

# ============================================================
# Majority Baseline
#
# IMPORTANT:
# Majority class is determined ONLY from TRAINING labels.
# ============================================================
def majority_baseline(Y_train, Y_test):
    train_labels = Y_train.detach().cpu().numpy()
    true_labels = Y_test.detach().cpu().numpy()
    train_counts = Counter(train_labels.tolist())
    majority_class = max(train_counts, key=train_counts.get)
    predictions = np.full(len(true_labels), majority_class, dtype=np.int64)
    return {
        "majority_class": majority_class,
        "majority_explainer": Explainers[majority_class],
        "accuracy": accuracy_score(true_labels, predictions),
        "precision_macro": precision_score(true_labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(true_labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(true_labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_labels, predictions, average="weighted", zero_division=0)
    }


# ============================================================
# Oracle Test Majority
#
# DIAGNOSTIC ONLY.
#
# This is NOT a fair predictive baseline because it uses
# the test labels to select the majority class.
#
# It tells us how imbalanced/easy the held-out set is.
# ============================================================

def test_majority_fraction(Y_test):
    counts = Counter(Y_test.tolist())
    majority_class = max(counts, key=counts.get)
    fraction = (counts[majority_class] / len(Y_test))
    return (majority_class, fraction)


# ============================================================
# Feature Distribution Shift
#
# Fit StandardScaler on TRAIN only.
#
# After scaling:
#
# train mean ~ 0
# train std  ~ 1
#
# Therefore test means tell us how far the held-out domain
# lies from the training distribution in standard-deviation
# units.
# ============================================================

def feature_shift_analysis(X_train, X_test):
    X_train_np = X_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    scaler = StandardScaler()
    scaler.fit(X_train_np[:, :42])
    X_test_scaled = scaler.transform(X_test_np[:, :42])
    # --------------------------------------------------------
    # Absolute mean z-shift for each feature
    # --------------------------------------------------------
    mean_shift_per_feature = np.abs(np.mean(X_test_scaled, axis=0))
    # ========================================================
    # Feature Groups
    # ========================================================
    latent_shift = np.mean(mean_shift_per_feature[0:32])
    dataset_stats_shift = np.mean(mean_shift_per_feature[32:37])
    gnn_eval_shift = np.mean(mean_shift_per_feature[37:41])
    gnn_parameter_shift = (mean_shift_per_feature[41])
    overall_shift = np.mean(mean_shift_per_feature)
    max_shift = np.max(mean_shift_per_feature)
    return {
        "feature_shift_overall": overall_shift,
        "feature_shift_latent": latent_shift,
        "feature_shift_dataset_stats": dataset_stats_shift,
        "feature_shift_gnn_eval": gnn_eval_shift,
        "feature_shift_gnn_parameters": gnn_parameter_shift,
        "feature_shift_max": max_shift
    }


# ============================================================
# Reconstruct Nested Grid Training Subsets
#
# This reproduces the exact procedure used in the Grid
# training script so that the original training labels can be
# recovered without saving separate split files.
# ============================================================
def create_nested_stratified_indices(Y_train_pool, training_scenarios, random_state):
    generator = torch.Generator()
    generator.manual_seed(random_state)
    class_indices = {}
    unique_classes = torch.unique(Y_train_pool)

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

        selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        subset_permutation = torch.randperm(len(selected_indices), generator=generator)
        selected_indices = selected_indices[subset_permutation]
        nested_indices[full_dataset_fraction] = selected_indices

    return nested_indices


# ============================================================
# Reconstruct Exact Grid Label Splits
# ============================================================
def reconstruct_grid_label_splits(current_weight_levels):
    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)
    x_file = grid_dataset_directory + "X_GridSearch_Weights_" + weight_file_name + ".pt"
    y_file = grid_dataset_directory + "Y_GridSearch_Weights_" + weight_file_name + ".pt"

    X_data = torch.stack(torch.load(x_file)).float()
    Y_data = torch.stack(torch.load(y_file)).long()

    (
        X_train_pool,
        X_test_raw,
        Y_train_pool,
        Y_test
    ) = train_test_split(
        X_data,
        Y_data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=Y_data
    )

    nested_training_indices = create_nested_stratified_indices(
        Y_train_pool=Y_train_pool,
        training_scenarios=training_scenarios,
        random_state=random_state
    )

    nested_training_labels = {}
    for train_fraction in training_scenarios.keys():
        train_indices = nested_training_indices[train_fraction]
        nested_training_labels[train_fraction] = Y_train_pool[train_indices]

    scenario_keys = list(training_scenarios.keys())
    for i in range(len(scenario_keys) - 1):
        smaller = set(nested_training_indices[scenario_keys[i]].tolist())
        larger = set(nested_training_indices[scenario_keys[i + 1]].tolist())
        assert smaller.issubset(larger), "Reconstructed Grid subsets are not nested."

    assert len(X_test_raw) == len(Y_test)
    return nested_training_labels, Y_test


# ============================================================
# Analyze Grid / Continuous Majority Baseline
# ============================================================
def analyze_standard_split(Y_train, Y_test, rf_metrics):
    majority = majority_baseline(Y_train, Y_test)
    accuracy_gain = rf_metrics["accuracy"] - majority["accuracy"]
    macro_f1_gain = rf_metrics["f1_macro"] - majority["f1_macro"]

    return {
        "rf_accuracy": rf_metrics["accuracy"],
        "rf_macro_f1": rf_metrics["f1_macro"],
        "rf_weighted_f1": rf_metrics["f1_weighted"],
        "majority_class": majority["majority_class"],
        "majority_explainer": majority["majority_explainer"],
        "majority_accuracy": majority["accuracy"],
        "majority_macro_f1": majority["f1_macro"],
        "majority_weighted_f1": majority["f1_weighted"],
        "accuracy_gain_over_majority": accuracy_gain,
        "macro_f1_gain_over_majority": macro_f1_gain
    }


# ============================================================
# Analyze One Generalization Split
# ============================================================
def analyze_generalization_split(X_train, Y_train, X_test, Y_test, rf_metrics):
    # ========================================================
    # Label Distributions
    # ========================================================
    (train_counts, train_distribution) = class_distribution(Y_train)
    (test_counts, test_distribution) = class_distribution(Y_test)
    label_shift = total_variation_distance(train_distribution, test_distribution)
    # ========================================================
    # Majority Baseline
    # ========================================================
    majority = majority_baseline(Y_train, Y_test)
    # ========================================================
    # Diagnostic Test Majority
    # ========================================================
    (test_majority_class, test_majority_accuracy) = test_majority_fraction(Y_test)
    # ========================================================
    # Feature Shift
    # ========================================================
    feature_shift = feature_shift_analysis(X_train, X_test)
    # ========================================================
    # Improvement Over Majority Baseline
    # ========================================================
    accuracy_gain = (rf_metrics["accuracy"] - majority["accuracy"])
    macro_f1_gain = (rf_metrics["f1_macro"] - majority["f1_macro"])
    result = {
        "rf_accuracy": rf_metrics["accuracy"],
        "rf_macro_f1": rf_metrics["f1_macro"],
        "rf_weighted_f1": rf_metrics["f1_weighted"],
        "majority_class": majority["majority_class"],
        "majority_explainer": majority["majority_explainer"],
        "majority_accuracy": majority["accuracy"],
        "majority_macro_f1": majority["f1_macro"],
        "majority_weighted_f1": majority["f1_weighted"],
        "accuracy_gain_over_majority": accuracy_gain,
        "macro_f1_gain_over_majority": macro_f1_gain,
        "test_majority_class": test_majority_class,
        "test_majority_explainer": Explainers[test_majority_class],
        "test_majority_fraction": test_majority_accuracy,
        "label_distribution_shift_TV": label_shift,

        **feature_shift
    }
    # --------------------------------------------------------
    # Store each class proportion as well
    # --------------------------------------------------------
    for class_index in range(num_classes):
        result[f"train_class_{class_index}_fraction"] = (train_distribution[class_index])
        result[f"test_class_{class_index}_fraction"] = (test_distribution[class_index])
    return result


# ============================================================
# Grid + Continuous Majority-Baseline Analysis
# ============================================================
grid_rows = []
continuous_rows = []

for (weight_level_name, current_weight_levels) in weight_levels.items():
    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)

    # Reconstruct the exact Grid training subsets and fixed Grid test set.
    nested_training_labels, Y_grid_test = reconstruct_grid_label_splits(current_weight_levels)

    # Load the corresponding Grid result file.
    grid_result_file = (
            grid_results_directory
            + "Random_Forest_Results_GridSearch_Weights_"
            + weight_file_name
            + ".pkl"
    )
    with open(grid_result_file, "rb") as file:
        grid_result = pickle.load(file)

    assert len(grid_result["train_size"]) == len(training_scenarios)

    for result_index, train_fraction in enumerate(training_scenarios.keys()):
        train_size_name = str(int(train_fraction * 100))
        Y_train = nested_training_labels[train_fraction]

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------
        grid_rf_metrics = {
            "accuracy": grid_result["accuracy"][result_index],
            "f1_macro": grid_result["f1_macro"][result_index],
            "f1_weighted": grid_result["f1_weighted"][result_index]
        }

        grid_analysis = analyze_standard_split(
            Y_train=Y_train,
            Y_test=Y_grid_test,
            rf_metrics=grid_rf_metrics
        )
        grid_analysis["experiment"] = "GRID"
        grid_analysis["weight_levels"] = weight_level_name
        grid_analysis["training_size"] = int(train_fraction * 100)
        grid_analysis["num_train_samples"] = len(Y_train)
        grid_analysis["num_test_samples"] = len(Y_grid_test)
        grid_rows.append(grid_analysis)

        # ----------------------------------------------------
        # Continuous
        #
        # The model was trained on the corresponding Grid
        # subset, so the fair majority baseline also uses the
        # majority class from that same Grid training subset.
        # ----------------------------------------------------
        continuous_rf_metrics = Continuous_results[weight_level_name][train_size_name]

        continuous_analysis = analyze_standard_split(
            Y_train=Y_train,
            Y_test=Y_continuous,
            rf_metrics=continuous_rf_metrics
        )
        continuous_analysis["experiment"] = "CONTINUOUS"
        continuous_analysis["weight_levels"] = weight_level_name
        continuous_analysis["training_size"] = int(train_fraction * 100)
        continuous_analysis["num_train_samples"] = len(Y_train)
        continuous_analysis["num_test_samples"] = len(Y_continuous)
        continuous_rows.append(continuous_analysis)

grid_df = pd.DataFrame(grid_rows)
grid_file = analysis_output_directory + "GridSearch_RF_vs_Majority.csv"
grid_df.to_csv(grid_file, index=False)
continuous_df = pd.DataFrame(continuous_rows)
continuous_file = analysis_output_directory + "Continuous_RF_vs_Majority.csv"
continuous_df.to_csv(continuous_file, index=False)

# ============================================================
# LODO Analysis
# ============================================================
lodo_rows = []
for (weight_level_name, current_weight_levels) in weight_levels.items():
    weight_file_name = "_".join(str(x) for x in current_weight_levels)
    for held_out_dataset in Datasets_Name:
        file_prefix = "LODO_" + "GridSearch_Weights_" + weight_file_name + "_HeldOut_" + held_out_dataset
        (X_train, Y_train, X_test, Y_test) = load_xy(lodo_dataset_directory + "X_Train_" + file_prefix + ".pt",
                                                     lodo_dataset_directory + "Y_Train_" + file_prefix + ".pt",
                                                     lodo_dataset_directory + "X_Test_" + file_prefix + ".pt",
                                                     lodo_dataset_directory + "Y_Test_" + file_prefix + ".pt")
        rf_metrics = LODO_results[weight_level_name][held_out_dataset]
        analysis = analyze_generalization_split(X_train, Y_train, X_test, Y_test, rf_metrics)
        analysis["experiment"] = "LODO"
        analysis["weight_levels"] = weight_level_name
        analysis["held_out_dataset"] = held_out_dataset
        analysis["held_out_gnn"] = None
        lodo_rows.append(analysis)

# ============================================================
# Save LODO Diagnostics
# ============================================================
lodo_df = pd.DataFrame(lodo_rows)
lodo_file = analysis_output_directory + "LODO_RF_Diagnostics.csv"
lodo_df.to_csv(lodo_file, index=False)

# ============================================================
# LOGO Analysis
# ============================================================
logo_rows = []
for (weight_level_name, current_weight_levels) in weight_levels.items():
    weight_file_name = "_".join(str(x) for x in current_weight_levels)
    for held_out_gnn in GNN_Models:
        file_prefix = "LOGO_" + "GridSearch_Weights_" + weight_file_name + "_HeldOut_" + held_out_gnn
        (X_train, Y_train, X_test, Y_test) = load_xy(logo_dataset_directory + "X_Train_" + file_prefix + ".pt",
                                                     logo_dataset_directory + "Y_Train_" + file_prefix + ".pt",
                                                     logo_dataset_directory + "X_Test_" + file_prefix + ".pt",
                                                     logo_dataset_directory + "Y_Test_" + file_prefix + ".pt")
        rf_metrics = LOGO_results[weight_level_name][held_out_gnn]
        analysis = analyze_generalization_split(X_train, Y_train, X_test, Y_test, rf_metrics)
        analysis["experiment"] = "LOGO"
        analysis["weight_levels"] = weight_level_name
        analysis["held_out_dataset"] = None
        analysis["held_out_gnn"] = held_out_gnn
        logo_rows.append(analysis)

# ============================================================
# Save LOGO Diagnostics
# ============================================================
logo_df = pd.DataFrame(logo_rows)
logo_file = analysis_output_directory + "LOGO_RF_Diagnostics.csv"
logo_df.to_csv(logo_file, index=False)
# ============================================================
# Combined LODO + LOGO Analysis
# ============================================================
combined_rows = []
for (weight_level_name, current_weight_levels) in weight_levels.items():
    weight_file_name = "_".join(str(x) for x in current_weight_levels)
    for held_out_dataset in Datasets_Name:
        for held_out_gnn in GNN_Models:
            file_prefix = (
                    "LODO_LOGO_" + "GridSearch_Weights_" + weight_file_name + "_HeldOutDataset_" + held_out_dataset
                    + "_HeldOutGNN_" + held_out_gnn)
            (X_train, Y_train, X_test, Y_test) = load_xy(lodo_logo_dataset_directory + "X_Train_" + file_prefix + ".pt",
                                                         lodo_logo_dataset_directory + "Y_Train_" + file_prefix + ".pt",
                                                         lodo_logo_dataset_directory + "X_Test_" + file_prefix + ".pt",
                                                         lodo_logo_dataset_directory + "Y_Test_" + file_prefix + ".pt")
            rf_metrics = LODO_LOGO_results[weight_level_name][held_out_dataset][held_out_gnn]
            analysis = analyze_generalization_split(X_train, Y_train, X_test, Y_test, rf_metrics)
            analysis["experiment"] = "LODO_LOGO"
            analysis["weight_levels"] = weight_level_name
            analysis["held_out_dataset"] = held_out_dataset
            analysis["held_out_gnn"] = held_out_gnn
            combined_rows.append(analysis)
# ============================================================
# Save Combined LODO + LOGO Diagnostics
# ============================================================
lodo_logo_df = pd.DataFrame(combined_rows)
lodo_logo_file = analysis_output_directory + "LODO_LOGO_RF_Diagnostics.csv"
lodo_logo_df.to_csv(lodo_logo_file, index=False)
# ============================================================
# Combine Generalization Diagnostics
# ============================================================

diagnostics_df = pd.concat(
    [
        lodo_df,
        logo_df,
        lodo_logo_df
    ],
    ignore_index=True
)

# ============================================================
# Print Important Columns
# ============================================================
important_columns = [
    "experiment",
    "weight_levels",
    "held_out_dataset",
    "held_out_gnn",

    "rf_accuracy",
    "majority_accuracy",
    "accuracy_gain_over_majority",

    "rf_macro_f1",
    "majority_macro_f1",
    "macro_f1_gain_over_majority",

    "test_majority_fraction",

    "label_distribution_shift_TV",

    "feature_shift_overall",
    "feature_shift_latent",
    "feature_shift_dataset_stats",
    "feature_shift_gnn_eval",
    "feature_shift_gnn_parameters"
]


print("\n" + "=" * 120)
print("GENERALIZATION DIAGNOSTICS")
print("=" * 120)
print(diagnostics_df[important_columns].to_string(index=False))


# ============================================================
# Aggregate by Experiment + Weight Grid
# ============================================================

summary_df = (
    diagnostics_df
    .groupby(
        [
            "experiment",
            "weight_levels"
        ]
    )
    .agg(
        rf_accuracy_mean=(
            "rf_accuracy",
            "mean"
        ),

        majority_accuracy_mean=(
            "majority_accuracy",
            "mean"
        ),

        accuracy_gain_mean=(
            "accuracy_gain_over_majority",
            "mean"
        ),

        rf_macro_f1_mean=(
            "rf_macro_f1",
            "mean"
        ),

        majority_macro_f1_mean=(
            "majority_macro_f1",
            "mean"
        ),

        macro_f1_gain_mean=(
            "macro_f1_gain_over_majority",
            "mean"
        ),

        label_shift_mean=(
            "label_distribution_shift_TV",
            "mean"
        ),

        feature_shift_mean=(
            "feature_shift_overall",
            "mean"
        ),

        latent_shift_mean=(
            "feature_shift_latent",
            "mean"
        ),

        dataset_stats_shift_mean=(
            "feature_shift_dataset_stats",
            "mean"
        ),

        gnn_eval_shift_mean=(
            "feature_shift_gnn_eval",
            "mean"
        ),

        gnn_parameter_shift_mean=(
            "feature_shift_gnn_parameters",
            "mean"
        )
    )
    .reset_index()
)


print("\n" + "=" * 120)
print("AGGREGATE DIAGNOSTIC SUMMARY")
print("=" * 120)
print(summary_df.to_string(index=False))
summary_file = analysis_output_directory + "RF_Generalization_Summary.csv"
summary_df.to_csv(summary_file, index=False)

# ============================================================
# Correlation Analysis
#
# Does stronger shift correspond to worse accuracy?
# ============================================================

correlation_columns = [
    "rf_accuracy",
    "rf_macro_f1",
    "label_distribution_shift_TV",
    "feature_shift_overall",
    "feature_shift_latent",
    "feature_shift_dataset_stats",
    "feature_shift_gnn_eval",
    "feature_shift_gnn_parameters",
    "test_majority_fraction"
]


correlation_matrix = (diagnostics_df[correlation_columns].corr())
print("\n" + "=" * 120)
print("CORRELATION MATRIX")
print("=" * 120)
print(correlation_matrix)
correlation_file = analysis_output_directory + "RF_Correlations.csv"
correlation_matrix.to_csv(correlation_file)

# ============================================================
# Final Paths
# ============================================================
print("\n" + "=" * 100)
print("SAVED RANDOM FOREST DIAGNOSTIC FILES")
print("=" * 100)

print("\nGrid Search vs Majority:")
print(grid_file)

print("\nContinuous vs Majority:")
print(continuous_file)

print("\nLODO Diagnostics:")
print(lodo_file)

print("\nLOGO Diagnostics:")
print(logo_file)

print("\nLODO + LOGO Diagnostics:")
print(lodo_logo_file)

print("\nGeneralization Summary:")
print(summary_file)

print("\nCorrelations:")
print(correlation_file)