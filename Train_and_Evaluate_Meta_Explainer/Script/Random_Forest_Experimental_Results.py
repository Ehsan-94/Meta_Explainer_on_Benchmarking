import os
import pickle
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
weight_levels = [
    "[1, 2]",
    "[1, 2, 3]",
    "[1, 2, 3, 4]"
]

metrics_to_report = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted"
]

# ============================================================
# Directories
# ============================================================
base_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Train_and_Evaluate_Meta_Explainer/"
    "Experimental Results/"
)

grid_results_directory = base_results_directory + "TestResults_of_Random_Forest_on_GridSearch_Weights/"
continuous_results_directory = base_results_directory + "TestResults_of_Random_Forest_on_Continuous_Weights/"
lodo_results_directory = base_results_directory + "TestResults_of_Random_Forest_LODO/"
logo_results_directory = base_results_directory + "TestResults_of_Random_Forest_LOGO/"
lodo_logo_results_directory = base_results_directory + "TestResults_of_Random_Forest_LODO_LOGO/"
# ============================================================
# Experimental Summary Output
# ============================================================
experimental_summary_directory = base_results_directory + "Experimental_Summary/"
rf_summary_directory = experimental_summary_directory + "Random_Forest/"
os.makedirs(rf_summary_directory, exist_ok=True)

# ============================================================
# Helper
# ============================================================
def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return (np.mean(values), np.std(values, ddof=1))

def format_mean_std(values):
    mean_value, std_value = mean_std(values)
    return (
        f"{mean_value:.4f} "
        f"± "
        f"{std_value:.4f}"
    )

# ============================================================
# 1. Grid-Search Results
# ============================================================
print("\n" + "=" * 100)
print("RANDOM FOREST GRID-SEARCH RESULTS")
print("=" * 100)

grid_results = {}
grid_rows = []
for weight_level_name in weight_levels:
    weight_file_name = (
        weight_level_name
        .replace("[", "")
        .replace("]", "")
        .replace(",", "")
        .replace(" ", "_")
    )
    result_file = grid_results_directory + "Random_Forest_Results_GridSearch_Weights_" + weight_file_name + ".pkl"
    assert os.path.exists(result_file), (
        f"Missing grid-search result file: {result_file}"
    )
    with open(result_file, "rb") as file:
        result = pickle.load(file)
    grid_results[weight_level_name] = result
    print("\nWeight levels:", weight_level_name)
    train_sizes = result["train_size"]
    for index, train_size in enumerate(train_sizes):
        print(f"\nTraining size: {train_size:.0f}%")
        for metric_name in metrics_to_report:
            print(
                f"  {metric_name}: "
                f"{result[metric_name][index]:.4f}"
            )
        row = {
            "weight_levels": weight_level_name,
            "training_size": int(train_size)
        }
        for metric_name in metrics_to_report:
            row[metric_name] = result[metric_name][index]
        grid_rows.append(row)
grid_df = pd.DataFrame(grid_rows)
grid_summary_file = rf_summary_directory + "GridSearch_RF.csv"
grid_df.to_csv(grid_summary_file, index=False)

# ============================================================
# 2. Continuous-Weight Results
# ============================================================
print("\n" + "=" * 100)
print("RANDOM FOREST CONTINUOUS-WEIGHT RESULTS")
print("=" * 100)
continuous_result_file = (continuous_results_directory
                          + "Random_Forest_GridSearch_Models_Continuous_Weights_Evaluation_Results.pkl")
assert os.path.exists(
    continuous_result_file
), (
    f"Missing continuous result file: "
    f"{continuous_result_file}"
)
with open(continuous_result_file, "rb") as file:
    continuous_results = pickle.load(file)
continuous_rows = []
for weight_level_name in weight_levels:
    print("\nWeight levels:", weight_level_name)
    for train_size_name, metrics in continuous_results[weight_level_name].items():
        print(f"\nTraining size: {train_size_name}%")
        for metric_name in metrics_to_report:
            print(
                f"  {metric_name}: "
                f"{metrics[metric_name]:.4f}"
            )
        row = {
            "weight_levels": weight_level_name,
            "training_size": int(train_size_name)
        }
        for metric_name in metrics_to_report:
            row[metric_name] = metrics[metric_name]
        continuous_rows.append(row)
continuous_df = pd.DataFrame(continuous_rows)
continuous_summary_file = rf_summary_directory + "Continuous_RF.csv"
continuous_df.to_csv(continuous_summary_file, index=False)

# ============================================================
# 3. LODO
#
# Aggregate over 6 held-out datasets.
# ============================================================
print("\n" + "=" * 100)
print("RANDOM FOREST LODO RESULTS")
print("=" * 100)
lodo_result_file = (
        lodo_results_directory
        + "Random_Forest_LODO_"
        + "GridSearch_Weights_"
        + "Evaluation_Results.pkl"
)

assert os.path.exists(
    lodo_result_file
), (
    f"Missing LODO result file: "
    f"{lodo_result_file}"
)

with open(lodo_result_file, "rb") as file:
    lodo_results = pickle.load(file)

lodo_summary = {}
for weight_level_name in weight_levels:
    lodo_summary[weight_level_name] = {}
    print("\nWeight levels:", weight_level_name)
    held_out_datasets = list(
        lodo_results[weight_level_name].keys())
    print("Number of held-out datasets:", len(held_out_datasets))
    for metric_name in metrics_to_report:
        values = [lodo_results[weight_level_name][dataset_name][metric_name] for dataset_name in held_out_datasets]
        mean_value, std_value = mean_std(values)

        lodo_summary[weight_level_name][metric_name] = {
            "mean": mean_value,
            "std": std_value
        }

        print(
            f"  "
            f"{metric_name}: "
            f"{mean_value:.4f} "
            f"± "
            f"{std_value:.4f}"
        )
lodo_rows_summary = []
for weight_level_name in weight_levels:
    row = {
        "weight_levels": weight_level_name,
        "num_held_out_contexts": 6
    }
    for metric_name in metrics_to_report:
        row[f"{metric_name}_mean"] = lodo_summary[weight_level_name][metric_name]["mean"]
        row[f"{metric_name}_std"] = lodo_summary[weight_level_name][metric_name]["std"]
    lodo_rows_summary.append(row)
lodo_df = pd.DataFrame(lodo_rows_summary)
lodo_summary_file = rf_summary_directory + "LODO_RF.csv"
lodo_df.to_csv(lodo_summary_file, index=False)

# ============================================================
# 4. LOGO
#
# Aggregate over 4 held-out GNNs.
# ============================================================

print("\n" + "=" * 100)
print("RANDOM FOREST LOGO RESULTS")
print("=" * 100)
logo_result_file = (
        logo_results_directory
        + "Random_Forest_LOGO_"
        + "GridSearch_Weights_"
        + "Evaluation_Results.pkl"
)

assert os.path.exists(
    logo_result_file
), (
    f"Missing LOGO result file: "
    f"{logo_result_file}"
)

with open(logo_result_file, "rb") as file:
    logo_results = pickle.load(file)

logo_summary = {}
for weight_level_name in weight_levels:
    logo_summary[weight_level_name] = {}
    print("\nWeight levels:", weight_level_name)
    held_out_gnns = list(logo_results[weight_level_name].keys())
    print("Number of held-out GNNs:", len(held_out_gnns))
    for metric_name in metrics_to_report:
        values = [logo_results[weight_level_name][gnn_name][metric_name] for gnn_name in held_out_gnns]
        mean_value, std_value = mean_std(values)
        logo_summary[weight_level_name][metric_name] = {
            "mean": mean_value,
            "std": std_value
        }
        print(
            f"  "
            f"{metric_name}: "
            f"{mean_value:.4f} "
            f"± "
            f"{std_value:.4f}"
        )
logo_rows_summary = []
for weight_level_name in weight_levels:
    row = {
        "weight_levels": weight_level_name,
        "num_held_out_contexts": 4
    }
    for metric_name in metrics_to_report:
        row[f"{metric_name}_mean"] = logo_summary[weight_level_name][metric_name]["mean"]
        row[f"{metric_name}_std"] = logo_summary[weight_level_name][metric_name]["std"]
    logo_rows_summary.append(row)
logo_df = pd.DataFrame(logo_rows_summary)
logo_summary_file = rf_summary_directory + "LOGO_RF.csv"
logo_df.to_csv(logo_summary_file, index=False)

# ============================================================
# 5. Combined LODO + LOGO
#
# Aggregate over:
#
# 6 datasets x 4 GNNs = 24 combinations.
# ============================================================

print("\n" + "=" * 100)
print("RANDOM FOREST COMBINED LODO + LOGO RESULTS")
print("=" * 100)
lodo_logo_result_file = (lodo_logo_results_directory + "Random_Forest_LODO_LOGO_" + "GridSearch_Weights_"
                         + "Evaluation_Results.pkl")

assert os.path.exists(
    lodo_logo_result_file
), (
    f"Missing LODO+LOGO result file: "
    f"{lodo_logo_result_file}"
)

with open(lodo_logo_result_file, "rb") as file:
    lodo_logo_results = pickle.load(file)

lodo_logo_summary = {}
for weight_level_name in weight_levels:
    lodo_logo_summary[weight_level_name] = {}
    print("\nWeight levels:", weight_level_name)
    for metric_name in metrics_to_report:
        values = []
        for dataset_name, gnn_results in (lodo_logo_results[weight_level_name].items()):
            for gnn_name, result in (gnn_results.items()):
                values.append(result[metric_name])
        assert len(values) == 24, (
            f"Expected 24 LODO+LOGO results, "
            f"found {len(values)}"
        )
        mean_value, std_value = mean_std(values)
        lodo_logo_summary[weight_level_name][metric_name] = {
            "mean": mean_value,
            "std": std_value
        }
        print(
            f"  "
            f"{metric_name}: "
            f"{mean_value:.4f} "
            f"± "
            f"{std_value:.4f}"
        )
lodo_logo_rows_summary = []
for weight_level_name in weight_levels:
    row = {
        "weight_levels": weight_level_name,
        "num_held_out_contexts": 24
    }
    for metric_name in metrics_to_report:
        row[f"{metric_name}_mean"] = lodo_logo_summary[weight_level_name][metric_name]["mean"]
        row[f"{metric_name}_std"] = lodo_logo_summary[weight_level_name][metric_name]["std"]
    lodo_logo_rows_summary.append(row)
lodo_logo_df = pd.DataFrame(lodo_logo_rows_summary)
lodo_logo_summary_file = rf_summary_directory + "LODO_LOGO_RF.csv"
lodo_logo_df.to_csv(lodo_logo_summary_file, index=False)

# ============================================================
# Compact Comparison
#
# I would focus especially on Accuracy and Macro F1.
# ============================================================
print("\n" + "=" * 100)
print("RANDOM FOREST GENERALIZATION SUMMARY")
print("=" * 100)

for weight_level_name in weight_levels:
    print("\nWeight levels:", weight_level_name)
    lodo_accuracy = format_mean_std([lodo_results[weight_level_name][dataset_name]["accuracy"] for dataset_name in
                                     (lodo_results[weight_level_name].keys())])
    print("  LODO Accuracy:      ", lodo_accuracy)
    lodo_macro_f1 = format_mean_std([lodo_results[weight_level_name][dataset_name]["f1_macro"] for dataset_name in
                                     (lodo_results[weight_level_name].keys())])
    print("  LODO Macro F1:      ", lodo_macro_f1)
    logo_accuracy = format_mean_std([logo_results[weight_level_name][gnn_name]["accuracy"] for gnn_name in
                                     (logo_results[weight_level_name].keys())])
    print("  LOGO Accuracy:      ", logo_accuracy)
    logo_macro_f1 = format_mean_std([logo_results[weight_level_name][gnn_name]["f1_macro"] for gnn_name in
                                     (logo_results[weight_level_name].keys())])
    print("  LOGO Macro F1:      ", logo_macro_f1)

    combined_accuracy = []
    combined_f1_macro = []

    for dataset_name, gnn_results in (lodo_logo_results[weight_level_name].items()):
        for gnn_name, result in (gnn_results.items()):
            combined_accuracy.append(result["accuracy"])
            combined_f1_macro.append(result["f1_macro"])
    print("  LODO+LOGO Accuracy: ", format_mean_std(combined_accuracy))
    print("  LODO+LOGO Macro F1: ", format_mean_std(combined_f1_macro))

# ============================================================
# Final Paths
# ============================================================

print("\n" + "=" * 100)
print("SAVED RANDOM FOREST SUMMARY FILES")
print("=" * 100)

print("\nGrid Search:")
print(grid_summary_file)

print("\nContinuous:")
print(continuous_summary_file)

print("\nLODO:")
print(lodo_summary_file)

print("\nLOGO:")
print(logo_summary_file)

print("\nLODO + LOGO:")
print(lodo_logo_summary_file)