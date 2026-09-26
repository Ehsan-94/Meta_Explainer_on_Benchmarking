import os
import pickle

import numpy as np
import torch


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

expected_feature_size_for_meta_explainer = (
    constants["expected_feature_size_for_meta_explainer"]
)

# ============================================================
# Weight Configurations
# ============================================================
weight_levels = {
    "[1, 2]": [1, 2],
    "[1, 2, 3]": [1, 2, 3],
    "[1, 2, 3, 4]": [1, 2, 3, 4]
}

# ============================================================
# Directories
# ============================================================

base_dataset_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
)

grid_search_directory = base_dataset_results_directory + "GridSearch_Weights/"
combined_output_directory = base_dataset_results_directory + "LODO_LOGO_GridSearch_Weights/"

os.makedirs(combined_output_directory, exist_ok=True)

# ============================================================
# Create One Combined LODO + LOGO Split
#
# TRAIN:
#
#   dataset != held_out_dataset
#   AND
#   gnn != held_out_gnn
#
# TEST:
#
#   dataset == held_out_dataset
#   AND
#   gnn == held_out_gnn
#
#
# Example:
#
# held_out_dataset = MUTAG
# held_out_gnn     = GIN
#
#
# Training:
#
# NCI1       x GCN
# NCI1       x DGCNN
# NCI1       x DIFFPOOL
#
# ENZYMES    x GCN
# ENZYMES    x DGCNN
# ENZYMES    x DIFFPOOL
#
# ...
#
#
# Test:
#
# MUTAG x GIN
#
#
# Excluded completely:
#
# MUTAG x GCN
# MUTAG x DGCNN
# MUTAG x DIFFPOOL
#
# all other datasets x GIN
#
# ============================================================

def create_lodo_logo_split(Meta_Explainer_Data, held_out_dataset, held_out_gnn):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for dataset_name in Datasets_Name:
        for gnn_name in GNN_Models:
            for weights_key, sample in Meta_Explainer_Data[dataset_name][gnn_name].items():
                feature_array = np.array(sample["features"], dtype=np.float32)
                label_index = int(sample["label_index"])
                # ====================================================
                # TEST:
                # exact held-out dataset-GNN pair
                # ====================================================
                if (dataset_name == held_out_dataset and gnn_name == held_out_gnn):
                    X_test.append(torch.from_numpy(feature_array))
                    Y_test.append(torch.tensor(label_index, dtype=torch.long))

                # ====================================================
                # TRAIN:
                #
                # dataset must NOT be the held-out dataset
                # AND
                # GNN must NOT be the held-out GNN
                # ====================================================
                elif (dataset_name != held_out_dataset and gnn_name != held_out_gnn):
                    X_train.append(torch.from_numpy(feature_array))
                    Y_train.append(torch.tensor(label_index, dtype=torch.long))

                # ====================================================
                # Everything else is intentionally excluded
                # ====================================================
                else:
                    continue
    return X_train, Y_train, X_test, Y_test

# ============================================================
# Create All Weight-Level Configurations
# ============================================================
total_splits_created = 0
for (weight_level_name, current_weight_levels) in weight_levels.items():
    print("\n" + "=" * 100)
    print(
        "Creating combined LODO + LOGO datasets "
        "for weight levels:",
        weight_level_name
    )
    print("=" * 100)

    # ========================================================
    # Weight-Level File Name
    # ========================================================
    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)

    # ========================================================
    # Load Structured Grid-Search Dataset
    # ========================================================
    structured_data_file = (
                grid_search_directory + "Meta_Explainer_Data_GridSearch_Weights_" + weight_file_name + ".pkl")

    assert os.path.exists(
        structured_data_file
    ), (
        f"Structured grid-search dataset not found: "
        f"{structured_data_file}"
    )
    with open(structured_data_file, "rb") as file:
        Meta_Explainer_Data = pickle.load(file)
    print("\nLoaded:")
    print(structured_data_file)

    # ========================================================
    # Number of Weight Vectors for Current Grid
    # ========================================================
    example_dataset = Datasets_Name[0]
    example_gnn = GNN_Models[0]
    number_of_weight_vectors = len(Meta_Explainer_Data[example_dataset][example_gnn])
    print("Number of unique normalized weight vectors:", number_of_weight_vectors)

    # ========================================================
    # Expected Sample Counts
    #
    # TRAIN:
    #
    # 5 remaining datasets
    # x
    # 3 remaining GNNs
    # x
    # W weight vectors
    #
    #
    # TEST:
    #
    # 1 held-out dataset
    # x
    # 1 held-out GNN
    # x
    # W weight vectors
    # ========================================================

    expected_train_samples = (len(Datasets_Name) - 1) * (len(GNN_Models) - 1) * number_of_weight_vectors
    expected_test_samples = number_of_weight_vectors
    print("Expected train samples per split:", expected_train_samples)
    print("Expected test samples per split:", expected_test_samples)

    # ========================================================
    # Hold Out Every Dataset-GNN Pair
    # ========================================================
    for held_out_dataset in Datasets_Name:
        for held_out_gnn in GNN_Models:
            print("\n" + "-" * 100)
            print("Weight levels:", weight_level_name)
            print("Held-out dataset:", held_out_dataset)
            print("Held-out GNN:", held_out_gnn)
            print("-" * 100)

            # ====================================================
            # Create Split
            # ====================================================
            X_train, Y_train, X_test, Y_test = create_lodo_logo_split(Meta_Explainer_Data=Meta_Explainer_Data,
                                                                      held_out_dataset=held_out_dataset,
                                                                      held_out_gnn=held_out_gnn)

            # ====================================================
            # Stack for Inspection / Sanity Checks
            # ====================================================
            X_train_tensor = torch.stack(X_train).float()
            Y_train_tensor = torch.stack(Y_train).long()
            X_test_tensor = torch.stack(X_test).float()
            Y_test_tensor = torch.stack(Y_test).long()
            # ====================================================
            # Print Shapes
            # ====================================================

            print("X train shape:", X_train_tensor.shape)
            print("Y train shape:", Y_train_tensor.shape)
            print("X test shape:", X_test_tensor.shape)
            print("Y test shape:", Y_test_tensor.shape)

            # ====================================================
            # Feature-Dimension Checks
            # ====================================================

            assert (X_train_tensor.shape[1] == expected_feature_size_for_meta_explainer), (
                "Unexpected number of training features.")
            assert (X_test_tensor.shape[1] == expected_feature_size_for_meta_explainer), (
                "Unexpected number of test features.")
            # ====================================================
            # X/Y Size Checks
            # ====================================================
            assert (X_train_tensor.shape[0] == Y_train_tensor.shape[0]), ("Training X/Y sample counts do not match.")
            assert (X_test_tensor.shape[0] == Y_test_tensor.shape[0]), ("Test X/Y sample counts do not match.")

            # ====================================================
            # Expected Sample-Count Checks
            # ====================================================
            assert (len(X_train) == expected_train_samples), (
                f"Unexpected train size for "
                f"{held_out_dataset} + {held_out_gnn}: "
                f"{len(X_train)} "
                f"instead of "
                f"{expected_train_samples}"
            )

            assert (len(X_test) == expected_test_samples), (
                f"Unexpected test size for "
                f"{held_out_dataset} + {held_out_gnn}: "
                f"{len(X_test)} "
                f"instead of "
                f"{expected_test_samples}"
            )

            # ====================================================
            # Preference Weight Checks
            # ====================================================
            train_weight_sums = X_train_tensor[:, -6:].sum(dim=1)
            test_weight_sums = X_test_tensor[:, -6:].sum(dim=1)
            assert torch.allclose(train_weight_sums, torch.ones_like(train_weight_sums), atol=1e-6), (
                "Some training preference vectors "
                "do not sum to 1."
            )
            assert torch.allclose(test_weight_sums, torch.ones_like(test_weight_sums), atol=1e-6), (
                "Some test preference vectors "
                "do not sum to 1."
            )

            # ====================================================
            # Finite-Value Checks
            # ====================================================
            assert torch.isfinite(X_train_tensor).all(), ("Training features contain NaN or Inf.")
            assert torch.isfinite(X_test_tensor).all(), ("Test features contain NaN or Inf.")

            # ====================================================
            # Label-Range Checks
            # ====================================================
            assert torch.all(Y_train_tensor >= 0)
            assert torch.all(Y_train_tensor < len(Explainers))
            assert torch.all(Y_test_tensor >= 0)
            assert torch.all(Y_test_tensor < len(Explainers))
            print("Combined LODO + LOGO sanity checks passed.")
            # ====================================================
            # File Prefix
            # ====================================================
            file_prefix = (
                    "LODO_LOGO_" + "GridSearch_Weights_" + weight_file_name + "_HeldOutDataset_" + held_out_dataset
                    + "_HeldOutGNN_" + held_out_gnn)
            # ====================================================
            # Output Files
            # ====================================================
            x_train_file = combined_output_directory + "X_Train_" + file_prefix + ".pt"
            y_train_file = combined_output_directory + "Y_Train_" + file_prefix + ".pt"
            x_test_file = combined_output_directory + "X_Test_" + file_prefix + ".pt"
            y_test_file = combined_output_directory + "Y_Test_" + file_prefix + ".pt"

            # ====================================================
            # Save
            # ====================================================
            torch.save(X_train, x_train_file)
            torch.save(Y_train, y_train_file)
            torch.save(X_test, x_test_file)
            torch.save(Y_test, y_test_file)
            total_splits_created += 1
            print("\nSaved:")
            print("X train:", x_train_file)
            print("Y train:", y_train_file)
            print("X test:", x_test_file)
            print("Y test:", y_test_file)

# ============================================================
# Final Summary
# ============================================================
expected_total_splits = len(weight_levels) * len(Datasets_Name) * len(GNN_Models)
assert (total_splits_created == expected_total_splits), (
    f"Expected {expected_total_splits} splits, "
    f"but created {total_splits_created}."
)
print("\n" + "=" * 100)
print(
    "All combined LODO + LOGO datasets "
    "created successfully."
)
print("Total number of splits:", total_splits_created)
print("Expected number of splits:", expected_total_splits)
print("Output directory:")
print(combined_output_directory)