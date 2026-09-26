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

expected_feature_size_for_meta_explainer = constants["expected_feature_size_for_meta_explainer"]

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
logo_output_directory = base_dataset_results_directory + "LOGO_GridSearch_Weights/"
os.makedirs(logo_output_directory, exist_ok=True)

# ============================================================
# Create One Leave-One-GNN-Out Split
# ============================================================
def create_logo_split(Meta_Explainer_Data, held_out_gnn):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for dataset_name in Datasets_Name:
        for gnn_name in GNN_Models:
            for weights_key, sample in Meta_Explainer_Data[dataset_name][gnn_name].items():
                feature_array = np.array(sample["features"], dtype=np.float32)
                label_index = int(sample["label_index"])
                if gnn_name == held_out_gnn:
                    X_test.append(torch.from_numpy(feature_array))
                    Y_test.append(torch.tensor(label_index, dtype=torch.long))
                else:
                    X_train.append(torch.from_numpy(feature_array))
                    Y_train.append(torch.tensor(label_index, dtype=torch.long))
    return X_train, Y_train, X_test, Y_test


# ============================================================
# Create Leave-One-GNN-Out Datasets
# ============================================================
for (weight_level_name, current_weight_levels) in weight_levels.items():

    print("\n" + "=" * 100)
    print(
        "Creating leave-one-GNN-out datasets "
        "for weight levels:",
        weight_level_name
    )
    print("=" * 100)
    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)
    structured_data_file = grid_search_directory + "Meta_Explainer_Data_GridSearch_Weights_" + weight_file_name + ".pkl"

    assert os.path.exists(
        structured_data_file
    ), (
        f"Structured grid-search file not found: "
        f"{structured_data_file}"
    )
    with open(structured_data_file, "rb") as file:
        Meta_Explainer_Data = pickle.load(file)


    # ========================================================
    # Hold Out Each GNN Once
    # ========================================================
    for held_out_gnn in GNN_Models:
        print("\n" + "-" * 100)
        print("Held-out GNN:", held_out_gnn)
        print("-" * 100)
        X_train, Y_train, X_test, Y_test = create_logo_split(Meta_Explainer_Data=Meta_Explainer_Data,
                                                             held_out_gnn=held_out_gnn)
        X_train_tensor = torch.stack(X_train).float()
        Y_train_tensor = torch.stack(Y_train).long()
        X_test_tensor = torch.stack(X_test).float()
        Y_test_tensor = torch.stack(Y_test).long()

        print("X train shape:", X_train_tensor.shape)
        print("Y train shape:", Y_train_tensor.shape)
        print("X test shape:", X_test_tensor.shape)
        print("Y test shape:", Y_test_tensor.shape)

        assert (X_train_tensor.shape[1] == expected_feature_size_for_meta_explainer)
        assert (X_test_tensor.shape[1] == expected_feature_size_for_meta_explainer)
        assert (X_train_tensor.shape[0] == Y_train_tensor.shape[0])
        assert (X_test_tensor.shape[0] == Y_test_tensor.shape[0])

        train_weight_sums = X_train_tensor[:, -6:].sum(dim=1)
        test_weight_sums = X_test_tensor[:, -6:].sum(dim=1)

        assert torch.allclose(train_weight_sums, torch.ones_like(train_weight_sums), atol=1e-6)
        assert torch.allclose(test_weight_sums, torch.ones_like(test_weight_sums), atol=1e-6)

        # ====================================================
        # Expected Sample Counts
        #
        # Test:
        #   6 datasets
        #   x 1 held-out GNN
        #   x number_of_weight_vectors
        #
        # Train:
        #   6 datasets
        #   x 3 remaining GNNs
        #   x number_of_weight_vectors
        # ====================================================
        example_dataset = Datasets_Name[0]
        number_of_weight_vectors = len(Meta_Explainer_Data[example_dataset][held_out_gnn])
        expected_test_samples = len(Datasets_Name) * number_of_weight_vectors
        expected_train_samples = len(Datasets_Name) * (len(GNN_Models) - 1) * number_of_weight_vectors
        assert (len(X_test) == expected_test_samples)
        assert (len(X_train) == expected_train_samples)
        print("Expected train samples:", expected_train_samples)
        print("Expected test samples:", expected_test_samples)

        # ====================================================
        # Strong Check:
        # Feature values must all be finite
        # ====================================================
        assert torch.isfinite(X_train_tensor).all()
        assert torch.isfinite(X_test_tensor).all()
        print("Leave-one-GNN-out sanity checks passed.")

        # ====================================================
        # File Prefix
        # ====================================================
        file_prefix = "LOGO_" + "GridSearch_Weights_" + weight_file_name + "_HeldOut_" + held_out_gnn


        # ====================================================
        # Save Files
        # ====================================================
        x_train_file = logo_output_directory + "X_Train_" + file_prefix + ".pt"
        y_train_file = logo_output_directory + "Y_Train_" + file_prefix + ".pt"
        x_test_file = logo_output_directory + "X_Test_" + file_prefix + ".pt"
        y_test_file = logo_output_directory + "Y_Test_" + file_prefix + ".pt"

        torch.save(X_train, x_train_file)
        torch.save(Y_train, y_train_file)
        torch.save(X_test, x_test_file)
        torch.save(Y_test, y_test_file)
        print("\nSaved:")
        print("X train:", x_train_file)
        print("Y train:", y_train_file)
        print("X test:", x_test_file)
        print("Y test:", y_test_file)

print("\n" + "=" * 100)
print("All leave-one-GNN-out datasets created successfully.")