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
with open(constants_file, "rb") as f:
    constants = pickle.load(f)
Datasets_Name = constants["Datasets_Name"]
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
lodo_output_directory = base_dataset_results_directory + "LODO_GridSearch_Weights/"
os.makedirs(lodo_output_directory, exist_ok=True)

# ============================================================
# Create One LODO Split
# ============================================================
def create_lodo_split(Meta_Explainer_Data, held_out_dataset):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for dataset_name in Datasets_Name:
        for gnn_name in (Meta_Explainer_Data[dataset_name].keys()):
            for weights_key, sample in (Meta_Explainer_Data[dataset_name][gnn_name].items()):
                feature_array = np.array(sample["features"], dtype=np.float32)
                label_index = sample["label_index"]
                if dataset_name == held_out_dataset:
                    X_test.append(torch.from_numpy(feature_array))
                    Y_test.append(torch.tensor(label_index, dtype=torch.long))
                else:
                    X_train.append(torch.from_numpy(feature_array))
                    Y_train.append(torch.tensor(label_index, dtype=torch.long))
    return X_train, Y_train, X_test, Y_test

# ============================================================
# Create LODO Datasets
# ============================================================
for (weight_level_name, current_weight_levels) in weight_levels.items():
    print("\n" + "=" * 100)
    print("Creating LODO datasets for weight levels:", weight_level_name)
    print("=" * 100)
    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)

    # ========================================================
    # Load Structured Grid Dataset
    # ========================================================
    structured_data_file = grid_search_directory + "Meta_Explainer_Data_GridSearch_Weights_" + weight_file_name + ".pkl"
    with open(structured_data_file, "rb") as file:
        Meta_Explainer_Data = pickle.load(file)

    # ========================================================
    # Leave Each Dataset Out Once
    # ========================================================
    for held_out_dataset in Datasets_Name:
        print("\nHeld-out dataset:", held_out_dataset)
        (X_train, Y_train, X_test, Y_test) = create_lodo_split(Meta_Explainer_Data=Meta_Explainer_Data,
                                                               held_out_dataset=held_out_dataset)

        # ====================================================
        # Stack Only for Checking
        # ====================================================
        X_train_tensor = torch.stack(X_train)
        Y_train_tensor = torch.stack(Y_train)
        X_test_tensor = torch.stack(X_test)
        Y_test_tensor = torch.stack(Y_test)
        print("X train shape:", X_train_tensor.shape)
        print("Y train shape:", Y_train_tensor.shape)
        print("X test shape:", X_test_tensor.shape)
        print("Y test shape:", Y_test_tensor.shape)

        # ====================================================
        # Sanity Checks
        # ====================================================
        assert (X_train_tensor.shape[1] == expected_feature_size_for_meta_explainer)
        assert (X_test_tensor.shape[1] == expected_feature_size_for_meta_explainer)
        assert (X_train_tensor.shape[0] == Y_train_tensor.shape[0])
        assert (X_test_tensor.shape[0] == Y_test_tensor.shape[0])
        train_weight_sums = X_train_tensor[:, -6:].sum(dim=1)
        test_weight_sums = X_test_tensor[:, -6:].sum(dim=1)

        assert torch.allclose(train_weight_sums, torch.ones_like(train_weight_sums), atol=1e-6)
        assert torch.allclose(test_weight_sums, torch.ones_like(test_weight_sums), atol=1e-6)

        # ====================================================
        # Expected Test Size
        # Each held-out dataset contains:
        # 4 GNNs x number_of_weight_vectors
        # ====================================================
        number_of_weight_vectors = len(
            Meta_Explainer_Data[held_out_dataset][next(iter(Meta_Explainer_Data[held_out_dataset]))])
        expected_test_samples = (number_of_weight_vectors * 4)
        assert (len(X_test) == expected_test_samples)
        print("Expected test samples:", expected_test_samples)
        print("LODO sanity checks passed.")

        # ====================================================
        # File Prefix
        # ====================================================
        file_prefix = "LODO_" + "GridSearch_Weights_" + weight_file_name + "_HeldOut_" + held_out_dataset

        # ====================================================
        # Save
        # ====================================================
        x_train_file = lodo_output_directory + "X_Train_" + file_prefix + ".pt"
        y_train_file = lodo_output_directory + "Y_Train_" + file_prefix + ".pt"
        x_test_file = lodo_output_directory + "X_Test_" + file_prefix + ".pt"
        y_test_file = lodo_output_directory + "Y_Test_" + file_prefix + ".pt"
        torch.save(X_train, x_train_file)
        torch.save(Y_train, y_train_file)
        torch.save(X_test, x_test_file)
        torch.save(Y_test, y_test_file)
        print("Saved:")
        print(x_train_file)
        print(y_train_file)
        print(x_test_file)
        print(y_test_file)