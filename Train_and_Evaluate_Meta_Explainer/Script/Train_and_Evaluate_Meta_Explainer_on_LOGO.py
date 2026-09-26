import os
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# Reproducibility
# ============================================================
random_state = 42

np.random.seed(random_state)
torch.manual_seed(random_state)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_state)

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

GNN_Models = constants["GNN_Models"]
Explainers = constants["Explainers"]

expected_feature_size_for_meta_explainer = constants["expected_feature_size_for_meta_explainer"]

# ============================================================
# Configuration
# ============================================================

weight_levels = {
    "[1, 2]": [1, 2],
    "[1, 2, 3]": [1, 2, 3],
    "[1, 2, 3, 4]": [1, 2, 3, 4]
}

input_size = 48
hidden_size = 64
output_size = len(Explainers)

num_epochs = 500

batch_size = 32
learning_rate = 0.001

PRINT_CLASS_DISTRIBUTION = True

# ============================================================
# Directories
# ============================================================
base_dataset_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
)

logo_dataset_directory = base_dataset_results_directory + "LOGO_GridSearch_Weights/"

base_training_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Train_and_Evaluate_Meta_Explainer/"
    "Experimental Results/"
)

logo_models_directory = base_training_results_directory + "Meta_Explainer_Models_Trained_LOGO/"
logo_results_directory = base_training_results_directory + "TestResults_of_Meta_Explainer_LOGO/"
os.makedirs(logo_models_directory, exist_ok=True)
os.makedirs(logo_results_directory, exist_ok=True)

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
# Scale First 42 Features
#
# Scaler is fitted ONLY on the three-GNN training set.
# ============================================================
def scale_features(X_train, X_test):
    X_train_np = X_train.cpu().numpy().copy()
    X_test_np = X_test.cpu().numpy().copy()
    scaler = StandardScaler()
    X_train_np[:, :42] = scaler.fit_transform(X_train_np[:, :42])
    X_test_np[:, :42] = scaler.transform(X_test_np[:, :42])
    X_train_scaled = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_np, dtype=torch.float32)

    return (X_train_scaled, X_test_scaled, scaler)


# ============================================================
# Train + Evaluate
# ============================================================
def train_and_evaluate(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, num_epochs, batch_size,
                       learning_rate):
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    meta_explainer = MetaExplainer(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta_explainer.parameters(), lr=learning_rate)

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
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device, dtype=torch.long)
            outputs = meta_explainer(batch_X)
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item() * batch_Y.size(0)
            predictions = torch.argmax(outputs, dim=1)
            test_correct += (predictions == batch_Y).sum().item()
            test_total += batch_Y.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())

    # ========================================================
    # Metrics
    # ========================================================
    avg_test_loss = test_loss / test_total
    accuracy = test_correct / test_total
    precision_macro = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    precision_weighted = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall_weighted = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
    metrics = {
        "test_loss": avg_test_loss,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }

    return meta_explainer, metrics


# ============================================================
# LOGO Experiment
# ============================================================
LOGO_results = {}
for (weight_level_name, current_weight_levels) in weight_levels.items():
    # print("\n" + "=" * 100)
    # print("Leave-one-GNN-out for weight levels:", weight_level_name)
    # print("=" * 100)
    LOGO_results[weight_level_name] = {}
    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)
    # ========================================================
    # Hold Out Each GNN
    # ========================================================
    for held_out_gnn in GNN_Models:
        # print("\n" + "=" * 100)
        # print("Held-out GNN:", held_out_gnn)
        # print("-" * 100)
        # ====================================================
        # File Prefix
        # ====================================================
        file_prefix = "LOGO_"+ "GridSearch_Weights_" + weight_file_name + "_HeldOut_" + held_out_gnn
        # ====================================================
        # Dataset Files
        # ====================================================
        x_train_file = logo_dataset_directory + "X_Train_" + file_prefix + ".pt"
        y_train_file = logo_dataset_directory + "Y_Train_" + file_prefix + ".pt"
        x_test_file = logo_dataset_directory + "X_Test_" + file_prefix + ".pt"
        y_test_file = logo_dataset_directory + "Y_Test_" + file_prefix + ".pt"
        # ====================================================
        # Existence Checks
        # ====================================================
        assert os.path.exists(
            x_train_file
        ), (
            f"Missing file: "
            f"{x_train_file}"
        )
        assert os.path.exists(
            y_train_file
        ), (
            f"Missing file: "
            f"{y_train_file}"
        )
        assert os.path.exists(
            x_test_file
        ), (
            f"Missing file: "
            f"{x_test_file}"
        )
        assert os.path.exists(
            y_test_file
        ), (
            f"Missing file: "
            f"{y_test_file}"
        )

        # ====================================================
        # Load Dataset
        # ====================================================
        X_train_list = torch.load(x_train_file)
        Y_train_list = torch.load(y_train_file)
        X_test_list = torch.load(x_test_file)
        Y_test_list = torch.load(y_test_file)
        X_train_raw = torch.stack(X_train_list).float()
        Y_train = torch.stack(Y_train_list).long()
        X_test_raw = torch.stack(X_test_list).float()
        Y_test = torch.stack(Y_test_list).long()
        # print("X train shape:", X_train_raw.shape)
        # print("Y train shape:", Y_train.shape)
        # print("X test shape:", X_test_raw.shape)
        # print("Y test shape:", Y_test.shape)
        # ====================================================
        # Sanity Checks
        # ====================================================
        assert (X_train_raw.shape[1] == expected_feature_size_for_meta_explainer)
        assert (X_test_raw.shape[1] == expected_feature_size_for_meta_explainer)
        assert (len(X_train_raw) == len(Y_train))
        assert (len(X_test_raw) == len(Y_test))

        # ----------------------------------------------------
        # Preference weights must sum to one
        # ----------------------------------------------------

        train_weight_sums = X_train_raw[:, -6:].sum(dim=1)
        test_weight_sums = X_test_raw[:, -6:].sum(dim=1)
        assert torch.allclose(train_weight_sums, torch.ones_like(train_weight_sums), atol=1e-6)
        assert torch.allclose(test_weight_sums, torch.ones_like(test_weight_sums), atol=1e-6)
        # ----------------------------------------------------
        # No NaN / Inf
        # ----------------------------------------------------
        assert torch.isfinite(X_train_raw).all()
        assert torch.isfinite(X_test_raw).all()
        # ====================================================
        # Class Distribution
        # ====================================================
        if PRINT_CLASS_DISTRIBUTION:
            train_counts = Counter(Y_train.tolist())
            test_counts = Counter(Y_test.tolist())
            # print("\nTraining class distribution:")
            for class_index in range(len(Explainers)):
                count = train_counts.get(class_index, 0)
                print(
                    f"  "
                    f"{class_index} "
                    f"({Explainers[class_index]}): "
                    f"{count} "
                    f"({count / len(Y_train):.2%})"
                )
            # print("\nHeld-out test class distribution:")
            for class_index in range(len(Explainers)):
                count = test_counts.get(class_index, 0)
                print(
                    f"  "
                    f"{class_index} "
                    f"({Explainers[class_index]}): "
                    f"{count} "
                    f"({count / len(Y_test):.2%})"
                )

        # ====================================================
        # Scale Using TRAINING DATA ONLY
        # ====================================================
        X_train, X_test, scaler = scale_features(X_train_raw, X_test_raw)

        # ====================================================
        # Verify Preference Features Were Not Changed
        # ====================================================
        assert torch.allclose(X_train[:, 42:48], X_train_raw[:, 42:48], atol=1e-7)
        assert torch.allclose(X_test[:, 42:48], X_test_raw[:, 42:48], atol=1e-7)

        # ====================================================
        # Train + Evaluate
        # ====================================================
        meta_explainer, metrics = train_and_evaluate(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
                                                     input_size=input_size, hidden_size=hidden_size,
                                                     output_size=output_size, num_epochs=num_epochs,
                                                     batch_size=batch_size, learning_rate=learning_rate)

        # ====================================================
        # Store Results
        # ====================================================
        LOGO_results[weight_level_name][held_out_gnn] = {
            "held_out_gnn": held_out_gnn,
            "weight_levels": current_weight_levels.copy(),
            "num_train_samples": len(X_train_raw),
            "num_test_samples": len(X_test_raw),
            **metrics}

        # ====================================================
        # Print Results
        # ====================================================
        print(
            f"\nTest Loss: "
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
        # Save Model
        # ====================================================
        model_file = logo_models_directory + "Meta_Explainer_" + file_prefix + ".pt"
        torch.save(meta_explainer.state_dict(), model_file)
        print("Saved model:", model_file)

        # ====================================================
        # Save Matching Scaler
        # ====================================================
        scaler_file = logo_models_directory + "Meta_Explainer_Scaler_" + file_prefix + ".pkl"
        with open(scaler_file, "wb") as file:
            pickle.dump(scaler, file)
        print("Saved scaler:", scaler_file)

# ============================================================
# Save All LOGO Results
# ============================================================
results_file = logo_results_directory + "Meta_Explainer_LOGO_" + "GridSearch_Weights_" + "Evaluation_Results.pkl"
with open(results_file, "wb") as file:
    pickle.dump(LOGO_results, file)
# print("\n" + "=" * 100)
# print("Leave-one-GNN-out training and evaluation completed.")
# print("Results saved to:")
# print(results_file)