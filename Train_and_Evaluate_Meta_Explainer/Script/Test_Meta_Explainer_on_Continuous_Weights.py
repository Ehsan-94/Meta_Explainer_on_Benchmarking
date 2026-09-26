import os
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

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
training_scenarios = [0.2, 0.4, 0.6, 0.8]
number_of_continuous_weight_vectors = 4000
input_size = 48
hidden_size = 64
output_size = len(Explainers)
batch_size = 256

# ============================================================
# Directories
# ============================================================

dataset_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
    "Continuous_Weights/"
)

base_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Train_and_Evaluate_Meta_Explainer/"
    "Experimental Results/"
)

models_directory = base_results_directory + "Meta_Explainer_Models_Trained_on_GridSearch_Weights/"
results_directory = base_results_directory + "TestResults_of_Meta_Explainer_on_Continuous_Weights/"

os.makedirs(results_directory, exist_ok=True)
os.makedirs(results_directory, exist_ok=True)

# ============================================================
# Meta-Explainer Architecture
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
# Load Continuous Test Dataset
# ============================================================
x_file = dataset_directory + "X_ContinuousWeights_" + str(number_of_continuous_weight_vectors) + "_WeightVectors.pt"
y_file = dataset_directory + "Y_ContinuousWeights_" + str(number_of_continuous_weight_vectors) + "_WeightVectors.pt"
X_list = torch.load(x_file)
Y_list = torch.load(y_file)
X_continuous_raw = torch.stack(X_list).float()
Y_continuous = torch.stack(Y_list).long()
print("\nContinuous X shape:", X_continuous_raw.shape)
print("Continuous Y shape:", Y_continuous.shape)


# ============================================================
# Sanity Checks
# ============================================================
assert (X_continuous_raw.shape[1] == expected_feature_size_for_meta_explainer)
assert (X_continuous_raw.shape[0] == Y_continuous.shape[0])
weight_sums = X_continuous_raw[:, -6:].sum(dim=1)
assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
print("Continuous dataset sanity checks passed.")


# ============================================================
# Class Distribution
# ============================================================
counts = Counter(Y_continuous.tolist())
total = len(Y_continuous)
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
# Evaluation Function
# ============================================================
def evaluate_model(model, X_test, Y_test, batch_size):
    test_dataset = (torch.utils.data.TensorDataset(X_test, Y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device, dtype=torch.long)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item() * batch_Y.size(0)
            predictions = torch.argmax(outputs, dim=1)
            test_correct += (predictions == batch_Y).sum().item()
            test_total += (batch_Y.size(0))
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())
    avg_test_loss = test_loss / test_total
    accuracy = (test_correct / test_total)
    precision_macro = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    precision_weighted = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall_weighted = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
    return {
        "test_loss": avg_test_loss,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }


# ============================================================
# Evaluate All 12 Grid-Trained Models
# ============================================================
continuous_results = {}
for (weight_level_name, current_weight_levels) in weight_levels.items():
    print("\n" + "=" * 100)
    print("Grid-trained models:", weight_level_name)
    print("=" * 100)
    continuous_results[weight_level_name] = {}
    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)
    file_names = ("GridSearch_Weights_" + weight_file_name)
    for train_size in training_scenarios:
        train_size_name = str(
            int(train_size * 100))
        print(
            "\nEvaluating "
            f"{weight_level_name} "
            f"with "
            f"{train_size_name}% training..."
        )

        # ====================================================
        # Matching Model File
        # ====================================================
        model_file = (
                models_directory + "Meta_Explainer_Model_" + file_names + "_Train_" + train_size_name + "Percent.pt")


        # ====================================================
        # Matching Scaler File
        # ====================================================
        scaler_file = (
                models_directory + "Meta_Explainer_Scaler_" + file_names + "_Train_" + train_size_name + "Percent.pkl")

        # ====================================================
        # Existence Checks
        # ====================================================
        assert os.path.exists(
            model_file
        ), (
            f"Model file not found: "
            f"{model_file}"
        )

        assert os.path.exists(
            scaler_file
        ), (
            f"Scaler file not found: "
            f"{scaler_file}"
        )

        # ====================================================
        # Load Matching Scaler
        # ====================================================
        with open(scaler_file, "rb") as file:
            scaler = pickle.load(file)

        # ====================================================
        # Start From RAW Continuous Features
        #
        # Every model receives the SAME raw test dataset,
        # but preprocessing uses that model's own scaler.
        # ====================================================
        X_continuous_np = X_continuous_raw.cpu().numpy().copy()
        original_weights = X_continuous_np[:, 42:48].copy()


        # ====================================================
        # Scale First 42 Features Only
        # ====================================================
        X_continuous_np[:, :42] = scaler.transform(X_continuous_np[:, :42])

        # ====================================================
        # Verify Preference Weights Were Not Modified
        # ====================================================
        assert np.allclose(X_continuous_np[:, 42:48], original_weights)
        X_continuous_scaled = torch.tensor(X_continuous_np, dtype=torch.float32)

        # ====================================================
        # Load Model
        # ====================================================
        model = MetaExplainer(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        # ====================================================
        # Evaluate
        # ====================================================
        metrics = evaluate_model(model=model, X_test=X_continuous_scaled, Y_test=Y_continuous, batch_size=batch_size)


        # ====================================================
        # Store Results
        # ====================================================
        continuous_results[weight_level_name][train_size_name] = metrics

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


# ============================================================
# Save Results
# ============================================================
results_file = (
        results_directory
        + "Meta_Explainer_GridSearch_Models_"
        + "Continuous_Weights_Evaluation_Results.pkl"
)

with open(results_file, "wb") as file:
    pickle.dump(continuous_results, file)

print("\n" + "=" * 100)
print("Continuous-weight evaluation completed.")
print("Results saved to:")
print(results_file)