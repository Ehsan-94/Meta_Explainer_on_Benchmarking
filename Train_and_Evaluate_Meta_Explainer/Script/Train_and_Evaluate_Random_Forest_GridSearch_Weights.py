import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split


# ============================================================
# Reproducibility
# ============================================================

random_state = 42

np.random.seed(random_state)
torch.manual_seed(random_state)


# ============================================================
# Experimental Setup
# ============================================================

Datasets_Name = ["MUTAG", "NCI1", "ENZYMES", "Graph-SST5", "PROTEINS", "IsCyclic"]

Explainers = [
    "GNNExplainer",
    "SubgraphX",
    "PGMExplainer",
    "CF2",
    "PGExplainer",
    "GraphMask",
    "XGNN",
    "GNNInterpreter",
]

weight_levels = {
    "[1, 2]": [1, 2],
    "[1, 2, 3]": [1, 2, 3],
    "[1, 2, 3, 4]": [1, 2, 3, 4],
}

test_size = 0.2
PRINT_CLASS_DISTRIBUTION = False

# Standard, intentionally simple Random Forest baseline.
number_of_trees = 100


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

models_directory = (
    base_results_directory
    + "Random_Forest_Models_Trained_on_GridSearch_Weights/"
)

results_directory = (
    base_results_directory
    + "TestResults_of_Random_Forest_on_GridSearch_Weights/"
)

os.makedirs(models_directory, exist_ok=True)
os.makedirs(results_directory, exist_ok=True)


# ============================================================
# Train and Evaluate Random Forest
# ============================================================

def train_and_evaluate_random_forest(X_train, Y_train, X_test, Y_test):
    """
    Train Random Forest on the exact same 48-dimensional inputs
    and labels used in the Meta-Explainer Grid experiment.

    Random Forest does not require feature standardization, so
    the original features are used directly.
    """

    X_train_np = X_train.detach().cpu().numpy()
    Y_train_np = Y_train.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy()
    Y_test_np = Y_test.detach().cpu().numpy()

    random_forest = RandomForestClassifier(
        n_estimators=number_of_trees,
        random_state=random_state,
        n_jobs=-1,
    )

    random_forest.fit(X_train_np, Y_train_np)

    predictions = random_forest.predict(X_test_np)
    probabilities_raw = random_forest.predict_proba(X_test_np)

    # sklearn returns probability columns only for classes seen in training.
    # Expand them to the full 8-explainer class space for log-loss.
    probabilities = np.zeros(
        (len(X_test_np), len(Explainers)),
        dtype=np.float64,
    )

    for probability_column, class_label in enumerate(random_forest.classes_):
        probabilities[:, int(class_label)] = probabilities_raw[:, probability_column]

    test_loss = log_loss(
        Y_test_np,
        probabilities,
        labels=np.arange(len(Explainers)),
    )

    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy_score(Y_test_np, predictions),
        "precision_macro": precision_score(
            Y_test_np, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            Y_test_np, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            Y_test_np, predictions, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            Y_test_np, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            Y_test_np, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            Y_test_np, predictions, average="weighted", zero_division=0
        ),
    }

    return random_forest, metrics


# ============================================================
# Create nested stratified subsets from the 80% training pool
#
# Kept identical to the uploaded Meta-Explainer Grid script.
# ============================================================

def create_nested_stratified_indices(Y_train_pool, training_scenarios, random_state):
    """
    Creates nested, approximately stratified training subsets:
    T20 subset T40 subset T60 subset T80
    """

    generator = torch.Generator()
    generator.manual_seed(random_state)

    class_indices = {}
    unique_classes = torch.unique(Y_train_pool)

    for class_index in unique_classes:
        indices = torch.where(Y_train_pool == class_index)[0]
        permutation = torch.randperm(len(indices), generator=generator)
        class_indices[int(class_index.item())] = indices[permutation]

    nested_indices = {}

    for full_dataset_fraction, pool_fraction in training_scenarios.items():
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
        subset_permutation = torch.randperm(
            len(selected_indices),
            generator=generator,
        )
        selected_indices = selected_indices[subset_permutation]
        nested_indices[full_dataset_fraction] = selected_indices

    return nested_indices


# ============================================================
# Training-size scenarios
# ============================================================

training_scenarios = {
    0.2: 0.25,
    0.4: 0.50,
    0.6: 0.75,
    0.8: 1.00,
}


# ============================================================
# Train for every weight-level configuration
# ============================================================

for weight_level_name, current_weight_levels in weight_levels.items():

    print("\n" + "=" * 100)
    print("Random Forest baseline for weight levels:", weight_level_name)
    print("=" * 100)

    weight_file_name = "_".join(str(weight) for weight in current_weight_levels)
    file_names = "GridSearch_Weights_" + weight_file_name

    directory_x = dataset_directory + "X_" + file_names + ".pt"
    directory_y = dataset_directory + "Y_" + file_names + ".pt"

    X_list = torch.load(directory_x)
    Y_list = torch.load(directory_y)

    X_data = torch.stack(X_list).float()
    Y_data = torch.stack(Y_list).long()

    weights_sum = X_data[:, -6:].sum(dim=1)

    assert torch.allclose(
        weights_sum,
        torch.ones_like(weights_sum),
        atol=1e-6,
    ), "Some preference-weight vectors do not sum to 1."

    if Y_data.dim() > 1 and Y_data.size(1) > 1:
        Y_data = torch.argmax(Y_data, dim=1).long()

    if PRINT_CLASS_DISTRIBUTION:
        counts = Counter(Y_data.tolist())
        total = len(Y_data)

        print("\nOverall class distribution:")

        for class_index in range(len(Explainers)):
            count = counts.get(class_index, 0)
            print(
                f"  {class_index} "
                f"({Explainers[class_index]}): "
                f"{count} ({count / total:.2%})"
            )

    results = defaultdict(list)
    results["explainer_order"] = Explainers.copy()
    results["weight_levels"] = current_weight_levels.copy()
    results["classifier"] = "RandomForestClassifier"
    results["n_estimators"] = number_of_trees
    results["random_state"] = random_state

    # ========================================================
    # ONE fixed 80% training pool + 20% test set
    #
    # Exact same split as the Meta-Explainer Grid experiment.
    # ========================================================

    (
        X_train_pool,
        X_test_raw,
        Y_train_pool,
        Y_test,
    ) = train_test_split(
        X_data,
        Y_data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=Y_data,
    )

    print("\nFixed train/test split created.")
    print("Training pool size:", len(X_train_pool))
    print("Fixed test-set size:", len(X_test_raw))
    print(
        "Fixed test percentage:",
        f"{len(X_test_raw) / len(X_data):.2%}",
    )

    if PRINT_CLASS_DISTRIBUTION:
        counts_test = Counter(Y_test.tolist())
        total_test = len(Y_test)

        print("\nFixed test-set distribution:")

        for class_index in range(len(Explainers)):
            count = counts_test.get(class_index, 0)
            print(
                f"  {class_index} "
                f"({Explainers[class_index]}): "
                f"{count} ({count / total_test:.2%})"
            )

    # ========================================================
    # Create exact same nested + stratified training subsets
    # ========================================================

    nested_training_indices = create_nested_stratified_indices(
        Y_train_pool=Y_train_pool,
        training_scenarios=training_scenarios,
        random_state=random_state,
    )

    # ========================================================
    # Verify nesting
    # ========================================================

    scenario_keys = list(training_scenarios.keys())

    for i in range(len(scenario_keys) - 1):
        smaller = set(nested_training_indices[scenario_keys[i]].tolist())
        larger = set(nested_training_indices[scenario_keys[i + 1]].tolist())

        assert smaller.issubset(larger), (
            f"Training subsets are not nested: "
            f"{scenario_keys[i]} -> {scenario_keys[i + 1]}"
        )

    print("\nNested training-subset check passed:")
    print("T20 subset T40 subset T60 subset T80")

    # ========================================================
    # Training-size scenarios
    # ========================================================

    for train_size in training_scenarios.keys():

        print("\n" + "-" * 100)
        print(
            "Training Random Forest with approximately "
            f"{int(train_size * 100)}% of the complete dataset..."
        )
        print("-" * 100)

        train_indices = nested_training_indices[train_size]
        X_train = X_train_pool[train_indices]
        Y_train = Y_train_pool[train_indices]

        print(
            "Actual training percentage:",
            f"{len(X_train) / len(X_data):.2%}",
        )

        if PRINT_CLASS_DISTRIBUTION:
            counts_train = Counter(Y_train.tolist())
            total_train = len(Y_train)

            print("Training-set distribution:")

            for class_index in range(len(Explainers)):
                count = counts_train.get(class_index, 0)
                print(
                    f"  {class_index} "
                    f"({Explainers[class_index]}): "
                    f"{count} ({count / total_train:.2%})"
                )

        # ====================================================
        # Train + Evaluate
        # ====================================================

        random_forest, metrics = train_and_evaluate_random_forest(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test_raw,
            Y_test=Y_test,
        )

        # ====================================================
        # Store Results
        # ====================================================

        results["train_size"].append(train_size * 100)
        results["actual_train_percentage"].append(
            100 * len(X_train) / len(X_data)
        )
        results["num_train_samples"].append(len(X_train))
        results["num_test_samples"].append(len(X_test_raw))

        for metric_name, value in metrics.items():
            results[metric_name].append(value)

        # ====================================================
        # Print Results
        # ====================================================

        print(f"Test Loss: {metrics['test_loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")

        # ====================================================
        # Save trained Random Forest
        #
        # Pickle is used because this is an sklearn model.
        # These exact Grid-trained models can later be loaded
        # for the Continuous evaluation.
        # ====================================================

        train_size_name = str(int(train_size * 100))

        model_file = (
            models_directory
            + "Random_Forest_Model_"
            + file_names
            + "_Train_"
            + train_size_name
            + "Percent.pkl"
        )

        with open(model_file, "wb") as file:
            pickle.dump(random_forest, file)

        print("Saved model:", model_file)

    # ========================================================
    # Save Results
    # ========================================================

    results_file = (
        results_directory
        + "Random_Forest_Results_"
        + file_names
        + ".pkl"
    )

    with open(results_file, "wb") as file:
        pickle.dump(dict(results), file)

    print("\nSaved results:")
    print(results_file)
