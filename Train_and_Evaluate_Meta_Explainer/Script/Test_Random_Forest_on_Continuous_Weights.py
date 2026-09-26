import os
import pickle
from collections import Counter

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


# ============================================================
# Reproducibility
# ============================================================

random_state = 42

np.random.seed(random_state)
torch.manual_seed(random_state)


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

Explainers = constants["Explainers"]
expected_feature_size_for_meta_explainer = (
    constants["expected_feature_size_for_meta_explainer"]
)


# ============================================================
# Configuration
# ============================================================

weight_levels = {
    "[1, 2]": [1, 2],
    "[1, 2, 3]": [1, 2, 3],
    "[1, 2, 3, 4]": [1, 2, 3, 4],
}

training_scenarios = [
    0.2,
    0.4,
    0.6,
    0.8,
]

number_of_continuous_weight_vectors = 4000


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

models_directory = (
    base_results_directory
    + "Random_Forest_Models_Trained_on_GridSearch_Weights/"
)

results_directory = (
    base_results_directory
    + "TestResults_of_Random_Forest_on_Continuous_Weights/"
)

os.makedirs(
    results_directory,
    exist_ok=True,
)


# ============================================================
# Load Continuous Test Dataset
# ============================================================

x_file = (
    dataset_directory
    + "X_ContinuousWeights_"
    + str(number_of_continuous_weight_vectors)
    + "_WeightVectors.pt"
)

y_file = (
    dataset_directory
    + "Y_ContinuousWeights_"
    + str(number_of_continuous_weight_vectors)
    + "_WeightVectors.pt"
)

X_list = torch.load(
    x_file
)

Y_list = torch.load(
    y_file
)

X_continuous = torch.stack(
    X_list
).float()

Y_continuous = torch.stack(
    Y_list
).long()

if (
    Y_continuous.dim() > 1
    and Y_continuous.size(1) > 1
):
    Y_continuous = torch.argmax(
        Y_continuous,
        dim=1,
    ).long()

print(
    "\nContinuous X shape:",
    X_continuous.shape,
)

print(
    "Continuous Y shape:",
    Y_continuous.shape,
)


# ============================================================
# Sanity Checks
# ============================================================

assert (
    X_continuous.shape[1]
    == expected_feature_size_for_meta_explainer
)

assert (
    X_continuous.shape[0]
    == Y_continuous.shape[0]
)

weight_sums = (
    X_continuous[
        :,
        -6:,
    ]
    .sum(
        dim=1
    )
)

assert torch.allclose(
    weight_sums,
    torch.ones_like(
        weight_sums
    ),
    atol=1e-6,
)

print(
    "Continuous dataset sanity checks passed."
)


# ============================================================
# Class Distribution
# ============================================================

counts = Counter(
    Y_continuous.tolist()
)

total = len(
    Y_continuous
)

print(
    "\nContinuous-test class distribution:"
)

for class_index in range(
    len(Explainers)
):
    count = counts.get(
        class_index,
        0,
    )

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

def evaluate_random_forest(
    model,
    X_test,
    Y_test,
):
    """
    Evaluate one Grid-trained Random Forest on the continuous
    preference-weight dataset.

    IMPORTANT:
    No scaler is loaded or applied. Random Forest was trained
    on the original 48-dimensional features and does not
    require standardization.
    """

    X_test_np = (
        X_test
        .detach()
        .cpu()
        .numpy()
    )

    Y_test_np = (
        Y_test
        .detach()
        .cpu()
        .numpy()
    )

    predictions = model.predict(
        X_test_np
    )

    probabilities_raw = (
        model.predict_proba(
            X_test_np
        )
    )

    # --------------------------------------------------------
    # Expand probabilities to all explainer classes.
    #
    # sklearn only returns probability columns for classes
    # observed during training.
    # --------------------------------------------------------

    probabilities = np.zeros(
        (
            len(X_test_np),
            len(Explainers),
        ),
        dtype=np.float64,
    )

    for (
        probability_column,
        class_label,
    ) in enumerate(
        model.classes_
    ):
        probabilities[
            :,
            int(class_label),
        ] = probabilities_raw[
            :,
            probability_column,
        ]

    test_loss = log_loss(
        Y_test_np,
        probabilities,
        labels=np.arange(
            len(Explainers)
        ),
    )

    accuracy = accuracy_score(
        Y_test_np,
        predictions,
    )

    precision_macro = precision_score(
        Y_test_np,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        Y_test_np,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        Y_test_np,
        predictions,
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        Y_test_np,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        Y_test_np,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        Y_test_np,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


# ============================================================
# Evaluate All 12 Grid-Trained Random Forest Models
#
# This exactly mirrors the Meta-Explainer continuous-weight
# experiment conceptually:
#
#   Grid-trained model
#       -> unseen continuous preference vectors
#
# No Random Forest is trained on the continuous dataset.
# ============================================================

continuous_results = {}


for (
    weight_level_name,
    current_weight_levels,
) in weight_levels.items():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Grid-trained Random Forest models:",
        weight_level_name,
    )

    print(
        "=" * 100
    )

    continuous_results[
        weight_level_name
    ] = {}

    weight_file_name = "_".join(
        str(weight)
        for weight in current_weight_levels
    )

    file_names = (
        "GridSearch_Weights_"
        + weight_file_name
    )

    for train_size in (
        training_scenarios
    ):

        train_size_name = str(
            int(
                train_size
                * 100
            )
        )

        print(
            "\nEvaluating "
            f"{weight_level_name} "
            f"with "
            f"{train_size_name}% training..."
        )

        # ====================================================
        # Matching Grid-Trained Random Forest Model
        # ====================================================

        model_file = (
            models_directory
            + "Random_Forest_Model_"
            + file_names
            + "_Train_"
            + train_size_name
            + "Percent.pkl"
        )

        # ====================================================
        # Existence Check
        # ====================================================

        assert os.path.exists(
            model_file
        ), (
            f"Model file not found: "
            f"{model_file}"
        )

        # ====================================================
        # Load Matching Random Forest
        # ====================================================

        with open(
            model_file,
            "rb",
        ) as file:

            model = pickle.load(
                file
            )

        # ====================================================
        # Evaluate on Same RAW Continuous Dataset
        #
        # Random Forest requires no feature scaling.
        # ====================================================

        metrics = evaluate_random_forest(
            model=model,
            X_test=X_continuous,
            Y_test=Y_continuous,
        )

        # ====================================================
        # Store Results
        # ====================================================

        continuous_results[
            weight_level_name
        ][
            train_size_name
        ] = metrics

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
    + "Random_Forest_GridSearch_Models_"
    + "Continuous_Weights_Evaluation_Results.pkl"
)

with open(
    results_file,
    "wb",
) as file:

    pickle.dump(
        continuous_results,
        file,
    )

print(
    "\n"
    + "=" * 100
)

print(
    "Random Forest continuous-weight evaluation completed."
)

print(
    "Results saved to:"
)

print(
    results_file
)
