import os
import pickle
from collections import Counter

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
    "[1, 2, 3, 4]": [1, 2, 3, 4],
}


# ============================================================
# Random Forest Configuration
# ============================================================

number_of_trees = 100

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

lodo_logo_dataset_directory = (
    base_dataset_results_directory
    + "LODO_LOGO_GridSearch_Weights/"
)

base_training_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Train_and_Evaluate_Meta_Explainer/"
    "Experimental Results/"
)

lodo_logo_models_directory = (
    base_training_results_directory
    + "Random_Forest_Models_Trained_LODO_LOGO/"
)

lodo_logo_results_directory = (
    base_training_results_directory
    + "TestResults_of_Random_Forest_LODO_LOGO/"
)

os.makedirs(
    lodo_logo_models_directory,
    exist_ok=True,
)

os.makedirs(
    lodo_logo_results_directory,
    exist_ok=True,
)


# ============================================================
# Train + Evaluate Random Forest
# ============================================================

def train_and_evaluate_random_forest(
    X_train,
    Y_train,
    X_test,
    Y_test,
):
    """
    Train Random Forest on the exact combined LODO+LOGO
    train/test files used by the Meta-Explainer experiment.

    The training context contains:
        5 datasets x 3 GNNs

    The test context is the held-out:
        dataset x GNN intersection

    Random Forest is tree-based, so no feature scaling is used.
    """

    X_train_np = (
        X_train
        .detach()
        .cpu()
        .numpy()
    )

    Y_train_np = (
        Y_train
        .detach()
        .cpu()
        .numpy()
    )

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

    random_forest = RandomForestClassifier(
        n_estimators=number_of_trees,
        random_state=random_state,
        n_jobs=-1,
    )

    random_forest.fit(
        X_train_np,
        Y_train_np,
    )

    predictions = random_forest.predict(
        X_test_np
    )

    probabilities_raw = (
        random_forest.predict_proba(
            X_test_np
        )
    )

    # --------------------------------------------------------
    # sklearn returns probabilities only for classes observed
    # in training. Expand to the complete explainer class set.
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
        random_forest.classes_
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

    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

    return (
        random_forest,
        metrics,
    )


# ============================================================
# Combined LODO + LOGO Experiment
# ============================================================

LODO_LOGO_results = {}

total_models_trained = 0


for (
    weight_level_name,
    current_weight_levels,
) in weight_levels.items():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Random Forest combined LODO + LOGO "
        "for weight levels:",
        weight_level_name,
    )

    print(
        "=" * 100
    )

    LODO_LOGO_results[
        weight_level_name
    ] = {}

    weight_file_name = "_".join(
        str(weight)
        for weight in current_weight_levels
    )

    # ========================================================
    # Hold Out Every Dataset
    # ========================================================

    for held_out_dataset in (
        Datasets_Name
    ):

        LODO_LOGO_results[
            weight_level_name
        ][
            held_out_dataset
        ] = {}

        # ====================================================
        # Hold Out Every GNN
        # ====================================================

        for held_out_gnn in (
            GNN_Models
        ):

            print(
                "\n"
                + "-" * 100
            )

            print(
                "Weight levels:",
                weight_level_name,
            )

            print(
                "Held-out dataset:",
                held_out_dataset,
            )

            print(
                "Held-out GNN:",
                held_out_gnn,
            )

            print(
                "-" * 100
            )

            # =================================================
            # File Prefix
            # =================================================

            file_prefix = (
                "LODO_LOGO_"
                + "GridSearch_Weights_"
                + weight_file_name
                + "_HeldOutDataset_"
                + held_out_dataset
                + "_HeldOutGNN_"
                + held_out_gnn
            )

            # =================================================
            # Dataset Files
            # =================================================

            x_train_file = (
                lodo_logo_dataset_directory
                + "X_Train_"
                + file_prefix
                + ".pt"
            )

            y_train_file = (
                lodo_logo_dataset_directory
                + "Y_Train_"
                + file_prefix
                + ".pt"
            )

            x_test_file = (
                lodo_logo_dataset_directory
                + "X_Test_"
                + file_prefix
                + ".pt"
            )

            y_test_file = (
                lodo_logo_dataset_directory
                + "Y_Test_"
                + file_prefix
                + ".pt"
            )

            # =================================================
            # File Existence Checks
            # =================================================

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

            # =================================================
            # Load Exact Same Split
            # =================================================

            X_train_list = torch.load(
                x_train_file
            )

            Y_train_list = torch.load(
                y_train_file
            )

            X_test_list = torch.load(
                x_test_file
            )

            Y_test_list = torch.load(
                y_test_file
            )

            X_train = torch.stack(
                X_train_list
            ).float()

            Y_train = torch.stack(
                Y_train_list
            ).long()

            X_test = torch.stack(
                X_test_list
            ).float()

            Y_test = torch.stack(
                Y_test_list
            ).long()

            # -------------------------------------------------
            # Support class-index or one-hot labels
            # -------------------------------------------------

            if (
                Y_train.dim() > 1
                and Y_train.size(1) > 1
            ):
                Y_train = torch.argmax(
                    Y_train,
                    dim=1,
                ).long()

            if (
                Y_test.dim() > 1
                and Y_test.size(1) > 1
            ):
                Y_test = torch.argmax(
                    Y_test,
                    dim=1,
                ).long()

            # =================================================
            # Sanity Checks
            # =================================================

            assert (
                X_train.shape[1]
                == expected_feature_size_for_meta_explainer
            )

            assert (
                X_test.shape[1]
                == expected_feature_size_for_meta_explainer
            )

            assert (
                len(X_train)
                == len(Y_train)
            )

            assert (
                len(X_test)
                == len(Y_test)
            )

            # -------------------------------------------------
            # Preference weights sum to one
            # -------------------------------------------------

            train_weight_sums = (
                X_train[
                    :,
                    -6:,
                ]
                .sum(
                    dim=1
                )
            )

            test_weight_sums = (
                X_test[
                    :,
                    -6:,
                ]
                .sum(
                    dim=1
                )
            )

            assert torch.allclose(
                train_weight_sums,
                torch.ones_like(
                    train_weight_sums
                ),
                atol=1e-6,
            )

            assert torch.allclose(
                test_weight_sums,
                torch.ones_like(
                    test_weight_sums
                ),
                atol=1e-6,
            )

            # -------------------------------------------------
            # No NaN / Inf
            # -------------------------------------------------

            assert torch.isfinite(
                X_train
            ).all()

            assert torch.isfinite(
                X_test
            ).all()

            # =================================================
            # Class Distribution
            # =================================================

            if PRINT_CLASS_DISTRIBUTION:

                train_counts = Counter(
                    Y_train.tolist()
                )

                test_counts = Counter(
                    Y_test.tolist()
                )

                print(
                    "\nTraining class distribution:"
                )

                for class_index in range(
                    len(Explainers)
                ):
                    count = train_counts.get(
                        class_index,
                        0,
                    )

                    print(
                        f"  "
                        f"{class_index} "
                        f"({Explainers[class_index]}): "
                        f"{count} "
                        f"({count / len(Y_train):.2%})"
                    )

                print(
                    "\nHeld-out test class distribution:"
                )

                for class_index in range(
                    len(Explainers)
                ):
                    count = test_counts.get(
                        class_index,
                        0,
                    )

                    print(
                        f"  "
                        f"{class_index} "
                        f"({Explainers[class_index]}): "
                        f"{count} "
                        f"({count / len(Y_test):.2%})"
                    )

            # =================================================
            # Train + Evaluate
            #
            # IMPORTANT:
            # No StandardScaler is used for Random Forest.
            # =================================================

            (
                random_forest,
                metrics,
            ) = train_and_evaluate_random_forest(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
            )

            # =================================================
            # Store Results
            #
            # Structure intentionally matches Meta-Explainer
            # LODO+LOGO results.
            # =================================================

            LODO_LOGO_results[
                weight_level_name
            ][
                held_out_dataset
            ][
                held_out_gnn
            ] = {
                "held_out_dataset":
                    held_out_dataset,

                "held_out_gnn":
                    held_out_gnn,

                "weight_levels":
                    current_weight_levels.copy(),

                "num_train_samples":
                    len(X_train),

                "num_test_samples":
                    len(X_test),

                **metrics,
            }

            # =================================================
            # Print Metrics
            # =================================================

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

            # =================================================
            # Save Random Forest Model
            # =================================================

            model_file = (
                lodo_logo_models_directory
                + "Random_Forest_"
                + file_prefix
                + ".pkl"
            )

            with open(
                model_file,
                "wb",
            ) as file:
                pickle.dump(
                    random_forest,
                    file,
                )

            print(
                "Saved model:",
                model_file,
            )

            total_models_trained += 1


# ============================================================
# Final Model-Count Check
# ============================================================

expected_total_models = (
    len(weight_levels)
    * len(Datasets_Name)
    * len(GNN_Models)
)

assert (
    total_models_trained
    == expected_total_models
), (
    f"Expected {expected_total_models} models, "
    f"but trained {total_models_trained}."
)


# ============================================================
# Save All Results
# ============================================================

results_file = (
    lodo_logo_results_directory
    + "Random_Forest_LODO_LOGO_"
    + "GridSearch_Weights_"
    + "Evaluation_Results.pkl"
)

with open(
    results_file,
    "wb",
) as file:
    pickle.dump(
        LODO_LOGO_results,
        file,
    )


# ============================================================
# Final Summary
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "Random Forest combined LODO + LOGO "
    "training and evaluation completed."
)

print(
    "Total models trained:",
    total_models_trained,
)

print(
    "Expected models:",
    expected_total_models,
)

print(
    "Results saved to:"
)

print(
    results_file
)
