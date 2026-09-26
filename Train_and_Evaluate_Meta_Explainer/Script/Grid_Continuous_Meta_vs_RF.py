import os
import pandas as pd


# ============================================================
# Directories
# ============================================================

base_results_directory = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Train_and_Evaluate_Meta_Explainer/"
    "Experimental Results/"
)

experimental_summary_directory = base_results_directory + "Experimental_Summary/"
meta_directory = experimental_summary_directory + "Meta_Explainer/"
rf_directory = experimental_summary_directory + "Random_Forest/"
paper_tables_directory = base_results_directory + "Paper_Tables/"
os.makedirs(paper_tables_directory, exist_ok=True)


# ============================================================
# Load Input CSV Files
# ============================================================

meta_grid = pd.read_csv(
    meta_directory
    + "GridSearch_Meta.csv"
)

meta_continuous = pd.read_csv(
    meta_directory
    + "Continuous_Meta.csv"
)

rf_grid = pd.read_csv(
    rf_directory
    + "GridSearch_RF.csv"
)

rf_continuous = pd.read_csv(
    rf_directory
    + "Continuous_RF.csv"
)


# ============================================================
# Columns Used in the Paper
# ============================================================

keys = [
    "weight_levels",
    "training_size"
]


meta_grid = meta_grid[
    keys + [
        "accuracy",
        "f1_macro"
    ]
    ].rename(
    columns={
        "accuracy": "Meta_Grid_Accuracy",
        "f1_macro": "Meta_Grid_Macro_F1"
    }
)


rf_grid = rf_grid[
    keys + [
        "accuracy",
        "f1_macro"
    ]
    ].rename(
    columns={
        "accuracy": "RF_Grid_Accuracy",
        "f1_macro": "RF_Grid_Macro_F1"
    }
)


meta_continuous = meta_continuous[
    keys + [
        "accuracy",
        "f1_macro"
    ]
    ].rename(
    columns={
        "accuracy": "Meta_Continuous_Accuracy",
        "f1_macro": "Meta_Continuous_Macro_F1"
    }
)


rf_continuous = rf_continuous[
    keys + [
        "accuracy",
        "f1_macro"
    ]
    ].rename(
    columns={
        "accuracy": "RF_Continuous_Accuracy",
        "f1_macro": "RF_Continuous_Macro_F1"
    }
)


# ============================================================
# Merge
# ============================================================

paper_table = (
    meta_grid
    .merge(
        rf_grid,
        on=keys,
        how="inner"
    )
    .merge(
        meta_continuous,
        on=keys,
        how="inner"
    )
    .merge(
        rf_continuous,
        on=keys,
        how="inner"
    )
)


# ============================================================
# Sort
# ============================================================

weight_order = {
    "[1, 2]": 0,
    "[1, 2, 3]": 1,
    "[1, 2, 3, 4]": 2
}

paper_table["_weight_order"] = (
    paper_table["weight_levels"]
    .map(weight_order)
)

paper_table = (
    paper_table
    .sort_values(
        [
            "_weight_order",
            "training_size"
        ]
    )
    .drop(
        columns=["_weight_order"]
    )
    .reset_index(drop=True)
)


# ============================================================
# Sanity Checks
# ============================================================
assert len(paper_table) == 12, (
    f"Expected 12 rows, "
    f"found {len(paper_table)}."
)

assert not paper_table.isnull().any().any(), (
    "Paper table contains missing values."
)


# ============================================================
# Save Paper Table
# ============================================================
output_file = paper_tables_directory + "Grid_Continuous_Meta_vs_RF.csv"
paper_table.to_csv(output_file, index=False)

# ============================================================
# Print
# ============================================================
print("\n" + "=" * 140)
print("GRID + CONTINUOUS: META-EXPLAINER VS RANDOM FOREST")
print("=" * 140)

print(
    paper_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:")
print(output_file)