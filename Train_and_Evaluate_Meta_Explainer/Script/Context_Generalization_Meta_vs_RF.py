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

diagnostics_directory = base_results_directory + "Generalization_Diagnostics/" + "Meta_Explainer/"
paper_tables_directory = base_results_directory + "Paper_Tables/"
os.makedirs(paper_tables_directory, exist_ok=True)


# ============================================================
# Helper: Normalize Weight-Level Strings
# ============================================================

def normalize_weight_levels(series):
    """
    Normalize strings such as:
        [1, 2]
        [1,2]
    into:
        [1,2]
    """
    return (
        series
        .astype(str)
        .str.replace(" ", "", regex=False)
    )


# ============================================================
# Protocol Files
# ============================================================

protocol_files = {
    "LODO": {
        "meta": "LODO_Meta.csv",
        "rf": "LODO_RF.csv",
    },

    "LOGO": {
        "meta": "LOGO_Meta.csv",
        "rf": "LOGO_RF.csv",
    },

    "LODO+LOGO": {
        "meta": "LODO_LOGO_Meta.csv",
        "rf": "LODO_LOGO_RF.csv",
    },
}


# ============================================================
# Load Meta-Explainer and RF Results
# ============================================================

all_protocols = []
for protocol, files in protocol_files.items():
    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------
    meta_df = pd.read_csv(meta_directory + files["meta"])
    rf_df = pd.read_csv(rf_directory + files["rf"])

    # --------------------------------------------------------
    # Normalize weight-level strings
    # --------------------------------------------------------
    meta_df["weight_levels"] = normalize_weight_levels(meta_df["weight_levels"])
    rf_df["weight_levels"] = normalize_weight_levels(rf_df["weight_levels"])

    # --------------------------------------------------------
    # Meta-Explainer: keep paper metrics
    # --------------------------------------------------------

    meta_df = meta_df[
        [
            "weight_levels",
            "accuracy_mean",
            "accuracy_std",
            "f1_macro_mean",
            "f1_macro_std",
        ]
    ].rename(
        columns={
            "accuracy_mean":
                "Meta_Accuracy_Mean",

            "accuracy_std":
                "Meta_Accuracy_Std",

            "f1_macro_mean":
                "Meta_Macro_F1_Mean",

            "f1_macro_std":
                "Meta_Macro_F1_Std",
        }
    )

    # --------------------------------------------------------
    # Random Forest: keep paper metrics
    # --------------------------------------------------------

    rf_df = rf_df[
        [
            "weight_levels",
            "accuracy_mean",
            "accuracy_std",
            "f1_macro_mean",
            "f1_macro_std",
        ]
    ].rename(
        columns={
            "accuracy_mean":
                "RF_Accuracy_Mean",

            "accuracy_std":
                "RF_Accuracy_Std",

            "f1_macro_mean":
                "RF_Macro_F1_Mean",

            "f1_macro_std":
                "RF_Macro_F1_Std",
        }
    )

    # --------------------------------------------------------
    # Merge Meta and RF
    # --------------------------------------------------------

    merged = meta_df.merge(
        rf_df,
        on="weight_levels",
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Add protocol
    # --------------------------------------------------------
    merged.insert(0, "Protocol", protocol,)
    all_protocols.append(merged)


# ============================================================
# Combine LODO, LOGO, and LODO+LOGO
# ============================================================
context_table = pd.concat(all_protocols, ignore_index=True,)


# ============================================================
# Sanity Check Before Adding Majority Baseline
# ============================================================

expected_protocols = {
    "LODO",
    "LOGO",
    "LODO+LOGO",
}

expected_weights = {
    "[1,2]",
    "[1,2,3]",
    "[1,2,3,4]",
}


assert set(context_table["Protocol"]) == expected_protocols, (
    "Unexpected protocols found in Meta/RF results: "
    f"{context_table['Protocol'].unique()}"
)

assert set(context_table["weight_levels"]) == expected_weights, (
    "Unexpected weight levels found in Meta/RF results: "
    f"{context_table['weight_levels'].unique()}"
)

assert len(context_table) == 9, (
    "Expected 9 Meta/RF context rows before majority merge, "
    f"found {len(context_table)}."
)


# ============================================================
# Load Training-Majority Baselines
# ============================================================

generalization_summary_file = diagnostics_directory + "Meta_Generalization_Summary.csv"
majority_df = pd.read_csv(generalization_summary_file)


print("\n" + "=" * 100)
print("DIAGNOSTIC SUMMARY")
print("=" * 100)

print("\nColumns:")
print(
    majority_df.columns.tolist()
)

print("\nExperiments:")
print(
    majority_df["experiment"].unique()
)

print("\nWeight levels:")
print(
    majority_df["weight_levels"].unique()
)


# ============================================================
# Normalize Diagnostic Experiment Names
# ============================================================

def normalize_experiment_name(name):

    name = str(name).strip()

    mapping = {
        # LODO
        "LODO": "LODO",

        # LOGO
        "LOGO": "LOGO",

        # Combined variants
        "LODO_LOGO": "LODO+LOGO",
        "LODO+LOGO": "LODO+LOGO",
        "LODO + LOGO": "LODO+LOGO",
        "LODO-LOGO": "LODO+LOGO",
        "Combined": "LODO+LOGO",
        "COMBINED": "LODO+LOGO",
    }

    return mapping.get(
        name,
        None,
    )


majority_df["Protocol"] = (
    majority_df["experiment"]
    .apply(normalize_experiment_name)
)


# ============================================================
# Detect Unknown Experiment Names
# ============================================================

unknown_mask = (
    majority_df["Protocol"]
    .isnull()
)

if unknown_mask.any():

    unknown_experiments = (
        majority_df.loc[
            unknown_mask,
            "experiment"
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Unknown experiment names in "
        "Meta_Generalization_Summary.csv: "
        f"{unknown_experiments}"
    )


# ============================================================
# Normalize Diagnostic Weight Levels
# ============================================================
majority_df["weight_levels"] = normalize_weight_levels(majority_df["weight_levels"])


# ============================================================
# Keep Majority Metrics Needed for Paper
# ============================================================

majority_df = majority_df[
    [
        "Protocol",
        "weight_levels",
        "majority_accuracy_mean",
        "majority_macro_f1_mean",
    ]
].rename(
    columns={
        "majority_accuracy_mean":
            "Majority_Accuracy",

        "majority_macro_f1_mean":
            "Majority_Macro_F1",
    }
)


# ============================================================
# Check for Duplicate Majority Rows
# ============================================================

duplicate_majority = (
    majority_df
    .duplicated(
        subset=[
            "Protocol",
            "weight_levels",
        ],
        keep=False,
    )
)

if duplicate_majority.any():

    print(
        "\nDuplicate majority rows:"
    )

    print(
        majority_df.loc[
            duplicate_majority
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "Duplicate Protocol/weight_levels combinations "
        "found in majority summary."
    )


# ============================================================
# Verify Majority Coverage Before Merge
# ============================================================

print("\n" + "=" * 100)
print("MAJORITY BASELINE KEYS")
print("=" * 100)

print(
    majority_df[
        [
            "Protocol",
            "weight_levels",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Merge Majority Baselines
# ============================================================

context_table = context_table.merge(
    majority_df,
    on=[
        "Protocol",
        "weight_levels",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# Check Majority Merge
# ============================================================

missing_majority = context_table[
    context_table[
        [
            "Majority_Accuracy",
            "Majority_Macro_F1",
        ]
    ]
    .isnull()
    .any(axis=1)
]


if not missing_majority.empty:

    print("\n" + "=" * 100)
    print("UNMATCHED MAJORITY BASELINES")
    print("=" * 100)

    print(
        missing_majority[
            [
                "Protocol",
                "weight_levels",
            ]
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "Some context-generalization rows could not "
        "be matched to a majority baseline."
    )


# ============================================================
# Sort
# ============================================================

protocol_order = {
    "LODO": 0,
    "LOGO": 1,
    "LODO+LOGO": 2,
}

weight_order = {
    "[1,2]": 0,
    "[1,2,3]": 1,
    "[1,2,3,4]": 2,
}


context_table["_protocol_order"] = (
    context_table["Protocol"]
    .map(protocol_order)
)

context_table["_weight_order"] = (
    context_table["weight_levels"]
    .map(weight_order)
)


# Check sort mappings
if context_table["_protocol_order"].isnull().any():

    raise ValueError(
        "Some protocol values could not be sorted."
    )

if context_table["_weight_order"].isnull().any():

    raise ValueError(
        "Some weight-level values could not be sorted."
    )


context_table = (
    context_table
    .sort_values(
        [
            "_protocol_order",
            "_weight_order",
        ]
    )
    .drop(
        columns=[
            "_protocol_order",
            "_weight_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# Final Column Order
# ============================================================

context_table = context_table[
    [
        "Protocol",
        "weight_levels",

        "Majority_Accuracy",
        "Majority_Macro_F1",

        "Meta_Accuracy_Mean",
        "Meta_Accuracy_Std",
        "Meta_Macro_F1_Mean",
        "Meta_Macro_F1_Std",

        "RF_Accuracy_Mean",
        "RF_Accuracy_Std",
        "RF_Macro_F1_Mean",
        "RF_Macro_F1_Std",
    ]
]


# ============================================================
# Final Sanity Checks
# ============================================================

assert len(context_table) == 9, (
    f"Expected 9 rows, "
    f"found {len(context_table)}."
)

assert not context_table.isnull().any().any(), (
    "Context-generalization paper table "
    "contains missing values."
)

assert set(context_table["Protocol"]) == expected_protocols, (
    "Unexpected protocols in final table."
)

assert set(context_table["weight_levels"]) == expected_weights, (
    "Unexpected weight levels in final table."
)


# ============================================================
# Check Expected Number of Rows Per Protocol
# ============================================================

rows_per_protocol = (
    context_table
    .groupby("Protocol")
    .size()
)

for protocol in expected_protocols:

    assert rows_per_protocol[protocol] == 3, (
        f"{protocol} should contain 3 rows, "
        f"but contains {rows_per_protocol[protocol]}."
    )


# ============================================================
# Save Paper CSV
# ============================================================
output_file = paper_tables_directory + "Context_Generalization_Meta_vs_RF.csv"
context_table.to_csv(output_file, index=False,)


# ============================================================
# Print Final Paper Table
# ============================================================

print("\n" + "=" * 160)
print(
    "CONTEXT GENERALIZATION: "
    "META-EXPLAINER VS RANDOM FOREST VS MAJORITY"
)
print("=" * 160)

print(
    context_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\nSaved:")
print(output_file)