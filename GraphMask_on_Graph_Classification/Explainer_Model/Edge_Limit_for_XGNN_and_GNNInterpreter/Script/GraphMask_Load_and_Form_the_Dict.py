import os
import torch


save_root = "/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking"
results_dir = os.path.join(save_root, "GraphMask_on_Graph_Classification", "Experimental_Results")

RUN_DATASETS = ["MUTAG", "NCI1", "ENZYMES", "Graph-SST5", "PROTEINS", "IsCyclic",]

RUN_GNNS = ["GCN_plus_GAP_Model", "DGCNN_Model", "DIFFPOOL_Model", "GIN_Model",]


explanations_size = {}

for dataset_name in RUN_DATASETS:
    for gnn_name in RUN_GNNS:

        save_path = os.path.join(results_dir, f"{dataset_name}_{gnn_name}_GraphMask.pt")

        if not os.path.exists(save_path):
            continue

        saved_data = torch.load(save_path, map_location="cpu")

        print("\n" + "=" * 100)
        print(dataset_name, "|", gnn_name)

        binary_explanations = saved_data["binary_edge_explanations"]
        saved_average_sizes = saved_data["average_explanation_size_per_class"]

        for class_index in range(saved_data["num_classes"]):

            calculated_sizes = []
            invalid_masks = 0
            length_mismatches = 0

            for graph_index, explanation in enumerate(binary_explanations[class_index]):
                explanation = torch.as_tensor(explanation)

                expected_num_edges = saved_data["num_edges"][graph_index]

                if explanation.numel() != expected_num_edges:
                    length_mismatches += 1

                is_binary = torch.all((explanation == 0) | (explanation == 1)).item()

                if not is_binary:
                    invalid_masks += 1

                    print(
                        "[INVALID MASK]",
                        "class =", class_index,
                        "graph =", graph_index,
                        "dtype =", explanation.dtype,
                        "shape =", tuple(explanation.shape),
                        "min =", explanation.min().item(),
                        "max =", explanation.max().item(),
                        "sum =", explanation.sum().item(),
                        "unique =", torch.unique(explanation)[:20].tolist(),
                    )

                calculated_sizes.append(
                    int(explanation.sum().item())
                )

            calculated_average = (
                sum(calculated_sizes) / len(calculated_sizes)
                if len(calculated_sizes) > 0
                else 0.0
            )

            print(
                f"class {class_index}:",
                f"saved_average={saved_average_sizes[class_index]},",
                f"recalculated_average={calculated_average},",
                f"min_size={min(calculated_sizes) if calculated_sizes else None},",
                f"max_size={max(calculated_sizes) if calculated_sizes else None},",
                f"invalid_masks={invalid_masks},",
                f"length_mismatches={length_mismatches}"
            )

output_dir = os.path.join(
    save_root,
    "GraphMask_on_Graph_Classification",
    "Explainer_Model",
    "Edge_Limit_for_XGNN_and_GNNInterpreter",
    "Experimental_Results"
)
print("explanations_size: ", explanations_size)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "GraphMask_Explanation_Sizes.pt")

torch.save(explanations_size, output_path)