import os
import torch


save_root = "/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking"
results_dir = os.path.join(save_root, "GraphMask_on_Graph_Classification", "Experimental_Results")

RUN_DATASETS = [
    "MUTAG",
    "NCI1",
    "ENZYMES",
    "Graph-SST5",
    "PROTEINS",
    "IsCyclic",
]

RUN_GNNS = [
    "GCN_plus_GAP_Model",
    "DGCNN_Model",
    "DIFFPOOL_Model",
    "GIN_Model",
]


explanations_size = {}

for dataset_name in RUN_DATASETS:
    explanations_size[dataset_name] = {}

    for gnn_name in RUN_GNNS:
        save_path = os.path.join(results_dir, f"{dataset_name}_{gnn_name}_GraphMask.pt")

        if not os.path.exists(save_path):
            explanations_size[dataset_name][gnn_name] = {}
            continue

        saved_data = torch.load(save_path, map_location="cpu")
        average_sizes = saved_data["average_explanation_size_per_class"]
        explanations_size[dataset_name][gnn_name] = {int(class_index): int(round(average_size)) for
                                                     class_index, average_size in average_sizes.items()}

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