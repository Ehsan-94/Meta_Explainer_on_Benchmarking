import torch
saved_data = torch.load(
    "/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/"
    "GraphMask_on_Graph_Classification/Experimental_Results/"
    "Graph-SST5_GCN_plus_GAP_Model_GraphMask.pt",
    map_location="cpu"
)

for class_index, masks in saved_data["binary_edge_explanations"].items():
    retained = sum(mask.sum().item() for mask in masks)
    total = sum(mask.numel() for mask in masks)

    print(
        f"Class {class_index} | "
        f"Retained edges: {retained}/{total} | "
        f"Retention: {retained / total:.4f} | "
        f"Sparsity: {1 - retained / total:.4f}"
    )
for class_index, masks in saved_data["edge_explanations"].items():

    values = torch.cat([
        mask.detach().cpu().float().reshape(-1)
        for mask in masks
    ])

    print(
        f"Class {class_index} | "
        f"Min: {values.min().item():.6f} | "
        f"Max: {values.max().item():.6f} | "
        f"Mean: {values.mean().item():.6f}"
    )