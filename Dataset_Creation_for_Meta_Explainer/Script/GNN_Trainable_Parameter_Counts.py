import os
import sys
import pickle
import torch

from torch_geometric.datasets import TUDataset


# ============================================================
# Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_path = "/data/cs.aau.dk/ey33jw/Explainability_Methods/"

dataset_names = [
    "MUTAG",
    "ENZYMES",
    "NCI1",
    "Graph-SST5",
    "PROTEINS",
    "IsCyclic"
]

gnn_names = [
    "GCN",
    "DGCNN",
    "DIFFPOOL",
    "GIN"
]


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(dataset_name):

    if dataset_name == "Graph-SST5":
        from dig.xgraph.dataset import SentiGraphDataset
        py_path = "/data/cs.aau.dk/ey33jw/"
        os.chdir(py_path)
        dataset = SentiGraphDataset(root="./Datasets_for_Explainability_Methods/", name="Graph-SST5")
        os.chdir(default_path)
    elif dataset_name == "IsCyclic":
        with open(
                "/data/cs.aau.dk/ey33jw/"
                "Datasets_for_Explainability_Methods/"
                "IsCyclic/iscyclic_graphs.pkl",
                "rb"
        ) as f:
            dataset = pickle.load(f)
    else:
        dataset = TUDataset(root="data/TUDataset", name=dataset_name)
    return dataset


# ============================================================
# Number of classes
# ============================================================
def find_num_classes_graph_labels(dataset):
    unique_labels = set()
    for graph in dataset:
        unique_labels.add(graph.y.tolist()[0])
    return len(unique_labels)


# ============================================================
# Count trainable parameters for GNN models
# ============================================================
def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Create exactly the GNN architectures used in the experiments
# ============================================================
def create_gnn(gnn_name, dataset, dataset_name, Bias=True, Weight_Initializer=1, act_fun="ReLu", dropout_rate=0.5):
    num_classes = find_num_classes_graph_labels(dataset)
    node_feat_size = dataset[0].x.size()[-1]
    # --------------------------------------------------------
    # GCN + GAP
    # --------------------------------------------------------
    if gnn_name == "GCN":
        sys.path.insert(0, default_path + "/Models/Script")
        import GCN_plus_GAP as gcn_plus_gap_model
        gcn_input_dim = {
            "MUTAG": node_feat_size,
            "NCI1": node_feat_size,
            "ENZYMES": 16,
            "Graph-SST5": 64,
            "PROTEINS": 16,
            "IsCyclic": node_feat_size
        }
        model = gcn_plus_gap_model.GCN_plus_GAP_Model(model_level="graph", num_classes=num_classes,
                                                      GNN_layers=[node_feat_size, gcn_input_dim[dataset_name]],
                                                      Bias=Bias, Weight_Initializer=Weight_Initializer, act_fun=act_fun,
                                                      dropout_rate=dropout_rate)
    # --------------------------------------------------------
    # DGCNN
    # --------------------------------------------------------
    elif gnn_name == "DGCNN":
        sys.path.insert(0, default_path + "/Models/Script")
        import DGCNN as dgcnn_model
        k_dgcnn = {
            "MUTAG": 17,
            "NCI1": 32,
            "ENZYMES": 32,
            "Graph-SST5": 19,
            "PROTEINS": 26,
            "IsCyclic": 20
        }
        dgcnn_input_dim = {
            "MUTAG": [32, 32, 32, 32],
            "NCI1": [32, 32, 32, 32],
            "ENZYMES": [32, 32, 32, 32],
            "Graph-SST5": [64, 64, 64, 64],
            "PROTEINS": [32, 32, 32, 32],
            "IsCyclic": [32, 32, 32, 32]
        }
        model = dgcnn_model.DGCNN_Model(GNN_layers=dgcnn_input_dim[dataset_name], Bias=Bias, num_classes=num_classes,
                                        mlp_act_fun="ReLu", dgcnn_act_fun="tanh", Weight_Initializer=Weight_Initializer,
                                        mlp_dropout_rate=dropout_rate, dgcnn_k=k_dgcnn[dataset_name],
                                        node_feat_size=node_feat_size, hid_channels=[16, 32], conv1d_kernels=[2, 5],
                                        ffn_layer_size=128, strides=[2, 1])


    # --------------------------------------------------------
    # DIFFPOOL
    # --------------------------------------------------------
    elif gnn_name == "DIFFPOOL":
        sys.path.insert(0, default_path + "/Models/Script/Layers/")

        import Batched_GraphSage_Layer
        import Batched_DIFFPOOL_Assignment
        import Batched_DIFFPOOL_Embedding
        import Batched_DIFFPOOL_Layer
        sys.path.insert(0, default_path + "/Models/Script")
        import DIFFPOOL as diffpool_model
        model = diffpool_model.DIFFPOOL_Model(embedding_input_dim=node_feat_size, embedding_num_block_layers=1,
                                              embedding_hid_dim=64, new_feature_size=64,
                                              assignment_input_dim=node_feat_size, assignment_num_block_layers=1,
                                              assignment_hid_dim=64, max_number_of_nodes=256,
                                              prediction_hid_layers=[50], concat_neighborhood=False,
                                              num_classes=num_classes, Weight_Initializer=Weight_Initializer, Bias=Bias,
                                              act_fun=act_fun, dropout_rate=dropout_rate, normalize_graphsage=False,
                                              aggregation="mean", concat_diffpools_outputs=True, num_pooling=1,
                                              pooling="mean")
    # --------------------------------------------------------
    # GIN
    # --------------------------------------------------------
    elif gnn_name == "GIN":

        sys.path.insert(0, default_path + "/Models/Script/Layers/")
        import GIN_MLP_Layers
        sys.path.insert(0, default_path + "/Models/Script")

        import GIN as gin_model

        gin_input_dim = {
            "MUTAG": node_feat_size,
            "NCI1": node_feat_size,
            "ENZYMES": 16,
            "Graph-SST5": 64,
            "PROTEINS": 16,
            "IsCyclic": node_feat_size
        }

        model = gin_model.GIN_Model(num_mlp_layers=4, Bias=Bias, num_slp_layers=2, mlp_act_fun=act_fun,
                                    mlp_input_dim=node_feat_size, mlp_hid_dim=gin_input_dim[dataset_name],
                                    mlp_output_dim=num_classes, dropout_rate=dropout_rate, joint_embeddings=False,
                                    Weight_Initializer=Weight_Initializer)
    else:
        raise ValueError(f"Unknown GNN model: {gnn_name}")

    return model.to(device)


# ============================================================
# Calculate parameter counts
# ============================================================
GNN_Parameter_Counts = {}
for dataset_name in dataset_names:

    print("\n" + "=" * 70)
    print("Dataset:", dataset_name)
    print("=" * 70)

    dataset = load_dataset(dataset_name)
    GNN_Parameter_Counts[dataset_name] = {}
    print("Node feature size:", dataset[0].x.size()[-1])

    print("Number of classes:", find_num_classes_graph_labels(dataset))
    for gnn_name in gnn_names:
        model = create_gnn(gnn_name=gnn_name, dataset=dataset, dataset_name=dataset_name, Bias=True,
                           Weight_Initializer=1, act_fun="ReLu", dropout_rate=0.5)
        num_parameters = count_trainable_parameters(model)
        GNN_Parameter_Counts[dataset_name][gnn_name] = num_parameters

        print(
            f"{gnn_name:10s}: "
            f"{num_parameters:,} trainable parameters"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# Final dictionary
# ============================================================

print("\n\nGNN Parameter Counts")
print("=" * 70)

for dataset_name, models in GNN_Parameter_Counts.items():
    print(f'\n"{dataset_name}": {{')
    for gnn_name, num_parameters in models.items():
        print(f'    "{gnn_name}": {num_parameters},')
    print("},")


# ============================================================
# Save results
# ============================================================

output_file = (
    "/data/cs.aau.dk/ey33jw/"
    "Meta_Explainer_on_Benchmarking/"
    "Dataset_Creation_for_Meta_Explainer/"
    "Experimental Results/"
    "Constants/"
    "GNN_Parameter_Counts.pkl"
)

with open(output_file, "wb") as f:
    pickle.dump(GNN_Parameter_Counts, f)

print("\nSaved to:")
print(output_file)