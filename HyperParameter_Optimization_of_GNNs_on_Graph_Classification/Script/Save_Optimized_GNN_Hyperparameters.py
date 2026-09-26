import os
import pickle

all_parameters = {1: {"MUTAG": {"lr": 0.00419721, "dropout": 0.26150688, "weight_initializer": 1, "bias": True},
                      "NCI1": {"lr": 0.00447584, "dropout": 0.367435201, "weight_initializer": 1, "bias": True},
                      "ENZYMES": {"lr": 0.00393719, "dropout": 0.314394521, "weight_initializer": 3, "bias": True},
                      "Graph-SST5": {"lr": 0.00398845, "dropout": 0.376534632, "weight_initializer": 2, "bias": True},
                      "PROTEINS": {"lr": 0.00411772, "dropout": 0.329187893, "weight_initializer": 1, "bias": True},
                      "IsCyclic": {"lr": 0.008468306, "dropout": 0.342759509, "weight_initializer": 3, "bias": False}},

                  2: {"MUTAG": {"lr": 0.00130896, "dropout": 0.66060395, "weight_initializer": 3, "bias": True},
                      "NCI1": {"lr": 0.00154723, "dropout": 0.52860329, "weight_initializer": 3, "bias": True},
                      "ENZYMES": {"lr": 0.00146179, "dropout": 0.55382278, "weight_initializer": 3, "bias": True},
                      "Graph-SST5": {"lr": 0.00113456, "dropout": 0.52848269, "weight_initializer": 2, "bias": True},
                      "PROTEINS": {"lr": 0.00124589, "dropout": 0.52350291, "weight_initializer": 1, "bias": True},
                      "IsCyclic": {"lr": 0.00434473, "dropout": 0.67939804, "weight_initializer": 3, "bias": True}},

                  3: {"MUTAG": {"lr": 0.00060551, "dropout": 0.65602164, "weight_initializer": 2, "bias": True},
                      "NCI1": {"lr": 0.00071152, "dropout": 0.58243376, "weight_initializer": 3, "bias": True},
                      "ENZYMES": {"lr": 0.00058299, "dropout": 0.55350198, "weight_initializer": 3, "bias": True},
                      "Graph-SST5": {"lr": 0.00052245, "dropout": 0.52451704, "weight_initializer": 2, "bias": True},
                      "PROTEINS": {"lr": 0.00063418, "dropout": 0.5303986, "weight_initializer": 1, "bias": True},
                      "IsCyclic": {"lr": 0.00804156, "dropout": 0.4877806, "weight_initializer": 3, "bias": True}},

                  4: {"MUTAG": {"lr": 0.00804157, "dropout": 0.48778068, "weight_initializer": 3, "bias": True},
                      "NCI1": {"lr": 0.00758931, "dropout": 0.3658565, "weight_initializer": 3, "bias": True},
                      "ENZYMES": {"lr": 0.0068349, "dropout": 0.34389162, "weight_initializer": 3, "bias": True},
                      "Graph-SST5": {"lr": 0.00920516, "dropout": 0.39022403, "weight_initializer": 3, "bias": True},
                      "PROTEINS": {"lr": 0.00864241, "dropout": 0.32800018, "weight_initializer": 1, "bias": True},
                      "IsCyclic": {"lr": 0.00901886, "dropout": 0.46620775, "weight_initializer": 3, "bias": True}},
                  }

save_dir = "/Meta_Explainer_on_Benchmarking/HyperParameter_Optimization_of_GNNs_on_Graph_Classification/Experimental Results"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "optimized_gnn_hyperparameters.pkl")

with open(save_path, "wb") as f:
    pickle.dump(all_parameters, f)

print(f"Hyperparameters saved to: {save_path}")