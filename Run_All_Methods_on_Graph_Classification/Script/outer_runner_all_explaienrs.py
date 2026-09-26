import os
import sys
import pickle
import torch
from importlib import reload

from charset_normalizer import from_path

from Datasets_for_Explainability_Methods.Load_Dataset_Script import load_datasets as Load_Dataset_Module
from Meta_Explainer_on_Benchmarking.Train_Models_on_Graph_Classification.Script import create_GNNs as Create_and_Load_GNNS_Module
from Meta_Explainer_on_Benchmarking.Run_All_Methods_on_Graph_Classification.Utils import running_file_checker as Runner_File_Checker


py_path = '/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/'
os.chdir(py_path)
print("Current Working Directory:", os.getcwd())



running_file_checker = Runner_File_Checker.file_checker(save_root="/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking")
gnn_oprtimized_hyperparameters_dir = ("/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/"
                                      "HyperParameter_Optimization_of_GNNs_on_Graph_Classification/Experimental_Results")

params_path = os.path.join(gnn_oprtimized_hyperparameters_dir, "optimized_gnn_hyperparameters.pkl")

with open(params_path, "rb") as f:
    all_parameters = pickle.load(f)


RUN_DATASETS = ["MUTAG", "NCI1", "Graph-SST5", "ENZYMES", "PROTEINS", "IsCyclic", ]
# RUN_DATASETS = ["NCI1", "ENZYMES", "PROTEINS", ]

RUN_GNNS = ["GCN_plus_GAP_Model", "DGCNN_Model", "DIFFPOOL_Model", "GIN_Model", ]

# RUN_EXPLAINERS = ["GNNExplainer", "PGExplainer", "GraphMask", "SubGraphX", "CF2", "PGMExplainer", "XGNN",
#                   "GNNInterpreter", ]
# RUN_EXPLAINERS = ["GNNExplainer", "PGExplainer", "GraphMask", "CF2", "PGMExplainer", "XGNN", "GNNInterpreter", ]
RUN_EXPLAINERS = ["GNNInterpreter", ]
GNN_NAME_TO_INDEX_dict = {
    "GCN_plus_GAP_Model": 1,
    "DGCNN_Model": 2,
    "DIFFPOOL_Model": 3,
    "GIN_Model": 4,
}

EXPLAINER_MODES_dict = {
    "GNNExplainer": "generate",                # generate | load
    "PGExplainer": "train",                 # train | test | load
    "GraphMask": "train",                   # train | test | load
    "SubGraphX": "generate",                   # generate | load
    "CF2": "generate",                         # generate | load
    "PGMExplainer": "generate",                # generate | load
    "XGNN": "generate",                        # generate | load
    "GNNInterpreter": "generate",          # generate | load
}


EXPLAINER_DATASET_BlockList = {"XGNN": {"Graph-SST5"},}

SKIP_COMBINATIONS = set()


batch_size = 16
gnn_model_loading_epoch = 5000
classifier_weight_decay = 1e-6

explainer_epoch = 1
explainer_learning_rate = 0.0001

save_root = "/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking"


for DataSet_name in RUN_DATASETS:

    print("\n")
    print("DATASET:", DataSet_name)

    loading_dataset = Load_Dataset_Module.load_dataset(dataset_name=DataSet_name, BATCH_SIZE=batch_size)

    (train_dataset, test_dataset, train_dataloader, test_dataloader, num_classes, entire_dataset) = loading_dataset()


    for GNN_name in RUN_GNNS:

        gnn_model_index = GNN_NAME_TO_INDEX_dict[GNN_name]

        print("\n")
        print("GNN:", GNN_name)

        loading_gnn_model = Create_and_Load_GNNS_Module.load_GNN_Models(gnn_model_index=gnn_model_index,
                                                                        dataset_name=DataSet_name,
                                                                        num_classes=num_classes,
                                                                        entire_dataset=entire_dataset,
                                                                        dropout_rate=
                                                                        all_parameters[gnn_model_index][DataSet_name][
                                                                            "dropout"],
                                                                        bias=
                                                                        all_parameters[gnn_model_index][DataSet_name][
                                                                            "bias"],
                                                                        weight_initializer=
                                                                        all_parameters[gnn_model_index][DataSet_name][
                                                                            "weight_initializer"],
                                                                        classifier_lr=
                                                                        all_parameters[gnn_model_index][DataSet_name][
                                                                            "lr"],
                                                                        act_fun="ReLu",
                                                                        gnn_model_loading_epoch=gnn_model_loading_epoch,
                                                                        classifier_weight_decay=classifier_weight_decay)

        GNN_Model_Optimizer, GNN_Model_Loaded = loading_gnn_model()
        GNN_Model_Loaded.eval()

        for Explainer_Name in RUN_EXPLAINERS:
            restricted_datasets_for_the_explainer = EXPLAINER_DATASET_BlockList.get(Explainer_Name)
            if (restricted_datasets_for_the_explainer is not None
                    and DataSet_name in restricted_datasets_for_the_explainer):
                print(f"[SKIP] {Explainer_Name} is not supported for dataset {DataSet_name}.")
                continue

            if (DataSet_name, Explainer_Name, GNN_name) in SKIP_COMBINATIONS:
                print(f"[SKIP] Manual exclusion: "f"{DataSet_name} | {GNN_name} | {Explainer_Name}")
                continue
            if not running_file_checker.should_run(dataset_name=DataSet_name, gnn_name=GNN_name,
                                                   explainer_name=Explainer_Name):
                continue

            print("\nRUN:", f"dataset={DataSet_name}", f"gnn={GNN_name}", f"explainer={Explainer_Name}")

                ########################################################################################################

            if Explainer_Name == "GNNExplainer":
                from Meta_Explainer_on_Benchmarking.GNNExplainer_on_Graph_Classification.Script import \
                    gnnexplainer_on_graph_classification as GNNExplainer_Module
                GNNExplainer_Module = reload(GNNExplainer_Module)

                GNE_manager = GNNExplainer_Module.GNNExplainer_Manager(GNN_Model=GNN_Model_Loaded,
                                                                       dataset_name=DataSet_name,
                                                                       num_classes=num_classes,
                                                                       Exp_Epoch=explainer_epoch,
                                                                       Exp_lr=explainer_learning_rate,
                                                                       save_root=save_root)

                GNE_saved_data = GNE_manager(test_dataset=test_dataset, GNE_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                                             binarization_threshold=0.5)

                print(GNN_name, " Model Explained by GNNExplainer", "\nTotal Runtime: ",
                      GNE_saved_data["total_runtime"], "\nAverage Explanation Runtime per Graph-Class: ",
                      GNE_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      GNE_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "PGExplainer":
                from Meta_Explainer_on_Benchmarking.PGExplainer_on_Graph_Classification.Script import \
                    pgexplainer_on_graph_classification as PGExplainer_Module
                PGExplainer_Module = reload(PGExplainer_Module)
                pgex_dim = {"GCN_plus_GAP_Model": {"MUTAG": test_dataset[0].x.size(-1),
                                                   "NCI1": test_dataset[0].x.size(-1), "ENZYMES": 16, "Graph-SST5": 64,
                                                   "PROTEINS": 16, "IsCyclic": test_dataset[0].x.size(-1)},
                            "DGCNN_Model": {"MUTAG": 32, "NCI1": 32, "ENZYMES": 32,
                                            "Graph-SST5": 64, "PROTEINS": 32, "IsCyclic": 32},
                            "DIFFPOOL_Model": {"MUTAG": test_dataset[0].x.size(-1), "NCI1": 37,
                                               "ENZYMES": test_dataset[0].x.size(-1),
                                               "Graph-SST5": test_dataset[0].x.size(-1),
                                               "PROTEINS": test_dataset[0].x.size(-1),
                                               "IsCyclic": test_dataset[0].x.size(-1)},
                            "GIN_Model": {"MUTAG": test_dataset[0].x.size(-1), "NCI1": test_dataset[0].x.size(-1),
                                          "ENZYMES": 16, "Graph-SST5": 64, "PROTEINS": 16,
                                          "IsCyclic": test_dataset[0].x.size(-1)}}

                PGE_manager = PGExplainer_Module.PGExplainer_Manager(GNN_Model=GNN_Model_Loaded,
                                                                     dataset_name=DataSet_name, num_classes=num_classes,
                                                                     Exp_Epoch=explainer_epoch,
                                                                     Exp_lr=explainer_learning_rate,
                                                                     node_feat_dim=pgex_dim[GNN_name][DataSet_name],
                                                                     save_root=save_root)

                PGE_manager(test_dataset=test_dataset, PGE_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                            binarization_threshold=0.5, explainer_training_dataset=test_dataloader)

                PGE_saved_data = PGE_manager(test_dataset=test_dataset, PGE_Mode="test", binarization_threshold=0.5)

                print(GNN_name, " Model Explained by PGExplainer", "\nTotal Runtime: ", PGE_saved_data["testing_runtime"],
                      "\nAverage Explanation Runtime per Graph-Class: ",
                      PGE_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      PGE_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "GraphMask":
                from Meta_Explainer_on_Benchmarking.GraphMask_on_Graph_Classification.Script import \
                    graphmask_on_graph_classification as GraphMask_Module
                GraphMask_Module = reload(GraphMask_Module)

                graphmask_explainers_input_dims = {
                    "GCN_plus_GAP_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": 16,
                        "Graph-SST5": 64,
                        "PROTEINS": 16,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "DGCNN_Model": {
                        "MUTAG": 32, "NCI1": 32, "ENZYMES": 32,
                        "Graph-SST5": 64, "PROTEINS": 32, "IsCyclic": 32
                    },
                    "DIFFPOOL_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": test_dataset[0].x.size(-1),
                        "PROTEINS": test_dataset[0].x.size(-1),
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "GIN_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": 16,
                        "Graph-SST5": 64,
                        "PROTEINS": 16,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    }
                }

                graphmask_explainers_hid_dims = {
                    "GCN_plus_GAP_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": 64,
                        "PROTEINS": 16,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "DGCNN_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": 64,
                        "PROTEINS": 32,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "DIFFPOOL_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": 64,
                        "PROTEINS": test_dataset[0].x.size(-1),
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "GIN_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": 64,
                        "PROTEINS": 16,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    }
                }

                GM_manager = GraphMask_Module.GraphMask_Manager(GNN_Model=GNN_Model_Loaded, dataset_name=DataSet_name,
                                                                num_classes=num_classes, Exp_Epoch=explainer_epoch,
                                                                Exp_lr=0.001, explainer_input_dim=
                                                                graphmask_explainers_input_dims[GNN_name][DataSet_name],
                                                                explainer_hid_dim=
                                                                graphmask_explainers_hid_dims[GNN_name][DataSet_name],
                                                                save_root=save_root)

                GM_manager(test_dataset=test_dataset, GM_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                           explainer_training_dataset=test_dataloader)

                GM_saved_data = GM_manager(test_dataset=test_dataset, GM_Mode="test")

                print(GNN_name, " Model Explained by GraphMask", "\nTesting Runtime: ", GM_saved_data["testing_runtime"],
                      "\nAverage Explanation Runtime per Graph-Class: ",
                      GM_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      GM_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "SubGraphX":
                from Meta_Explainer_on_Benchmarking.SubGraphX_on_Graph_Classification.Script import \
                    subgraphx_on_graph_classification as SubGraphX_Module
                SubGraphX_Module = reload(SubGraphX_Module)

                SGX_manager = SubGraphX_Module.SubGraphX_Manager(GNN_Model=GNN_Model_Loaded, dataset_name=DataSet_name,
                                                                 num_classes=num_classes, num_hops=2, rollout_count=20,
                                                                 min_children_threshold=5, ubc1_c_coef=10.0,
                                                                 expand_count_threshold=5, high2low=True,
                                                                 sample_num=100, save_root=save_root)

                SGX_saved_data = SGX_manager(test_dataset=test_dataset, SGX_Mode=EXPLAINER_MODES_dict[Explainer_Name])

                print(GNN_name, " Model Explained by SubGraphX", "\nTotal Runtime: ", SGX_saved_data["total_runtime"],
                      "\nAverage Explanation Runtime per Graph-Class: ",
                      SGX_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      SGX_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "CF2":
                from Meta_Explainer_on_Benchmarking.CF2_on_Graph_Classification.Script import \
                    cf2_on_graph_classification as CF2_Module
                CF2_Module = reload(CF2_Module)

                CF2_manager = CF2_Module.CF2_Manager(GNN_Model=GNN_Model_Loaded, dataset_name=DataSet_name,
                                                     num_classes=num_classes, explainer_epochs=explainer_epoch,
                                                     input_dim=test_dataset[0].x[0].size()[0],
                                                     hid_dim=test_dataset[0].x[0].size()[0], output_dim=num_classes,
                                                     save_root=save_root)

                CF2_saved_data = CF2_manager(test_dataset=test_dataset, CF2_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                                             binarization_threshold=0.5)

                print(GNN_name, " Model Explained by CF2", "\nTotal Runtime: ", CF2_saved_data["total_runtime"],
                      "\nAverage Explanation Runtime per Graph-Class: ",
                      CF2_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      CF2_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "PGMExplainer":
                from Meta_Explainer_on_Benchmarking.PGMExplainer_on_Graph_Classification.Script import \
                    pgmexplainer_on_graph_classification as PGMExplainer_Module
                PGMExplainer_Module = reload(PGMExplainer_Module)

                PGM_manager = PGMExplainer_Module.PGMExplainer_Manager(GNN_Model=GNN_Model_Loaded,
                                                                       dataset_name=DataSet_name,
                                                                       num_classes=num_classes,
                                                                       perturb_feature_list=[None], perturb_mode="mean",
                                                                       perturb_indicator="abs",
                                                                       num_samples_rule="num_nodes",
                                                                       noise_offset_percentage=50, top_node=5,
                                                                       p_value_threshold=0.05, save_root=save_root)

                PGM_saved_data = PGM_manager(PGM_Mode=EXPLAINER_MODES_dict[Explainer_Name], test_dataset=test_dataset)

                print(GNN_name, " Model Explained by PGMExplainer", "\nTotal Runtime: ",
                      PGM_saved_data["total_runtime"], "\nAverage Explanation Runtime per Graph-Class: ",
                      PGM_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      PGM_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "XGNN":
                from Meta_Explainer_on_Benchmarking.XGNN_on_Graph_Classification.Script import \
                    xgnn_on_graph_classification as XGNN_Module
                XGNN_Module = reload(XGNN_Module)

                datasets_max_number_of_nodes = {
                    "MUTAG": 28,
                    "NCI1": 111,
                    "PROTEINS": 620,
                    "Graph-SST5": 56,
                    "ENZYMES": 126,
                    "IsCyclic": 40
                }

                XGNN_manager = XGNN_Module.XGNN_Manager(GNN_Model=GNN_Model_Loaded, dataset_name=DataSet_name,
                                                        num_classes=num_classes, explainer_epochs=explainer_epoch,
                                                        max_generation_iterations=10,
                                                        num_node_features=test_dataset[0].x.size(-1),
                                                        max_number_of_nodes=datasets_max_number_of_nodes[DataSet_name],
                                                        random_start=True, rollout_count=10, hyp_for_rollout=1,
                                                        hyp_for_rules=2, dropout_rate=0.5, explainer_lr=0.01, b1=0.9,
                                                        b2=0.999, weight_decay=5e-4, save_root=save_root,
                                                        random_seed=42)

                XGNN_Mode = EXPLAINER_MODES_dict[Explainer_Name]
                if XGNN_Mode == "generate":
                    XGNN_saved_data = XGNN_manager(XGNN_Mode="generate", reference_dataset=test_dataset)
                elif XGNN_Mode == "load":
                    XGNN_saved_data = XGNN_manager(XGNN_Mode="load")
                else:
                    raise ValueError("XGNN_Mode must be 'generate' or 'load'.")

                print(GNN_name, " Model Explained by XGNN", "\nTotal Runtime: ", XGNN_saved_data["total_runtime"],
                      "\nAverage Explanation Runtime per Class: ",
                      XGNN_saved_data["average_explanation_runtime_per_class"], )

                ########################################################################################################

            elif Explainer_Name == "GNNInterpreter":
                from Meta_Explainer_on_Benchmarking.GNNInterpreter_on_Graph_Classification.Script import \
                    gnninterpreter_on_graph_classification as GNNInterpreter_Module
                GNNInterpreter_Module = reload(GNNInterpreter_Module)
                explanations_size = GNNInterpreter_Module.get_averaged_explanation_sizes()
                # explanations_size["Graph-SST5"]["DIFFPOOL_Model"][4] = 2
                GNNI_max_nodes = explanations_size[DataSet_name][GNN_name]

                if not GNNI_max_nodes:
                    print(f"[SKIP] GNNInterpreter has no explanation-size configuration for "
                          f"{DataSet_name} | {GNN_name}.")
                    continue

                GNNInt_manager = GNNInterpreter_Module.GNNInterpreter_Manager(GNN_Model=GNN_Model_Loaded,
                                                                              dataset_name=DataSet_name,
                                                                              num_classes=num_classes,
                                                                              explanation_epochs=explainer_epoch,
                                                                              max_nodes=GNNI_max_nodes,
                                                                              num_node_classes=test_dataset[0].x.size(
                                                                                  -1), num_edge_classes=None,
                                                                              temperature=0.15, learning_node_feat=True,
                                                                              learning_edge_feat=False, budget=10,
                                                                              budget_order=2, budget_beta=1,
                                                                              target_probability_min=0.9,
                                                                              batch_size_for_same_sized_graphs=1,
                                                                              save_root=save_root, random_seed=42)

                GNNInt_saved_data = GNNInt_manager(GNNI_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                                                   test_dataset=test_dataset)

                print(GNN_name, " Model Explained by GNNInterpreter", "\nTotal Runtime: ",
                      GNNInt_saved_data["total_runtime"], "\nAverage Explanation Runtime per Class: ",
                      GNNInt_saved_data["average_explanation_runtime_per_class"],)

        del GNN_Model_Loaded
        del GNN_Model_Optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()