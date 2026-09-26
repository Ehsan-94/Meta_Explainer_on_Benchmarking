import os
import sys
import torch









class load_GNN_Models:
    def __init__(self, gnn_model_index, dataset_name, num_classes, entire_dataset, dropout_rate, weight_initializer,
                 bias, act_fun, classifier_weight_decay, classifier_lr, gnn_model_loading_epoch):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout_rate = dropout_rate
        self.dataset_name = dataset_name
        self.entire_dataset = entire_dataset
        self.num_classes = num_classes
        self.weight_initializer = weight_initializer
        self.bias = bias
        self.act_fun = act_fun
        self.classifier_weight_decay = classifier_weight_decay
        self.classifier_lr = classifier_lr
        self.GNN_Models_Name = dict([(1, "GCN_plus_GAP_Model"), (2, "DGCNN_Model"), (3, "DIFFPOOL_Model"),
                                     (4, "GIN_Model")])
        self.GNN_Model_Name = self.GNN_Models_Name[gnn_model_index]
        self.gnn_model_loading_epoch = gnn_model_loading_epoch
        self.Datsets_Name_Dict = {1: "MUTAG", 2: "NCI1", 3: "Graph-SST5", 4: "ENZYMES", 5: "PROTEINS", 6: "IsCyclic"}
        self.Weight_Initializer = {1: "XAVIER", 2: "Kaiming", 3: "Uniform"}
        self.Loss_Function = {1: "CrossEntropy", 2: "MAE", 3: "MSE"}
        self.default_path = '/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/'
        sys.path.insert(0, self.default_path + '/Models/Script/Layers')

    def __call__(self):

        ###################################################################################################          GCN

        if self.GNN_Model_Name == "GCN_plus_GAP_Model":
            import GCN_Layer as gcn_layer
            import GlobalAveragePooling as globalaveragepooling
            import IdenticalPooling as identicalpooling

            sys.path.insert(0, self.default_path + '/Models/Script')
            import GCN_plus_GAP as gcn_plus_gap_model
            node_feat_size = self.entire_dataset[0].x.size()[-1]
            gcn_input_dim = {'MUTAG': node_feat_size, 'NCI1': node_feat_size, 'ENZYMES': 16, 'Graph-SST5': 64,
                             "PROTEINS": 16, 'IsCyclic': node_feat_size}
            self.GNN_Model = gcn_plus_gap_model.GCN_plus_GAP_Model(model_level="graph", num_classes=self.num_classes,
                                                                   GNN_layers=[node_feat_size,
                                                                               gcn_input_dim[self.dataset_name]],
                                                                   Weight_Initializer=self.weight_initializer,
                                                                   Bias=self.bias, act_fun=self.act_fun,
                                                                   dropout_rate=self.dropout_rate).to(self.device)

        ###################################################################################################        DGCNN

        if self.GNN_Model_Name == "DGCNN_Model":
            import DGCNN_Layer as dgcnn_layer
            import DGCNN_GNN_Layers as dgcnn_gnn_layers
            import DGCNN_SortPooling_Layer as sortpooling_layer
            import DGCNN_MLP as dgcnn_mlp

            sys.path.insert(0, self.default_path + '/Models/Script')
            import DGCNN as dgcnn_model

            k_dgcnn = {'MUTAG': 17, 'NCI1': 32, 'ENZYMES': 32, 'Graph-SST5': 19, 'PROTEINS': 26, 'IsCyclic': 20}
            node_feat_size = self.entire_dataset[0].x.size()[-1]
            dgcnn_input_dim = {'MUTAG': [32, 32, 32, 32], 'NCI1': [32, 32, 32, 32], 'ENZYMES': [32, 32, 32, 32],
                               'Graph-SST5': [64, 64, 64, 64], 'PROTEINS': [32, 32, 32, 32],
                               'IsCyclic': [32, 32, 32, 32]}

            self.GNN_Model = dgcnn_model.DGCNN_Model(GNN_layers=dgcnn_input_dim[self.dataset_name], Bias=self.bias,
                                                     num_classes=self.num_classes, mlp_act_fun="ReLu",
                                                     dgcnn_act_fun="tanh", Weight_Initializer=self.weight_initializer,
                                                     mlp_dropout_rate=self.dropout_rate,
                                                     dgcnn_k=k_dgcnn[self.dataset_name],
                                                     node_feat_size=node_feat_size, hid_channels=[16, 32],
                                                     conv1d_kernels=[2, 5], ffn_layer_size=128,
                                                     strides=[2, 1]).to(self.device)

        ###################################################################################################     DIFFPOOL

        if self.GNN_Model_Name == "DIFFPOOL_Model":
            import Batched_GraphSage_Layer as batched_graphsage_layer
            import Batched_DIFFPOOL_Assignment as batched_diffpool_assignment
            import Batched_DIFFPOOL_Embedding as batched_diffpool_embedding
            import Batched_DIFFPOOL_Layer as batched_diffpool_layer
            sys.path.insert(0, self.default_path + '/Models/Script')
            import DIFFPOOL as diffpool_model
            node_feat_size = self.entire_dataset[0].x.size()[-1]
            self.GNN_Model = diffpool_model.DIFFPOOL_Model(embedding_input_dim=node_feat_size,
                                                           embedding_num_block_layers=1, embedding_hid_dim=64,
                                                           new_feature_size=64, assignment_input_dim=node_feat_size,
                                                           assignment_num_block_layers=1, assignment_hid_dim=64,
                                                           max_number_of_nodes=256, prediction_hid_layers=[50],
                                                           concat_neighborhood=False, num_classes=self.num_classes,
                                                           Weight_Initializer=self.weight_initializer, Bias=self.bias,
                                                           act_fun=self.act_fun, dropout_rate=self.dropout_rate,
                                                           normalize_graphsage=False, aggregation="mean",
                                                           concat_diffpools_outputs=True, num_pooling=1,
                                                           pooling="mean").to(self.device)

        ###################################################################################################          GIN

        if self.GNN_Model_Name == "GIN_Model":
            import GIN_MLP_Layers as gin_mlp_layers
            sys.path.insert(0, self.default_path + '/Models/Script')
            import GIN as gin_model
            node_feat_size = self.entire_dataset[0].x.size()[-1]

            gin_input_dim = {'MUTAG': node_feat_size, 'NCI1': node_feat_size, 'ENZYMES': 16, 'Graph-SST5': 64,
                             'PROTEINS': 16, 'IsCyclic': node_feat_size}
            self.GNN_Model = gin_model.GIN_Model(num_mlp_layers=4, Bias=self.bias, num_slp_layers=2,
                                                 mlp_act_fun=self.act_fun, mlp_input_dim=node_feat_size,
                                                 mlp_hid_dim=gin_input_dim[self.dataset_name], joint_embeddings=False,
                                                 mlp_output_dim=self.num_classes, dropout_rate=self.dropout_rate,
                                                 Weight_Initializer=self.weight_initializer).to(self.device)

        GNN_Model_state_dict = torch.load("/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/"
                                          "Train_Models_on_Graph_Classification/Experimental_Results/" +
                                          self.GNN_Model_Name + "/" + self.GNN_Model_Name + "_" + self.dataset_name +
                                          "_" + str(self.gnn_model_loading_epoch) + ".pt", map_location=self.device)
        # print("self.bias: ", self.bias)
        if self.bias == False:
            keys_to_remove = [key for key in GNN_Model_state_dict['model_state_dict'].keys() if 'bias' in key]
            for key in keys_to_remove:
                del GNN_Model_state_dict['model_state_dict'][key]
        # if 'eps' in self.GNN_Model.state_dict().keys():
        #     print("True for model")
        # if 'eps' in GNN_Model_state_dict['model_state_dict'].keys():
        #     print("True for loading")
        # for key in GNN_Model_state_dict['model_state_dict'].keys():
        #     print(key)
        # print("self.GNN_Model: ", self.GNN_Model.state_dict().keys())
        # print(GNN_Model_state_dict['model_state_dict'])
        self.GNN_Model.load_state_dict(GNN_Model_state_dict['model_state_dict'])
        self.GNN_Model_Optimizer = torch.optim.Adam(self.GNN_Model.parameters(), lr=self.classifier_lr,
                                                    weight_decay=self.classifier_weight_decay)

        return self.GNN_Model_Optimizer, self.GNN_Model
