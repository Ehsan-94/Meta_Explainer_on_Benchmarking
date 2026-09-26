from colorit import *
from torch._C import dtype
import torch
import csv
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
from time import perf_counter
from sklearn import metrics
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset


default_path = '/data/cs.aau.dk/ey33jw/Explainability_Methods/'
sys.path.insert(0, default_path+'/Models/Script/Layers/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#################################################################################################                    GCN

# import GCN_Layer as gcn_layer
# import GlobalAveragePooling as globalaveragepooling
# import IdenticalPooling as identicalpooling
#
# sys.path.insert(0, default_path+'/Models/Script')
# import GCN_plus_GAP as gcn_plus_gap_model
#
# GNN_Model = gcn_plus_gap_model.GCN_plus_GAP_Model(model_level='graph', GNN_layers=[7, 7], num_classes=2, Bias=True,
#                                                   act_fun='ReLu', Weight_Initializer=3, dropout_rate=0.1).to(device)

#################################################################################################                  DGCNN

# import DGCNN_Layer as dgcnn_layer
# import DGCNN_GNN_Layers as dgcnn_gnn_layers
# import DGCNN_SortPooling_Layer as sortpooling_layer
# import DGCNN_MLP as dgcnn_mlp
#
# sys.path.insert(0, default_path+'/Models/Script')
# import DGCNN as dgcnn_model
#
# k_dgcnn={'MUTAG': 17, 'NCI1': 32, 'ENZYMES': 32, 'Graph-SST5': 19}
#
#
# GNN_Model = dgcnn_model.DGCNN_Model(GNN_layers=[32, 32, 32, 32], num_classes=2, mlp_act_fun='ReLu',
#                                     dgcnn_act_fun='tanh', mlp_dropout_rate=0.5, Weight_Initializer=3, Bias=False,
#                                     dgcnn_k=17, node_feat_size=7, hid_channels=[16, 32], conv1d_kernels=[2, 5],
#                                     ffn_layer_size=128, strides=[2, 1]).to(device)



#################################################################################################               DIFFPOOL

# import Batched_GraphSage_Layer as batched_graphsage_layer
# import Batched_DIFFPOOL_Assignment as batched_diffpool_assignment
# import Batched_DIFFPOOL_Embedding as batched_diffpool_embedding
# import Batched_DIFFPOOL_Layer as batched_diffpool_layer
# sys.path.insert(0, default_path+'/Models/Script')
# import DIFFPOOL as diffpool_model
#
# GNN_Model = diffpool_model.DIFFPOOL_Model(embedding_input_dim=7, embedding_num_block_layers=1, embedding_hid_dim=64,
#                                           new_feature_size=64, assignment_input_dim=7, assignment_num_block_layers=1,
#                                           assignment_hid_dim=64, max_number_of_nodes=256, prediction_hid_layers=[50],
#                                           concat_neighborhood=False, num_classes=2, Weight_Initializer=1, Bias=True,
#                                           dropout_rate=0.1, normalize_graphsage=False, aggregation="mean",
#                                           act_fun="ReLu", concat_diffpools_outputs=True, num_pooling=1, pooling="mean").to(device)

#################################################################################################                    GIN

# import GIN_MLP_Layers as gin_mlp_layers
sys.path.insert(0, default_path+'/Models/Script')
import GIN as gin_model
GNN_Model = gin_model.GIN_Model(num_mlp_layers=4, Bias=True, num_slp_layers=2, mlp_input_dim=7, mlp_hid_dim=7,
                                mlp_output_dim=2, mlp_act_fun="ReLu", dropout_rate=0.1, joint_embeddings=False,
                                Weight_Initializer=1).to(device)


GNN_Model_Optimizer = torch.optim.Adam(GNN_Model.parameters(), lr=0.001, weight_decay=1e-6)

dataset = TUDataset(root='data/TUDataset', name='MUTAG')
df = pandas.read_csv("/data/cs.aau.dk/ey33jw/Datasets_for_Explainability_Methods/" +
                     "Train and Test Indexes on Graph Classification/Experimental Results/train_test_indexes_MUTAG.csv")
node_feat_size = len(dataset[0].x[0])
read_training_list_indexes__ = df['Train Indexes']
read_test_list_indexes__ = df['Test Indexes']
read_test_list_indexes__ = read_test_list_indexes__.dropna()
read_test_list_indexes = []
read_training_list_indexes = []
for element in read_test_list_indexes__:
    read_test_list_indexes.append(int(element))
for element in read_training_list_indexes__:
    read_training_list_indexes.append(int(element))


print(read_training_list_indexes)
print(read_test_list_indexes)

train_dataset = []
test_dataset = []
for index in read_training_list_indexes:
    train_dataset.append(dataset[index])
for index in read_test_list_indexes:
    test_dataset.append(dataset[index])


print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
batch_size=20
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


criterion = torch.nn.CrossEntropyLoss()
def loss_calculations(preds, gtruth):
    loss_per_epoch = criterion(preds, gtruth)
    return loss_per_epoch



def visualize_losses(GNN_Model_losses, epoch_history):
    GNN_Model_losses_list = torch.stack(GNN_Model_losses).cpu().detach().numpy()

    fig = plt.figure(figsize=(27,20))

    ax = plt.subplot2grid((3, 1), (0, 0), colspan=1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(" Loss in Epoch: " + str(epoch_history))

    ax.plot(GNN_Model_losses_list, color='r')

    # plt.savefig('/content/drive/My Drive/Explainability Methods/'+str(Explainability_name)+' on ' + str(Task_name) + '/Experimental Results/' + File_Name + 'Loss_til_epoch_{:04d}.png'.format(epoch_history))
    # plt.show()
    plt.close(fig)




def train_step(data):
    GNN_Model_loss_batch = []
    Pred_Labels = []
    Real_Labels = []

    GNN_Model.train()
    GNN_Model.zero_grad()
    for batch_of_graphs in data:
        batch_of_graphs = batch_of_graphs.to(device)
        if GNN_Model.__class__.__name__ == "GCN_plus_GAP_Model":
            Output_of_Hidden_Layers, pooling_layer_output, ffn_output, soft = GNN_Model(batch_of_graphs, None)
            batch_loss = loss_calculations(soft, batch_of_graphs.y)
            Pred_Labels.extend(soft.argmax(dim=1).detach().tolist())
        if GNN_Model.__class__.__name__ == "DGCNN_Model":
            final_GNN_layer_output, sortpooled_embedings, output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, output_h1, dropout_output_h1, output_h2, softmaxed_h2 = GNN_Model(batch_of_graphs, None)
            # print("softmaxed_h2: ", softmaxed_h2.size())
            batch_loss = loss_calculations(softmaxed_h2, batch_of_graphs.y)
            Pred_Labels.extend(softmaxed_h2.argmax(dim=1).detach().tolist())
        if GNN_Model.__class__.__name__ == "DIFFPOOL_Model":
            concatination_list_of_poolings, prediction_output_without_softmax, prediction_output  = GNN_Model(batch_of_graphs, None)
            # print("softmaxed_h2: ", softmaxed_h2.size())
            batch_loss = loss_calculations(prediction_output, batch_of_graphs.y)
            Pred_Labels.extend(prediction_output.argmax(dim=1).detach().tolist())
        if GNN_Model.__class__.__name__ == "GIN_Model":
            mlps_output_embeds, mlp_outputs_globalSUMpooled, lin1_output, lin1_output_dropouted, lin2_output, lin2_output_softmaxed = GNN_Model(batch_of_graphs, None)
            # print("softmaxed_h2: ", softmaxed_h2.size())
            batch_loss = loss_calculations(lin2_output_softmaxed, batch_of_graphs.y)
            Pred_Labels.extend(lin2_output_softmaxed.argmax(dim=1).detach().tolist())

        Real_Labels.extend(batch_of_graphs.y.detach().tolist())
        GNN_Model_loss_batch.append(batch_loss)

        batch_loss.backward()
        GNN_Model_Optimizer.step()

    return torch.mean(torch.tensor(GNN_Model_loss_batch)), metrics.accuracy_score(Real_Labels, Pred_Labels)



GNN_Model_training_Acc_per_epoch = []
GNN_Model_training_time_per_epoch = []
def train(EPOCHS, load_index, data):
    GNN_Model_training_loss_per_epoch = []

    for epoch in range(EPOCHS):
        t1 = perf_counter()
        GNN_Model_training_loss, training_acc = train_step(data)
        GNN_Model_training_time_per_epoch.append(perf_counter()-t1)
        print(f'Epoch: {epoch+1:03d}, Model Loss: {GNN_Model_training_loss:.4f}, Accuracy: {training_acc:.2f}')

        GNN_Model_training_loss_per_epoch.append(GNN_Model_training_loss)
        GNN_Model_training_Acc_per_epoch.append(training_acc)
        #break

        if (epoch + load_index + 1) % 50 == 0 and epoch > 0:
            visualize_losses(GNN_Model_training_loss_per_epoch, epoch + load_index + 1)
        # if (epoch + load_index + 1) % 100 == 0 and epoch > 0:
        #     torch.save({'epoch': epoch+load_index+1, 'model_state_dict': GNN_Model.state_dict(), 'optimizer_state_dict': GNN_Model_Optimizer.state_dict(), 'loss': GNN_Model_training_loss_per_epoch,}, "/content/drive/My Drive/Explainability Methods/" + str(Explainability_name) + " on " + str(Task_name) + "/Model/" + File_Name + str(epoch + load_index + 1)+".pt")

print(GNN_Model.state_dict().keys())
print(GNN_Model.state_dict()['gin_mlp_layers.0.gin_mlp_layers.0.weight'])

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA is not available. Running on CPU.")

EPOCHS = 5000
load_index = 0

train(EPOCHS, load_index, train_dataloader)


print(GNN_Model.state_dict()['gin_mlp_layers.0.gin_mlp_layers.0.weight'])

from Meta_Explainer_on_Benchmarking.GNNInterpreter_on_Graph_Classification.Script import gnninterpreter_on_graph_classification as GNNInterpreter_Module
explanations_size = GNNInterpreter_Module.get_averaged_explanation_sizes()
edge_limit = explanations_size["MUTAG"]["GCN_plus_GAP_Model"][0]
print(explanations_size["Graph-SST5"]["DIFFPOOL_Model"][4])
print(explanations_size)