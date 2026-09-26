import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import pickle  # you need this for saving your results

# ================= Config =================
RECALL_AVERAGING = "macro"   # <- changed from "weighted"
# ==========================================

# 1) Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
ablation_index_list = [0, 1, 2, 3, 4, 5]
for abl_indx in ablation_index_list:
    weight_levels = [1, 2, 3, 4]
    ablation_index = abl_indx
    num_metrics = 6
    experiment_title = "Ablation_Study"
    weights_distribution = "_GridSearch_Weights_"
    ablation_metrics_dict = {
        0: "Fidelity+",
        1: "Fidelity-",
        2: "Contrastivity",
        3: "Sparsity",
        4: "Stability",
        5: "ExplanationTime"
    }
    input_file_names  = (experiment_title + weights_distribution
                         + '_'.join(str(i) for i in weight_levels)
                         + "_ablated_index_" + str(ablation_index))
    output_file_names = (experiment_title + weights_distribution
                         + '_'.join(str(i) for i in weight_levels)
                         + "_ablated_metric_" + str(ablation_metrics_dict[ablation_index]))
    directory_x = (
        "/data/cs.aau.dk/ey33jw/Explainability_Methods/"
        "Dataset_Creation_for_MetaExplainer/Experimental Results/"
        f"X_{input_file_names}.pt"
    )
    directory_y = (
        "/data/cs.aau.dk/ey33jw/Explainability_Methods/"
        "Dataset_Creation_for_MetaExplainer/Experimental Results/"
        f"Y_{input_file_names}.pt"
    )
    X_list = torch.load(directory_x)
    Y_list = torch.load(directory_y)

    X_data = torch.stack(X_list).float()
    Y_data = torch.stack(Y_list).float()
    if Y_data.dim() > 1 and Y_data.size(1) > 1:
        Y_data = torch.argmax(Y_data, dim=1)

    class MetaExplainer(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MetaExplainer, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    def train_and_evaluate(train_loader, test_loader, input_size, hidden_size, output_size, num_epochs):
        # 2) Move model to device
        meta_explainer = MetaExplainer(input_size, hidden_size, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(meta_explainer.parameters(), lr=0.001)

        # Training loop
        for epoch in range(num_epochs):
            meta_explainer.train()
            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                outputs = meta_explainer(batch_X)
                loss = criterion(outputs, batch_Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluation
        meta_explainer.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_test_outputs = []
        all_test_labels = []

        with torch.no_grad():
            for test_X, test_Y in test_loader:
                test_X = test_X.to(device)
                test_Y = test_Y.to(device)

                test_outputs = meta_explainer(test_X)
                test_loss += criterion(test_outputs, test_Y).item()
                _, test_predicted = torch.max(test_outputs, 1)
                test_correct += (test_predicted == test_Y).sum().item()
                test_total += test_Y.size(0)

                all_test_outputs.extend(test_predicted.cpu().numpy())
                all_test_labels.extend(test_Y.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total

        # --- Metrics ---
        precision_w = precision_score(all_test_labels, all_test_outputs, average='weighted')
        recall_m    = recall_score(all_test_labels, all_test_outputs, average=RECALL_AVERAGING)  # "macro"
        f1_w        = f1_score(all_test_labels, all_test_outputs, average='weighted')

        # (Optional sanity check: weighted recall == accuracy for single-label tasks)
        # recall_w = recall_score(all_test_labels, all_test_outputs, average='weighted')
        # assert abs(recall_w - test_accuracy) < 1e-9

        return avg_test_loss, test_accuracy, precision_w, recall_m, f1_w

    hidden_size = 64
    output_size = len(torch.unique(Y_data))
    num_epochs = 500

    scenarios = [0.2, 0.4, 0.6, 0.8]
    test_size = 0.2
    results = defaultdict(list)

    for train_size in scenarios:
        print(f"\nTraining with {int(train_size * 100)}% of the data...")

        X_train, X_test, Y_train, Y_test = train_test_split(
            X_data, Y_data,
            test_size=test_size,
            train_size=train_size,
            random_state=42,
            shuffle=True
        )

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset  = torch.utils.data.TensorDataset(X_test, Y_test)
        test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        avg_test_loss, test_accuracy, precision, recall, f1 = train_and_evaluate(
            train_loader,
            test_loader,
            input_size=X_train.shape[1],
            hidden_size=hidden_size,
            output_size=output_size,
            num_epochs=num_epochs
        )

        results['train_size'].append(train_size * 100)
        results['test_loss'].append(avg_test_loss)
        results['test_accuracy'].append(test_accuracy)
        results['precision_weighted'].append(precision)
        results['recall_macro'].append(recall)      # <- explicit naming
        results['f1_weighted'].append(f1)

        print(f"Test Loss: {avg_test_loss:.4f}, "
              f"Acc: {test_accuracy:.2%}, "
              f"Prec(w): {precision:.4f}, "
              f"Rec(macro): {recall:.4f}, "
              f"F1(w): {f1:.4f}")

    results_file = (
        "/data/cs.aau.dk/ey33jw/Explainability_Methods/"
        "Dataset_Representation_Learning/Experimental Results/"
        f"meta_explainer_results_{output_file_names}.pkl"
    )
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
