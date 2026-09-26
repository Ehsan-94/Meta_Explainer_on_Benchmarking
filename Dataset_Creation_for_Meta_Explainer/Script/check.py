import torch
# Load and verify
num_samples = 100
weight_levels = [1, 2]
num_metrics = 6
base_name = "Weights_"
weights_distribution = "UniformDistribution_"
file_names = weights_distribution + str(num_samples) + "_" + base_name + '_'.join(str(i) for i in weight_levels)
X = torch.load("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Creation_for_MetaExplainer/Experimental Results/" +
               "X_" + file_names + ".pt")
Y = torch.load("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Creation_for_MetaExplainer/Experimental Results/" +
               "Y_" + file_names + ".pt")

print(len(X))   # e.g., tensor of feature vector
print(Y[0])         # One-hot vector for explainer label
