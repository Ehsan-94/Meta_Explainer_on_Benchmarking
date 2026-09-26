import pickle
import pandas as pd



# ------------------------------------------------------------------- Weighting Schemes
# weight_levels = [1, 2, 3, 4]
# experiment_title = ""
#
# weights_distribution = "UniformDistribution"
# if weights_distribution == "UniformDistribution":
#     num_samples = 4000
#     file_names = "meta_explainer_results_" + weights_distribution + "_" + str(num_samples) + "_Samples_Weights_" + '_'.join(str(i) for i in weight_levels)
# elif weights_distribution == "GridSearch":
#     file_names = "meta_explainer_results_" + experiment_title + weights_distribution + "_Weights_" + '_'.join(str(i) for i in weight_levels)
# directory = "/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Representation_Learning/Experimental Results/"
#
# input_file_names = directory + file_names + ".pkl"

# ------------------------------------------------------------------- Ablation Study
ablation_index_list = [0, 1, 2, 3, 4, 5]

for ablt_indx in ablation_index_list:
    weight_levels = [1, 2, 3, 4]
    ablation_index = ablt_indx

    num_metrics = 6
    experiment_title = "Ablation_Study"
    weights_distribution = "_GridSearch_Weights_"
    ablation_metrics_dict = {0: "Fidelity+", 1: "Fidelity-", 2: "Contrastivity", 3: "Sparsity", 4: "Stability",
                             5: "ExplanationTime"}
    directory = "/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Representation_Learning/Experimental Results/"
    file_names = "meta_explainer_results_" + experiment_title + weights_distribution + '_'.join(str(i) for i in weight_levels) + "_ablated_metric_" + str(ablation_metrics_dict[ablation_index])
    input_file_names = directory + file_names + ".pkl"

    # 2) Load your results dict
    with open(input_file_names, "rb") as f:
        results = pickle.load(f)

    # 3) Build a DataFrame
    df = pd.DataFrame(results)

    # 4) (Optional) format train_size as “XX%”
    df["train_size"] = df["train_size"].astype(int).astype(str) + "%"

    # 5a) In a Jupyter notebook, just display it:
    # display(df)

    # 5b) Or print a Markdown table to the console:
    print(df.to_markdown(index=False))

    # 5c) Or save to CSV:
    df.to_csv(directory + f"{file_names}.csv", index=False)
