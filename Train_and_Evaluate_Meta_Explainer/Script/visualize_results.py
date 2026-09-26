import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

weight_combo = "_weights1234"
directory = "/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Representation_Learning/Experimental Results/"
results_file = directory + f"meta_explainer_results{weight_combo}.pkl"

with open(results_file, "rb") as f:
    results = pickle.load(f)

font_path = '/Explainability_Methods/Dataset_Representation_Learning/Experimental Results/times-new-roman.ttf'
times_new_roman_font_22 = fm.FontProperties(fname=font_path, size=22)
times_new_roman_font_20 = fm.FontProperties(fname=font_path, size=20)
times_new_roman_font_16 = fm.FontProperties(fname=font_path, size=16)


plt.figure(figsize=(10, 6))


plt.plot(results['train_size'], results['test_loss'], marker='s', color='b', label='Test Loss')
plt.plot(results['train_size'], results['test_accuracy'], marker='o', color='r', label='Test Accuracy')
plt.plot(results['train_size'], results['precision'], marker='^', color='g', label='Precision')
plt.plot(results['train_size'], results['recall'], marker='*', color='c', label='Recall')
plt.plot(results['train_size'], results['f1'], marker='D', color='m', label='F1 Score')

plt.xticks(fontproperties=times_new_roman_font_16)
plt.yticks(fontproperties=times_new_roman_font_16)

plt.title('Meta-Explainer Performance vs. Training Data Size', fontdict={'fontproperties': times_new_roman_font_20})
plt.xlabel('Training Data Size (%)', fontdict={'fontproperties': times_new_roman_font_20})
plt.ylabel('Performance Metric Value', labelpad=1, fontdict={'fontproperties': times_new_roman_font_22})
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(prop=times_new_roman_font_16)
plt.grid(True)

# Display the plot
# plt.show()

directory_plot = ("/data/cs.aau.dk/ey33jw/Explainability_Methods/Dataset_Representation_Learning/Experimental Results/MetaExplainer_Evaluation_times" + str(weight_combo) + ".pdf")
plt.savefig(directory_plot)
# plt.show()
plt.close()