import os


class file_checker:
    def __init__(self, save_root):
        self.save_root = save_root

    def get_result_path(self, dataset_name, gnn_name, explainer_name, task_name="Graph_Classification"):
        save_dir = os.path.join(self.save_root, f"{explainer_name}_on_{task_name}", "Experimental_Results")
        save_path = os.path.join(save_dir, f"{dataset_name}_{gnn_name}_{explainer_name}.pt")

        return save_path

    def result_exists(self, dataset_name, gnn_name, explainer_name, task_name="Graph_Classification", verbose=True):
        save_path = self.get_result_path(dataset_name=dataset_name, gnn_name=gnn_name, explainer_name=explainer_name,
                                         task_name=task_name)
        exists = os.path.isfile(save_path)
        if exists and verbose:
            print(f"[SKIP] Already completed: "f"dataset={dataset_name} "f"gnn={gnn_name} "f"explainer={explainer_name}")

        return exists

    def should_run(self, dataset_name, gnn_name, explainer_name, task_name="Graph_Classification", verbose=True):
        return not self.result_exists(dataset_name=dataset_name, gnn_name=gnn_name, explainer_name=explainer_name,
                                      task_name=task_name, verbose=verbose)
