from dataclasses import dataclass, asdict,is_dataclass
import os
import shutil
import json
import torch
import csv

@dataclass
class ExperimentParams:
    model_id: str
    short_name:str
    description: str
    input_size: int
    output_size: int
    lr: float
    batch_size: int
    num_epochs: int
    clip_grad_norm: float

class ExperimentManager:
    def __init__(self):
        self.root_path = 'experiments/'
        os.makedirs(self.root_path, exist_ok=True)
        self.index_file = os.path.join(self.root_path, 'experiment_index.json')
        self.experiment_indices = self._load_or_initialize_experiment_indices()
        # Initialize variables for managing a single experiment
        self.current_experiment_folder = None
        self.current_params = None

    def _load_or_initialize_experiment_indices(self):
        if not os.path.exists(self.index_file):
            with open(self.index_file, 'w') as f:
                json.dump({}, f)
            return {}
        else:
            with open(self.index_file, 'r') as f:
                return json.load(f)

    def _save_experiment_indices(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.experiment_indices, f, indent=4)

    def _increment_experiment_index(self, model_id):
        self.experiment_indices[model_id] = self.experiment_indices.get(model_id, 0) + 1
        self._save_experiment_indices()
        return self.experiment_indices[model_id]

    def create_experiment(self, params):
        if not is_dataclass(params) or not isinstance(params, ExperimentParams):
            raise ValueError("params must be a dataclass instance derived from ExperimentParams.")
        self.current_params = params
        index = self._increment_experiment_index(params.model_id)
        self.current_experiment_folder = os.path.join(self.root_path, f"{params.model_id}/{params.short_name}_{index}")
        os.makedirs(self.current_experiment_folder, exist_ok=True)
        
        # Copy model_id folder contents
        model_id_path = os.path.join(params.model_id)
        if os.path.exists(model_id_path):
            for item in os.listdir(model_id_path):
                s = os.path.join(model_id_path, item)
                d = os.path.join(self.current_experiment_folder, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        
        # Save params
        self.save_experiment_params()

    def save_experiment_params(self):
        params_file = os.path.join(self.current_experiment_folder, 'params.json')
        with open(params_file, 'w') as f:
            json.dump(asdict(self.current_params), f, indent=4)

    def save_plot(self, plotting):
        # save plots
        plotting.save_plot(self.current_experiment_folder, "plots.png")

    def save_model_info(self, model):
        # save model info
        model_str = str(model)
        model_info_file = os.path.join(self.current_experiment_folder, 'model_info.txt')
        with open(model_info_file, 'w') as f:
            f.write(model_str)

    def save_experiment_without_model(self, model, plotting):
        self.save_model_info(model)
        self.save_plot(plotting)

    def save_model(self, model):
        # save model
        model_path = os.path.join(self.current_experiment_folder, f"{os.path.basename(self.current_experiment_folder)}_model.pth")
        torch.save(model.state_dict(), model_path)

    def load_experiment(self, model_id, short_name, index):
        experiment_folder = os.path.join(self.root_path, f"{model_id}/{short_name}_{index}")
        params_file = os.path.join(experiment_folder, 'params.json')
        with open(params_file, 'r') as f:
            params = json.load(f)
        self.current_experiment_folder = experiment_folder
        # Assuming params are compatible with ExperimentParams; adjust as needed for specialization
        self.current_params = ExperimentParams(**params)
        return self.current_params
    
    def delete_current_experiment(self):
        if self.current_experiment_folder is None:
            print("No current experiment to delete.")
            return

        # Extract model_id and index from the current experiment folder
        model_id = self.current_params.model_id
        index_str = self.current_experiment_folder.rstrip('/').split('_')[-1]
        
        try:
            index = int(index_str)
        except ValueError:
            print(f"Could not determine experiment index from folder name: {self.current_experiment_folder}")
            return

        # Delete the experiment folder
        try:
            shutil.rmtree(self.current_experiment_folder)
            print(f"Deleted experiment folder: {self.current_experiment_folder}")
        except OSError as e:
            print(f"Error deleting experiment folder: {self.current_experiment_folder}\n{e}")
            return

        # Update the experiment indices and save
        if model_id in self.experiment_indices:
            # Assuming the experiment to delete is the last one created for simplicity
            # For more complex scenarios, additional logic would be needed
            self.experiment_indices[model_id] -= 1
            if self.experiment_indices[model_id] <= 0:
                del self.experiment_indices[model_id]  # Remove model_id if no experiments left
            self._save_experiment_indices()

        # Reset current experiment tracking
        self.current_experiment_folder = None
        self.current_params = None

    def save_validation_metrics(self, metrics):
        if self.current_experiment_folder is None:
            print("No current experiment set. Cannot save validation metrics.")
            return

        validation_metrics_file = os.path.join(self.current_experiment_folder, 'validation_metrics.csv')
        self._save_metrics_to_csv(metrics, validation_metrics_file)

    def save_test_metrics(self, metrics):
        if self.current_experiment_folder is None:
            print("No current experiment set. Cannot save test metrics.")
            return

        test_metrics_file = os.path.join(self.current_experiment_folder, 'test_metrics.csv')
        self._save_metrics_to_csv(metrics, test_metrics_file)
        print(f"Test metrics saved to {test_metrics_file}")

    def _save_metrics_to_csv(self, metrics, file_path):
        """
        Appends the given metrics to a CSV file. If the file does not exist, it is created.
        
        Parameters:
        - metrics: Tuple containing the metrics (accuracy, precision, recall, f1)
        - file_path: The path to the CSV file
        - metric_type: A string indicating the type of metrics (for header purposes)
        """
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write the header only if the file is being created
                header = [f"{m}" for m in ["Accuracy", "Precision", "Recall", "F1"]]
                writer.writerow(header)
            writer.writerow(metrics)