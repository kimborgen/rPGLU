import torch
import math
import numpy as np
from aeon.datasets import load_classification

def get_datasets(params, device):
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()
    train_iterator = DataIterator(X_train, y_train, params, device, shuffle_on_reset=True)
    val_iterator = DataIterator(X_val, y_val, params, device, shuffle_on_reset=False)
    test_iterator = DataIterator(X_test, y_test, params, device, shuffle_on_reset=False)
    return train_iterator, val_iterator, test_iterator

class DataIterator:
    def __init__(self, X, y, params, device, shuffle_on_reset=False):
        self.X = X
        self.y = y
        self.batch_size = params.batch_size
        self.device = device
        self.current = 0
        self.shuffle_on_reset = shuffle_on_reset

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.X):
            raise StopIteration
        else:
            X_batch = self.X[self.current:self.current+self.batch_size]
            y_batch = self.y[self.current:self.current+self.batch_size]
            self.current += self.batch_size

            X_tensor = torch.from_numpy(X_batch).float().to(self.device)
            y_tensor = torch.from_numpy(y_batch).long().to(self.device)

             # Transpose X_tensor to shape (time_steps, batch_size, features)
            X_tensor = X_tensor.transpose(1, 2).transpose(0, 1)
            
            return X_tensor, y_tensor

    def reset(self):
        self.current = 0
        if self.shuffle_on_reset:
                indices = np.random.permutation(len(self.X))
                self.X = self.X[indices]
                self.y = self.y[indices]

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)
    
def load_datasets():
    # This will not download, because Arrowhead is already in the directory.
    # Change the extract path or name to downloads
    X_train, y_train, meta_data_train = load_classification("Tiselac", split="train", extract_path="./ts_data", return_metadata=True)
    X_test, y_test, meta_data_test = load_classification("Tiselac", split="test", extract_path="./ts_data", return_metadata=True)
    
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Determine the size of the validation set
    validation_size = int(0.05 * X_train.shape[0])

    # Create a random permutation of indices and split into validation and training indices
    indices = torch.randperm(X_train.shape[0])
    validation_indices, train_indices = indices[:validation_size], indices[validation_size:]

    # Create the validation set
    X_val, y_val = X_train[validation_indices], y_train[validation_indices]

    # Update the training set to exclude the validation set
    X_train, y_train = X_train[train_indices], y_train[train_indices]

    # Shuffle the train dataset
    shuffle_indices = torch.randperm(train_indices.size(0))
    X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]

    # change y from str 1-indexed to int 0-indexed
    y_train = y_train.astype(int) - 1
    y_test = y_test.astype(int) - 1
    y_val = y_val.astype(int) - 1

    X_train_norm, X_val_norm, X_test_norm = normalize(X_train, X_val, X_test)

    return X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test

def normalize(X_train, X_val, X_test):
    """
    Takes in the entire dataset, calculates the min and max for each feature across all timesteps and across all splits.
    and normalizes the dataset to the range [0, 1]
    """


    X_train_stats = calculate_feature_statistics(X_train)
    X_val_stats = calculate_feature_statistics(X_val)
    X_test_stats = calculate_feature_statistics(X_test)

    num_features = 10
    global_min = np.zeros(num_features)
    global_max = np.zeros(num_features)
    
    for feature_idx in range(num_features):
        global_min[feature_idx] = min(X_train_stats['min'][feature_idx], X_val_stats['min'][feature_idx], X_test_stats['min'][feature_idx])
        global_max[feature_idx] = max(X_train_stats['max'][feature_idx], X_val_stats['max'][feature_idx], X_test_stats['max'][feature_idx])
    
    #print(global_min, global_max)

    X_train_norm = normalize_features_to_neg_1_pos_1(X_train, global_min, global_max)
    X_val_norm = normalize_features_to_neg_1_pos_1(X_val, global_min, global_max)
    X_test_norm = normalize_features_to_neg_1_pos_1(X_test, global_min, global_max)
    
    print(f"Min/max of each feature in train: {calculate_feature_statistics(X_train_norm)}")
    print(f"Min/max of each feature in val: {calculate_feature_statistics(X_val_norm)}")
    print(f"Min/max of each feature in test: {calculate_feature_statistics(X_test_norm)}")
    
    print(" Shape of X_train_norm = ", X_train_norm.shape)
    print(" Shape of X_test_norm = ", X_test_norm.shape)
    print(" Shape of X_val_norm = ", X_val_norm.shape)

    return X_train_norm, X_val_norm, X_test_norm

def normalize_features_to_0_1(X, global_min, global_max):
    """
    Normalize the dataset using the global minimum and maximum values for each feature,
    scaling each feature to the range [0, 1].

    Args:
    X (np.ndarray): The dataset to normalize.
    global_min (np.ndarray): Global minimum values for each feature.
    global_max (np.ndarray): Global maximum values for each feature.

    Returns:
    np.ndarray: The normalized dataset.
    """
    X_normalized = np.zeros_like(X)
    num_samples, num_features, num_timesteps = X.shape
    
    for feature_idx in range(num_features):
        min_val = global_min[feature_idx]
        max_val = global_max[feature_idx]
        # Ensure division is meaningful; if min and max are the same, feature is constant
        if min_val != max_val:
            X_normalized[:, feature_idx, :] = (X[:, feature_idx, :] - min_val) / (max_val - min_val)
        else:
            # For constant features, set to 0 (or another value that makes sense in your context)
            X_normalized[:, feature_idx, :] = 0
    
    return X_normalized

def normalize_features_to_neg_1_pos_1(X, global_min, global_max):
    """
    Normalize the dataset using the global minimum and maximum values for each feature,
    scaling each feature to the range [-1, 1].

    Args:
    X (np.ndarray): The dataset to normalize, shape (samples, features, timesteps).
    global_min (np.ndarray): Global minimum values for each feature.
    global_max (np.ndarray): Global maximum values for each feature.

    Returns:
    np.ndarray: The normalized dataset.
    """
    X_normalized = np.zeros_like(X)
    num_samples, num_features, num_timesteps = X.shape

    for feature_idx in range(num_features):
        min_val = global_min[feature_idx]
        max_val = global_max[feature_idx]

        # Ensure division is meaningful; if min and max are the same, feature is constant
        if min_val != max_val:
            # New formula to scale to [-1, 1]
            X_normalized[:, feature_idx, :] = 2 * (X[:, feature_idx, :] - min_val) / (max_val - min_val) - 1
        else:
            # For constant features, set to 0 (or another value that makes sense in your context)
            X_normalized[:, feature_idx, :] = 0

    return X_normalized

def calculate_feature_statistics(X):
    """
    Calculate min and max values for each feature across all timesteps.

    Args:
    X (np.ndarray): Input data of shape (samples, features, timesteps).

    Returns:
    dict: A dictionary containing the min and max values for each feature.
    """
    num_samples, num_features, num_timesteps = X.shape
    feature_stats = {'min': np.zeros(num_features), 'max': np.zeros(num_features)}

    for feature_idx in range(num_features):
        feature_data = X[:, feature_idx, :]
        feature_stats['min'][feature_idx] = feature_data.min()
        feature_stats['max'][feature_idx] = feature_data.max()

    return feature_stats

if __name__ == "__main__":
    load_datasets()