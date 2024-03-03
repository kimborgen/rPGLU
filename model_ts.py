from aeon.datasets import load_classification
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mlp
mlp.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import math
import numpy as np
from torch.optim.lr_scheduler import StepLR


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
                
        self.fc1 = nn.Linear(10, 50)
        #self.bn1 = nn.BatchNorm1d(100)  # BatchNorm for the first layer
        self.fc2 = nn.Linear(50, 50)
        #self.bn2 = nn.BatchNorm1d(100)  # BatchNorm for the second layer
        self.fc3 = nn.Linear(50, 9)

        #self.fc4 = nn.Linear(100, 9)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        fc1 = self.fc1(x)
        #bn1 = self.bn1(fc1)
        r1 = F.relu(fc1)
        d1 = self.dropout(r1)

        fc2 = self.fc2(d1)
        #bn2 = self.bn2(fc2)
        r2 = F.relu(fc2)
        d2 = self.dropout(r2)

        fc3 = self.fc3(d2)
        #r3 = F.relu(fc3)

        #logits = self.fc4(r3)
        return fc3
    
class BaselineNetFull(nn.Module):
    def __init__(self):
        super(BaselineNetFull, self).__init__()
                
        self.fc1 = nn.Linear(230, 8192)
        #self.bn1 = nn.BatchNorm1d(100)  # BatchNorm for the first layer
        self.fc2 = nn.Linear(8192, 8192)
        #self.bn2 = nn.BatchNorm1d(100)  # BatchNorm for the second layer
        self.fc3 = nn.Linear(8192, 9)

        #self.fc4 = nn.Linear(100, 9)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        fc1 = self.fc1(x)
        #bn1 = self.bn1(fc1)
        r1 = F.relu(fc1)
        d1 = self.dropout(r1)

        fc2 = self.fc2(d1)
        #bn2 = self.bn2(fc2)
        r2 = F.relu(fc2)
        d2 = self.dropout(r2)

        fc3 = self.fc3(d2)
        #r3 = F.relu(fc3)

        #logits = self.fc4(r3)
        return fc3 
    

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dr_lstm, dr_fc, num_timesteps):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, dropout=dr_lstm)
        self.dropout = nn.Dropout(dr_fc)
        si = hidden_size * num_timesteps
        self.fc = nn.Linear(si, si * 12)
        self.fc2 = nn.Linear(si * 12, si * 12)
        self.fc3 = nn.Linear(si * 12, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # x.size(1) is batch_size
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape ( seq_length, batch size, hidden_size)
        #last = out[-1, :, :]
        # Decode the hidden state of the last time step
        out_reshaped = out.transpose(0,1)
        out_reshaped2 = out_reshaped.reshape(out.size(1), -1)
        d1 = self.dropout(out_reshaped2)
        fc1 = self.fc(d1)
        r1 = F.relu(fc1)
        d2 = self.dropout(r1)
        fc2 = self.fc2(d2)
        r2 = F.relu(fc2)
        d3 = self.dropout(r2)
        fc3 = self.fc3(d3)

        return fc3

class ZeroWithSigmoidGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Perform the forward pass operation: gated * 0
        ctx.save_for_backward(input)
        return input * 0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Compute the surrogate gradient using the sigmoid function
        surrogate_grad = grad_output * torch.sigmoid(input)
        return surrogate_grad
    
class NegativeReLUGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward pass, output is zeroed out
        ctx.save_for_backward(input)
        return input * 0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Compute the gradient using negative ReLU
        grad_input = grad_output.clone()
        grad_input[input > 0] = -input[input > 0]
        return grad_input

"""
class LinearGated(nn.Module):
    def __init__(self, device, batch_size, num_out):
        super(LinearGated, self).__init__()
        self.device = device
        #self.inp = inp
        self.num_out = num_out
        #self.spike_decay_rate = spike_decay_rate
        self.batch_size = batch_size


        #self.fc = nn.Linear(inp, out)
          #add device
        #self.reset()
        mu, sigma = 0, 1000  # For threshold
        mu_decay, sigma_decay = 0.4, 0.5  # For decay rate
        #mu_spike, sigma_spike = 0, -1.0
        self.tresh = nn.Parameter(torch.abs(torch.rand(size=(self.num_out,), device=self.device, requires_grad=True)) * sigma + mu)
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(self.num_out,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)
        #self.spike_decay_rate = nn.Parameter(torch.abs(torch.rand(size=(self.out,), device=self.device, requires_grad=True)) * sigma_spike + mu_spike)
        pass 

    def forward(self, x, potential):

        #normed = F.normalize(x)
        # linear
        # lin = self.fc(x)
        # linear negative

        # sigmoid to normalize between 0 and 1.
        # activated = F.sigmoid(x)
        #activated = F.relu(x)
        #activated = F.tanh(x)
        activated = x 

        # calculate new potential
        potential = potential + activated

        #gated_potential = F.relu(potential)

        # Zero those where potential < tresh
        #gated_bool = torch.ge(potential, self.tresh)
        gated = F.relu(potential - self.tresh)
        #gated = potential * gated0    

        
    
        # reduce the potential of the open gates with spike_decay_rate
        #post_gated = gated * self.spike_decay_rate

        post_gated = NegativeReLUGradientFunction.apply(gated)

        # all the gates that were not activated by reli

        # reduce the potential of the closed gates with decay_rate
        #non_gated_bool = ~gated_bool # negation operator
        non_gated_bool = (gated == 0).type(torch.float32)  # or use .type(torch.int32) for integer output
        non_gated = potential * non_gated_bool
        post_non_gated = non_gated * self.decay_rate

        # now combine the two to the new potential 
        new_potential = post_gated + post_non_gated
        potential = F.relu(new_potential)

        return gated, potential
"""

# Define the mathematical function for the inverted bump function
def inverted_bump_function_math(x):
    # Function is 1 outside the range -1 to 1
    f = torch.ones_like(x)
    # Within the range -1 to 1, apply the inverted bump function
    mask = (x > -1) & (x < 1)
    f[mask] = 1 - torch.exp(-1 / (1 - x[mask] ** 2))
    return f

# Define the derivative of the inverted bump function
def derivative_inverted_bump_function_math(x):
    # Derivative is zero outside the range -1 to 1
    df = torch.zeros_like(x)
    # Within the range -1 to 1, calculate the derivative
    mask = (x > -1) & (x < 1)
    df[mask] = 2 * x[mask] * torch.exp(-1 / (1 - x[mask] ** 2)) / (1 - x[mask] ** 2) ** 2
    return df

class InvertedBumpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for use in backward pass
        ctx.save_for_backward(x)
        
        # Forward pass: return 1 where x is not 0, otherwise 0 (as a placeholder behavior)
        output = torch.where(x != 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        x, = ctx.saved_tensors
        
        # Calculate the derivative of the inverted bump function
        derivative = derivative_inverted_bump_function_math(x)
        
        # Apply the chain rule (multiply by incoming gradient)
        grad_input = derivative * grad_output
        
        return grad_input

class BumpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for use in backward pass
        ctx.save_for_backward(x)
        
        # Forward pass: return 1 where x is 0, otherwise 0
        output = torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        x, = ctx.saved_tensors
        
        # Create a mask for where the bump function is active (-1 < x < 1)
        mask = (x > -1) & (x < 1)
        
        # Initialize gradient as zero for all elements
        grad_x = torch.zeros_like(x)
        
        # Compute gradient only for the masked elements
        grad_x[mask] = torch.exp(-1 / (1 - (x[mask] ** 2)))
        # Normalize the bump function to have a maximum of 1
        grad_x[mask] = grad_x[mask] / torch.max(grad_x[mask])
        # Adjust the gradient for the bump function
        grad_x[mask] = grad_output[mask] * (-2 * x[mask] / (1 - x[mask] ** 2) ** 2) * grad_x[mask]
        
        return grad_x
    
class LinearGated(nn.Module):
    def __init__(self, device, batch_size, num_out):
        super(LinearGated, self).__init__()
        self.device = device
        #self.inp = inp
        self.num_out = num_out
        #self.spike_decay_rate = spike_decay_rate
        self.batch_size = batch_size


        #self.fc = nn.Linear(inp, out)
          #add device
        #self.reset()
        mu, sigma = 0, 0.9  # For threshold
        mu_decay, sigma_decay = 0.4, 0.5  # For decay rate
        #mu_spike, sigma_spike = 0, -1.0
        self.tresh = nn.Parameter(torch.abs(torch.rand(size=(self.num_out,), device=self.device, requires_grad=True)) * sigma + mu)
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(self.num_out,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)
        #self.spike_decay_rate = nn.Parameter(torch.abs(torch.rand(size=(self.out,), device=self.device, requires_grad=True)) * sigma_spike + mu_spike)
        pass 

    def forward(self, x, potential): 

        # calculate new potential
        potential = potential + x

        treshed = potential - self.tresh

        # Zero those where potential < tresh
        gated = F.relu(treshed)
        
        # now apply this gated filter to the potential
        gated_bool = InvertedBumpFunction.apply(gated)

        # This is now what we return from this layer
        filtered = gated_bool * potential
        activated = F.relu(filtered)

        # Now we wish to reduce the potential of the open gates with spike_decay_rate

        # reduce the potential of the closed gates with decay_rate
        non_gated_bool = BumpFunction.apply(gated)
        non_gated = potential * non_gated_bool
        potential_non_gated = non_gated * self.decay_rate
        
        # Remove negative potentials 
        new_potential = F.relu(potential_non_gated)

        return activated, new_potential

class GatedNet(nn.Module):
    def __init__(self, device, batch_size, inp, out):
        super(GatedNet, self).__init__()

        self.batch_size = batch_size
        self.inp = inp 
        self.out = out
        self.device = device
        self.dropout = 0.2

        self.hs = inp * 64
        self.fc1 = nn.Linear(inp, self.hs)
        self.norm1 = nn.LayerNorm(self.hs)
        self.lg1 = LinearGated(device, batch_size, self.hs)
        self.d1 = nn.Dropout(self.dropout)

        self.fc2 = nn.Linear(self.hs,self.hs)
        self.norm2 = nn.LayerNorm(self.hs)
        self.lg2 = LinearGated(device, batch_size, self.hs)
        self.d2 = nn.Dropout(self.dropout)

        self.fc3 = nn.Linear(self.hs, out)


    def forward(self, data):
        # Define the forward pass here


        pot1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        pot2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)

        out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps



            x = data[step]

            fc1 = self.fc1(x)
            #norm1 = self.norm1(fc1)
            lg1, pot1 = self.lg1(fc1, pot1)
            d1 = self.d1(lg1)

            fc2 = self.fc2(d1)
            #norm2 = self.norm2(fc2)
            lg2, pot2 = self.lg2(fc2, pot2)
            d2 = self.d2(lg2)

            fc3 = self.fc3(d2)

            out.append(fc3)

        last = out[-1]
        #stacked = torch.stack(out)
        #x = stacked.sum(dim=0)
        
        return last
    
class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class Dataset:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

def load_data():
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

    # normalize
    X_train_stats = calculate_feature_statistics(X_train)
    X_val_stats = calculate_feature_statistics(X_val)
    X_test_stats = calculate_feature_statistics(X_test)

    
    #X_val_s = custom_min_max_normalize_np(X_val, min_val=-1000, max_val=2475)

    num_features = 10
    global_min = np.zeros(num_features)
    global_max = np.zeros(num_features)
    
    for feature_idx in range(num_features):
        global_min[feature_idx] = min(X_train_stats['min'][feature_idx], X_val_stats['min'][feature_idx], X_test_stats['min'][feature_idx])
        global_max[feature_idx] = max(X_train_stats['max'][feature_idx], X_val_stats['max'][feature_idx], X_test_stats['max'][feature_idx])
    
    #print(global_min, global_max)

    X_train_norm = normalize_features(X_train, global_min, global_max)
    X_val_norm = normalize_features(X_val, global_min, global_max)
    X_test_norm = normalize_features(X_test, global_min, global_max)
    #print(calculate_feature_statistics(X_train_norm))
    #print(calculate_feature_statistics(X_val_norm))
    #print(calculate_feature_statistics(X_test_norm))
    
    print(" Shape of X_train_norm = ", X_train_norm.shape)
    print(" Shape of y_train = ", y_train.shape)

    print(" Shape of X_test_norm = ", X_test_norm.shape)
    print(" Shape of y_test = ", y_test.shape)

    print(" Shape of X_val_norm = ", X_val_norm.shape)
    print(" Shape of y_val = ", y_val.shape)

  

    datasets = Dataset(X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test)

    return datasets

def normalize_features(X, global_min, global_max):
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

def calculate_feature_statistics(X):
    """
    Calculate min and max values for each feature across all timesteps.

    Args:
    X (np.ndarray): Input data of shape (samples, features, timesteps).

    Returns:
    dict: A dictionary containing the min and max values for each feature.
    """
    num_features = X.shape[1]
    feature_stats = {'min': np.zeros(num_features), 'max': np.zeros(num_features)}

    for feature_idx in range(num_features):
        feature_data = X[:, feature_idx, :]
        feature_stats['min'][feature_idx] = feature_data.min()
        feature_stats['max'][feature_idx] = feature_data.max()

    return feature_stats

def evaluate_model(model, datasets, device, evaluation_fn):
    model.eval()  # Set the model to evaluation mode
    val_iterator = DataIterator(datasets.X_val, datasets.y_val, batch_size=16, device=device)
    
    predictions = []
    truths = []
    
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in val_iterator:
            # Forward pass
            outputs = evaluation_fn(X_batch, model)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect the predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            truths.extend(y_batch.cpu().numpy())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(truths, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(truths, predictions, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

def test_model(model, datasets, device, evaluation_fn):
    model.eval()  # Set the model to evaluation mode
    test_iterator = DataIterator(datasets.X_test, datasets.y_test, batch_size=128, device=device)
    
    predictions = []
    truths = []
    
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in test_iterator:
            # Forward pass
            outputs = evaluation_fn(X_batch, model)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect the predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            truths.extend(y_batch.cpu().numpy())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(truths, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(truths, predictions, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1
 
class DataIterator:
    def __init__(self, X, y, batch_size, device, shuffle_on_reset=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
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

def ev_net_stepped(X, net):
    out = list()
    for step in range(X.size(0)):
        x = X[step]
        o = net(x)
        out.append(o)
    out = torch.stack(out)
    out = out.transpose(0, 1)  # Reshape from (23, 16, 9) to (16, 23, 9)
    out_sum = out.sum(dim=1)       # Sum over the second dimension to get shape (16, 9)
    return out_sum

def ev_net_fullflat(X, net):
    # X is of shape (23, 128, 10)
    X_reshaped = X.transpose(0, 1).reshape(X.size(1), -1)
    # X is of shape (128, 23 * 10)
    o = net(X_reshaped)
    return o

def ev_net_full(X, net):
    # X should be of shape (seq, batch_size, features)
    o = net(X)
    return o

def ev_GatedNet(X, net):
    return net(X)

class TrainingPlots:
    def __init__(self, seq, experiment_name, plot_loss=True, plot_accuracy=True, plot_gradients=True):
        self.plot_loss = plot_loss
        self.plot_accuracy = plot_accuracy
        self.plot_gradients = plot_gradients
        self.seq = seq

        self.loss_hist = []
        self.acc_hist = []
        self.prec_hist = []
        self.recall_hist = []
        self.f1_hist = []
        self.grads_dict = {}

        # Count the number of plots to display
        self.num_plots = sum([plot_loss, plot_accuracy, plot_gradients])
        self.fig, self.axes = plt.subplots(self.num_plots, 1, figsize=(10, 5 * self.num_plots))
        
        self.fig.suptitle(experiment_name)
        self.fig.canvas.manager.set_window_title(experiment_name)


        if self.num_plots == 1:
            self.axes = [self.axes]  # Ensure axes is always a list

    def update_loss(self, loss):
        if self.plot_loss:
            self.loss_hist.append(loss)

    def update_accuracy_metrics(self, accuracy, precision, recall, f1):
        if self.plot_accuracy:
            self.acc_hist.append(accuracy)
            self.prec_hist.append(precision)
            self.recall_hist.append(recall)
            self.f1_hist.append(f1)

    def update_gradients(self, net):
        if not self.plot_gradients:
            return

        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.norm().item()
                if name in self.grads_dict:
                    self.grads_dict[name].append(grad_norm)
                else:
                    self.grads_dict[name] = [grad_norm]

    def reset_gradients(self):
        self.grads_dict = dict()

    def plot_all(self):
        plot_idx = 0  # Index to keep track of which subplot to use

        if self.plot_loss and plot_idx < self.num_plots:
            ax = self.axes[plot_idx]
            ax.clear()
            ax.plot(self.loss_hist[-self.seq:], label='Training Loss')
            ax.set_title('Training Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend()
            plot_idx += 1

        if self.plot_accuracy and plot_idx < self.num_plots:
            ax = self.axes[plot_idx]
            ax.clear()
            ax.plot(self.acc_hist, label='Accuracy')
            ax.plot(self.prec_hist, label='Precision')
            ax.plot(self.recall_hist, label='Recall')
            ax.plot(self.f1_hist, label='F1 Score')
            ax.set_title('Validation Metrics')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')
            ax.legend()
            plot_idx += 1

        if self.plot_gradients and plot_idx < self.num_plots:
            ax = self.axes[plot_idx]
            ax.clear()
            for name, grad_history in self.grads_dict.items():
                ax.plot(grad_history[-self.seq:], label=f'{name} grad')
            ax.set_title('Gradient Norms')
            ax.legend()
            #max_grad = max(self.grads_dict[name][-self.seq:] for name in self.grads_dict)
            #ax.set_ylim([0, max_grad])

        self.fig.canvas.draw()
        plt.pause(0.1)

def train_baseline(datasets):

    fig_loss, ax_loss = plt.subplots()
    fig_acc, ax_acc = plt.subplots()
    fig_grads, ax_grads = plt.subplots()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #net = BaselineNet()
    net = BaselineNetFull()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  

    loss_hist = list()
    acc_hist = list()
    prec_hist = list()
    recall_hist = list()
    f1_hist = list()
    batch_size = 128 
    num_epochs = 100

    train_iter = DataIterator(datasets.X_train, datasets.y_train, batch_size, device, shuffle_on_reset=True)
    test_iter = DataIterator(datasets.X_val, datasets.y_val, batch_size, device)
    
    grad_plots = GradPlots()
    

    for epoch in range(num_epochs):
        if epoch != 0:
            train_iter.reset()
            #scheduler.step()
        with tqdm(total=len(train_iter), desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True) as pbar:
            
            for i, (X, y) in enumerate(train_iter):
                pbar.set_postfix_str(f"i={i}")
                pbar.update(1)

                net.train()
                #out_sum = ev_net_stepped(X, net)
                out = ev_net_fullflat(X, net)
                loss = loss_fn(out, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                #print("loss: ", loss.item())
                loss_hist.append(loss.item())
                grad_plots.update(net)

            grad_plots.plot(ax_grads, fig_grads)
            grad_plots.reset()

            ax_loss.clear()  # Clear the loss axis
            ax_loss.plot(loss_hist, label='Training Loss')  # Plot the training loss
            ax_loss.set_title('Training Loss')
            ax_loss.set_xlabel('Iteration')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            fig_loss.canvas.draw()  # Redraw the loss figure
            plt.pause(0.01)  # Pause for a short period to update the plot

            """
            plt.subplot(1, 2, 1)  # Prepare subplot for loss plot
            plt.clf()  # Clear the current figure
            plt.plot(loss_hist, label='Training Loss')  # Plot the training loss
            plt.title('Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.tight_layout()  # Adjust subplots to fit into figure area.
            plt.draw()  # Redraw the current figure
            plt.pause(0.01)  # Pause for a short period to update the plot
            """

            #if len(loss_hist) >= 2000:
            #    avg_last_1000 = sum(loss_hist[-1000:]) / 1000
            #    avg_first_1000 = sum(loss_hist[:1000]) / 1000
            #    if avg_last_1000 <= 100 * avg_first_1000:
            #        del loss_hist[:1000]
            #        # Adjust the x-axis to keep the number
            loss_hist = list()

            accuracy, precision, recall, f1 = evaluate_model(net, datasets, device, ev_net_fullflat)
            print(f"Validation - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
            acc_hist.append(accuracy)
            prec_hist.append(precision)
            recall_hist.append(recall)
            f1_hist.append(f1)

            ax_acc.clear()  # Clear the accuracy axis
            ax_acc.plot(acc_hist, label='Accuracy')  # Plot the validation accuracy
            ax_acc.plot(prec_hist, label='Precission')  # Plot the validation accuracy
            ax_acc.plot(recall_hist, label='Recall')  # Plot the validation recall
            ax_acc.plot(f1_hist, label='F1 Score')  # Plot the validation F1 score
            
            ax_acc.set_title('Validation')
            ax_acc.set_xlabel('Iteration')
            ax_acc.set_ylabel('y')
            ax_acc.legend()
            fig_acc.canvas.draw()  # Redraw the accuracy figure
            plt.pause(0.01)  # Pause for a short period to update the plot
    #plt.show()
    # test
    accuracy, precision, recall, f1 = test_model(net, datasets, device, ev_net_fullflat)
    print(f"Test - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
 

def train_lstm(datasets):

    experiment_name = "3layers, hs 64, dr_lstm 0.1, epoch 150, bs 128, all_hs 3fc * 12"
    input_size = 10
    num_classes = 9
    hidden_size = 64
    num_layers = 3
    lr=0.002
    dr_lstm = 0.1
    dr_fc = 0.3
    batch_size = 128
    num_epochs = 150

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    net = LSTMNet(input_size, hidden_size, num_layers, num_classes, dr_lstm, dr_fc, 23).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  

    

    train_iter = DataIterator(datasets.X_train, datasets.y_train, batch_size, device, shuffle_on_reset=True)
    #test_iter = DataIterator(datasets.X_val, datasets.y_val, batch_size, device)
    
    plots = TrainingPlots(seq=len(train_iter), experiment_name=experiment_name)    

    for epoch in range(num_epochs):
        if epoch != 0:
            train_iter.reset()
            #scheduler.step()
        with tqdm(total=len(train_iter), desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True) as pbar:
            
            for i, (X, y) in enumerate(train_iter):
                pbar.set_postfix_str(f"i={i}")
                pbar.update(1)

                net.train()
                #out_sum = ev_net_stepped(X, net)
                out = ev_net_full(X, net)
                loss = loss_fn(out, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                

                plots.update_loss(loss.item())
                #print("loss: ", loss.item())
                #loss_hist.append(loss.item())
                plots.update_gradients(net)

            accuracy, precision, recall, f1 = evaluate_model(net, datasets, device, ev_net_full)
            print(f"Validation - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
            plots.update_accuracy_metrics(accuracy, precision, recall, f1)
            plots.plot_all()

    #plt.show()
    # test
    plt.show()
    accuracy, precision, recall, f1 = test_model(net, datasets, device, ev_net_full)
    print(f"Test - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
 

def custom_min_max_normalize_np(array, min_val=-255, max_val=255):
    """
    Normalize a NumPy array with known min and max values to the range [0, 1].
    
    Args:
    array (np.ndarray): The NumPy array to normalize.
    min_val (float): The minimum value in the original range of the array.
    max_val (float): The maximum value in the original range of the array.

    Returns:
    np.ndarray: A NumPy array with values normalized to [0, 1].
    """
    # Normalize array to the [0, 1] range using the known min and max
    normalized_array = (array - min_val) / (max_val - min_val)
    
    return normalized_array

def train_gated(datasets):

    experiment_name = "gated 3 layers, hs * 3, dr 0.2, epoch 100, bs 128, lr 0.002"
    input_size = 10
    num_classes = 9
    lr=0.001
    batch_size = 128
    num_epochs = 100

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    net = GatedNet(device, batch_size, input_size, num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  

    

    train_iter = DataIterator(datasets.X_train, datasets.y_train, batch_size, device, shuffle_on_reset=True)
    #test_iter = DataIterator(datasets.X_val, datasets.y_val, batch_size, device)
    
    plots = TrainingPlots(seq=len(train_iter), experiment_name=experiment_name)    

    for epoch in range(num_epochs):
        if epoch != 0:
            train_iter.reset()
            #scheduler.step()
        with tqdm(total=len(train_iter), desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True) as pbar:
            
            for i, (X, y) in enumerate(train_iter):
                pbar.set_postfix_str(f"i={i}")
                pbar.update(1)

                net.train()
                #out_sum = ev_net_stepped(X, net)
                out = net(X)
                loss = loss_fn(out, y)
                optimizer.zero_grad()
                loss.backward()

                """
                 # Print all gradients
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        print(f"Gradient of {name} is {param.grad}")
                """
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                

                plots.update_loss(loss.item())
                #print("loss: ", loss.item())
                #loss_hist.append(loss.item())
                plots.update_gradients(net)

            accuracy, precision, recall, f1 = evaluate_model(net, datasets, device, ev_GatedNet)
            print(f"Validation - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
            plots.update_accuracy_metrics(accuracy, precision, recall, f1)
            plots.plot_all()

    #plt.show()
    # test
    plt.show()
    accuracy, precision, recall, f1 = test_model(net, datasets, device, ev_GatedNet)
    print(f"Test - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
 


def main():
    datasets = load_data()

    
    #train_baseline(datasets)
    #train_lstm(datasets)
    train_gated(datasets)



if __name__ == "__main__":
    main()