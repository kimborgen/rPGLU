from aeon.datasets import load_classification
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    
    print(" Shape of X_train = ", X_train.shape)
    print(" Shape of y_train = ", y_train.shape)

    print(" Shape of X_test = ", X_test.shape)
    print(" Shape of y_test = ", y_test.shape)

    print(" Shape of X_val = ", X_val.shape)
    print(" Shape of y_val = ", y_val.shape)

  

    datasets = Dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    return datasets

def evaluate_model(model, datasets, device):
    model.eval()  # Set the model to evaluation mode
    val_iterator = DataIterator(datasets.X_val, datasets.y_val, batch_size=16, device=device)
    
    predictions = []
    truths = []
    
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in val_iterator:
            # Forward pass
            outputs = ev_net_stepped(X_batch, model)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect the predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            truths.extend(y_batch.cpu().numpy())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(truths, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(truths, predictions, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

def test_model(model, datasets, device):
    model.eval()  # Set the model to evaluation mode
    test_iterator = DataIterator(datasets.X_test, datasets.y_test, batch_size=128, device=device)
    
    predictions = []
    truths = []
    
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in test_iterator:
            # Forward pass
            outputs = ev_net_stepped(X_batch, model)
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

class GradPlots:
    def __init__(self):
        self.grads_dict = {}

    def update(self, net):
        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.norm().item()
                if name in self.grads_dict:
                    self.grads_dict[name].append(grad_norm)
                else:
                    self.grads_dict[name] = [grad_norm]

    def plot(self, ax, fig):
        for name, grad_history in self.grads_dict.items():
            ax.plot(grad_history, label=f'{name} grad')
        ax.set_title('Gradient Norms')
        ax.legend()
        fig.canvas.draw()  # Redraw the accuracy figure
        plt.pause(0.01)  # Pause for a short period to update the plot

        max_grad = max(max(self.grads_dict[name][-607:]) for name in self.grads_dict)
        ax.set_ylim([0, max_grad])


def train_baseline(datasets):

    fig_loss, ax_loss = plt.subplots()
    fig_acc, ax_acc = plt.subplots()
    fig_grads, ax_grads = plt.subplots()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = BaselineNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  

    loss_hist = list()
    acc_hist = list()
    prec_hist = list()
    recall_hist = list()
    f1_hist = list()
    batch_size = 128 
    num_epochs = 60

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
                out_sum = ev_net_stepped(X, net)
                loss = loss_fn(out_sum, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                #print("loss: ", loss.item())
                loss_hist.append(loss.item())
                grad_plots.update(net)

            grad_plots.plot(ax_grads, fig_grads)

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

            if len(loss_hist) >= 2000:
                avg_last_1000 = sum(loss_hist[-1000:]) / 1000
                avg_first_1000 = sum(loss_hist[:1000]) / 1000
                if avg_last_1000 <= 100 * avg_first_1000:
                    del loss_hist[:1000]
                    # Adjust the x-axis to keep the number

            accuracy, precision, recall, f1 = evaluate_model(net, datasets, device)
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
    accuracy, precision, recall, f1 = test_model(net, datasets, device)
    print(f"Test - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
 



def main():
    datasets = load_data()

    
    train_baseline(datasets)




if __name__ == "__main__":
    main()