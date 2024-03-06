import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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



def main():
    datasets = load_data()

    
    #train_baseline(datasets)
    #train_lstm(datasets)
    #train_gated(datasets)



if __name__ == "__main__":
    main()