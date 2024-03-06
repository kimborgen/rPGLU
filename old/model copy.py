# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
#from torchsummary import summary
import os
from tqdm import tqdm
import torch.nn.functional as torchF
import matplotlib.pyplot as plt
import random

import tonic 
import torchvision
import tonic.transforms as transforms


from torch.utils.data import DataLoader
from tonic import DiskCachedDataset



class LinearGated(nn.Module):
    def __init__(self, device, batch_size, out, tresh, decay_rate, spike_decay_rate):
        super().__init__()
        self.device = device
        #self.inp = inp
        self.out = out
        self.tresh = tresh
        self.decay_rate = decay_rate
        self.spike_decay_rate = spike_decay_rate
        self.batch_size = batch_size

        #self.fc = nn.Linear(inp, out)
          #add device

    
    def reset(self):
        self.potential = torch.zeros(self.batch_size, self.out, requires_grad=True, device=self.device)

    def forward(self, x):

        #normed = F.normalize(x)
        # linear
        # lin = self.fc(x)
        # linear negative

        # sigmoid to normalize between 0 and 1.
        # activated = F.sigmoid(x)
        #activated = F.relu(x)
        activated = F.tanh(x)

        # calculate new potential
        potential = self.potential + activated

        # Zero those where potential < tresh
        gated_bool = potential > self.tresh
        gated = potential * gated_bool    
    
        # reduce the potential of the open gates with spike_decay_rate
        post_gated = gated * self.spike_decay_rate

        # reduce the potential of the closed gates with decay_rate
        non_gated_bool = ~gated_bool # negation operator
        non_gated = potential * non_gated_bool
        post_non_gated = non_gated * self.decay_rate

        # now combine the two to the new potential 
        new_potential = post_gated + post_non_gated
        self.potential = new_potential

        return gated

class Net(nn.Module):
    def __init__(self, device, batch_size, inp, out, tresh, decay, spike):
        super(Net, self).__init__()

        hs = inp * 3
        self.d0 = nn.Dropout(0.2)
        self.c1 = nn.Conv2d(2, 12, 5)
        self.m1 = nn.MaxPool2d(2)
        # m1 shape (batch_size, 12, (32-4)/2, (32-4)/2)
        self.d1 = nn.Dropout(0.2)
        self.lg1 = LinearGated(device, batch_size, 12 * 15 * 15, tresh, decay, spike)

        self.c2 = nn.Conv2d(12, 32, 5)
        self.m2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout(0.2)
        self.lg2 = LinearGated(device, batch_size, 32 * 5 * 5, tresh, decay, spike)
        #self.flat = nn.Flatten()

        self.fc1 = nn.Linear(32 * 5 * 5, 10)
        self.d3 = nn.Dropout(0.2)
        self.lg3 = LinearGated(device, batch_size, 10, tresh,decay,spike)

        #self.d1 = nn.Dropout(0.3)
        #self.lg1 = LinearGated(device, batch_size,  inp, hs, tresh, decay, spike)
        #self.d2 = nn.Dropout(0.3)
        #self.lg2 = LinearGated(device, batch_size, hs, hs, tresh, decay, spike)
        #self.d3 = nn.Dropout(0.3)
        #self.lg3 = LinearGated(device, batch_size, hs, out, tresh, decay, spike)


    def reset(self):
        self.lg1.reset()
        self.lg2.reset()
        self.lg3.reset()

    def forward(self, data):
        # Define the forward pass here
    
        #x = data[step].view(data[step].size(0), -1)

        x = data
        y = self.d0(x)
                    
        c1 = self.c1(y)
        m1 = self.m1(c1)
        m1_flat = m1.view(m1.size(0), -1)
        d1 = self.d1(m1_flat)
        lg1 = self.lg1(d1)
        lg1_unflatten = lg1.view(m1.size())

        c2 = self.c2(lg1_unflatten)
        m2 = self.m2(c2)
        m2_flat = m2.view(m2.size(0), -1)
        d2 = self.d2(m2_flat)
        lg2 = self.lg2(d2)
        
        fc1 = self.fc1(lg2)
        d3 = self.d3(fc1)
        o = self.lg3(d3)

        return o

def main():

    sensor_size = tonic.datasets.NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                        transforms.ToFrame(sensor_size=sensor_size, 
                                                            time_window=1000)
                                        ])

    trainset = tonic.datasets.NMNIST(save_to='./tmp/data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./tmp/data', transform=frame_transform, train=False)


    transform = tonic.transforms.Compose([torch.from_numpy,
                                        torchvision.transforms.RandomRotation([-10,10])])

    cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')

    # no augmentations for the testset
    cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

    batch_size = 1
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    inp = 2312
    out = 10
    #lg = LinearGated(device, inp, out, 0.5, 0.1, 0.5)
    tresh = 0.05
    decay = 0.7
    spike = 0.001

    net = Net(device, batch_size, inp, out, tresh, decay, spike)

    #optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00002, betas=(0.9, 0.999))
    #loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 500
    num_iters = 50

    loss_hist = []
    acc_hist = []
    torch.autograd.set_detect_anomaly(True)
    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)

            out = []

            net.reset()

            for step in range(data.size(0)):  # data.size(0) = number of time steps
                # flatten
                net.train()
                o = net(data[step])
                loss_val = loss_fn(o, targets)
                optimizer.zero_grad()
                loss_val.backward(retain_graph=True)
                optimizer.step()
                #out.append(o)
                loss_hist.append(loss_val.item())
    
                print(f"Epoch {epoch}, Iteration {i}, Step {step} \nTrain Loss: {loss_val.item():.2f}")

                for name, parameter in net.named_parameters():
                    if parameter.grad is not None:
                        print(f"{name} gradient: {parameter.grad.norm().item()}")

            #stacked = torch.stack(out)
            #x = stacked.sum(dim=0)
        

            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            #optimizer.step()

            # Store loss history for future plotting
            

            #acc = SF.accuracy_rate(spk_rec, targets) 
            #acc_hist.append(acc)
            #print(f"Accuracy: {acc * 100:.2f}%\n")

            # This will end training after 50 iterations by default\\
            plt.clf()  # Clear the current figure
            plt.plot(loss_hist)  # Plot the updated data
            plt.draw()  # Redraw the current figure
            plt.pause(0.1)  # Pause for a short period to update the plot

            #sm = F.softmax(spk_rec)
            
            if i == 10:
                break



if __name__ == "__main__":
    main()
        

        
