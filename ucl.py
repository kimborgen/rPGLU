# imports
#import snntorch as snn
#from snntorch import surrogate
#from snntorch import backprop
#from snntorch import functional as SF
#from snntorch import utils
#from snntorch import spikeplot as splt

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

from tslearn.datasets import extract_from_zip_url



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
        activated = F.normalize(x)

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

        

    def forward(self, data):
        # Define the forward pass here
        self.lg1.reset()
        self.lg2.reset()
        self.lg3.reset()

        out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps
            # flatten
            #x = data[step].view(data[step].size(0), -1)

            x = data[step]
            y = F.normalize(x)
                        
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

            out.append(o)

        stacked = torch.stack(out)
        x = stacked.sum(dim=0)
        
        return x

def main():



if __name__ == "__main__":
    main()
        

        
