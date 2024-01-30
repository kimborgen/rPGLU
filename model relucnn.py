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

from torchviz import make_dot, make_dot_from_trace

import time

from torch.utils.tensorboard import SummaryWriter

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


class LinearGated(nn.Module):
    def __init__(self, device, batch_size, out, initial_tresh, initial_decay_rate, spike_decay_rate):
        super(LinearGated, self).__init__()
        self.device = device
        #self.inp = inp
        self.out = out
        self.initial_tresh = initial_tresh
        self.initial_decay_rate = initial_decay_rate
        #self.spike_decay_rate = spike_decay_rate
        self.batch_size = batch_size


        #self.fc = nn.Linear(inp, out)
          #add device
        #self.reset()
        mu, sigma = 0, 0.04  # For threshold
        mu_decay, sigma_decay = 0.4, 0.5  # For decay rate
        mu_spike, sigma_spike = 0, -1.0
        self.tresh = nn.Parameter(torch.abs(torch.rand(size=(self.out,), device=self.device, requires_grad=True)) * sigma + mu)
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(self.out,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)
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

class NormalNet(nn.Module):
    def __init__(self, device, batch_size, inp, out, tresh, decay, spike):
        super(NormalNet, self).__init__()

        self.batch_size = batch_size
        self.inp = inp 
        self.out = out
        self.device = device
        self.dropout = 0.2
        self.lg1_s = 12 * 15 * 15
        self.lg2_s = 32 * 5 * 5 

        hs = inp * 3
        self.d0 = nn.Dropout(self.dropout)
        self.c1 = nn.Conv2d(2, 12, 5)
        self.m1 = nn.MaxPool2d(2)
        # m1 shape (batch_size, 12, (32-4)/2, (32-4)/2)
        #self.d1 = nn.Dropout(self.dropout)
        #self.lg1 = LinearGated(device, batch_size, self.lg1_s, tresh, decay, spike)

        self.c2 = nn.Conv2d(12, 32, 5)
        self.m2 = nn.MaxPool2d(2)
        #self.d2 = nn.Dropout(self.dropout)
        #self.lg2 = LinearGated(device, batch_size, self.lg2_s, tresh, decay, spike)
        #self.flat = nn.Flatten()

        self.fc1 = nn.Linear(self.lg2_s, self.lg2_s)
        self.d3 = nn.Dropout(self.dropout)
        #self.lg3 = LinearGated(device, batch_size, 10, tresh,decay,spike)

        #self.d1 = nn.Dropout(0.3)
        #self.lg1 = LinearGated(device, batch_size,  inp, hs, tresh, decay, spike)
        #self.d2 = nn.Dropout(0.3)
        #self.lg2 = LinearGated(device, batch_size, hs, hs, tresh, decay, spike)
        #self.d3 = nn.Dropout(0.3)
        #self.lg3 = LinearGated(device, batch_size, hs, out, tresh, decay, spike)
        self.fc2 = nn.Linear(self.lg2_s * 2, 10)

        

    def forward(self, data):
        # Define the forward pass here


        #pot1 = torch.zeros((self.batch_size, 12 * 15 * 15), device=self.device, requires_grad=True)
        #pot2 = torch.zeros((self.batch_size, 32 * 5 * 5), device=self.device, requires_grad=True)
        #pot1 = torch.zeros((self.batch_size, 10), device=self.device, requires_grad=True)


        out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps
            # flatten
            #x = data[step].view(data[step].size(0), -1)

            x = data[step]
            y = self.d0(x)
                        
            c1 = self.c1(y)
            m1 = self.m1(c1)
            r1 = F.relu(m1)
            #m1_flat = m1.view(m1.size(0), -1)
            #norm1 = F.normalize(m1_flat)
            #d1 = self.d1(norm1)
            #lg1, pot1 = self.lg1(d1, pot1)
            #lg1_unflatten = lg1.view(m1.size())

            c2 = self.c2(r1)
            m2 = self.m2(c2)
            r2 = F.relu(m2)
            v = r2.view(m2.size(0), -1)

            #combined = torch.cat((lg1, lg2), dim=1)
            
            fc1 = self.fc1(v)
            #norm3 = F.normalize(fc1)
            #d3 = self.d3(norm3)
            #o, pot1 = self.(fc1, pot1)
            r3 = F.relu(fc1)

            combined = torch.cat((v, r3), dim=1)
            fc2 = self.fc2(combined)
            r4 = F.relu(fc2)

            out.append(r4)

        stacked = torch.stack(out)
        x = stacked.sum(dim=0)
        
        return x


class Net(nn.Module):
    def __init__(self, device, batch_size, inp, out, tresh, decay, spike):
        super(Net, self).__init__()

        self.batch_size = batch_size
        self.inp = inp 
        self.out = out
        self.device = device
        self.dropout = 0.2
        self.lg1_s = 12 * 15 * 15
        self.lg2_s = 32 * 5 * 5 

        hs = inp * 3
        self.d0 = nn.Dropout(self.dropout)
        self.c1 = nn.Conv2d(2, 12, 5)
        self.m1 = nn.MaxPool2d(2)
        # m1 shape (batch_size, 12, (32-4)/2, (32-4)/2)
        #self.d1 = nn.Dropout(self.dropout)
        #self.lg1 = LinearGated(device, batch_size, self.lg1_s, tresh, decay, spike)

        self.c2 = nn.Conv2d(12, 32, 5)
        self.m2 = nn.MaxPool2d(2)
        #self.d2 = nn.Dropout(self.dropout)
        #self.lg2 = LinearGated(device, batch_size, self.lg2_s, tresh, decay, spike)
        #self.flat = nn.Flatten()

        self.fc1 = nn.Linear(self.lg2_s, 10)
        self.d3 = nn.Dropout(self.dropout)
        self.lg3 = LinearGated(device, batch_size, 10, tresh,decay,spike)

        #self.d1 = nn.Dropout(0.3)
        #self.lg1 = LinearGated(device, batch_size,  inp, hs, tresh, decay, spike)
        #self.d2 = nn.Dropout(0.3)
        #self.lg2 = LinearGated(device, batch_size, hs, hs, tresh, decay, spike)
        #self.d3 = nn.Dropout(0.3)
        #self.lg3 = LinearGated(device, batch_size, hs, out, tresh, decay, spike)

        

    def forward(self, data):
        # Define the forward pass here


        #pot1 = torch.zeros((self.batch_size, 12 * 15 * 15), device=self.device, requires_grad=True)
        #pot2 = torch.zeros((self.batch_size, 32 * 5 * 5), device=self.device, requires_grad=True)
        pot1 = torch.zeros((self.batch_size, 10), device=self.device, requires_grad=True)


        out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps
            # flatten
            #x = data[step].view(data[step].size(0), -1)

            x = data[step]
            y = self.d0(x)
                        
            c1 = self.c1(y)
            m1 = self.m1(c1)
            r1 = F.relu(m1)
            #m1_flat = m1.view(m1.size(0), -1)
            #norm1 = F.normalize(m1_flat)
            #d1 = self.d1(norm1)
            #lg1, pot1 = self.lg1(d1, pot1)
            #lg1_unflatten = lg1.view(m1.size())

            c2 = self.c2(r1)
            m2 = self.m2(c2)
            r2 = F.relu(m2)
            v = r2.view(m2.size(0), -1)

            #combined = torch.cat((lg1, lg2), dim=1)
            
            fc1 = self.fc1(v)
            #norm3 = F.normalize(fc1)
            #d3 = self.d3(norm3)
            o, pot1 = self.lg3(fc1, pot1)

            out.append(o)

        stacked = torch.stack(out)
        x = stacked.sum(dim=0)
        
        return x
    
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

def forward_pass(net, data):  
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(data.size(0)):  # data.size(0) = number of time steps
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)
  
  return torch.stack(spk_rec)

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
    tresh = 0.0
    decay = 0.7
    spike = 0.001

    #net = NormalNet(device, batch_size, inp, out, tresh, decay, spike)
    # neuron and simulation parameters
    spike_grad = surrogate.atan()
    beta = 0.5

    #  Initialize Network
    net = nn.Sequential(nn.Conv2d(2, 12, 5),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(12, 32, 5),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(32*5*5, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)

    #optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002, betas=(0.9, 0.999))
    #loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    #loss_fn = nn.CrossEntropyLoss()
    #loss_fn = SmoothCrossEntropyLoss(smoothing=0.2)
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    num_epochs = 10
    num_iters = 40

    loss_hist = []
    acc_hist = []

    print(net)

    writer = SummaryWriter()

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)
    #prof.start()


    # training loop
    for epoch in range(num_epochs):
        mean_train_loss = 3
        for i, (data, targets) in enumerate(iter(trainloader)):
            #prof.step()
            data = data.to(device)
            targets = targets.to(device)
            #targets = (1 - 0.1) * targets + 0.1 / 10
            net.train()
            #net.lg1.train()
            #net.lg2.train()
            #net.lg3.train()
            #spk_rec = net(data)
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)
            mean_train_loss = (mean_train_loss + loss_val.item()) / (i + 1)
            
            #dot = make_dot(spk_rec.mean(), params=dict(net.named_parameters()), show_attrs=True, show_saved=True)
            #dot.render("computational graph", format="png")
            #dot.view()
            # Calculate uniqueness of spk_rec
            #max_val = torch.max(spk_rec)
            #uniqueness = torch.sum((spk_rec == max_val).float()) / spk_rec.numel()
            # Multiply loss_val with uniqueness
            #loss_adj = loss_val * uniqueness

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()

            # Log gradients to TensorBoard
            #for name, parameter in net.named_parameters():
            #    if parameter.grad is not None:
            #        writer.add_scalar(f'Gradients/{name}', parameter.grad.norm(), epoch * len(trainloader) + i)

            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
    
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            print(net.named_parameters())

            for name, parameter in net.named_parameters():
                if parameter.grad is not None:
                    print(f"{name} gradient: {parameter.grad.norm().item()}")

            #acc = SF.accuracy_rate(spk_rec, targets) 
            #acc_hist.append(acc)
            #print(f"Accuracy: {acc * 100:.2f}%\n")

            # This will end training after 50 iterations by default\\
            plt.clf()  # Clear the current figure
            plt.plot(loss_hist)  # Plot the updated data
            plt.draw()  # Redraw the current figure
            plt.pause(0.1)  # Pause for a short period to update the plot

            sm = F.softmax(spk_rec)
            
            if i == num_iters:
                break
        b = False
        if b:
            break
        #if mean_train_loss < 2:
        #        break
    
    torch.no_grad()
    hits = 0
    net.eval()
    for i, (data, targets) in enumerate(iter(trainloader)):
        data = data.to(device)
        targets = targets.to(device)
        spk_rec = forward_pass(net, data)
        spk_rec = spk_rec.sum(dim=0).unsqueeze(0)

        loss_val = loss_fn(spk_rec, targets) 
        predicted_label = torch.argmax(spk_rec)
        hit = predicted_label == targets
        if hit:
            hits += 1

        if i == num_iters:
            break

    print(f"hits {hits} of {num_iters}")

    prof.stop()
 


if __name__ == "__main__":
    main()
        

        
