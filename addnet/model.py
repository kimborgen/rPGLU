import sys
from pathlib import Path
parent_folder_path = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import print_params
from dataclasses import dataclass
from experiment_helper import ExperimentParams
from training import train_model
from sglu.components import InvertedBumpFunction, BumpFunction

class LinearAdd(nn.Module):
    def __init__(self, params, device, inp, out):
        super(LinearAdd, self).__init__()
        self.params = params
        self.device = device 

        self.lin = nn.Linear(inp, out)

    def forward(self, x, h):
        fc = self.lin(x)
        h_n = h + fc
        out = F.tanh(h_n)
        return out, h_n        

class LinearAddDecay(nn.Module):
    def __init__(self, params, device, inp, out):
        super(LinearAddDecay, self).__init__()
        self.params = params
        self.device = device 

        self.lin = nn.Linear(inp, out)
        sigma_decay = 0.7
        mu_decay = 0.1
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(out,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)

    def forward(self, x, h):
        fc = self.lin(x)
        h_d = h * self.decay_rate
        h_n = h_d + fc
        out = F.tanh(h_n)
        return out, h_n

class DoubleLinearDecay(nn.Module):
    def __init__(self, params, device, inp, out):
        super(DoubleLinearDecay, self).__init__()
        self.params = params
        self.device = device 

        self.lin = nn.Linear(inp + out, out)
        sigma_decay = 0.7
        mu_decay = 0.1
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(out,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)

    def forward(self, x, h):
        h_d = h * self.decay_rate
        all_together_now = torch.cat((x, h_d), dim=1)
        fc = self.lin(all_together_now)
        h_n = h_d + fc
        out = F.tanh(h_n)
        return out, h_n
    
class LinearGated(nn.Module):
    def __init__(self, params, device, width):
        super(LinearGated, self).__init__()
        self.device = device
        self.num_out = params.output_size
        self.batch_size = params.batch_size

        mu, sigma = 0, 0.3  # For threshold
        mu_decay, sigma_decay = 0.1, 0.7  # For decay rate
        self.tresh = nn.Parameter(torch.abs(torch.rand(size=(width,), device=self.device, requires_grad=True)) * sigma + mu)
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(width,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)

    def forward(self, x, potential): 

        # calculate new potential
        potential = potential + x

        treshed = potential - self.tresh

        # Zero those where potential < tresh
        gated = F.relu(treshed)
        
        # now apply this gated filter to the potential
        gated_bool = InvertedBumpFunction.apply(gated)

        # This is now what we return from this layer
        activated = gated_bool * potential

        # Now we wish to reduce the potential of the open gates with spike_decay_rate

        # reduce the potential of the closed gates with decay_rate
        non_gated_bool = BumpFunction.apply(gated)
        non_gated = potential * non_gated_bool
        potential_non_gated = non_gated * self.decay_rate
    
        return activated, potential_non_gated
    
class InhibitNet(nn.Module):
    def __init__(self, params, device, inp, out):
        super(InhibitNet, self).__init__()
        self.device = device
        self.params = device
        self.lin = nn.Linear(inp, out)
        self.inp = inp
        self.out = out

        mu = 0.7 # center
        sigma = 0.2  # variance from center
        self.tresh = nn.Parameter(torch.abs(torch.rand(size=(out,), device=self.device, requires_grad=True)) * sigma + mu) 
    def forward(self, x): 
        # nominal ReLU forward
        fc = self.lin(x)

        # nominal activation function
        activated = F.sigmoid(fc)

        # treshold act
        treshed = activated - self.tresh

        # Zero those where activated < tresh
        gated = F.relu(treshed)
        
        # Squish to 0 positive values, 1 negative, bump function backwards
        gated_bool = BumpFunction.apply(gated)
    
        return activated, gated_bool

class InhibitAddDecay(nn.Module):
    def __init__(self, params, device, inp, out):
        super(InhibitAddDecay, self).__init__()
        self.params = params
        self.device = device 

        self.lin = nn.Linear(inp, out)
        self.inhibit = InhibitNet(params, device, inp, out)

        sigma_decay = 0.7
        mu_decay = 0.1
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(out,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)

    def forward(self, x_a, h, x_i):
        # Add network
        fc = self.lin(x_a)

        # Inhibit network        
        x_i_next, inhibit = self.inhibit(x_i)

        # inhibit h
        h_i = h * inhibit

        # decay remaining h
        h_decayed = h_i * self.decay_rate
        
        # combine
        h_next = h_decayed + fc

        # activate
        out = F.tanh(h_next)

        return out, h_next, x_i_next


class InhibitAddDecayExpert(nn.Module):
    def __init__(self, params, device, inp, out):
        super(InhibitAddDecay, self).__init__()
        self.params = params
        self.device = device 

        self.lin = nn.Linear(inp, out)
        self.inhibit = InhibitNet(params, device, inp, out)

        sigma_decay = 0.7
        mu_decay = 0.1
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(out,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)

    def forward(self, x_a, h, x_i):
        # Add network
        fc = self.lin(x_a)

        # Inhibit network        
        x_i_next, inhibit = self.inhibit(x_i)

        # inhibit h
        h_i = h * inhibit

        # decay remaining h
        h_decayed = h_i * self.decay_rate
        
        # combine
        h_next = h_decayed + fc

        # activate
        out = F.tanh(h_next)

        return out, h_next, x_i_next

class AddNet(nn.Module):
    def __init__(self, params, device):
        super(AddNet, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.dropout = 0.2
        self.hs = params.hidden_size

        self.d = nn.Dropout(params.dr)
        self.fc1 = InhibitAddDecay(params, device, self.inp, self.hs)
        self.fc2 = InhibitAddDecay(params, device, self.hs, self.hs)
        self.fc3 = nn.Linear(self.hs, self.out)   


    def forward(self, data):
        # Define the forward pass here


        h1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        h2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #h3 = torch.zeros((data.size(1), self.out), device=self.device, requires_grad=True) 

        out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps

            x = data[step]

            an1, h1, inh1 = self.fc1(x, h1, x)
            d1 = self.d(an1)
            di1 = self.d(inh1)
            an2, h2, inh2 = self.fc2(d1, h2, di1)
            d2 = self.d(an2)
            #di2 = self.d(inh2)
            fc3 = self.fc3(d2)

            out.append(fc3)
        
        return out[-1]
    
def ev_AddNet(X, net):
    return net(X)

@dataclass
class AddNetParams(ExperimentParams):
    hidden_size: int
    dr: float
    weight_decay: float

def get_sglstm_params():
    return AddNetParams(
        model_id="addnet",
        short_name="addnet",
        description="AddNet, 3 layer, last layer output size",
        input_size=10,
        output_size=9,
        lr=0.01,
        batch_size=128, 
        num_epochs=100,
        hidden_size=64,
        clip_grad_norm=1.0,
        dr=0.2,
        weight_decay=1e-05
    )

def train_addNet():
    params = get_sglstm_params()  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = AddNet(params, device).to(device)
    print_params(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    train_model(params, device, net, loss_fn, optimizer, ev_AddNet, print_grads=True)

if __name__=="__main__":
    train_addNet()