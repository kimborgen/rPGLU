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

class SGLUFFCNNet(nn.Module):
    def __init__(self, params, device):
        super(SGLUFFCNNet, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.dropout = 0.2
        self.hs = params.hidden_size

        self.fc1 = nn.Linear(self.inp, self.hs)
        self.norm1 = nn.LayerNorm(self.hs)
        self.lg1 = LinearGated(self.device, self.batch_size, self.hs)
        self.d1 = nn.Dropout(self.dropout)

        self.fc2 = nn.Linear(self.hs,self.hs)
        self.norm2 = nn.LayerNorm(self.hs)
        self.lg2 = LinearGated(self.device, self.batch_size, self.hs)
        self.d2 = nn.Dropout(self.dropout)

        self.fc3 = nn.Linear(self.hs, self.hs)


        si = self.hs * 23 # num timesteps
        self.fd1 = nn.Dropout(self.dropout)
        self.ffc = nn.Linear(si, si * 2)
        self.fd2 = nn.Dropout(self.dropout)
        self.ffc2 = nn.Linear(si *2, self. out)



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

        stacked = torch.stack(out)
        # Step 1: Transpose to make 'batch_size' the first dimension
        tensor_transposed = stacked.transpose(0, 1)  # New shape: (batch_size, timesteps, features)
        # Step 2: Reshape to concatenate 'timesteps' and 'features'
        gnn_out = tensor_transposed.reshape(tensor_transposed.shape[0], 23 * self.hs)
        fd1 = self.fd1(gnn_out)
        ffc1 = self.ffc(fd1)
        fr1 = F.relu(ffc1)
        fd2 = self.fd2(fr1)
        ffc2 = self.ffc2(fd2) 
        
        return ffc2
    
class SGLUNet(nn.Module):
    def __init__(self, params, device):
        super(SGLUNet, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.dropout = 0.2
        self.hs = params.hidden_size

        self.fc1 = nn.Linear(self.inp, self.hs)
        self.norm1 = nn.LayerNorm(self.hs)
        self.lg1 = LinearGated(self.device, self.batch_size, self.hs)
        self.d1 = nn.Dropout(self.dropout)

        self.fc2 = nn.Linear(self.hs,self.hs)
        self.norm2 = nn.LayerNorm(self.hs)
        self.lg2 = LinearGated(self.device, self.batch_size, self.hs)
        self.d2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(self.hs, self.out)

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

        return out[-1]
    
class AR_SGLUNet(nn.Module):
    def __init__(self, params, device):
        super(AR_SGLUNet, self).__init__() 

        self.params = params
        self.device = device
        self.hs = params.hidden_size * 2

        self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        self.lg1 = LinearGated(params, device, params.hidden_size)
        self.d1 = nn.Dropout(params.dr)

        self.fc2 = nn.Linear(self.hs, self.hs)
        #self.norm2 = nn.LayerNorm(params.hidden_size)
        self.lg2 = LinearGated(params, device, self.hs)
        self.d2 = nn.Dropout(params.dr)
        self.fc3 = nn.Linear(self.hs, params.hidden_size)

        self.fc_out = nn.Linear(params.hidden_size, params.output_size)

    def forward(self, data):
        # Define the forward pass here

        pot1 = torch.zeros((data.size(1), self.params.hidden_size), device=self.device, requires_grad=True)
        pot2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)

        out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps

            x = data[step]

            fc1 = self.fc1(x)

            lg1, pot1 = self.lg1(fc1, pot1)
            d1 = self.d1(lg1)

            if len(out) > 0:
                with_previous = torch.cat((d1, out[-1]), dim=1)
            else:
                empty = torch.zeros(data.size(1), self.params.hidden_size, device=self.device, requires_grad=True)
                with_previous = torch.cat((d1, empty), dim=1)

            fc2 = self.fc2(with_previous)
            lg2, pot2 = self.lg2(fc2, pot2)
            d2 = self.d2(lg2)

            fc3 = self.fc3(d2)

            out.append(fc3)
        last = out[-1]
        fc_out = self.fc_out(last)

        return fc_out


def ev_GatedNet(X, net):
    return net(X)

@dataclass
class SGLUFFCParams(ExperimentParams):
    hidden_size: int

def get_sgluffc_params():
    input_size = 10
    hidden_size = input_size * 64

    return SGLUFFCParams(
        model_id="sglu",
        short_name="sglu_ffc",
        description="gated 3 layers, hs=10*64, dr 0.2, epoch 100, bs 128, lr 0.0002, 2 layer ANN out si=hs*2",
        input_size=input_size,
        output_size=9,
        lr=0.0002,
        batch_size=128,
        num_epochs=100,
        hidden_size=hidden_size,
        clip_grad_norm=1.0
    )

@dataclass
class SGLUParams(ExperimentParams):
    hidden_size: int

def get_SGLUParams():
    input_size = 10
    hidden_size = 1024

    return SGLUParams(
        model_id="sglu",
        short_name="sglu_last",
        description="3 layer sglu using the last timestep as output",
        input_size=input_size,
        output_size=9,
        lr=0.0002,
        batch_size=128,
        num_epochs=100,
        clip_grad_norm=1.0,
        hidden_size=hidden_size
    )

def get_sgluffc_params():
    input_size = 10
    hidden_size = input_size * 64

    return SGLUFFCParams(
        model_id="sglu",
        short_name="sglu_ffc",
        description="gated 3 layers, hs=10*64, dr 0.2, epoch 100, bs 128, lr 0.0002, 2 layer ANN out si=hs*2",
        input_size=input_size,
        output_size=9,
        lr=0.0002,
        batch_size=128,
        num_epochs=100,
        hidden_size=hidden_size,
        clip_grad_norm=1.0
    )

@dataclass
class AR_SGLUParams(ExperimentParams):
    hidden_size: int
    dr: float

def get_AR_SGLUParams():
    input_size = 10
    hidden_size = 1337

    return AR_SGLUParams(
        model_id="sglu",
        short_name="sglu_last",
        description="3 layer sglu using the last timestep as output",
        input_size=input_size,
        output_size=9,
        lr=0.002,
        batch_size=128,
        num_epochs=1000,
        clip_grad_norm=1.0,
        hidden_size=hidden_size,
        dr=0.2
    )

def train_gated():
    params = get_AR_SGLUParams()  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = AR_SGLUNet(params, device).to(device)
    print_params(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    train_model(params, device, net, loss_fn, optimizer, ev_GatedNet)

if __name__=="__main__":
    train_gated()