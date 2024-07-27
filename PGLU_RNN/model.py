import sys
from pathlib import Path
parent_folder_path = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import print_params, scaled_bell_distribution_capped_at_2_times_sigma
from dataclasses import dataclass
from experiment_helper import ExperimentParams
from training import train_model
from components import InvertedBumpFunction, BumpFunction

class PGLU_tresh(nn.Module):
    def __init__(self, input_size, output_size, init_tresh_center, init_tresh_sigma, init_decay_center, init_decay_sigma):
        super(PGLU_tresh, self).__init__()

        self.W = nn.Linear(input_size, output_size)
        self.tresh = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(output_size, init_tresh_center, init_tresh_sigma), requires_grad=True)
        self.decay_rate = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(output_size, init_decay_center, init_decay_sigma), requires_grad=True)

    def forward(self, x, potential): 

        new_x = self.W(x)

        # calculate new potential
        potential = potential + new_x

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
    
class PGLU(nn.Module):
    def __init__(self, input_size, output_size, init_decay_center, init_decay_sigma):
        super(PGLU, self).__init__()

        self.W = nn.Linear(input_size, output_size)
        self.decay_rate = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(output_size, init_decay_center, init_decay_sigma), requires_grad=True)

    def forward(self, x, potential): 

        new_x = self.W(x)

        # calculate new potential
        potential = potential + new_x

        # Activation function
        activated = F.relu(potential)

        # Remove activated neurons from potential
        potential_new = potential - activated

        # Reduce potential with decay_rate
        potential_new = potential_new * self.decay_rate

        return activated, potential_new

    
class Net(nn.Module):
    def __init__(self, params, device):
        super(Net, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.hs = params.hidden_size

        self.pglu1 = PGLU(self.inp, self.hs, params.init_decay_center, params.init_decay_sigma)
        #self.pglu2 = PGLU(self.device, self.hs, self.hs, self.dropout)
        self.rnn = nn.RNNCell(self.hs, self.hs)
        
        self.fc_out = nn.Linear(self.hs, self.out)

    def forward(self, data):
        # Define the forward pass here

        hidden1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #c1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        pot1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #pot2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #hidden2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)

        #out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps

            x = data[step]

            #fc1 = self.fc1(x)

            activated, pot1 = self.pglu1(x, pot1)
            #d1 = self.d1(hidden1)

            #hidden2, pot2 = self.pglu2(hidden1, hidden2, pot2)
            hidden1 = self.rnn(activated, hidden1)

            #fc2 = self.fc2(d1)
            #hidden2, pot2 = self.pglu2(hidden1, hidden2, pot2)
            #d2 = self.d2(pglu2)

            # at the last step, compute the out network
            if step == data.size(0) - 1:
                return self.fc_out(hidden1)
            
class Net_v2(nn.Module):
    def __init__(self, params, device):
        super(Net_v2, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.hs = params.hidden_size

        self.pglu1 = PGLU(self.hs, self.hs, params.init_decay_center, params.init_decay_sigma)
        #self.pglu2 = PGLU(self.device, self.hs, self.hs, self.dropout)
        self.rnn = nn.RNNCell(self.inp, self.hs)
        
        self.fc_out = nn.Linear(self.hs, self.out)

    def forward(self, data):
        # Define the forward pass here

        hidden1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #c1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        pot1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #pot2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #hidden2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)

        #out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps

            x = data[step]

            #fc1 = self.fc1(x)

            
            #d1 = self.d1(hidden1)

            #hidden2, pot2 = self.pglu2(hidden1, hidden2, pot2)
            rnn_out, hidden1 = self.rnn(x, hidden1)
            
            hidden1, pot1 = self.pglu1(rnn_out, pot1)

            #fc2 = self.fc2(d1)
            #hidden2, pot2 = self.pglu2(hidden1, hidden2, pot2)
            #d2 = self.d2(pglu2)

            # at the last step, compute the out network
            if step == data.size(0) - 1:
                return self.fc_out(hidden1)

def ev_GatedNet(X, net):
    return net(X)

@dataclass
class PGLUParams(ExperimentParams):
    hidden_size: int
    init_decay_center: float
    init_decay_sigma: float

class PGLU_treshParams(ExperimentParams):
    hidden_size: int
    init_tresh_center: float
    init_tresh_sigma: float
    init_decay_center: float
    init_decay_sigma: float

def get_params():
    return PGLUParams(
        model_id="pglu_basic_reccurent",
        short_name="pglu_basic_reccurent_last",
        description="The most basic recurrent version of PGLU. Simplified PGLU without treshold.",
        input_size=10,
        output_size=9,
        lr=0.002,
        batch_size=128,
        num_epochs=15,
        hidden_size=32,
        clip_grad_norm=1.0,
        init_decay_center=0.7,
        init_decay_sigma=0.1,
    )

def train_gated():
    params = get_params() 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    experiements = [
        {"hidden_size": 32, "init_decay_center": 0, "init_decay_sigma": 0},
        {"hidden_size": 32, "init_decay_center": 0.3, "init_decay_sigma": 0.1},
        {"hidden_size": 32, "init_decay_center": 0.4, "init_decay_sigma": 0.1},
        {"hidden_size": 32, "init_decay_center": 0.5, "init_decay_sigma": 0.1},
        {"hidden_size": 32, "init_decay_center": 0.6, "init_decay_sigma": 0.1},
        {"hidden_size": 32, "init_decay_center": 0.7, "init_decay_sigma": 0.1},
        {"hidden_size": 32, "init_decay_center": 0.8, "init_decay_sigma": 0.1},
    ]

    for exp in experiements:
        params.init_decay_center = exp["init_decay_center"]
        params.init_decay_sigma = exp["init_decay_sigma"]
        params.hidden_size = exp["hidden_size"]

        net = Net(params, device).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

        train_model(params, device, net, loss_fn, optimizer, ev_GatedNet, automate=True, automate_save_model=False)

if __name__=="__main__":
    train_gated()