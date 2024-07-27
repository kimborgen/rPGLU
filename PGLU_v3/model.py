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

class PTGLU(nn.Module):
    def __init__(self, input_size, output_size, init_tresh_center, init_tresh_sigma, init_decay_center, init_decay_sigma, bias):
        super(PTGLU, self).__init__()

        self.W_ptglu_in = nn.Linear(input_size + output_size, output_size * 2, bias=bias)
        self.tresh = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(output_size * 2, init_tresh_center, init_tresh_sigma), requires_grad=True)
        self.decay_rate = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(output_size * 2, init_decay_center, init_decay_sigma), requires_grad=True)

        self.update_gate = nn.Linear(output_size * 3, output_size)
        self.reset_gate = nn.Linear(output_size * 3, output_size)
        self.candidate = nn.Linear(output_size * 3, output_size)

    def forward(self, x, h_prev, pot_prev): 

        combined = torch.cat([x, h_prev], dim=1)

        new_x = self.W_ptglu_in(combined)

        # calculate new potential
        pot_tmp = pot_prev + new_x

        treshed = pot_tmp - self.tresh

        # Zero those where potential < tresh
        gated = F.relu(treshed)
        
        # now apply this gated filter to the potential
        gated_bool = InvertedBumpFunction.apply(gated)

        # This is now what we return from this layer
        activated = gated_bool * pot_tmp
        
        # Now we wish to reduce the potential of the open gates with spike_decay_rate

        # reduce the potential of the closed gates with decay_rate
        non_gated_bool = BumpFunction.apply(gated)
        non_gated = pot_tmp * non_gated_bool
        pot_next = non_gated * self.decay_rate


        # Should the update be gated, or directly applied? Or should the previous activation be the hidden state?
        # The goal is model short-temporal memory with this potential and add it to a long-term memory such that the last hidden state will contain all the information needed for classification
        """ 
        Option 1: Simple add activation to hidden state. The numbers will become large, but it will contain all the information needed for classification. Can be shrunk with a linear layer, normed, or minmaxed.
        Option 2: The temporal signals are captured by the potential and the activations. We can combine them and a linear layer to update the hidden state. 
        Option 3: Reset and update gate like in GRU, but then in the candidate state use the activations? Or maybe a secondary activations?
        """ 

        #GRU 
        gru_combined = torch.cat([activated, h_prev], dim=1)
        z = F.sigmoid(self.update_gate(gru_combined))
        r = F.sigmoid(self.reset_gate(gru_combined))
        reset_hidden = h_prev * r
        candidate = torch.cat([activated, reset_hidden], dim=1)
        n = F.tanh(self.candidate(candidate))

        h_next = (1 - z) * h_prev + z * n
        return h_next, pot_next
    

class Net(nn.Module):
    def __init__(self, params, device):
        super(Net, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.hs = params.hidden_size

        self.pglu1 = PTGLU(self.inp, self.hs, params.init_tresh_center, params.init_tresh_sigma, params.init_decay_center, params.init_decay_sigma, bias=params.pglu_linear_bias)
        #self.pglu2 = PGLU(self.device, self.hs, self.hs, self.dropout)
        
        self.fc_out = nn.Linear(self.hs, self.out)

    def forward(self, data):
        # Define the forward pass here

        #hidden1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #c1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        pot1 = torch.zeros((data.size(1), self.hs * 2), device=self.device, requires_grad=True)
        hidden1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #pot2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #hidden2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)

        #out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps

            x = data[step]

            #fc1 = self.fc1(x)

            hidden1, pot1 = self.pglu1(x, hidden1, pot1)
            #d1 = self.d1(hidden1)

            #hidden2, pot2 = self.pglu2(hidden1, hidden2, pot2)


            #fc2 = self.fc2(d1)
            #hidden2, pot2 = self.pglu2(hidden1, hidden2, pot2)
            #d2 = self.d2(pglu2)

            # at the last step, compute the out network
            if step == data.size(0) - 1:
                return self.fc_out(hidden1)

def ev_GatedNet(X, net):
    return net(X)

@dataclass
class PGLU_Params(ExperimentParams):
    hidden_size: int
    init_decay_center: float
    init_decay_sigma: float

@dataclass 
class PGLU_tresh_Params(ExperimentParams):
    hidden_size: int
    init_tresh_center: float
    init_tresh_sigma: float
    init_decay_center: float
    init_decay_sigma: float
    pglu_linear_bias: bool

def get_params():
    return PGLU_tresh_Params(
        model_id="ptglu_v3",
        short_name="ptglu_v3_last",
        description="x = cat(x,h_prev) and GRU, 2*hs",
        input_size=10,
        output_size=9,
        lr=0.002,
        batch_size=128,
        num_epochs=100,
        hidden_size=64,
        clip_grad_norm=1.0,
        init_decay_center=0.7,
        init_decay_sigma=0.1,
        init_tresh_center=0.5,
        init_tresh_sigma=0.1,
        pglu_linear_bias=False,
    )

def train_gated():
    params = get_params() 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    experiements = [
        {"hidden_size": 64, "init_decay_center": 0.7, "init_decay_sigma": 0.1, "init_tresh_center": 0.4, "init_tresh_sigma": 0.1},
    ]

    for exp in experiements:
        params.init_decay_center = exp["init_decay_center"]
        params.init_decay_sigma = exp["init_decay_sigma"]
        params.init_tresh_center = exp["init_tresh_center"]
        params.init_tresh_sigma = exp["init_tresh_sigma"]
        params.hidden_size = exp["hidden_size"]

        net = Net(params, device).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

        train_model(params, device, net, loss_fn, optimizer, ev_GatedNet, automate=True, automate_save_model=False)

if __name__=="__main__":
    train_gated()