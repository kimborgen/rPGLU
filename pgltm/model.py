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

class PGLU(nn.Module):
    def __init__(self, device, input_size, params):
        super(PGLU, self).__init__()
        self.device = device
        self.input_size = input_size

        """
        Misc
        """
        #self.dropout_layer = nn.Dropout(p=params.dropout_rate)

        """
        Potential
        """
        self.tresh = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(params.hidden_size, params.init_tresh_center, params.init_tresh_sigma), requires_grad=True)
        self.decay_rate = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(params.hidden_size, params.init_decay_center, params.init_decay_sigma), requires_grad=True)


        """
        ?
        """
        # input gate
        self.W_i = nn.Linear(input_size, params.hidden_size)

        """
        Gates
        """

        # # Reset gate
        self.W_r = nn.Linear(2 * params.hidden_size, params.hidden_size)
        self.reset = nn.Sigmoid()
        
        # Update gate
        self.W_z = nn.Linear(2 * params.hidden_size, params.hidden_size)
        self.update = nn.Sigmoid()

        # Candidate gate
        self.W_c = nn.Linear(2 * params.hidden_size, params.hidden_size)
        self.candidate_hidden_state = nn.Tanh()

    def forward(self, x, h_prev, potential_prev): 
        # Linear layer for input gate
        processed_input = self.W_i(x)
    
        # calculate new potential
        potential = potential_prev + processed_input

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

        
        """
        Now gru stuff
        """
        # Concatenate activated and previous hidden state
        combined = torch.cat((activated, h_prev), dim=1)
        
        # Compute reset gate
        r = self.reset(self.W_r(combined))
        
        # Compute update gate
        z = self.update(self.W_z(combined))
        
        # Compute new memory content
        n = self.candidate_hidden_state(self.W_c(torch.cat((activated, r * h_prev), dim=1)))
        
        # Compute new hidden state
        h_new = (1 - z) * h_prev + z * n
    
        return h_new, potential_non_gated
    

class PGLU_b(nn.Module):
    def __init__(self, device, input_size, hidden_size, dropout_rate):
        super(PGLU_b, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        """
        Misc
        """
        self.dropout_layer = nn.Dropout(p=dropout_rate)

        """
        Potential
        """
        # mu = variability
        # sigma = center point
        mu, sigma = 0, 0.3  # For threshold
        mu_decay, sigma_decay = 0.1, 0.7  # For decay rate
        self.tresh = nn.Parameter(torch.abs(torch.rand(size=(hidden_size,), device=device, requires_grad=True)) * sigma + mu)
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(hidden_size,), device=device, requires_grad=True)) * sigma_decay + mu_decay)

        """
        ?
        """
        # input gate
        #self.W_i = nn.Linear(input_size, hidden_size)

        """
        Gates
        """

        # # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset = nn.Sigmoid()
        
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.update = nn.Sigmoid()

        # Candidate gate
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_hidden_state = nn.Tanh()

    def forward(self, x, h_prev, potential_prev): 
        """
        Now gru stuff
        """
        # Concatenate activated and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        
        # Compute reset gate
        r = self.reset(self.W_r(combined))
        
        # Compute update gate
        z = self.update(self.W_z(combined))
        
        # Compute new memory content
        n = self.candidate_hidden_state(self.W_c(torch.cat((x, r * h_prev), dim=1)))
        
        # Compute new hidden state
        h_new = (1 - z) * h_prev + z * n
    
        # calculate new potential
        potential = potential_prev + h_new

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
    
class PGLTM(nn.Module):
    def __init__(self, params, device):
        super(PGLTM, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.dropout = 0.2
        self.hs = params.hidden_size

        self.pglu1 = PGLU(self.device, self.inp, params)
        #self.pglu2 = PGLU(self.device, self.hs, self.hs, self.dropout)
        
        self.fc_out = nn.Linear(self.hs, self.out)

    def forward(self, data):
        # Define the forward pass here

        pot1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
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
class PGLTMParams(ExperimentParams):
    hidden_size: int
    init_decay_center: float
    init_decay_sigma: float
    init_tresh_center: float
    init_tresh_sigma: float

def get_pgltm_params():
    input_size = 10

    return PGLTMParams(
        model_id="pgltm",
        short_name="pgltm_last",
        description="normed dataset. New init distribution for tresh and pot",
        input_size=input_size,
        output_size=9,
        lr=0.002,
        batch_size=128,
        num_epochs=100,
        hidden_size=16,
        clip_grad_norm=1.0,
        init_decay_center=0.7,
        init_decay_sigma=0.1,
        init_tresh_center=0.3,
        init_tresh_sigma=0.05
    )

def train_gated():
    params = get_pgltm_params() 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = PGLTM(params, device).to(device)
    print_params(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    train_model(params, device, net, loss_fn, optimizer, ev_GatedNet)

if __name__=="__main__":
    train_gated()