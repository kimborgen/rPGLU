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

class LinearGated(nn.Module):
    def __init__(self, device, width):
        super(LinearGated, self).__init__()
        self.device = device
        self.width = width

        self.tresh = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(width, 0.3, 0.05), requires_grad=True)
        self.decay_rate = nn.Parameter(scaled_bell_distribution_capped_at_2_times_sigma(width, 0.7, 0.1), requires_grad=True)

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

class PGGRU(nn.Module):
    def __init__(self, device, input_size, hidden_size, dropout_rate):
        super(PGGRU, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        """
        Misc
        """
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        """
        ?
        """
        # input gate
        #self.W_i = nn.Linear(input_size, hidden_size)

        """
        Gates
        """

        # # Reset gate
        # self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        # self.reset = LinearGated(device, hidden_size)
        
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.update = LinearGated(device, hidden_size)

        # Candidate gate
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_hidden_state = nn.Tanh()

    def forward(self, x, h_prev, potential_prev_z): 
        """
        Now gru stuff
        """

        # Concatenate activated and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        
        # Compute reset gate
        # r_l = self.W_r(combined)
        # r, potential_new_r = self.reset(r_l, potential_prev_r)
        # # Since the output of the PGLU is 0 or a value between 0.3 (treshold) and 1, we must take the complement to figure out how much to reset with. A value of 0 means we don't reset and a value of 1 means we reset fully, while 0.3 means we reset partially (30%).
        # r_o = (1 - r) * h_prev

        # Compute update gate
        z_l = self.W_z(combined)
        z, potential_new_z = self.update(z_l, potential_prev_z)
        # ok problem. In the first iterations, the update gate will not trigger as potential needs to build up, thus the hidden state will not be updated, and thus in the next iteration it won't be updated as much as well.
        # potential solutions
        """
        Weigh the input more, so that the potential is more likely to build up, but this should happen automagically.
        Maybe use the potential instead? However this defeats a bit of the purpose of the potential. 

        Ok, for testing, lets move the complement to the candidate branch. Thus assuming that in most steps, the update gate produces 0, it will completly forget the previous hidden state and populate it with the new state.
        This should make it incapable of long-term memory, but it should be able to learn short-term dependices. And the dataset that is used for testing is pretty short-termed so I expect it to work well.
        
        Ideas for long-term:
        - Also concat the potential so the nn has more to work with in the combined input.
        - Make a new hidden state for long-term dependices.

        So simply switching the complement did not work, next remove the reset gate 
        """

        # Compute new memory content
        n_l = self.W_c(combined)
        n = self.candidate_hidden_state(n_l)
        
        # Compute new hidden state
        h_new = z * h_prev + (1 - z) * n
    
        return h_new, potential_new_z

class PGGRUNet(nn.Module):
    def __init__(self, params, device):
        super(PGGRUNet, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.dropout = 0.2
        self.hs = params.hidden_size

        self.pggru1 = PGGRU(self.device, self.inp, self.hs, self.dropout)
        #self.pglu2 = PGLU(self.device, self.hs, self.hs, self.dropout)
        
        self.fc_out = nn.Linear(self.hs, self.out)

    def forward(self, data):
        # Define the forward pass here

        # pot1_r = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        pot1_z = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        hidden1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #pot2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        #hidden2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)

        #out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps

            x = data[step]

            #fc1 = self.fc1(x)

            hidden1, pot1_z = self.pggru1(x, hidden1, pot1_z)
            #d1 = self.d1(pglu1)

            #fc2 = self.fc2(d1)
            #hidden2, pot2 = self.pglu2(hidden1, hidden2, pot2)
            #d2 = self.d2(pglu2)

            # at the last step, compute the out network
            if step == data.size(0) - 1:
                return self.fc_out(hidden1)

def ev_GatedNet(X, net):
    return net(X)

@dataclass
class PGGRUParams(ExperimentParams):
    hidden_size: int

def get_pggru_params():
    return PGGRUParams(
        model_id="pggru",
        short_name="pggru_last",
        description="Normed dataset. New init distribution for tresh and pot",
        input_size=10,
        output_size=9,
        lr=0.0002,
        batch_size=128,
        num_epochs=100,
        hidden_size=16,
        clip_grad_norm=1.0
    )

def train_gated():
    params = get_pggru_params() 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = PGGRUNet(params, device).to(device)
    print_params(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    train_model(params, device, net, loss_fn, optimizer, ev_GatedNet)

if __name__=="__main__":
    train_gated()