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


class Plz(nn.Module):
    def __init__(self, params, device, width):
        super(Plz, self).__init__()




class LinearGated(nn.Module):
    def __init__(self, params, device, width):
        super(LinearGated, self).__init__()
        self.device = device
        #self.num_out = params.output_size
        #self.batch_size = params.batch_size

        mu, sigma = 0, 0.5  # For threshold
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
z
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
    
class SGLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, params, device):
        super(SGLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.d = nn.Dropout(params.dr_lstm)

        # Input gate layers
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True)
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.b_i = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
        self.sg_i = LinearGated(params, device, hidden_size)

        # Forget gate layers
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True)
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.b_f = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
        self.sg_f = LinearGated(params, device, hidden_size)

        # Output gate layers
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True)
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.b_o = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
        self.sg_o = LinearGated(params, device, hidden_size)

        # Cell state layers
        self.W_ic = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True)
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.b_c = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)

        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / torch.sqrt(torch.tensor(self.hidden_size))
        print(self.named_parameters())
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h_t, c_t, pot_f, pot_i, pot_o):
        """
        x: Input tensor of shape (batch, input_size)
        h_t: hidden 
        c_t: cell state
        """

        # dropout
        x = self.d(x)
        
        # Forget gate
        f_t, pot_f = self.sg_f(x @ self.W_if + h_t @ self.W_hf + self.b_f, pot_f)
        
        # Input gate
        i_t, pot_i = self.sg_i(x @ self.W_ii + h_t @ self.W_hi + self.b_i, pot_i)
        g_t = torch.tanh(x @ self.W_ic + h_t @ self.W_hc + self.b_c)
        
        # Output gate
        o_t, pot_o= self.sg_o(x @ self.W_io + h_t @ self.W_ho + self.b_o, pot_o)
        
        # Update cell state
        c_next = f_t * c_t + i_t * g_t
        
        # Update hidden state
        h_next = o_t * torch.tanh(c_next)
        
        return h_next, c_next, pot_f, pot_i, pot_o
    
class SGLSTMNet(nn.Module):
    def __init__(self, params, device):
        super(SGLSTMNet, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.dropout = 0.2
        self.hs = params.hidden_size

        self.sglstm = SGLSTM(self.inp, self.hs, params, device)
        self.fc_out = nn.Linear(self.hs, self.out)

    def forward(self, data):
        f = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        i = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        o = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        h, c = (torch.zeros(data.size(1), self.hs).to(self.device), torch.zeros(data.size(1), self.hs).to(self.device))
        out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps

            x = data[step]
            h, c, f, i, o = self.sglstm(x, h, c, f, i, o)
            out.append(h)

        last = out[-1]
        fc_out = self.fc_out(last)
        return fc_out

def ev_SGLSTM(X, net):
    return net(X)

@dataclass
class SGLSTMParams(ExperimentParams):
    hidden_size: int
    dr_lstm: float

def get_sglstm_params():
    input_size = 10
    hidden_size = input_size * 64

    return SGLSTMParams(
        model_id="sglstm",
        short_name="sglstm",
        description="single SGLSTM, 1 layer ANN out",
        input_size=input_size,
        output_size=9,
        lr=0.0002,
        batch_size=128,
        num_epochs=100,
        hidden_size=hidden_size,
        clip_grad_norm=1.0,
        dr_lstm= 0.2
    )

def train_gated():
    params = get_sglstm_params()  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = SGLSTMNet(params, device).to(device)
    print_params(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    train_model(params, device, net, loss_fn, optimizer, ev_SGLSTM)

if __name__=="__main__":
    train_gated()