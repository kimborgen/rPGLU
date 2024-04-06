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

class RNN(nn.Module):
    def __init__(self, params, device):
        super(RNN, self).__init__()
        self.hs = params.hidden_size
        self.inp = params.input_size
        self.out = params.output_size
        self.device = device
        
        # Initialize weights
        self.W_ih = nn.Parameter(torch.randn(self.inp, self.hs))
        self.W_hh = nn.Parameter(torch.randn(self.hs, self.hs))
        self.b_i = nn.Parameter(torch.randn(self.hs))
        self.b_h = nn.Parameter(torch.randn(self.hs))

    def forward(self, x_t, h):
       return torch.tanh(x_t @ self.W_ih + self.b_i + h @ self.W_hh + self.b_h)
       

def prnt(module, grad_input, grad_output):
    #print(module)
    #print(grad_input)
    #print(grad_output)
    return (F.layer_norm(grad_input[0], grad_input[0].shape), F.layer_norm(grad_input[1], grad_input[1].shape))

class SimpleRNN(nn.Module):
    def __init__(self, params, device):
        super(SimpleRNN, self).__init__()
        self.hs = params.hidden_size
        self.inp = params.input_size
        self.out = params.output_size
        self.device = device
        
        self.rnn = RNN(params, device)
        self.rnn.register_full_backward_hook(prnt)

        # Output layer
        self.linear = nn.Linear(self.hs, self.out)
        
    def forward(self, data):
        # Batch size for dynamic sizing
        batch_size = data.size(1)
        
        # Initialize hidden state with zeros
        h = torch.zeros(batch_size, self.hs, device=self.device, requires_grad=True)
        
        # Manually iterate through time steps
        for step in range(data.size(0)):
            x_t = data[step]
            h = self.rnn(x_t, h)
        
        # Compute the output (assuming output at the last time step)
        out = self.linear(h)
        return out
    
def ev_rnn(X, net):
    return net(X)

@dataclass
class RNNParams(ExperimentParams):
    hidden_size: int

def get_rnn_params():
    return RNNParams(
        model_id="rnn",
        short_name="simple_rnn",
        description="dirt simple rnn",
        input_size=10,
        output_size=9,
        lr=0.02,
        batch_size=128,
        num_epochs=100,
        clip_grad_norm=1.0,
        hidden_size=64
    )

def train_lstm():
    params = get_rnn_params()  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = SimpleRNN(params, device).to(device)
    print_params(net)

    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1) 

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    train_model(params, device, net, loss_fn, optimizer, ev_rnn)

train_lstm()