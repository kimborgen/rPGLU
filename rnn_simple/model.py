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

class RNNNet(nn.Module):
    def __init__(self, params):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(params.input_size, params.hidden_size, params.num_layers, batch_first=False, dropout=params.dr)
        self.fc_out = nn.Linear(params.hidden_size, params.output_size)
        self.params = params
        
    def forward(self, x):
        # Initialize hidden and cell states
        # x.size(1) is batch_size
        h0 = torch.zeros(self.params.num_layers, x.size(1), self.params.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)  # out: tensor of shape ( seq_length, batch size, hidden_size)
        last = out[-1, :, :]
        # Decode the hidden state of the last time step
        fc_out = self.fc_out(last)
        return fc_out
    
def ev_net(X, net):
    return net(X)

@dataclass
class RNNParams(ExperimentParams):
    hidden_size: int
    num_layers: int
    steps: int
    dr: float

def get_rnn_last_params():
    return RNNParams(
        model_id="rnn",
        short_name="rnn_last",
        description="Normed dataset",
        input_size=10,
        output_size=9,
        lr=0.002,
        batch_size=128,
        num_epochs=100,
        clip_grad_norm=1.0,
        hidden_size=32,
        dr = 0.1,
        steps = 23,
        num_layers=1
    )

def train_rnn():
    params = get_rnn_last_params()  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    experiments = [
        {"hidden_size": 32, "num_layers": 1},
        {"hidden_size": 64, "num_layers": 1},
        {"hidden_size": 128, "num_layers": 1},
        {"hidden_size": 512, "num_layers": 1},
        {"hidden_size": 32, "num_layers": 2},
        {"hidden_size": 64, "num_layers": 2},
        {"hidden_size": 128, "num_layers": 2},
        {"hidden_size": 512, "num_layers": 2},
    ]

    for exp in experiments:
        params.hidden_size = exp["hidden_size"]
        params.num_layers = exp["num_layers"]
        
        net = RNNNet(params).to(device)

        #scheduler = StepLR(optimizer, step_size=5, gamma=0.1) 

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

        train_model(params, device, net, loss_fn, optimizer, ev_net, print_grads=False, automate=True, automate_save_model=False)

train_rnn()