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

class GRUNet(nn.Module):
    def __init__(self, params):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(params.input_size, params.hidden_size, params.num_layers, batch_first=False, dropout=params.dr)
        self.fc_out = nn.Linear(params.hidden_size, params.output_size)
        self.params = params
        
    def forward(self, x):
        # Initialize hidden and cell states
        # x.size(1) is batch_size
        h0 = torch.zeros(self.params.num_layers, x.size(1), self.params.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.gru(x, h0)  # out: tensor of shape ( seq_length, batch size, hidden_size)
        last = out[-1, :, :]
        # Decode the hidden state of the last time step
        fc_out = self.fc_out(last)
        return fc_out
    
def ev_gru(X, net):
    return net(X)

@dataclass
class GRUParams(ExperimentParams):
    hidden_size: int
    num_layers: int
    steps: int
    dr: float

def get_gru_last_params():
    return GRUParams(
        model_id="gru",
        short_name="gru_last",
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

def train_gru():
    params = get_gru_last_params()  
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
        
        net = GRUNet(params).to(device)

        #scheduler = StepLR(optimizer, step_size=5, gamma=0.1) 

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

        train_model(params, device, net, loss_fn, optimizer, ev_gru, print_grads=False, automate=True, automate_save_model=False)

train_gru()