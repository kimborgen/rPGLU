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

class LSTMFFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dr_lstm, dr_fc, num_timesteps):
        super(LSTMFFCNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, dropout=dr_lstm)
        self.dropout = nn.Dropout(dr_fc)
        si = hidden_size * num_timesteps
        self.fc = nn.Linear(si, si * 2)
        self.fc2 = nn.Linear(si * 2, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # x.size(1) is batch_size
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape ( seq_length, batch size, hidden_size)
        #last = out[-1, :, :]
        # Decode the hidden state of the last time step
        out_reshaped = out.transpose(0,1)
        out_reshaped2 = out_reshaped.reshape(out.size(1), -1)
        d1 = self.dropout(out_reshaped2)
        fc1 = self.fc(d1)
        r1 = F.relu(fc1)
        d2 = self.dropout(r1)
        fc2 = self.fc2(d2)

        return fc2

class LSTMNet(nn.Module):
    def __init__(self, params):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(params.input_size, params.hidden_size, params.num_layers, batch_first=False, dropout=params.dr_lstm)
        self.fc_out = nn.Linear(params.hidden_size, params.output_size)
        self.params = params
        
    def forward(self, x):
        # Initialize hidden and cell states
        # x.size(1) is batch_size
        h0 = torch.zeros(self.params.num_layers, x.size(1), self.params.hidden_size).to(x.device)
        c0 = torch.zeros(self.params.num_layers, x.size(1), self.params.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape ( seq_length, batch size, hidden_size)
        last = out[-1, :, :]
        # Decode the hidden state of the last time step
        fc_out = self.fc_out(last)
        return fc_out
    
def ev_lstm(X, net):
    return net(X)

@dataclass
class LSTMParams(ExperimentParams):
    hidden_size: int
    num_layers: int
    dr_lstm: float
    steps: int

def get_lstm_last_params():
    return LSTMParams(
        model_id="lstm",
        short_name="lstm_last",
        description="Last lstm layer out",
        input_size=10,
        output_size=9,
        lr=0.002,
        batch_size=128,
        num_epochs=100,
        clip_grad_norm=1.0,
        hidden_size=945,
        dr_lstm = 0.1,
        steps = 23,
        num_layers=2
    )

def train_lstm():
    params = get_lstm_last_params()  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = LSTMNet(params).to(device)
    print_params(net)

    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1) 

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    train_model(params, device, net, loss_fn, optimizer, ev_lstm)

train_lstm()