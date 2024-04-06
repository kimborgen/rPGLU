import sys
from pathlib import Path
parent_folder_path = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_folder_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils import print_params
from dataclasses import dataclass
from experiment_helper import ExperimentParams
from training import train_model
from sglu.components import InvertedBumpFunction, BumpFunction

class LinearSqueeze(nn.Module):
    def __init__(self, params, device, inp, out):
        super(LinearSqueeze, self).__init__()
        self.params = params
        self.device = device 

        self.lin = nn.Linear(inp, out)

    def forward(self, x, h):
        fc = self.lin(x)
        _fc = x = h + fc
        h_n = F.tanh(_fc)
        return h_n        

class AddNet(nn.Module):
    def __init__(self, params, device):
        super(AddNet, self).__init__()

        self.batch_size = params.batch_size
        self.inp = params.input_size 
        self.out = params.output_size
        self.device = device
        self.dropout = 0.2
        self.hs = params.hidden_size

        self.d = nn.Dropout(params.dr)
        self.fc1 = LinearSqueeze(params, device, self.inp, self.hs)
        self.fc2 = LinearSqueeze(params, device, self.hs, self.hs)
        self.fc3 = LinearSqueeze(params, device, self.hs, self.out)    


    def forward(self, data):
        # Define the forward pass here


        h1 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        h2 = torch.zeros((data.size(1), self.hs), device=self.device, requires_grad=True)
        h3 = torch.zeros((data.size(1), self.out), device=self.device, requires_grad=True) 

        out = []

        for step in range(data.size(0)):  # data.size(0) = number of time steps

            x = data[step]

            h1 = self.fc1(x, h1)
            d1 = self.d(h1)
            h2 = self.fc2(d1, h2)
            d2 = self.d(h2)
            h3 = self.fc3(d2, h3)

            out.append(h3)
        
        return out[-1]
    
def ev_AddNet(X, net):
    return net(X)

@dataclass
class AddNetParams(ExperimentParams):
    hidden_size: int
    dr: float

def get_sglstm_params():
    return AddNetParams(
        model_id="stnet",
        short_name="stnet",
        description="Short Term Net, 3 layer, last layer output size",
        input_size=10,
        output_size=9,
        lr=0.002,
        batch_size=128,
        num_epochs=100,
        hidden_size=640,
        clip_grad_norm=1.0,
        dr= 0.2
    )

def train_addNet():
    params = get_sglstm_params()  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = AddNet(params, device).to(device)
    #print_params(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    train_model(params, device, net, loss_fn, optimizer, ev_AddNet)

if __name__=="__main__":
    train_addNet()