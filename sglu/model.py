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

# Define the mathematical function for the inverted bump function
def inverted_bump_function_math(x):
    # Function is 1 outside the range -1 to 1
    f = torch.ones_like(x)
    # Within the range -1 to 1, apply the inverted bump function
    mask = (x > -1) & (x < 1)
    f[mask] = 1 - torch.exp(-1 / (1 - x[mask] ** 2))
    return f

# Define the derivative of the inverted bump function
def derivative_inverted_bump_function_math(x):
    # Derivative is zero outside the range -1 to 1
    df = torch.zeros_like(x)
    # Within the range -1 to 1, calculate the derivative
    mask = (x > -1) & (x < 1)
    df[mask] = 2 * x[mask] * torch.exp(-1 / (1 - x[mask] ** 2)) / (1 - x[mask] ** 2) ** 2
    return df

class InvertedBumpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for use in backward pass
        ctx.save_for_backward(x)
        
        # Forward pass: return 1 where x is not 0, otherwise 0 (as a placeholder behavior)
        output = torch.where(x != 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        x, = ctx.saved_tensors
        
        # Calculate the derivative of the inverted bump function
        derivative = derivative_inverted_bump_function_math(x)
        
        # Apply the chain rule (multiply by incoming gradient)
        grad_input = derivative * grad_output
        
        return grad_input

class BumpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for use in backward pass
        ctx.save_for_backward(x)
        
        # Forward pass: return 1 where x is 0, otherwise 0
        output = torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        x, = ctx.saved_tensors
        
        # Create a mask for where the bump function is active (-1 < x < 1)
        mask = (x > -1) & (x < 1)
        
        # Initialize gradient as zero for all elements
        grad_x = torch.zeros_like(x)
        
        # Compute gradient only for the masked elements
        grad_x[mask] = torch.exp(-1 / (1 - (x[mask] ** 2)))
        # Normalize the bump function to have a maximum of 1
        grad_x[mask] = grad_x[mask] / torch.max(grad_x[mask])
        # Adjust the gradient for the bump function
        grad_x[mask] = grad_output[mask] * (-2 * x[mask] / (1 - x[mask] ** 2) ** 2) * grad_x[mask]
        
        return grad_x
    
class LinearGated(nn.Module):
    def __init__(self, device, batch_size, num_out):
        super(LinearGated, self).__init__()
        self.device = device
        #self.inp = inp
        self.num_out = num_out
        #self.spike_decay_rate = spike_decay_rate
        self.batch_size = batch_size


        #self.fc = nn.Linear(inp, out)
          #add device
        #self.reset()
        mu, sigma = 0, 0.3  # For threshold
        mu_decay, sigma_decay = 0.1, 0.7  # For decay rate
        #mu_spike, sigma_spike = 0, -1.0
        self.tresh = nn.Parameter(torch.abs(torch.rand(size=(self.num_out,), device=self.device, requires_grad=True)) * sigma + mu)
        self.decay_rate = nn.Parameter(torch.abs(torch.rand(size=(self.num_out,), device=self.device, requires_grad=True)) * sigma_decay + mu_decay)
        #self.spike_decay_rate = nn.Parameter(torch.abs(torch.rand(size=(self.out,), device=self.device, requires_grad=True)) * sigma_spike + mu_spike)
        pass 

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

class GatedNet(nn.Module):
    def __init__(self, params, device):
        super(GatedNet, self).__init__()

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
    
class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def ev_GatedNet(X, net):
    return net(X)

@dataclass
class SGLUParams(ExperimentParams):
    hidden_size: int

def get_params():
    input_size = 10
    hidden_size = input_size * 64

    return SGLUParams(
        model_id="sglu",
        short_name="sglu_ffc",
        description="gated 3 layers, hs=10*64, dr 0.2, epoch 100, bs 128, lr 0.0002, 2 layer ANN out si=hs*2",
        input_size=input_size,
        output_size=9,
        lr=0.0002,
        batch_size=128,
        num_epochs=100,
        hidden_size=hidden_size,
        clip_grad_norm=1.0
    )

def train_gated():
    params = get_params()  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    net = GatedNet(params, device).to(device)
    print_params(net)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    train_model(params, device, net, loss_fn, optimizer, ev_GatedNet)

if __name__=="__main__":
    train_gated()