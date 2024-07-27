import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if torch.sum(grad_x[mask]) != 0:
            grad_x[mask] = grad_x[mask] / torch.max(grad_x[mask])
            # Adjust the gradient for the bump function
        grad_x[mask] = grad_output[mask] * (-2 * x[mask] / (1 - x[mask] ** 2) ** 2) * grad_x[mask]
        
        return grad_x
    
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