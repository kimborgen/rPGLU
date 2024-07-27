import matplotlib as mlp
mlp.use("TkAgg")
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

# Correcting the definition of the sigmoid function to be centered at y=-1


def custom_grad_scale(x):
    def sig_neg(x):
        return 1000*torch.sigmoid((x+1.5)) - 1000

    def sig_pos(x):
        return 1000*torch.sigmoid((x-1.5)) - 29

    def squ_tanh(x):
        return torch.tanh(5 * x)
        
    # Masks
    mask_tanh = (x > -1) & (x < 1)
    mask_sig_neg = (x <= -1)
    mask_sig_pos = (x >= 1)

    # Apply custom tanh to specific elements
    result = torch.zeros_like(x)
    result[mask_tanh] = squ_tanh(x[mask_tanh])
    result[mask_sig_neg] = x[mask_sig_neg].cli
    result[mask_sig_pos] = sig_pos(x[mask_sig_pos])
    return result

x_values_corrected = torch.linspace(-3, 3, 1000)
y_values_corrected = custom_grad_scale(x_values_corrected)

# Plot the corrected custom sigmoid function
plt.plot(x_values_corrected.numpy(), y_values_corrected.numpy())
plt.title('SnakedLinearUnit')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()