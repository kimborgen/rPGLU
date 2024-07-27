from prettytable import PrettyTable
import torch
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_paramters_all(model):
    return sum(p.numel() for p in model.parameters())

def print_params(model):
    table = PrettyTable(["Modules", "Parameters", "Requires grad"])
    total_params = 0
    total_params_grad = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        table.add_row([name, param, parameter.requires_grad])
        total_params+=param
        if parameter.requires_grad:
            total_params_grad += param

    print(table)
    print(f"Total Params: {total_params}")
    print(f"Total Grad Params: {total_params_grad}")
    return total_params

def scaled_bell_distribution_capped_at_2_times_sigma(size, center, sigma):
    return torch.clamp(torch.randn(size) * (sigma/1.5) + center, min=center-2*sigma, max=center+2*sigma)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = scaled_bell_distribution_capped_at_2_times_sigma(10000, 0.7, 0.1)
    print(result.sort())

    plt.hist(result.numpy(), bins=50)
    plt.title(f"Distribution centered at {0.7} with variability {0.1}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
