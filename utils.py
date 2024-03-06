from prettytable import PrettyTable

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
