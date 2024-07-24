from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataset import DataIterator
import torch
from dataset import get_datasets
from plotting import TrainingPlots
from tqdm import tqdm
from experiment_helper import ExperimentManager
import signal
import csv
import signal
import time
from threading import Timer
import torch.nn.functional as F

def evaluate_model(model, val_iterator, evaluation_fn):
    model.eval()  # Set the model to evaluation mode
    
    predictions = []
    truths = []
    
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in val_iterator:
            # Forward pass
            outputs = evaluation_fn(X_batch, model)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect the predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            truths.extend(y_batch.cpu().numpy())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(truths, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(truths, predictions, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

def test_model(model, test_iterator, evaluation_fn):
    model.eval()  # Set the model to evaluation mode
    
    predictions = []
    truths = []
    
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in test_iterator:
            # Forward pass
            outputs = evaluation_fn(X_batch, model)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect the predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            truths.extend(y_batch.cpu().numpy())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(truths, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(truths, predictions, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

def get_Yn(prompt):
    user_input = input(prompt + " (Y/n): ").strip()
    # If input is empty, 'Y', or 'y', return True; if 'n', return False
    if user_input == '' or user_input.lower() == 'y':
        return True
    elif user_input.lower() == 'n':
        return False
    else:
        print("Invalid input. Please enter 'Y' for Yes or 'n' for No.")
        return get_Yn(prompt)  # Recursively ask for input again if invalid



def signal_handler(signum, frame):
    print("Ctrl+C pressed. Press again within 0.5 seconds to abort. Steady hands matey!")
  
    # Change the signal handler for subsequent Ctrl+C presses
    signal.signal(signal.SIGINT, signal_handler_second)
    time.sleep(0.5)
    signal.signal(signal.SIGINT, signal_handler)
    print("No second Ctrl+C detected, continuing training our robot overlords")

def signal_handler_second(signum, frame):
    raise KeyboardInterrupt
    

def train_model(params, device, net, loss_fn, optimizer, eval_net, print_grads=False, print_loss=False, layernorm_grads=False, post_loss_fn=lambda x,z: None):
    
    train_iter, val_iter, test_iter = get_datasets(params, device)
    plotting = TrainingPlots(seq=len(train_iter), experiment_name=params.short_name + ": " + params.description)

    em = ExperimentManager()
    em.create_experiment(params)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        for epoch in range(params.num_epochs):
            if epoch != 0:
                train_iter.reset()
            with tqdm(total=len(train_iter), desc=f"Epoch {epoch+1}/{params.num_epochs}", position=0, leave=True) as pbar:
                for i, (X, y) in enumerate(train_iter):
                    pbar.set_postfix_str(f"i={i}")
                    pbar.update(1)

                    net.train()
                    #out_sum = ev_net_stepped(X, net)
                    out = net(X)
                    loss = loss_fn(out, y)
                    optimizer.zero_grad()
                    loss.backward()

                    if params.clip_grad_norm != 0.0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=params.clip_grad_norm)

                    if print_grads:
                        # Print all gradients
                        for name, param in net.named_parameters():
                            if param.requires_grad:
                                print(f"Gradient of {name} is {param.grad}")

                    if layernorm_grads:
                        for name, param in net.named_parameters():
                            if param.requires_grad:
                                print(f"Gradient sum of {name} is {torch.sum(param.grad)}")
                                param.grad = F.layer_norm(param.grad, param.grad.shape)
                                print(f"Normed gradient sum of {name} is {torch.sum(param.grad)}")
                    
                    if print_loss:
                        print(f"Loss: {loss}")

                    
                    post_loss_fn(net, params)
                    

                    optimizer.step()

                    plotting.update_loss(loss.item())
                    plotting.update_gradients(net)

                accuracy, precision, recall, f1 = evaluate_model(net, val_iter, eval_net)
                print(f"Validation - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
                em.save_validation_metrics((accuracy, precision, recall, f1))
                val_iter.reset()
                plotting.update_accuracy_metrics(accuracy, precision, recall, f1)
                plotting.plot_all()
    except KeyboardInterrupt:
        try:
            if not get_Yn("Experiment aborted, continue?"):
                em.delete_current_experiment()
                exit()
        except KeyboardInterrupt:
            try:
                em.delete_current_experiment()
                exit()
            except KeyboardInterrupt:
                print("cleanup aborted, only some parts of the current experiment was saved")
                exit()

    accuracy, precision, recall, f1 = test_model(net, test_iter, eval_net)
    print(f"Test - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
    em.save_test_metrics((accuracy, precision, recall, f1))
    em.save_experiment_without_model(net, plotting)

    if get_Yn("Save model?"):
        em.save_model(net)

    print("So long and thanks for all the gpu!")

