import matplotlib as mlp
#mlp.use("TkAgg")
import matplotlib.pyplot as plt
import os
import numpy as np

class TrainingPlots:
    def __init__(self, seq, experiment_name, plot_loss=True, plot_accuracy=True, plot_gradients=True):
        self.plot_loss = plot_loss
        self.plot_accuracy = plot_accuracy
        self.plot_gradients = plot_gradients
        self.seq = seq


        self.rolling_loss = 50
        self.rolling_acc = 4
        self.loss_hist = []
        self.loss_hist_avg_rolling = []
        self.acc_hist = []
        self.acc_hist_avg_rolling = []
        self.prec_hist = []
        self.recall_hist = []
        self.f1_hist = []
        self.grads_dict = {}

        # Count the number of plots to display
        self.num_plots = sum([plot_loss, plot_accuracy, plot_gradients])
        self.fig, self.axes = plt.subplots(self.num_plots, 1, figsize=(10, 5 * self.num_plots))
        
        self.fig.suptitle(experiment_name)
        self.fig.canvas.manager.set_window_title(experiment_name)


        if self.num_plots == 1:
            self.axes = [self.axes]  # Ensure axes is always a list

    def update_loss(self, loss):
        if self.plot_loss:
            self.loss_hist.append(loss)
            if len(self.loss_hist) > self.rolling_loss:
                self.loss_hist_avg_rolling.append(np.average(self.loss_hist[-self.rolling_loss:]))


    def update_accuracy_metrics(self, accuracy, precision, recall, f1):
        if self.plot_accuracy:
            self.acc_hist.append(accuracy)
            self.prec_hist.append(precision)
            self.recall_hist.append(recall)
            self.f1_hist.append(f1)
            
            if len(self.acc_hist) > self.rolling_acc:
                self.acc_hist_avg_rolling.append(np.average(self.acc_hist[-self.rolling_acc:]))


    def update_gradients(self, net):
        if not self.plot_gradients:
            return

        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.norm().item()
                if name in self.grads_dict:
                    self.grads_dict[name].append(grad_norm)
                else:
                    self.grads_dict[name] = [grad_norm]

    def reset_gradients(self):
        self.grads_dict = dict()

    def plot_all(self):
        plot_idx = 0  # Index to keep track of which subplot to use

        if self.plot_loss and plot_idx < self.num_plots:
            ax = self.axes[plot_idx]
            ax.clear()
            ax.plot(self.loss_hist[-self.seq:], label='Training Loss')
           
            ax.plot(self.loss_hist_avg_rolling[-self.seq+self.rolling_loss:])  
            ax.set_title('Training Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend()
            plot_idx += 1

        if self.plot_accuracy and plot_idx < self.num_plots:
            ax = self.axes[plot_idx]
            ax.clear()
            ax.plot(self.acc_hist, label='Accuracy')
            ax.plot(self.prec_hist, label='Precision')
            ax.plot(self.recall_hist, label='Recall')
            ax.plot(self.f1_hist, label='F1 Score')
            ax.plot(self.acc_hist_avg_rolling, label='Rolling average')
            ax.set_title('Validation Metrics')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')
            ax.legend()
            plot_idx += 1

        if self.plot_gradients and plot_idx < self.num_plots:
            ax = self.axes[plot_idx]
            ax.clear()
            for name, grad_history in self.grads_dict.items():
                ax.plot(grad_history[-self.seq:], label=f'{name} grad')
            ax.set_title('Gradient Norms')
            ax.legend()
            #max_grad = max(self.grads_dict[name][-self.seq:] for name in self.grads_dict)
            #ax.set_ylim([0, max_grad])

        self.fig.canvas.draw()
        plt.pause(0.1)
    
    def plot_and_pause(self):
        plt.show()
        input("Press any key to continue")

    def save_plot(self, experiment_folder, file_name="training_plots.png"):
        """
        Saves the current plots to an image file within the specified experiment folder.

        Parameters:
        - experiment_folder: str, the path to the experiment folder where the plot image will be saved.
        - file_name: str, the name of the file to save the plots as. Default is "training_plots.png".
        """
        plot_path = os.path.join(experiment_folder, file_name)
        self.fig.savefig(plot_path)
        print(f"Plot saved to {plot_path}")