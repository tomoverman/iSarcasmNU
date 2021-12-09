import numpy as np
import matplotlib.pyplot as plt


############################################
##  Plotting and Visualization Functions  ##
############################################

def plot_loss_and_accuracy(model_name, losses, accuracies, outdir,
                           data_path="", save_suffix="", plot_logloss=False):
    """
    Plots the results of training a model. Given lists LOSSES and ACCURACIES, 
    or a path to a file containing this data. Saves the figure in the directory
    specified by OUTDIR.
    """
    
    if data_path:
        xs, losses, accuracies = np.genfromtxt(data_path).T
        print(f"Loading results located at {data_path}")

    fig, [ax1, ax2] = plt.subplots(2, 1)
    xs = range(len(losses))
    
    if plot_logloss:
        ax1.semilogy(xs, losses)
    else:
        ax1.plot(xs, losses)
        
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{model_name.upper()} Training Loss")
    
    ax2.plot(xs, accuracies)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"{model_name.upper()} Training Accuracy")

    fig.tight_layout()

    fpath = f"{outdir}/training_results_{model_name}{save_suffix}.png"
    plt.savefig(fpath)
    print(f"Plots saved to {fpath}")
