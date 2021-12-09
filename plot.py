
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser(description='iSarcasmNU Visualizations')

    parser.add_argument("model",    type=str, choices=["cnn", "3cnn", "lstm", "lstm_att", "siarn", "miarn"])

    parser.add_argument("action",   type=str, choices=["plot_training_results"])

    parser.add_argument("--load_model",         type=str,       default="",
                        help="path to saved model file")

    parser.add_argument("--outdir",             type=str,       default="",
                        help="path to directory to save output")

    parser.add_argument("--suffix",             type=str,       default="",
                        help="suffix to append to saved filenames")

    parser.add_argument("--training_results",   type=str,       default="",
                        help="path to file containing training results")

    parser.add_argument("--plot_logloss",            action="store_true",
                        help="if given, plot log of the training loss")

    args = parser.parse_args()

    model_name          = args.model.lower()
    action              = args.action
    load_path           = args.load_model
    outdir              = args.outdir if args.outdir else f"out/{model_name}"
    save_suffix         = "_" + args.suffix if args.suffix else args.suffix
    train_results_path  = args.training_results
    plot_logloss        = args.plot_logloss

    # Make output directory if needed
    os.makedirs(outdir, exist_ok=True)
    
    # Actions
    if action == "plot_training_results":
        plot_loss_and_accuracy(model_name, None, None, outdir, data_path=train_results_path,
                               save_suffix=save_suffix, plot_logloss=plot_logloss)


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


if __name__ == "__main__":
    main()
