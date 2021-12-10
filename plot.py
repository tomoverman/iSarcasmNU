
import argparse
import os
import numpy as np
from plotting_utils import plot_loss_and_accuracy


def main():

    parser = argparse.ArgumentParser(description='iSarcasmNU Visualizations')

    parser.add_argument("model",    type=str, choices=["cnn", "3cnn", "lstm", "lstm_att", "siarn", "miarn", "siarn3"])

    parser.add_argument("action",   type=str, choices=["plot_training_results"])

    parser.add_argument("--training_results",   type=str,       default="",
                        help="path to file containing training results")

    parser.add_argument("--outdir",             type=str,       default="",
                        help="path to directory to save output")

    parser.add_argument("--suffix",             type=str,       default="",
                        help="suffix to append to saved filenames")

    parser.add_argument("--plot_logloss",            action="store_true",
                        help="if given, plot log of the training loss")

    args = parser.parse_args()

    model_name          = args.model.lower()
    action              = args.action
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


if __name__ == "__main__":
    main()
