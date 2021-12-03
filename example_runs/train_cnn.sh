# Train a CNN model over a specified number of epochs.
# Run with the command
#      sh example_runs/train_cnn.sh <n>
# where n is the desired number of training epochs.
# Saves the model in the file pretrained/cnn_<n>.pth
# Saves training and testing data (losses and accuracies) in the directory out/cnn
# Also generates a plot of this data in the directory out/cnn

python main.py cnn train --num_epochs $1 --save_model pretrained/cnn_$1.pth --outdir out/cnn