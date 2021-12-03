# Train a model over a specified number of epochs and with a given batch size.
#    Arg 1: model type
#    Arg 2: num_epochs
#    Arg 3: batch size

# Run with the command
#    sh example_runs/train_model.sh <model_name> <num_epochs> <batch_size>

# Saves the model in the file pretrained/<model_name>_<num_epochs>_<batch_size>.pth
# Saves training and testing data (losses and accuracies) in the directory out/<model_name>
# Also generates a plot of this data in the directory out/<model_name>

python main.py $1 train --save_model pretrained/$1_$2_$3.pth --num_epochs $2 --batch_size $3 --suffix $2_$3
