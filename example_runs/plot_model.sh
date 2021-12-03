# Plot the training results of a model, from a data file.
#    Arg 1: model_name
#    Arg 2: path_to_training_results

# Run with the command
#    sh example_runs/plot_model.sh <model_name> <path_to_training_results>

# Loads and plots the data in the given text file.
# Generates a plot of this data in the directory out/<model_name>
# Saved output will contain the suffix 'from_file' so as not to overwrite original output

python main.py $1 plot_training_results --training_results $2 --suffix from_file
