# Plot the results of training a CNN model.
# Run with the command
#      sh example_runs/plot_cnn.sh <path_to_training_results>
# Loads and plots the data in the given text file.
# Generates a plot of this data in the directory out/cnn_test_results

python main.py cnn plot_training_results --training_results $1 --outdir out/cnn_test_results