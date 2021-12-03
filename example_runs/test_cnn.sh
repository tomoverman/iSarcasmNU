# Test a pretrained CNN model.
# Run with the command
#      sh example_runs/test_cnn.sh <path_to_model>
# Loads the model in the file <path_to_model>
# Saves testing results in the directory out/cnn_test_results, 
# which will be created if it doesn't already exist.

python main.py cnn test --load_model $1 --outdir out/cnn_test_results