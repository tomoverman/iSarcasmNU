# Test a pretrained model.
#    Arg 1: model_name
#    Arg 2: path_to_model

# Run with the command
#    sh example_runs/test_model.sh <model_name> <path_to_model>

# Loads the model in the file <path_to_model>
# Saves testing results in the directory out/<model_name>
# Also generates a plot of this data in the directory out/<model_name>
# Saved output will contain the suffix 'test_preloaded' so as not to overwrite original data

python main.py $1 test --load_model $2 --suffix test_preloaded
