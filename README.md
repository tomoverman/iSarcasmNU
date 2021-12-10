# iSarcasmNU
<h1>Deep learning for sarcasm detection</h1>
Models trained on iSarcasm and Ptacek datasets. Both datasets are comprised of tweets and corresponding sarcasm labels.
<br>
<b>Models implemented:</b>
<ol>
  <li>LSTM</li>
  <li>LSTM with Attention</li>
  <li>CNN</li>
  <li>3CNN</li>
  <li>SIARN</li>
  <li>MIARN</li>
  <li>SIARN3</li>
</ol>

## Installation
We recommend the use of conda environments for installation. After downloading, cd into the project directory and run

```
conda env create -f environment.yml
conda activate isarcasmenv
```

## Usage
The script main.py provides the starting point for training and testing of a model. The user must specify the type of model and the action to take, as follows

```
python main.py <model> <action> <options>
```

Models: ```cnn```, ```3cnn```, ```lstm```, ```lstm_att```, ```siarn```, ```miarn```, and ```siarn3```.

Actions: ```train```, ```test```, and ```long_train```

Common Options:
  - ```--data_train_fpath```: path to training data (defaults to data/ptacek_data_train.csv)
  - ```--data_test_fpath```: path to training data (defaults to data/ptacek_data_test.csv)
  - ```--cuda```: flag indicating that a GPU should be used if available (if not given, runs on CPU)
  - ```--num_epochs```: number of training epochs (defaults to 512)
  - ```--batch_size```: batch size to use (defaults to 30)
  - ```--load_model```: path to the pretrained model file that will be loaded, when testing
  - ```--save_model```: path to which the model will be saved after training (will not save if not given)
  - ```--outdir```: path to the directory to which output training data will be saved (defaults to out/<model>)
  - ```--suffix```: a suffix to append to any output filenames
  - ```--storage_step```: for long training, step size of epochs in which model parameters are saved (default is 1)

Please see the source code for additional options. For the purpose of outputting results, note that any models saved using the ```--save_model``` option require that the prefix directories exist, while the prefix for output specified with the ```--outdir``` option will be created at runtime if it does not already exist. 

## Examples

### Train an LSTM
  
  Train the LSTM over 30 epochs with a batch size of 256, and save the resulting model. Use a GPU if available.

```
python main.py lstm train --num_epochs 30 --batch_size 256 --cuda --save_model pretrained/lstm_example.pth --suffix example
```

This will generate the following output:

  - ```pretrained/lstm_example.pth```: file containing the saved model parameters.
  - ```out/lstm/training_results_lstm_example.txt```: a text file containing training losses and accuracies at the end of each epoch.
  - ```out/lstm/training_results_lstm_example.png```: a plot of the training loss and accuracy over the course of training.
  - ```out/lstm/testing_results_lstm_example.txt```: a text file containing the testing precision, recall, accuracy, and F-score.

### Train a CNN
  
  Train the CNN over 30 epochs with a batch size of 512, specifying the output directory and a suffix for any output files. Use a GPU if available.

```
python main.py cnn train --num_epochs 30 --batch_size 512 --cuda --outdir out/example \
                         --save_model pretrained/cnn_example.pth --suffix example
```

This will generate the following output:
  
  - ```pretrained/cnn_example.pth```
  - ```out/example/training_results_cnn_example.txt```
  - ```out/example/training_results_cnn_example.png```
  - ```out/example/testing_results_cnn_example.txt```

### Test a pretrained CNN
  
  Load a pretrained model from the specified path and test it on the default testing data.

```
python main.py cnn test --load_model pretrained/cnn_example.pth
```

Results are output to the terminal.
  
### Train an LSTM with Attention, with periodic saves

  Train the model over 100 epochs, saving the model state every 5 epochs, and select as a final model the one with the highest F-score, using a subset of the training data as a validation set.
  
```
python main.py lstm_att long_train --num_epochs 100 --storage_step 5 --save_model pretrained/lstm_att_best.pth \
                                   --suffix long --outdir out/example --validation_criterion fscore --cuda
```

This will generate the following output:

- ```out/long_train/lstm_att/{5,10,15,...,100}.pth```: 20 model files saved during the course of training.
- ```pretrained/lstm_att_best.pth```: the best model, based on the validation F-score.
- ```out/example/training_results_lstm_att_long.txt```: training results of the best model.
- ```out/example/training_results_lstm_att_long.png```: a plot of the training results for the best model.
- ```out/example/testing_results_lstm_att_long.txt```: testing results for the best model.

## Pretrained Models

The directory ```pretrained/``` contains a collection of pretrained models in the form of <model_name>.pth files. These can be loaded and tested as shown in the previous examples.

## Data and Preprocessing

The data intended for use in this project can be aquired using the Twitter API, and may require an access token. The scripts in the preprocessing module handle the download and cleaning of data. Please refer to the source code for additional details regarding preprocessing steps.

## Plotting and Visualization

plot.py is intended to handle visualization of results. Post hoc plotting can be handled in this way, reading results from output training data files. Please see the source code for usage information.

## Authors and Acknowledgment
  Authors: Addison Howe and Tom Overman
  
  Citations:
  - Silviu Oprea and Walid Magdy. isarcasm: A dataset of intended sarcasm. http://arxiv.org/abs/1911.03123
  - Yi Tay, Luu Anh Tuan, Siu Cheung Hui, and Jian Su. Reasoning with sarcasm by reading in-between. http://arxiv.org/abs/1805.02856
  

  
