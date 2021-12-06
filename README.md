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
</ol>

## Installation
We recommend the use of conda environments for installation. After downloading, cd into the project directory and run

```
conda env create -f environment.yml
conda activate isarcasmenv
```

## Examples
Train an LSTM model on the GPU over 30 epochs with a batch size of 256, and save the resulting model to the file pretrained/lstm_30_256.pth.
An output data file containing the training losses and accuracies will be saved to out/lstm/training_results_lstm_30_256.txt.
In addition, a plot of this data is generated and saved to the file out/lstm/training_results_lstm_30_256.png.
Finally, the model is tested on the testing data and results are saved to the file out/lstm/testing_results_lstm_30_256.txt.

```
python main.py lstm train --save_model pretrained/lstm_30_256.pth --num_epochs 30 --batch_size 256 --cuda
```

Train a CNN model on the CPU over 30 epochs with a batch size of 512.
Name the saved model pretrained/examplecnn.pth and save all other output to the directory myoutput. 
In addition, append all generated output filenames with "\_example" so that the resulting outputs are

- pretrained/examplecnn.pth
- myoutput/training_results_miarn_example.txt
- myoutput/training_results_miarn_example.png
- myoutput/testing_results_miarn_example.txt

```
python main.py cnn train --save_model pretrained/examplecnn.pth --num_epochs 30 --batch_size 512 --outdir myoutput --suffix example
```

Test a pretrained CNN model. Results are output to the terminal.

```
python main.py cnn test --load_model pretrained/examplecnn.pth
```
