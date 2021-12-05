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
Train an LSTM over 30 epochs with a batch size of 256.

```
python main.py lstm train --save_model <path_to_save.pth> --num_epochs 30 --batch_size 256
```

Test a pretrained CNN.

```
python main.py cnn test --load_model <path_to_pretrained.pth>
```
