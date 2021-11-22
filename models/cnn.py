import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

	def __init__(self, params):

		super(CNN, self).__init__()

		self.num_filters = params['num_filters']
		self.filter_size = params['filter_size']
		self.embed_size  = params['embed_size']
		self.vocab_size  = params['vocab_size']
		self.seq_len	 = params['seq_len']

		# Embedding layer definition
		self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

		# Convolution, Activation, Maxpooling layer
		self.layer1 = nn.Sequential(
			nn.Conv1d(self.seq_len, self.num_filters, self.filter_size, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2, stride=2))

		# Linear feed forward
		self.fc = nn.Linear(self.embed_size // 2 * self.num_filters, 1)

	def forward(self, x):
		out = self.embedding(x)
		out = self.layer1(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out.squeeze()







