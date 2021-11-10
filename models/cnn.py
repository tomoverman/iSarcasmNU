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

		# Embedding layer definition  # BCEcrossentropy l2 regularizer in rnsprop
		self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

		# Convolution
		self.conv = nn.Conv1d(self.seq_len, self.num_filters, self.filter_size)

		# Max Pool
		self.pool = nn.MaxPool1d(self.filter_size, 1)

		# Linear feed forward
		self.linear = nn.Linear(self.num_filters, self.embed_size)

	def forward(self, x):
		r1 = self.embedding(x)
		r1 = torch.transpose(r1, 2, 1)

		conv_out = self.conv(r1)

		return self.linear(conv_out)







