import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_output_size(in_size, padding, dilation, kernel_size, stride):
	out_size = math.floor(((in_size + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1)
	return out_size


class CNN(nn.Module):

	def __init__(self, filter_count, filter_size, embed_size, vocab_size, seq_len, stride=1):
		"""
		:param filter_count: int - output size for convolution+maxpool layer
		:param filter_size: int - kernel size for convolution
		:param embed_size: int - size of the embedding layer
		:param vocab_size: int - size of the vocabulary set
		:param seq_len: int - length of the input token sequence
		:param stride: int - stride for convolution+maxpool layer. Defaults to 1.
		"""

		super(CNN, self).__init__()

		self.filter_count = filter_count
		self.filter_size  = filter_size
		self.embed_size   = embed_size
		self.vocab_size   = vocab_size
		self.seq_len	  = seq_len
		self.stride 	  = stride
		self.padding  	  = 0
		self.dilation     = 1

		# Embedding layer definition
		self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

		# Convolution, Activation, Maxpooling layer
		self.layer1 = nn.Sequential(
			nn.Conv1d(self.seq_len, self.filter_count, self.filter_size, padding=self.padding),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=self.filter_size, stride=stride))

		# Linear feed forward followed by softmax activation
		out_size_conv = get_output_size(self.embed_size, self.padding, self.dilation, self.filter_size, self.stride)
		out_size_pool = get_output_size(out_size_conv, self.padding, self.dilation, self.filter_size, self.stride)
		self.fc = nn.Linear(out_size_pool * self.filter_count, 1)
		self.act = nn.Sigmoid()

	def forward(self, x):
		out = self.embedding(x)
		out = self.layer1(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		out = self.act(out)
		return out.squeeze()


class CNNLayered(nn.Module):

	def __init__(self, filter_counts, filter_sizes, embed_size, vocab_size, seq_len, strides=None):
		"""
		:param filter_counts: list of length num_filters - output size for each convolution+maxpool layer
		:param filter_sizes: list of length num_filters - kernel size for each convolution
		:param embed_size: int - size of the embedding layer
		:param vocab_size: int - size of the vocabulary set
		:param seq_len: int - length of the input token sequence
		:param strides: list of length num_filters - stride for each convolution+maxpool layer. Defaults to all 1.
		"""

		super(CNNLayered, self).__init__()

		self.num_filters 	= len(filter_counts)
		self.filter_counts 	= filter_counts
		self.filter_sizes 	= filter_sizes
		self.embed_size   	= embed_size
		self.vocab_size   	= vocab_size
		self.seq_len	  	= seq_len
		self.strides 	  	= strides if strides else [1 for _ in range(self.num_filters)]
		self.padding        = 0
		self.dilation		= 1

		# Embedding layer definition
		self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

		# Convolution, Activation, Maxpooling layers
		self.layers = []
		out_sizes = []
		for size, count, stride in zip(self.filter_sizes, self.filter_counts, self.strides):
			out_size_conv = get_output_size(self.embed_size, self.padding, self.dilation, size, stride)
			out_size_pool = get_output_size(out_size_conv, self.padding, self.dilation, size, stride)
			out_sizes.append(out_size_pool)

			self.layers.append(nn.Sequential(
				nn.Conv1d(self.seq_len, count, size),
				nn.ReLU(),
				nn.MaxPool1d(kernel_size=size, stride=stride))
			)

		# Linear feed forward followed by softmax activation
		out_size = sum([s * c for s, c in zip(out_sizes, self.filter_counts)])
		self.fc = nn.Linear(out_size, 1)
		self.act = nn.Sigmoid()

	def forward(self, x):
		out = self.embedding(x)

		layer_outs = []
		for layer in self.layers:
			layer_outs.append(layer(out))

		out = torch.cat(layer_outs, 2)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		out = self.act(out)
		return out.squeeze()


