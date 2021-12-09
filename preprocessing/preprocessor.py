import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from torch.utils.data import Dataset
import torch


##########################
##  Preprocessor Class  ##
##########################

class Preprocessor:

	PAD_TOKEN = "<pad>"
	UNK_TOKEN = "<unk>"
	PAD_IDX = 0
	UNK_IDX = 1
	START_IDX = 2


	def __init__(self, seq_len=40, min_len=5, num_tokens=-1):

		self.seq_len   = seq_len
		self.min_len   = min_len
		self.num_tokens = num_tokens

		self.df_train = None
		self.df_test  = None
		self.tweets_train = None
		self.tweets_test  = None
		self.labels_train = None
		self.labels_test  = None

		self.x_train = None
		self.y_train = None

		self.vocabulary = None
		self.V = None

	def initialize(self):
		# Clean training and testing data
		self.tweets_train, self.labels_train = self._clean_data(self.tweets_train, self.labels_train)
		self.tweets_test,  self.labels_test  = self._clean_data(self.tweets_test,  self.labels_test)

		# Build vocabulary
		self.vocabulary, self.fdist = self.build_vocabulary(self.tweets_train, self.seq_len,
															self.min_len, self.num_tokens)
		self.V = len(self.vocabulary)

		# Process raw test and training data
		self.x_train, kept_idxs = self.process(self.tweets_train)
		self.y_train = [self.labels_train[i] for i in kept_idxs]
		self.x_test, kept_idxs  = self.process(self.tweets_test)
		self.y_test  = [self.labels_test[i] for i in kept_idxs]

	def load_data(self, train_data_fpath, test_data_fpath):
		"""
		Load data from the specified training and testing filepaths, if given, and store the read
		tweets and labels. Expects filepaths to point to csv files with 2 columns, containing 
		the text and label. If not given, uses the filepaths specified at instantiation.
		"""
		df_train = pd.read_csv(train_data_fpath)
		df_test  = pd.read_csv(test_data_fpath)
		self.tweets_train  = df_train['text'].values
		self.tweets_test   = df_test['text'].values
		self.labels_train  = df_train['sarcasm_label'].values
		self.labels_test   = df_test['sarcasm_label'].values

	def _clean_data(self, data, labels):
		"""
		Given data, a list of strings, with corresponding labels, remove all entries in data
		containing 'http', and replace all substrings of the form @* with @user.

		Returns a pair: the cleaned data list, and the corresponding labels
		"""
		cleaned_data = replace_ats(data)
		filtered_data, kept_idxs = remove_urls(cleaned_data)
		filtered_labels = [labels[i] for i in kept_idxs]
		return filtered_data, filtered_labels

	def build_vocabulary(self, data, seq_len, min_len, num_tokens=-1):
		"""
		Build a vocabulary from the given data, where data is a list of strings:
			Tokenize the data.
			Replace tokens occurring only once with UNK_TOKEN.
			Truncate to length specified by seq_len.
			Remove all tokenized texts with fewer than min_len tokens.
			Determine top num_tokens most frequent tokens if given (not including unk token)
		Returns the vocabulary mapping tokens to indices, as well as the frequency distribution
		of tokens making up the truncated and filtered data.
		Vocabulary will contain two special tokens, the unknown and padding tokens.
		"""
		
		# Tokenize
		tokenizer = TweetTokenizer()
		tk_data = [[tok.lower() for tok in tokenizer.tokenize(text)] for text in data]

		# Get distribution of tokens
		fdist = self.get_fdist(tk_data)

		# Get tokens occurring only once in the entire corpus
		hapaxes = set(fdist.hapaxes())

		# Truncate each tokenized text at seq_len tokens
		tk_data = self.truncate(tk_data, seq_len)

		# Replace tokens occurring only once with UNK_TOKEN
		for i, tk_text in enumerate(tk_data):
			tk_data[i] = [tk if tk not in hapaxes else self.UNK_TOKEN for tk in tk_text]

		# Remove all tokenized texts with fewer than min_len tokens
		tk_data = [tk_text for tk_text in tk_data if len(tk_text) >= min_len]

		# Recompute distribution
		fdist = self.get_fdist(tk_data)

		# Remove UNK_TOKEN from distribution
		if self.UNK_TOKEN in fdist:
			fdist.pop(self.UNK_TOKEN)

		# Set num_tokens to be the number of unique tokens if not given.
		if num_tokens < 1 or num_tokens > len(fdist):
			num_tokens = len(fdist)
		
		# Ensure vocabulary contains PAD and UNK tokens
		vocabulary = {
			self.PAD_TOKEN: self.PAD_IDX,
			self.UNK_TOKEN: self.UNK_IDX
		}

		# Add tokens to the vocabulary
		common_tks = fdist.most_common(num_tokens)
		i = self.START_IDX
		for word_freq_pair in common_tks:
			vocabulary[word_freq_pair[0]] = i
			i += 1

		# Recompute distribution
		fdist = self.get_fdist(tk_data)

		return vocabulary, fdist
		
	def get_fdist(self, data):
		"""
		Return a frequency distribution of the tokens in data, a list of tokenized strings.
		"""
		fdist = nltk.FreqDist()
		for tk_text in data:
			for token in tk_text:
				fdist[token] += 1
		return fdist

	def tokenize(self, data):
		"""
		Tokenizes the given data, a list of strings, using the vocabulary of tokens. 
		If a token in the given data is not in the vocabulary, replaces it with the UNK_TOKEN.
		Returns a list of lists, where each sublist is a tokenized text.
		Note that the tokenization includes taking the lowercase of the token returned by nltk.
		"""
		tokenizer = TweetTokenizer()
		tk_data = [[tok.lower() for tok in tokenizer.tokenize(text)] for text in data]
		tk_data = [[tk if tk in self.vocabulary else self.UNK_TOKEN for tk in tk_text] for tk_text in tk_data]
		return tk_data

	def truncate(self, tk_data, trunc_len):
		"""
		Given data, a list of lists of tokens, and a truncation length, truncate each tokenized text
		in data to the truncation length.
		"""
		return [tk_text[0:min(len(tk_text), trunc_len)] for tk_text in tk_data]

	def remove_short(self, tk_data, min_len):
		"""
		Given data, a list of lists of tokens, and a minimum length, remove all tokenized texts with
		length lower than specified minimum.
		Returns the filtered data, and a list of the indices kept.
		"""
		kept_idxs = [i for i in range(len(tk_data)) if len(tk_data[i]) >= min_len]
		filtered_data = [tk_data[i] for i in kept_idxs]
		return filtered_data, kept_idxs

	def pad(self, tk_data, length):
		"""
		Pad each tokenized string in data with PAD_TOKEN if 
		length is less than the specified length.

		Returns padded data.
		"""
		return [[self.PAD_TOKEN for _ in range(length - len(tk_text))] + tk_text for tk_text in tk_data]

	def process(self, data, convert_to_idxs=True):
		"""
		Given raw data, a list of strings, tokenize, truncate, filter out short-length
		tokenized values, pad, and convert tokens to indices unless flag is specified 
		otherwise. In that case, returns tokens instead of corresponding indices.
		Return a pair: list of indices (or tokens), and indices of kept values from input data.
		"""
		tk_data = self.tokenize(data)
		tk_data = self.truncate(tk_data, self.seq_len)
		tk_data, kept_idxs = self.remove_short(tk_data, self.min_len)
		tk_data = self.pad(tk_data, self.seq_len)
		if convert_to_idxs:
			return [[self.vocabulary[token] for token in tk_text] for tk_text in tk_data], kept_idxs
		else:
			return tk_data, kept_idxs

	def get_dataset_train(self,use_gpu,validation):
		return TokenizedDataset(self.x_train, self.y_train, use_gpu, validation)

	def get_dataset_valid(self,use_gpu):
		return TokenizedDataset(self.x_train, self.y_train, use_gpu, valid="Valid")

	def get_dataset_test(self,use_gpu):
		return TokenizedDataset(self.x_test, self.y_test, use_gpu, valid=False)


#####################
##  Dataset Class  ##
#####################

class TokenizedDataset(Dataset):

	def __init__(self, x, y, use_gpu, valid):
		if valid=="Valid":
			# case when building the validation set, take 15% of the training set to use as validation set.
			spot = int(np.floor(len(x)*.85))
			x=x[spot:]
			y=y[spot:]
		elif valid:
			# case when building the training set and we need to leave some of the dataset for validation
			spot = int(np.floor(len(x) * .85))
			x = x[0:spot]
			y = y[0:spot]
		else:
			# case for testing set and if we are building training set without validation set, no change to x,y
			pass
		if not use_gpu:
			self.x = np.array(x)
			self.y = np.array(y)
		else:
			self.x = torch.tensor(np.array(x)).to("cuda")
			self.y = torch.tensor(np.array(y)).to("cuda")

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


########################
##  Helper Functions  ##
########################

def remove_urls(lst):
	"""
	Given a list of strings, remove all entries containing 'http'. 
	Does not mutate given list.
	Returns a pair: the filtered list, and the indices of the kept values.
	"""
	kept_idxs = [i for i in range(len(lst)) if 'http' not in lst[i]]
	return [lst[i] for i in kept_idxs], kept_idxs
	
	
def replace_ats(lst):
	"""
	Replace all occurrences of @* with @user in the strings in list lst. 
	Does not mutate given list.
	"""
	return [replace_ats_in_string(s) for s in lst]


def replace_ats_in_string(text):
	"""
	Replace all occurrences of @* with @user in the given string.
	"""
	words = text.split()
	words = [w[0] + 'user' if w[0] == '@' and len(w) > 1 else w for w in words]
	return " ".join(words)
