import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.LSTMSarcasm import LSTMSarcasm
import csv
from preprocessing.preprocessor import Preprocessor

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return idxs


# training_data=[]
# with open('../data/data_train.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     next(reader)
#     for row in reader:
#         training_data.append((row[0],int(row[1])))
#
# ###TEMPORARY
# training_data=training_data[:-7]
#
# #use tweet tokenizer!!! Documents with http removed as well as
#
# word_to_ix = {}
# # For each words-list (sentence) and tags-list in each tuple of training_data
# for tweet, label in training_data:
#     for word in tweet.split(' '):
#         if word not in word_to_ix:  # word has not been assigned an index yet
#             word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
#
# #create the final X_train with padded and truncated elements
# #use 40 for max sentence length
# tweet_length = 70
# X_train = np.zeros((len(training_data), tweet_length))
# Y_train = np.zeros(len(training_data))
# c=0
# for tweet,label in training_data:
#     tweet_in = prepare_sequence(tweet.split(' '), word_to_ix)
#     X_train[c] = np.array(list(np.zeros(tweet_length-len(tweet_in))) + tweet_in)
#     Y_train[c] = label
#     c+=1

#use the preprocessing functions
train_fpath = "data/data_train.csv"
test_fpath  = "data/data_test.csv"

seq_len = 40
min_len = 5

prep = Preprocessor(seq_len=seq_len, min_len=min_len)
prep.load_data(train_fpath, test_fpath)
prep.initialize()

X_train, kept_idxs = prep.process(prep.tweets_train, convert_to_idxs=True)
Y_train = [prep.labels_train[i] for i in kept_idxs]
X_test, kept_idxs  = prep.process(prep.tweets_test,  convert_to_idxs=True)
Y_test  = [prep.labels_test[i] for i in kept_idxs]

X_train=np.array(X_train)
Y_train=np.array(Y_train)


#create the torch dataloaders
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
batch_size = 200
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)


EMBEDDING_DIM = 100
HIDDEN_DIM = 100


model = LSTMSarcasm(EMBEDDING_DIM, HIDDEN_DIM, len(prep.vocabulary))
loss_function = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=10**-8)

#gradient clipping stuff to prevent exploding gradient
clip=5
print(model(train_data[0:30][0].type(torch.LongTensor)))
for epoch in range(30):
    for inputs, labels in train_loader:
        # zero the grads
        model.zero_grad()

        # return output
        inputs = inputs.type(torch.LongTensor)
        output = model(inputs)

        # loss and backprop
        loss = loss_function(output.squeeze(), labels.float())
        loss.backward()

        # prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    print(loss.item())

print(model(train_data[0:30][0].type(torch.LongTensor)))
print(train_data[0:30][1])