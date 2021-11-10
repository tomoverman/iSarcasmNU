import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from LSTMSarcasm import LSTMSarcasm
import csv

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return idxs


training_data=[]
with open('../data/data_train.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        training_data.append((row[0],int(row[1])))

###TEMPORARY
training_data=training_data[:-7]

#use tweet tokenizer!!! Documents with http removed as well as

word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for tweet, label in training_data:
    for word in tweet.split(' '):
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index

#create the final X_train with padded and truncated elements
#use 40 for max sentence length
tweet_length = 70
X_train = np.zeros((len(training_data), tweet_length))
Y_train = np.zeros(len(training_data))
c=0
for tweet,label in training_data:
    tweet_in = prepare_sequence(tweet.split(' '), word_to_ix)
    X_train[c] = np.array(list(np.zeros(tweet_length-len(tweet_in))) + tweet_in)
    Y_train[c] = label
    c+=1

#create the torch dataloaders
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
batch_size = 200
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)


EMBEDDING_DIM = 100
HIDDEN_DIM = 100


model = LSTMSarcasm(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix))
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#gradient clipping stuff to prevent exploding gradient
clip=5
print(model(train_data[0:30][0].type(torch.LongTensor)))
for epoch in range(300):
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