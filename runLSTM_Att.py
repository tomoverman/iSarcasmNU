import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.LSTM_Att import LSTM_Att
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
train_fpath = "data/ptacek_data_train.csv"
test_fpath  = "data/ptacek_data_test.csv"

# train_fpath = "data/data_train.csv"
# test_fpath  = "data/data_test.csv"

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
X_test=np.array(X_test)
Y_test=np.array(Y_test)


#create the torch dataloaders
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
batch_size = 32
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)


EMBEDDING_DIM = 100
HIDDEN_DIM = 100


model = LSTM_Att(EMBEDDING_DIM, HIDDEN_DIM, len(prep.vocabulary),seq_len)
loss_function = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=10**-8)

#gradient clipping stuff to prevent exploding gradient
clip=5

for epoch in range(5):
    for inputs, labels in train_loader:
        # zero the grads
        model.zero_grad()

        # return output
        inputs = inputs.type(torch.LongTensor)
        output = model(inputs)

        # loss and backprop
        loss = loss_function(output.squeeze(), labels.float())
        optimizer.zero_grad()
        loss.backward()

        # prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    print(loss.item())


#find training and testing accuracies
train_predictions = torch.round(model(train_data[:][0].type(torch.LongTensor))).detach().numpy()
train_accuracy = np.sum(train_predictions==Y_train)/Y_train.shape[0]
print("Train Accuracy: " + str(train_accuracy))

test_predictions = torch.round(model(test_data[:][0].type(torch.LongTensor))).detach().numpy()
test_accuracy = np.sum(test_predictions==Y_test)/Y_test.shape[0]
print("Test Accuracy: " + str(test_accuracy))


#find precision, recall, and F-score for testing and training datasets

#find train precision
#precision is (true positives) / ( (true positives) + (false positives))
#positives in this case is that the tweet has sarcasm
true_positives = np.sum(Y_train[train_predictions == Y_train]==1)
false_positives = np.sum(Y_train[train_predictions==1]==0)
print("Training Precision: " + str(true_positives/(true_positives + false_positives)))

#find train recall
#recall is (true positives) / ( (true positives) + (false negatives))
false_negatives = np.sum(Y_train[train_predictions==0]==1)
print("Training Recall: " + str(true_positives/(true_positives + false_negatives)))

#find test precision
#precision is (true positives) / ( (true positives) + (false positives))
#positives in this case is that the tweet has sarcasm
true_positives = np.sum(Y_test[test_predictions == Y_test]==1)
false_positives = np.sum(Y_test[test_predictions==1]==0)
print("Testing Precision: " + str(true_positives/(true_positives + false_positives)))

#find test recall
#recall is (true positives) / ( (true positives) + (false negatives))
false_negatives = np.sum(Y_test[test_predictions==0]==1)
print("Testing Recall: " + str(true_positives/(true_positives + false_negatives)))