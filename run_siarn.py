import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.siarn import SIARN
import csv
from preprocessing.preprocessor import Preprocessor

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return idxs
def get_accuracy(y_pred, y_true):
	return (y_pred == y_true).sum().float() / y_true.shape[0]

def get_false_positives(y_pred, y_true):
	return torch.logical_and(y_pred != y_true, y_pred == 1).sum().int()

def get_false_negatives(y_pred, y_true):
	return torch.logical_and(y_pred != y_true, y_pred == 0).sum().int()

def get_true_positives(y_pred, y_true):
	return torch.logical_and(y_pred == y_true, y_pred == 1).sum().int()

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
batch_size = 256
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


EMBEDDING_DIM = 100
HIDDEN_DIM = 100


model = SIARN(EMBEDDING_DIM, HIDDEN_DIM, len(prep.vocabulary),seq_len)
loss_function = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=10**-8)

#gradient clipping stuff to prevent exploding gradient
clip=5

for epoch in range(30):
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

###############
##  Testing  ##
###############

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    for xs, labels in test_loader:
        outputs = model(xs)
        pred_labels = torch.round(outputs).int()
        total += len(labels)
        correct += (pred_labels == labels).sum().int()
        false_negatives += get_false_negatives(pred_labels, labels)
        false_positives += get_false_positives(pred_labels, labels)
        true_positives += get_true_positives(pred_labels, labels)

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
accuracy = correct / total

print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test Accuracy: {accuracy}")


# #find training and testing accuracies
# # split into two sets to prevent memory overload
# train_predictions1 = torch.round(model(train_data[0:X_train.shape[0]//2][0].type(torch.LongTensor))).detach().numpy()
# train_predictions2 = torch.round(model(train_data[X_train.shape[0]//2:][0].type(torch.LongTensor))).detach().numpy()
# train_predictions = np.concatenate(train_predictions1, train_predictions2)
# train_accuracy = np.sum(train_predictions==Y_train)/Y_train.shape[0]
# print("Train Accuracy: " + str(train_accuracy))
#
# test_predictions1 = torch.round(model(test_data[0:X_test.shape[0]//2][0].type(torch.LongTensor))).detach().numpy()
# test_predictions2 = torch.round(model(test_data[X_test.shape[0]//2:][0].type(torch.LongTensor))).detach().numpy()
# test_predictions = np.concatenate(test_predictions1, test_predictions2)
# test_accuracy = np.sum(test_predictions==Y_test)/Y_test.shape[0]
# print("Test Accuracy: " + str(test_accuracy))
#
#
# #find precision, recall, and F-score for testing and training datasets
#
# #find train precision
# #precision is (true positives) / ( (true positives) + (false positives))
# #positives in this case is that the tweet has sarcasm
# true_positives = np.sum(Y_train[train_predictions == Y_train]==1)
# false_positives = np.sum(Y_train[train_predictions==1]==0)
# print("Training Precision: " + str(true_positives/(true_positives + false_positives)))
#
# #find train recall
# #recall is (true positives) / ( (true positives) + (false negatives))
# false_negatives = np.sum(Y_train[train_predictions==0]==1)
# print("Training Recall: " + str(true_positives/(true_positives + false_negatives)))
#
# #find test precision
# #precision is (true positives) / ( (true positives) + (false positives))
# #positives in this case is that the tweet has sarcasm
# true_positives = np.sum(Y_test[test_predictions == Y_test]==1)
# false_positives = np.sum(Y_test[test_predictions==1]==0)
# print("Testing Precision: " + str(true_positives/(true_positives + false_positives)))
#
# #find test recall
# #recall is (true positives) / ( (true positives) + (false negatives))
# false_negatives = np.sum(Y_test[test_predictions==0]==1)
# print("Testing Recall: " + str(true_positives/(true_positives + false_negatives)))