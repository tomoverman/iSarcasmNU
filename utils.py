import argparse
import os
from models.cnn import CNN, CNNLayered
from models.LSTMSarcasm import LSTMSarcasm
from models.LSTM_Att import LSTM_Att
from models.SIARN import SIARN
# from models.MIARN import MIARN
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


MODELS = {
    "cnn": CNN,
    "3cnn": CNNLayered,
    "lstm": LSTMSarcasm,
    "lstm_att": LSTM_Att,
    "siarn": SIARN,
    # "miarn": MIARN
}


def select_model(model_name, embed_size, vocab_size, seq_len, load_path=""):
    """
    Given the name of the desired model, and an optional path to a pretrained model, construct
    and return the model, loading its parameters from the .pth file if specified.
    """

    if model_name == "cnn":
        # Parameters for CNN
        filter_count = 100
        filter_size = 3
        params = [filter_count, filter_size, embed_size, vocab_size, seq_len]

    elif model_name == "3cnn":
        # Parameters for 3CNN
        filter_counts   = [100, 100, 100]
        filter_sizes    = [3, 4, 5]
        params = [filter_counts, filter_sizes, embed_size, vocab_size, seq_len]

    elif model_name == "lstm":
        # Parameters for LSTM
        hidden_dim = 100
        params = [embed_size, hidden_dim, vocab_size]

    elif model_name == "lstm_att":
        # Parameters for LSTM_ATT
        hidden_dim = 100
        params = [embed_size, hidden_dim, vocab_size, seq_len]

    elif model_name == "siarn":
        # Parameters for SIARN
        hidden_dim = 100
        params = [embed_size, hidden_dim, vocab_size, seq_len]

    elif model_name == 'miarn':
        # Parameters for MIARN
        # hidden_dim = 100
        # params = []
        pass

    # Construct the model
    model = MODELS[model_name](*params)
    
    # If specified, load a pretrained model
    if load_path:
        model.load_state_dict(torch.load(load_path))
        model.eval()

    return model


def train(model, num_epochs, train_loader, optimizer, loss_function, clip):
    """
    Train the given model over a number of epochs, using the data in train_loader.
    Uses the specified optimizer and los function. Clip gives the amount to clip
    the gradient to avoid exploding gradients.
    Returns two lists: the training losses and the accuracies at each iteration.
    """

    train_losses = []
    accuracies   = []
    
    model.train()
    for epoch in range(1, num_epochs + 1):
        for i, (xs, labels) in enumerate(train_loader):

            # Perform forward pass
            outputs = model(xs)
            pred_labels = convert_output_to_label(outputs)

            # Compute and store loss and accuracy
            loss = loss_function(outputs, labels.float())
            accuracy = get_accuracy(pred_labels, labels)
            train_losses.append(loss.item())
            accuracies.append(accuracy)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # Computes accuracy only once at the end of each epoch
        # pred_labels = convert_output_to_label(outputs)
        # accuracy = get_accuracy(pred_labels, labels)
        # train_losses.append(loss.item())
        # accuracies.append(accuracy)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')

    return train_losses, accuracies


def test(model, test_loader):
    """
    Test the given model on the test data in test_loader.
    Returns the precision, recall, accuracy, and F-score.
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        false_positives = 0
        false_negatives = 0
        true_positives  = 0
        for xs, labels in test_loader:
            outputs = model(xs)
            pred_labels = convert_output_to_label(outputs)
            total += len(labels)
            correct += (pred_labels == labels).sum().int()
            false_negatives += get_false_negatives(pred_labels, labels)
            false_positives += get_false_positives(pred_labels, labels)
            true_positives  += get_true_positives(pred_labels, labels)

    # Calculate precision, recall, accuracy, and F-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = correct / total
    fscore = 2. * (precision * recall) / (precision + recall)

    print(f"Test Precision: {precision}")
    print(f"Test Recall:    {recall}")
    print(f"Test Accuracy:  {accuracy}")
    print(f"Test F-score:   {fscore}")

    return precision, recall, accuracy, fscore


def plot_training_data(model_name, losses, accuracies, outdir, data_path=""):
    """
    Plots the results of training a model. Given lists LOSSES and ACCURACIES, 
    or a path to a file containing this data. Saves the figure in the directory
    specified by OUTDIR.
    """
    
    if data_path: 
        xs, losses, accuracies = np.genfromtxt(data_path).T

    fig, [ax1, ax2] = plt.subplots(2, 1)
    xs = range(len(losses))
    
    ax1.plot(xs, losses)
    ax1.set_xlabel("iter")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{model_name} Training Loss")
    
    ax2.plot(xs, accuracies)
    ax2.set_xlabel("iter")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"{model_name} Training Accuracy")

    fig.tight_layout()

    outpath = f"{outdir}/img_training_results_{model_name}.png"
    plt.savefig(outpath)


def save_training_results(model_name, losses, accuracies, outdir):
    np.savetxt(f"{outdir}/training_results_{model_name}.txt", 
               np.array([range(len(losses)), losses, accuracies]).T,
               header='iter, loss, accuracy')


def save_testing_results(model_name, precision, recall, accuracy, fscore, outdir):
    np.savetxt(f"{outdir}/testing_results_{model_name}.txt", 
               np.array([precision, recall, accuracy, fscore]),
               header='precision, recall, accuracy, fscore')


########################
##  Helper functions  ##
########################

def convert_output_to_label(output):
    return torch.round(output).int()

def get_accuracy(y_pred, y_true):
    return (y_pred == y_true).sum().float() / y_true.shape[0]

def get_false_positives(y_pred, y_true):
    return torch.logical_and(y_pred != y_true, y_pred == 1).sum().int()

def get_false_negatives(y_pred, y_true):
    return torch.logical_and(y_pred != y_true, y_pred == 0).sum().int()

def get_true_positives(y_pred, y_true):
    return torch.logical_and(y_pred == y_true, y_pred == 1).sum().int()
