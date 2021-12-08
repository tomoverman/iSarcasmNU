import argparse
import os
import time
from models.cnn import CNN, CNNLayered
from models.lstm import LSTM, LSTMAtt
from models.SIARN import SIARN
from models.MIARN import MIARN
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


MODELS = {
    "cnn": CNN,
    "3cnn": CNNLayered,
    "lstm": LSTM,
    "lstm_att": LSTMAtt,
    "siarn": SIARN,
    "miarn": MIARN
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
        hidden_dim = 100
        params = [embed_size, hidden_dim, vocab_size, seq_len]

    # Construct the model
    model = MODELS[model_name](*params)
    print(f"Constructed model {model_name.upper()}.")
    
    # If specified, load a pretrained model
    if load_path:
        model.load_state_dict(torch.load(load_path))
        model.eval()
        print(f"Model loaded from {load_path}")

    return model


def train_model(model, num_epochs, train_loader, optimizer, loss_function, clip):
    """
    Train the given model over a number of epochs, using the data in train_loader.
    Uses the specified optimizer and los function. Clip gives the amount to clip
    the gradient to avoid exploding gradients.
    Returns two lists: the training losses and the accuracies at each iteration.
    """

    print("Training model...")
    time0 = time.time()
    t0 = time0

    train_losses = []
    accuracies   = []
    
    model.train()
    for epoch in range(1, num_epochs + 1):
        for i, (xs, labels) in enumerate(train_loader):

            # Perform forward pass
            outputs = model(xs)

            # Compute loss and perform backprop
            loss = loss_function(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()

            # Prevent exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # Compute accuracy and save it as well as the training loss
        pred_labels = convert_output_to_label(outputs)
        accuracy = get_accuracy(pred_labels, labels)
        train_losses.append(loss.item())
        accuracies.append(accuracy)        
        t1 = time.time()
        time_elapsed = (t1 - t0)
        t0 = t1

        print(f'\tEpoch [{epoch}/{num_epochs}], Time: {time_elapsed:.2f} sec, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')

    time_elapsed = time.time() - time0
    print(f"    Time Elapsed: {time_elapsed:.2f} sec")

    return train_losses, accuracies


def long_train_model(model, model_name, train_loader, optimizer, loss_function, clip):
    """
    Train the given model over a number of epochs, using the data in train_loader.
    Uses the specified optimizer and los function. Clip gives the amount to clip
    the gradient to avoid exploding gradients.
    Returns two lists: the training losses and the accuracies at each iteration.
    """

    base_path = "out/long_train/" + model_name + "/"
    print("Training model...")
    time0 = time.time()
    t0 = time0

    train_losses = []
    accuracies = []

    model.train()
    # train for many, many epochs to get a large parameter space
    num_epochs=200
    for epoch in range(1, num_epochs + 1):
        for i, (xs, labels) in enumerate(train_loader):
            # Perform forward pass
            outputs = model(xs)

            # Compute loss and perform backprop
            loss = loss_function(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()

            # Prevent exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # Compute accuracy and save it as well as the training loss
        pred_labels = convert_output_to_label(outputs)
        accuracy = get_accuracy(pred_labels, labels)
        train_losses.append(loss.item())
        accuracies.append(accuracy)
        t1 = time.time()
        time_elapsed = (t1 - t0)
        t0 = t1

        print(
            f'\tEpoch [{epoch}/{num_epochs}], Time: {time_elapsed:.2f} sec, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')

        #save model params every 5 epochs
        if epoch%5==0:
            torch.save(model.state_dict(), base_path + str(epoch) + ".pth")

    time_elapsed = time.time() - time0
    print(f"    Time Elapsed: {time_elapsed:.2f} sec")

    return train_losses, accuracies

def evaluate_long_train(model_name, valid_loader, embed_size, vocab_size, seq_len):
    base_path = "out/long_train/" + model_name + "/"
    num_epochs=200
    accuracy=np.zeros(len(range(5,num_epochs+1,5)))
    for i,epoch in enumerate(range(5,num_epochs+1,5)):
        model_path=base_path + str(epoch) + ".pth"
        model = select_model(model_name, embed_size, vocab_size, seq_len, model_path)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for xs, labels in valid_loader:
                outputs = model(xs)
                pred_labels = convert_output_to_label(outputs)
                total += len(labels)
                correct += (pred_labels == labels).sum().int()

        accuracy[i]=correct/total

    # choose best model based on accuracy
    best_i = np.argmax(accuracy)

    return accuracy[best_i], base_path + str(best_i*5+5) + ".pth"

def test_model(model, test_loader):
    """
    Test the given model on the test data in test_loader.
    Returns the precision, recall, accuracy, and F-score.
    """
    print("Testing model...")
    time0 = time.time()

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

    time_elapsed = time.time() - time0
    print(f"    Time Elapsed: {time_elapsed:.2f} sec")
    
    # Calculate precision, recall, accuracy, and F-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = correct / total
    fscore = 2. * (precision * recall) / (precision + recall)

    print("    Testing Results:")
    print(f"\tPrecision: {precision:.4f}")
    print(f"\tRecall:    {recall:.4f}")
    print(f"\tAccuracy:  {accuracy:.4f}")
    print(f"\tF-score:   {fscore:.4f}")

    return precision, recall, accuracy, fscore


def plot_loss_and_accuracy(model_name, losses, accuracies, outdir, data_path="", save_suffix=""):
    """
    Plots the results of training a model. Given lists LOSSES and ACCURACIES, 
    or a path to a file containing this data. Saves the figure in the directory
    specified by OUTDIR.
    """
    
    if data_path:
        xs, losses, accuracies = np.genfromtxt(data_path).T
        print(f"Loading results located at {data_path}")

    fig, [ax1, ax2] = plt.subplots(2, 1)
    xs = range(len(losses))
    
    ax1.plot(xs, losses)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{model_name.upper()} Training Loss")
    
    ax2.plot(xs, accuracies)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"{model_name.upper()} Training Accuracy")

    fig.tight_layout()

    fpath = f"{outdir}/training_results_{model_name}{save_suffix}.png"
    plt.savefig(fpath)
    print(f"Plots saved to {fpath}")


def save_training_results(model_name, losses, accuracies, outdir, save_suffix=""):
    fpath = f"{outdir}/training_results_{model_name}{save_suffix}.txt"
    np.savetxt(fpath, np.array([range(len(losses)), losses, accuracies]).T, 
                      header='epoch, loss, accuracy')
    print(f"Training results saved to {fpath}")


def save_testing_results(model_name, precision, recall, accuracy, fscore, outdir, save_suffix=""):
    fpath = f"{outdir}/testing_results_{model_name}{save_suffix}.txt"
    np.savetxt(fpath, np.array([precision, recall, accuracy, fscore]), 
                      header='precision, recall, accuracy, fscore')
    print(f"Testing results saved to {fpath}")


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
