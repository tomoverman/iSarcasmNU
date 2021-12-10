import argparse
import os
import time
from models.cnn import CNN, CNNLayered
from models.lstm import LSTM, LSTMAtt
from models.siarn import SIARN
from models.miarn import MIARN
from models.siarn3 import SIARN3
import numpy as np
import torch
import torch.nn as nn


MODELS = {
    "cnn": CNN,
    "3cnn": CNNLayered,
    "lstm": LSTM,
    "lstm_att": LSTMAtt,
    "siarn": SIARN,
    "miarn": MIARN,
    "siarn3": SIARN3
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

    elif model_name == "siarn3":
        # Parameters for SIARN
        hidden_dim = 100
        params = [embed_size, hidden_dim, vocab_size, seq_len]

    # Construct the model
    model = MODELS[model_name](*params)
    
    # If specified, load a pretrained model
    if load_path:
        model.load_state_dict(torch.load(load_path))
        model.eval()

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


def long_train_model(model, model_name, train_loader, optimizer, loss_function, clip, num_epochs, storage_step):
    """
    Train the given model over a number of epochs, using the data in train_loader.
    Uses the specified optimizer and loss function. Clip gives the amount to clip
    the gradient to avoid exploding gradients.
    Returns two lists: the training losses and the accuracies at each iteration.
    """

    base_path = "out/long_train/" + model_name + "/"
    os.makedirs(base_path, exist_ok=True)
    print("Training model...")
    time0 = time.time()
    t0 = time0

    train_losses = []
    accuracies = []

    model.train()
    # train for many, many epochs to get a large parameter space
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
        if epoch%storage_step==0:
            torch.save(model.state_dict(), base_path + str(epoch) + ".pth")

    time_elapsed = time.time() - time0
    print(f"    Time Elapsed: {time_elapsed:.2f} sec")

    return train_losses, accuracies


def evaluate_long_train(model_name, valid_loader, embed_size, vocab_size, seq_len, use_gpu, 
                        num_epochs, storage_step, valid_criterion):
    
    base_path = "out/long_train/" + model_name + "/"

    accuracies = np.zeros(len(range(storage_step,num_epochs+1, storage_step)))
    fscores    = np.zeros(len(range(storage_step,num_epochs+1, storage_step)))

    for i, epoch in enumerate(range(storage_step,num_epochs+1, storage_step)):
        model_path = base_path + str(epoch) + ".pth"
        model = select_model(model_name, embed_size, vocab_size, seq_len, model_path)
        if use_gpu:
            model.cuda()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            false_positives = 0
            false_negatives = 0
            true_positives  = 0
            for xs, labels in valid_loader:
                outputs = model(xs)
                pred_labels = convert_output_to_label(outputs)
                if valid_criterion == "accuracy":
                    total += len(labels)
                    correct += (pred_labels == labels).sum().int()
                elif valid_criterion == "fscore":
                    false_negatives += get_false_negatives(pred_labels, labels)
                    false_positives += get_false_positives(pred_labels, labels)
                    true_positives  += get_true_positives(pred_labels, labels)

        if valid_criterion == "accuracy":
            accuracy  = correct / total
            accuracies[i] = correct / total
        elif valid_criterion == "fscore":
            precision = true_positives / (true_positives + false_positives)
            recall    = true_positives / (true_positives + false_negatives)
            fscore    = 2. * (precision * recall) / (precision + recall)
            fscores[i]    = fscore
        
    # Choose best model based on specified criterion    
    if valid_criterion == "accuracy":
        best_i = np.argmax(accuracies)
        return accuracies[best_i], base_path + str(best_i*storage_step+storage_step) + ".pth"
    elif valid_criterion == "fscore":
        best_i = np.nanargmax(fscores)
        return fscores[best_i], base_path + str(best_i*storage_step+storage_step) + ".pth"


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
