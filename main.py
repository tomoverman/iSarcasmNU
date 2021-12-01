
import argparse
from preprocessing.preprocessor import Preprocessor
from models.cnn import CNN, CNNLayered
from models.LSTMSarcasm import LSTMSarcasm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Location of training and testing data
data_path = "~/Documents/Projects/iSarcasmNU/data"
# data_train_fpath = f"{data_path}/data_train.csv"
# data_test_fpath  = f"{data_path}/data_test.csv"
data_train_fpath = f"{data_path}/ptacek_data_train.csv"
data_test_fpath  = f"{data_path}/ptacek_data_test.csv"


# Helper functions

def convert_output_to_label(cnn_output):
    return torch.round(cnn_output).int()

def get_accuracy(y_pred, y_true):
    return (y_pred == y_true).sum().float() / y_true.shape[0]

def get_false_positives(y_pred, y_true):
    return torch.logical_and(y_pred != y_true, y_pred == 1).sum().int()

def get_false_negatives(y_pred, y_true):
    return torch.logical_and(y_pred != y_true, y_pred == 0).sum().int()

def get_true_positives(y_pred, y_true):
    return torch.logical_and(y_pred == y_true, y_pred == 1).sum().int()


def run(model_name, batch_size):

    seq_len = 40
    min_len = 5
    embed_size  = 100

    # Preprocess data
    prep = Preprocessor(seq_len=seq_len, min_len=min_len)
    prep.load_data(data_train_fpath, data_test_fpath)
    prep.initialize()
    vocab_size  = prep.V

    if model_name.lower() == "cnn":
        # Parameters for CNN
        num_filters = 100
        filter_size = 3
        model = CNN(num_filters, filter_size, embed_size, vocab_size, seq_len)

    elif model_name.lower() == "3cnn":
        # Parameters for 3CNN
        filter_counts   = [100, 100, 100]
        filter_sizes    = [3, 4, 5]
        model = CNNLayered(filter_counts, filter_sizes, embed_size, vocab_size, seq_len)

    elif model_name.lower() == "lstm":
    	# Parameters for LSTM
    	hidden_dim = 100
    	model = LSTMSarcasm(embed_size, hidden_dim, vocab_size)

    elif model_name.lower() == "lstm_att":
    	# Parameters for LSTM_ATT
    	pass

    elif model_name.lower() == "siarn":
    	# Parameters for SIARN
    	pass

    elif model_name.lower() == 'miarn':
        pass



    ################
    ##  Training  ##
    ################

    # Hyperparameters
    if batch_size < 0:
        batch_size = 512
    num_epochs = 2
    learning_rate = 0.001
    reg_l2 = 1e-8

    clip = 5

    # Optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=reg_l2)

    # Loss function
    criterion = nn.BCELoss()
    
    # Load datasets
    train_loader = DataLoader(dataset=prep.get_dataset_train(), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=prep.get_dataset_test(),  batch_size=batch_size, shuffle=False)
    test_dataset = prep.get_dataset_test()
    
    train_losses = []
    test_losses  = []
    accuracies   = []
    
    model.train()
    for epoch in range(1, num_epochs + 1):
        for i, (xs, labels) in enumerate(train_loader):

            tr_loss = 0

            # Perform forward pass
            outputs = model(xs)
            pred_labels = convert_output_to_label(outputs)

            # Compute loss and accuracy
            loss = criterion(outputs, labels.float())
            accuracy = get_accuracy(pred_labels, labels)

            train_losses.append(loss.item())
            accuracies.append(accuracy)

            # Loss and backprop
            optimizer.zero_grad()
            loss.backward()

            # Prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()


        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')

    
    ###############
    ##  Testing  ##
    ###############
    
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

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = correct / total

    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training and Testing')
    parser.add_argument("model", type=str, choices=["cnn", "3cnn", "lstm", "lstm_att", "siarn"])
    parser.add_argument("--batch_size", type=int, default=-1)
    args = parser.parse_args()

    run(args.model, args.batch_size)
