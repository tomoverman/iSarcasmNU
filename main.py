
import argparse
from preprocessing.preprocessor import Preprocessor
from models.cnn import CNN, CNNLayered
from models.LSTMSarcasm import LSTMSarcasm
from models.LSTM_Att import LSTM_Att
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def main():

    parser = argparse.ArgumentParser(description='Model Training and Testing')
    parser.add_argument("model", type=str, choices=["cnn", "3cnn", "lstm", "lstm_att", "siarn", "miarn"])
    parser.add_argument("--data_train_fpath",   type=str,       default="./data/ptacek_data_train.csv")
    parser.add_argument("--data_test_fpath",    type=str,       default="./data/ptacek_data_test.csv")
    parser.add_argument("--batch_size",         type=int,       default=512)
    parser.add_argument("--num_epochs",         type=int,       default=30)
    parser.add_argument("--seq_len",            type=int,       default=40)
    parser.add_argument("--min_len",            type=int,       default=5)
    parser.add_argument("--embed_size",         type=int,       default=100)
    parser.add_argument("--learning_rate",      type=float,     default=0.001)
    parser.add_argument("--regularization",     type=float,     default=1e-8)
    parser.add_argument("--clip",               type=int,       default=5)
    args = parser.parse_args()

    model_name          = args.model
    data_train_fpath    = args.data_train_fpath
    data_test_fpath     = args.data_test_fpath
    batch_size          = args.batch_size
    num_epochs          = args.num_epochs 
    seq_len             = args.seq_len 
    min_len             = args.min_len 
    embed_size          = args.embed_size 
    learning_rate       = args.learning_rate 
    reg_l2              = args.regularization 
    clip                = args.clip
    

    # Preprocess data
    prep = Preprocessor(seq_len=seq_len, min_len=min_len)
    prep.load_data(data_train_fpath, data_test_fpath)
    prep.initialize()
    vocab_size  = prep.V

    # Construct model
    if model_name.lower() == "cnn":
        # Parameters for CNN
        filter_size = 3
        num_filters = 100
        model = CNN(num_filters, filter_size, embed_size, vocab_size, seq_len)

    elif model_name.lower() == "3cnn":
        # Parameters for 3CNN
        filter_sizes    = [3, 4, 5]
        filter_counts   = [100, 100, 100]
        model = CNNLayered(filter_counts, filter_sizes, embed_size, vocab_size, seq_len)

    elif model_name.lower() == "lstm":
    	# Parameters for LSTM
    	hidden_dim = 100
    	model = LSTMSarcasm(embed_size, hidden_dim, vocab_size)

    elif model_name.lower() == "lstm_att":
    	# Parameters for LSTM_ATT
        hidden_dim = 100
        model = LSTM_Att(embed_size, hidden_dim, vocab_size, seq_len)

    elif model_name.lower() == "siarn":
        hidden_dim = 100
        model = SIARN(embed_size, hidden_dim, vocab_size, seq_len)

    elif model_name.lower() == 'miarn':
        # Parameters for MIARN
        pass
    
    # Specify optimizer and loss function
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=reg_l2)
    criterion = nn.BCELoss()
    
    # Load datasets
    train_loader = DataLoader(dataset=prep.get_dataset_train(), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=prep.get_dataset_test(),  batch_size=batch_size, shuffle=False)

    # Train model    
    train_losses, test_losses, accuracies = train(model, num_epochs, train_loader, optimizer, criterion, clip)

    # Test model
    precision, recall, accuracy, fscore = test(model, test_loader)
    

def train(model, num_epochs, train_loader, optimizer, loss_function, clip):
    train_losses = []
    test_losses  = []
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

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')

    return train_losses, test_losses, accuracies


def test(model, test_loader):
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
    print(f"Test Recall: {recall}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Test F-score: {fscore}")

    return precision, recall, accuracy, fscore


########################
##  Helper functions  ##
########################

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


if __name__ == "__main__":
    main()
