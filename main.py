
import argparse
import os
from preprocessing.preprocessor import Preprocessor
from utils import select_model, train_model, test_model, plot_loss_and_accuracy, \
                  save_training_results, save_testing_results, long_train_model, evaluate_long_train
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def main():

    parser = argparse.ArgumentParser(description='iSarcasmNU')

    parser.add_argument("model",    type=str, choices=["cnn", "3cnn", "lstm", "lstm_att", "siarn", "miarn"])
    # long_train means to run for a large number of epochs, saving the model parameters peiordically, then select the best
    # model parameters using a validation set
    parser.add_argument("action",   type=str, choices=["train", "test", "plot_training_results", "long_train"])

    parser.add_argument("--data_train_fpath",   type=str,       default="./data/ptacek_data_train.csv",
                        help="path to training data")
    parser.add_argument("--data_test_fpath",    type=str,       default="./data/ptacek_data_test.csv",
                        help="path to testing data")

    parser.add_argument("--num_epochs",         type=int,       default=30,
                        help="number of training epochs")
    parser.add_argument("--batch_size",         type=int,       default=512,
                        help="testing and training batch size")
    parser.add_argument("--seq_len",            type=int,       default=40,
                        help="fixed length of input (number of tokens in tweet)")
    parser.add_argument("--min_len",            type=int,       default=5,
                        help="minimum number of tokens in input sequence")
    parser.add_argument("--embedding_size",     type=int,       default=100,
                        help="dimension of the embedding space")

    parser.add_argument("--learning_rate",      type=float,     default=0.001,
                        help="learning rate hyperparameter")
    parser.add_argument("--regularization",     type=float,     default=1e-8,
                        help="L2 regularization hyperparameter")

    parser.add_argument("--clip",               type=int,       default=5,
                        help="clipping amount to prevent exploding gradients")

    parser.add_argument("--save_model",         type=str,       default="",
                        help="file path to save model parameters to)")
    parser.add_argument("--load_model",         type=str,       default="",
                        help="path to saved model file")
    parser.add_argument("--outdir",             type=str,       default="",
                        help="path to directory to save output")
    parser.add_argument("--training_results",   type=str,       default="",
                        help="path to file containing training results")
    parser.add_argument("--suffix",             type=str,       default="",
                        help="suffix to append to saved filenames")
    parser.add_argument("--logloss",            action="store_true",
                        help="if given, plot log of the training loss")
    
    parser.add_argument("--cuda",               action="store_true",
                        help="flag indicating to use GPU if available")
    parser.add_argument("--storage_step", type=int, default=1,
                        help="step size of epochs in which model parameters are saved during long train")

    args = parser.parse_args()

    model_name          = args.model.lower()
    action              = args.action
    data_train_fpath    = args.data_train_fpath
    data_test_fpath     = args.data_test_fpath
    batch_size          = args.batch_size
    num_epochs          = args.num_epochs 
    seq_len             = args.seq_len 
    min_len             = args.min_len 
    embed_size          = args.embedding_size 
    learning_rate       = args.learning_rate 
    reg_l2              = args.regularization 
    clip                = args.clip
    save_model_path     = args.save_model
    load_path           = args.load_model
    outdir              = args.outdir if args.outdir else f"out/{model_name}"
    train_results_path  = args.training_results
    save_suffix         = "_" + args.suffix if args.suffix else args.suffix
    use_logplot         = args.logloss
    cuda                = args.cuda
    storage_step        = args.storage_step

    # Determine whether to use CUDA based on input arguments and if cuda device is available
    use_gpu = (cuda and torch.cuda.is_available())
    
    # Preprocess data
    prep = Preprocessor(seq_len=seq_len, min_len=min_len)
    prep.load_data(data_train_fpath, data_test_fpath)
    prep.initialize()
    vocab_size  = prep.V

    # Select model
    model = select_model(model_name, embed_size, vocab_size, seq_len, load_path)
    if use_gpu:
        model.cuda()
    
    # Specify optimizer and loss function
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=reg_l2)
    criterion = nn.BCELoss()

    # Make output directory if needed
    os.makedirs(outdir, exist_ok=True)
    
    # Actions
    if action == "train":
        
        gpu_or_cpu = "GPU" if use_gpu else "CPU"
        print(f"Training model {model_name.upper()} on {gpu_or_cpu} with hyperparameters: \
                \n\tnum_epochs:         {num_epochs}\
                \n\tbatch_size:         {batch_size}\
                \n\tlearning_rate:      {learning_rate}\
                \n\tL2_regularization:  {reg_l2}")

        # Load data, then train and test the model
        validation = False
        train_loader = DataLoader(dataset=prep.get_dataset_train(use_gpu,validation), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(dataset=prep.get_dataset_test(use_gpu),  batch_size=batch_size, shuffle=False)
        train_losses, accuracies = train_model(model, num_epochs, train_loader, optimizer, criterion, clip)
        precision, recall, accuracy, fscore = test_model(model, test_loader)

        # Save model if desired
        if save_model_path:
            torch.save(model.state_dict(), save_model_path)
        
        # Save and plot results
        save_training_results(model_name, train_losses, accuracies, outdir, save_suffix=save_suffix)
        plot_loss_and_accuracy(model_name, train_losses, accuracies, outdir, 
                               save_suffix=save_suffix, logloss=use_logplot)
        save_testing_results(model_name, precision, recall, accuracy, fscore, outdir, save_suffix=save_suffix)

    elif action == "long_train":
        gpu_or_cpu = "GPU" if use_gpu else "CPU"
        print(f"Training model {model_name.upper()} on {gpu_or_cpu} with hyperparameters: \
                        \n\tbatch_size:         {batch_size}\
                        \n\tlearning_rate:      {learning_rate}\
                        \n\tL2_regularization:  {reg_l2}")

        # Load data, then train and test the model
        validation = True
        train_loader = DataLoader(dataset=prep.get_dataset_train(use_gpu, validation), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=prep.get_dataset_valid(use_gpu), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=prep.get_dataset_test(use_gpu), batch_size=batch_size, shuffle=False)

        train_losses, accuracies = long_train_model(model, model_name, train_loader, optimizer, criterion, clip, num_epochs, storage_step)

        save_training_results(model_name, train_losses, accuracies, outdir, save_suffix=save_suffix)
        plot_loss_and_accuracy(model_name, train_losses, accuracies, outdir, 
                               save_suffix=save_suffix, logloss=use_logplot)

        valid_acc, best_model_path = evaluate_long_train(model_name, valid_loader, embed_size, vocab_size, seq_len, use_gpu,num_epochs, storage_step)

        best_model = select_model(model_name, embed_size, vocab_size, seq_len, best_model_path)

        if use_gpu:
            best_model.cuda()

        precision, recall, accuracy, fscore = test_model(best_model, test_loader)
        save_testing_results(model_name, precision, recall, accuracy, fscore, outdir, save_suffix=save_suffix)

    elif action == "test":
        print(f"Testing model {model_name.upper()} loaded from {load_path}")
        
        # Load test data and test the model
        test_loader  = DataLoader(dataset=prep.get_dataset_test(use_gpu),  batch_size=batch_size, shuffle=False)
        precision, recall, accuracy, fscore = test_model(model, test_loader)

    elif action == "plot_training_results":
        plot_loss_and_accuracy(model_name, None, None, outdir, data_path=train_results_path,
                               save_suffix=save_suffix, logloss=use_logplot)


if __name__ == "__main__":
    main()
