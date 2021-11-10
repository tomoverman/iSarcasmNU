
from preprocessing.preprocessor import Preprocessor
from models.cnn import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Location of training and testing data
data_path = "~/Documents/Projects/iSarcasmNU/data"
data_train_fpath = f"{data_path}/data_train.csv"
data_test_fpath  = f"{data_path}/data_test.csv"


def convert_output_to_label(cnn_output):
	return torch.round(torch.sigmoid(cnn_output)).int()

def get_accuracy(y_pred, y_true):
	return (y_pred == y_true).sum().float() / y_true.shape[0]

def get_false_positives(y_pred, y_true):
	return torch.logical_and(y_pred != y_true, y_pred == 1).sum().int()

def get_false_negatives(y_pred, y_true):
	return torch.logical_and(y_pred != y_true, y_pred == 0).sum().int()

def get_true_positives(y_pred, y_true):
	return torch.logical_and(y_pred == y_true, y_pred == 1).sum().int()


def main():
		
	# Preprocess data
	seq_len = 40
	min_len = 5
	prep = Preprocessor(seq_len=seq_len, min_len=min_len)
	prep.load_data(data_train_fpath, data_test_fpath)
	prep.initialize()
	
	x_train = torch.Tensor(prep.x_train).int()
	y_train = torch.Tensor(prep.y_train)
	x_test  = torch.Tensor(prep.x_test).int()
	y_test  = torch.Tensor(prep.y_test)

	# Parameters for CNN
	params = {}
	params['num_filters'] = 100
	params['filter_size'] = 3
	params['embed_size'] = 100
	params['vocab_size'] = prep.V
	params['seq_len']	= seq_len

	# Construct the CNN
	cnn = CNN(params)

	################
	##  Training  ##
	################

	# Hyperparameters
	num_epochs = 30
	batch_size = 16
	learning_rate = 0.001
	reg_l2 = 1e-8

	# Optimizer
	optimizer = optim.RMSprop(cnn.parameters(), lr=learning_rate, weight_decay=reg_l2)

	# Loss function
	criterion = nn.BCEWithLogitsLoss()
	
	# Load datasets
	train_loader = DataLoader(dataset=prep.get_dataset_train(), batch_size=batch_size, shuffle=True)
	test_loader  = DataLoader(dataset=prep.get_dataset_test(),  batch_size=batch_size, shuffle=False)
	
	train_losses = []
	test_losses  = []
	acc_list	 = []
	
	cnn.train()

	for epoch in range(1, num_epochs + 1):
		for i, (xs, labels) in enumerate(train_loader):

			tr_loss = 0

			# Perform forward pass
			outputs = cnn(xs)
			pred_labels = convert_output_to_label(outputs)

			# Compute loss and accuracy
			loss = criterion(outputs, labels.float())
			accuracy = get_accuracy(pred_labels, labels)

			train_losses.append(loss.item())
			acc_list.append(accuracy)

			# Optimization step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


		print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')

	
	###############
	##  Testing  ##
	###############
	
	cnn.eval()
	with torch.no_grad():
	    correct = 0
	    total = 0
	    false_positives = 0
	    false_negatives = 0
	    true_positives  = 0
	    for xs, labels in test_loader:
	        outputs = cnn(xs)
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


main()
