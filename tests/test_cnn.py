import unittest
from preprocessing.preprocessor import Preprocessor
from models.cnn import CNN


DATA_PATH = "~/Documents/Projects/iSarcasmNU/data"  # data location


class TestCNN(unittest.TestCase):

	def test_1(self):
		train_fpath = f"{DATA_PATH}/data_train.csv"
		test_fpath  = f"{DATA_PATH}/data_test.csv"

		seq_len = 40
		min_len = 5

		prep = Preprocessor(seq_len=seq_len, min_len=min_len)
		prep.load_data(train_fpath, test_fpath)
		prep.initialize()

		x_train, kept_idxs = prep.process(prep.tweets_train, convert_to_idxs=False)
		y_train = [prep.labels_train[i] for i in kept_idxs]
		x_test, kept_idxs  = prep.process(prep.tweets_test,  convert_to_idxs=False)
		y_test  = [prep.labels_test[i] for i in kept_idxs]

		assert len(x_train[0]) == 40
		assert len(x_train[-1]) == 40

	
if __name__ == '__main__':
	unittest.main()
