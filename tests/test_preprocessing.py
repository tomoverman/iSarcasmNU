import unittest
from preprocessing.preprocessor import Preprocessor


DATA_PATH = "~/Documents/Projects/iSarcasmNU/tests/test_data"  # location of training and testing data


class TestPreprocessing(unittest.TestCase):

	def test_1(self):
		train_fpath = f"{DATA_PATH}/testdata_train_1.csv"
		test_fpath  = f"{DATA_PATH}/testdata_test_1.csv"

		seq_len = 10
		min_len = 3

		prep = Preprocessor(seq_len=seq_len, min_len=min_len)
		prep.load_data(train_fpath, test_fpath)
		prep.initialize()

		x_train, kept_idxs = prep.process(prep.tweets_train, convert_to_idxs=False)
		y_train = [prep.labels_train[i] for i in kept_idxs]
		x_test, kept_idxs  = prep.process(prep.tweets_test,  convert_to_idxs=False)
		y_test  = [prep.labels_test[i] for i in kept_idxs]

		assert len(x_train) == 6
		assert len(x_train[0]) == 10
		assert len(x_train[-1]) == 10
		assert prep.UNK_TOKEN not in x_train[0]
		assert prep.UNK_TOKEN not in x_train[-1]
		assert x_train[-1][-1] == "banana"
		assert prep.fdist["banana"] == 1

	
if __name__ == '__main__':
	unittest.main()
