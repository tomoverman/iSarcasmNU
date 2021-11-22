import numpy as np
import requests
import csv
import time

token_fpath = "token.txt"  # Path to file containing authorization bearer token

in_data_test_fpath   = "isarcasm_test.csv"
in_data_train_fpath  = "isarcasm_train.csv"
out_data_test_fpath  = "data_test.csv"
out_data_train_fpath = "data_train.csv"

#below is to get the data for the ptacket dataset
# in_data_test_fpath   = "ptacek_test.csv"
# in_data_train_fpath  = "ptacek_train.csv"
# out_data_test_fpath  = "ptacek_data_test.csv"
# out_data_train_fpath = "ptacek_data_train.csv"

def is_sarcastic(s):
	return int(s == "sarcastic")


def get_texts(fpath_in, bearer_token):
	"""Given path to isarcasm csv file, containing rows of form [id, sarcasm_label, sarcasm_category],
	and user's bearer token, return a 2d array DATA containing entries [text, label] where text is the
	text of a valid tweet, and label {0, 1} indicating if it is sarcasm. Also return list ERRORS with 
	IDs of tweets not found."""
	
	headers = {"Authorization": "Bearer {}".format(bearer_token)}
	url_base = "https://api.twitter.com/2/tweets?ids="
	
	tweet_ids = []
	id_label_map = {}
	id_category_map = {}
	
	# Read from input csv file
	with open(fpath_in, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for i, row in enumerate(reader):
			if i > 0:
				tweet_id, label, category = row
				tweet_ids.append(tweet_id)
				id_label_map[tweet_id] = label
				id_category_map[tweet_id] = category

	n = len(tweet_ids)
	stride = 100
	data = []
	errors = []
	# rate_limit_delay=900
	for start_idx in range(0, n, stride):
		#below is to get around twitter api rate limits
		# if start_idx%(200*stride)==0:
		# 	time.sleep(rate_limit_delay)
		# Get group of ids
		ids = tweet_ids[start_idx:min(start_idx + stride, n)]
		# Seperate ids in a comma separated string
		cat_ids = ",".join(ids)
		# Query from twitter and convert to JSON
		response = requests.request("GET", url_base + cat_ids, headers=headers).json()
		# Append data with [text, id] pairs
		data += [[x['text'], is_sarcastic(id_label_map[x['id']])] for x in response['data']]
		# If errors, store the tweet ids
		if 'errors' in response:
			errors += [int(x['value']) for x in response['errors']]

	return data, errors


def write_out(data, fpath_out):
	"""Write [text, sarcasm_label] rows to a csv file given by fpath_out"""
	with open(fpath_out, 'w', newline='') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(["text", "sarcasm_label"])
		writer.writerows(data)


def main():
	with open(token_fpath, 'r') as f:
		bearer_token = f.read()

	data_train, errors_train  = get_texts(in_data_train_fpath, bearer_token)
	data_test,  errors_test   = get_texts(in_data_test_fpath,  bearer_token)
	
	write_out(data_train, out_data_train_fpath)
	write_out(data_test,  out_data_test_fpath)


main()


