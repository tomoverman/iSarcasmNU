import numpy as np
import requests
import csv

token_fpath = "token.txt"
in_data_test_fpath  = "isarcasm_test.csv"
in_data_train_fpath = "isarcasm_train.csv"
out_data_test_fpath = "data_test.csv"
out_data_train_fpath = "data_train.csv"

with open(token_fpath, 'r') as f:
	bearer_token = f.read()

def is_sarcastic(s):
	return int(s == "sarcastic")

def get_texts(fpath_in, bearer_token):
	
	headers = {"Authorization": "Bearer {}".format(bearer_token)}
	url_base = "https://api.twitter.com/2/tweets?ids="
	
	tweet_ids = []
	id_category_map = {}
	id_label_map = {}
	
	with open(fpath_in, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for i, row in enumerate(reader):
			if i == 0:
				print(f'Column names are {", ".join(row)}')
			else:
				tweet_id, label, category = row
				tweet_ids.append(tweet_id)
				id_label_map[tweet_id] = label
				id_category_map[tweet_id] = category

	n = len(tweet_ids)
	stride = 100
	data = []
	errors = []
	for start_idx in range(0, n, stride):
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
	with open(fpath_out, 'w', newline='') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerows(data)



data_test, errors_test = get_texts(in_data_test_fpath, bearer_token)
print(len(data_test))
print(len(errors_test))

data_train, errors_train = get_texts(in_data_train_fpath, bearer_token)
print(len(data_train))
print(len(errors_train))

write_out(data_test, out_data_test_fpath)
write_out(data_train, out_data_train_fpath)




