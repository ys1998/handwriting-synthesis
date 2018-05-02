"""
Utility functions.
"""
import tensorflow as tf
import numpy as np
import os
from six.moves import cPickle

"""
Function to load data from disk and preprocess it.
Args:
	string_file - path where strings are stored
	points_file - path where corresponding points are saved
	batch_size - size of each batch
Returns:
	extracted data from specified files.
"""
def load_data(string_file, points_file, batch_size):
	assert batch_size > 0, "Invalid batch size."
	a = np.load(points_file, encoding='bytes')
	n_batches = len(a) // batch_size
	
	strings = []
	with open(string_file, 'r') as f:
		texts = f.readlines()
	strings = [s.strip('\n') for s in texts]
	
	# Obtain maximum sequence length
	max_seq_len = max([pt.shape[0] for pt in a])
	max_str_len = max([len(s) for s in strings])
	
	# Encode strings to one-hot vector(s)
	char_mapping = map_strings(strings, path='save/mapping')
	lst_C = [generate_char_encoding(s, char_mapping, max_str_len) for s in strings]

	# Convert all sequences to max_seq_len by padding with [0., 0., 1] 
	# i.e. zero offset and end-of-stroke is true
	for idx, pt in enumerate(a):
		if pt.shape[0] < max_seq_len:
			a[idx] = np.concatenate([pt, np.tile([0,0,1], [max_seq_len - pt.shape[0], 1])], axis=0)
	
	sts = []; pts = []
	# Group into batches
	for bn in range(n_batches):
		sts.append(np.stack(lst_C[bn*batch_size:(bn+1)*batch_size]))
		pts.append(np.stack(a[bn*batch_size:(bn+1)*batch_size]))
	return (sts, pts)

"""
Function to generate the input for the prediction step
Args:
	string - the string which has to be fed to the model
	start_pt - a np.array of [x, y, end_of_stroke]
	batch_size - size of each batch
Returns:
	tuple of encoded strings and the points
"""
def prediction_input(string, start_pt, batch_size):
	assert batch_size > 0, "Invalid batch_size"
	strings = []
	strings.append(string)
	for i in range(batch_size - 1):
		strings.append('')
	max_seq_len = 1
	max_str_len = len(string)
	a = start_pt
	a.resize([1, 1, 3])

	# Encode strings to one-hot vector(s)
	char_mapping = map_strings(strings, path='save/mapping')
	lst_C = [generate_char_encoding(s, char_mapping, max_str_len) for s in strings]

	# Convert all sequences to max_seq_len by padding with [0., 0., 1] 
	# i.e. zero offset and end-of-stroke is true
	padding = np.tile([0, 0, 1], [batch_size-1, 1])
	padding.resize(batch_size-1, 1, 3)
	pts = np.concatenate([a, padding], axis=0)

	sts = np.stack(lst_C[0:-1])

	return (lst_C, pts)


"""
Function to convert string to one-hot encoded matrix.
Args:
	string - string to be encoded
	char_mapping - character -> number mapping
Returns:
	one-hot encoded matrix for given string.
"""
def generate_char_encoding(string, char_mapping, length=None):
	n_chars = len(char_mapping)
	indices = []
	for c in string:
		assert c in char_mapping, "Unknown character encountered."
		indices.append(char_mapping[c])
	one_hot = np.zeros([len(string), n_chars])
	one_hot[list(range(len(string))), indices] = 1.0
	if length is not None and length > one_hot.shape[0]:
		one_hot = np.concatenate([one_hot, np.zeros([length - one_hot.shape[0], n_chars])], axis=0)
	return one_hot

"""
Function to map characters from a collection of strings to 
a number for one-hot encoding.
Args:
	lst_str - list of strings to be mapped
	path - path where generated mapping is saved
Returns:
	character -> number mapping for given strings.
"""
def map_strings(lst_str, path=None):
	if os.path.exists(path):
		with open(path, 'rb') as f:
			mapping = cPickle.load(f)
		return mapping
	else:
		mapping = {}
		cntr = 0
		for s in lst_str:
			for c in s:
				if c not in mapping:
					mapping[c] = cntr
					cntr += 1
		if path is not None:
			with open(path, 'wb') as f:
				cPickle.dump(mapping, f)
		return mapping
