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
	a = np.load(points_file, encoding='bytes')
	n_batches = len(a) // batch_size
	
	strings = []
	with open(string_file, 'r') as f:
		texts = f.readlines()
	strings = [s.strip('\n') for s in texts]
	
	# Encode strings to one-hot vector(s)
	char_mapping = map_strings(strings, path='save/mapping.pkl')
	lst_C = [generate_char_encoding(s, char_mapping) for s in strings]

	# Obtain maximum sequence length
	max_seq_len = max([pt.shape[0] for pt in a])
	# Convert all sequences to max_seq_len by padding with [0., 0., 1] 
	# i.e. zero offset and end-of-stroke is true
	points = 
	return (strings, pts)

"""
Function to convert string to one-hot encoded matrix.
Args:
    string - string to be encoded
    char_mapping - character -> number mapping
Returns:
    one-hot encoded matrix for given string.
"""
def generate_char_encoding(string, char_mapping):
	n_chars = len(char_mapping)
	indices = []
	for idx, c in enumerate(string):
		assert c in char_mapping, "Unknown character encountered."
		indices.append(char_mapping[c])
	return tf.transpose(tf.one_hot(indices, n_chars))

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
		with open(path, 'r') as f:
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
			with open(path, 'w') as f:
				cPickle.dump(mapping, f)
		return mapping
