"""
Utility functions.
"""
import tensorflow as tf
import numpy as np
import os
from six.moves import cPickle

"""
Class to load batches.
"""
class BatchLoader(object):
	def __init__(self, string_file, points_file, save_dir, batch_size, seq_len, max_str_len):
		assert batch_size > 0, "Invalid batch size."

		self.batch_size = batch_size
		self.seq_len = seq_len

		temp = np.load(points_file, encoding='bytes')
		self.a = np.array([np.concatenate([np.array([[0., 0., 1.]]),data[:,[1,2,0]]], axis=0) for data in temp])
		
		self.strings = []
		with open(string_file, 'r') as f:
			texts = f.readlines()
		self.strings = [s.strip('\n') for s in texts]
			
		# Encode strings to one-hot vector(s)
		self.char_mapping = map_strings(self.strings, path=os.path.join(save_dir, 'mapping'))
		self.lst_C = [generate_char_encoding(s, self.char_mapping, max_str_len) for s in self.strings]

		self.indices = np.random.choice(len(self.a), size=batch_size, replace=False)
		self.st_times = np.zeros(batch_size, dtype=np.int32)

	def get_batch(self):
		reset_reqd = False
		reset = np.ones([self.batch_size, 1], dtype=np.float32)
		batch = np.zeros([self.batch_size, self.seq_len + 1, 3], dtype=np.float32)
		C = np.zeros([self.batch_size, self.lst_C[0].shape[0], self.lst_C[0].shape[1]], dtype=np.float32)
		for i in range(self.batch_size):
			if self.st_times[i] + self.seq_len + 1 > self.a[self.indices[i]].shape[0]:
				# Need to reset this 'i'
				reset_reqd = True
				self.st_times[i] = 0
				# Move to a new string
				self.indices[i] = np.random.choice(len(self.a))
				reset[i] = 0
			batch[i, :, :] = self.a[self.indices[i]][self.st_times[i]:self.st_times[i]+self.seq_len+1]
			C[i, :, :] = self.lst_C[self.indices[i]]
			self.st_times[i] += self.seq_len
		return batch, C, reset, reset_reqd

	@property
	def n_chars(self):
		return self.lst_C[0].shape[1]
"""
Function to load data from disk and preprocess it.
Args:
	string_file - path where strings are stored
	points_file - path where corresponding points are saved
	batch_size - size of each batch
Returns:
	extracted data from specified files.
"""
def load_data(string_file, points_file, save_dir, batch_size, max_str_len):
	assert batch_size > 0, "Invalid batch size."
	temp = np.load(points_file, encoding='bytes')
	a = np.array([data[:,[1,2,0]] for data in temp])
	n_batches = len(a) // batch_size
	
	strings = []
	with open(string_file, 'r') as f:
		texts = f.readlines()
	strings = [s.strip('\n') for s in texts]
	
	# Obtain maximum sequence length
	max_seq_len = max([pt.shape[0] for pt in a])
	
	# Encode strings to one-hot vector(s)
	char_mapping = map_strings(strings, path=os.path.join(save_dir, 'mapping'))
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
	start_pt - a np.array of [x, y, end_of_stroke]
	batch_size - size of each batch
Returns:
	padded tensor of points
"""
def prediction_input(start_pt, batch_size):
	assert batch_size > 0, "Invalid batch_size"
	a = start_pt
	a.resize([1, 1, 3])

	# Pad starting point with zero vectors to convert it
	# into dimension [batch_size, 1, 3]
	padding = np.tile([0, 0, 0], [batch_size - 1, 1])
	padding.resize(batch_size - 1, 1, 3)
	pts = np.concatenate([a, padding], axis=0)
	return pts

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
