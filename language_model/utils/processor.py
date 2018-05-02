"""This file loads text, processes it and converts it into batches."""
from utils.trie import Trie
from utils.strings import FILES, LOGS, ERRORS

from six.moves import cPickle
import codecs
import numpy as np
import os
import sys
import multiprocessing as mp
from multiprocessing import sharedctypes
import time


class DataLoader(object):
	"""Contains functionality to convert text to integers."""

	def __init__(self, args):
		"""Standard __init__ method."""
		self.args = args
		input_path = os.path.join(args.data_dir, args.dataset + ".train.txt")
		lm_path = os.path.join(args.data_dir, args.lm)
		print(LOGS[0])
		# codecs is needed due to utf-8 setting
		# This is essential for non-ASCII characters
		with open(input_path, "r") as f:
			text = f.read()

		# Fix data to add </s> tokens
		# It is assumed that file has <unk> tokens
		self.text = text.replace('\n', ' </s> ').split()

		# vocab_file needs to be explicitly provided in data_dir
		# Re-Use old vocabulary file
		print(LOGS[1])
		saved_vocab = os.path.join(args.data_dir, args.vocab)
		with open(saved_vocab, 'r') as f:
			self.rev_vocab = f.read().split()

		self.vocab = vocab = {word: i for i, word in enumerate(self.rev_vocab)}

		args.vocab_size = self.vocab_size = len(self.vocab)

		if args.mode == 'test':
			return

		# Convert the text tokens to integers based on vocab map
		self.data = np.array(list(map(self.vocab.get, self.text)))

		# Load n-gram ARPA file
		self.tr = tr = Trie()
		tr.load_arpa(lm_path, vocab)


class BatchLoader(object):
	"""This class is used to build batches of data."""

	def __init__(self, args, data_loader):
		"""Standard __init__ function."""
		# Copying pointers
		self.args = args
		self.data_loader = data_loader
		self.vocab = vocab = data_loader.vocab
		self.vocab_size = data_loader.vocab_size
		data = data_loader.data
		self.batch_size = batch_size = args.config.batch_size
		self.timesteps = timesteps = args.config.timesteps

		self.num_batches = num_batches = int(len(data) / (batch_size * timesteps))
		# When the data (tensor) is too small
		if num_batches == 0:
			print(ERRORS[0])
			sys.exit()
		# Pruning away excess data, generally a few tokens
		if args.loss_mode != 'l1':
			print("Loading contexts...")
			self.contexts = contexts = []
			for i in range(len(data)):
				if i == 0:
					# For the first token, context is <s>
					context = [vocab['<s>'], data[0]]
				elif data[i] == vocab['</s>']:
					context = [vocab['<s>']]
				elif data[i - 1] == vocab['</s>']:
					context = [vocab['<s>'], data[i]]
				else:
					context = [data[i - 1], data[i]]
				contexts.append(context)

		xdata = data[:num_batches * batch_size * timesteps]
		ydata = np.copy(xdata)

		# Building output tokens - next token predictors
		ydata[:-1] = xdata[1:]
		ydata[-1] = xdata[0]

		self.xdata = xdata
		self.ydata = ydata

		# Splitting x, y and constants into batches.
		# Frequency batches are not generated now to save memory.
		self.x_batches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
		self.y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)

		self.pointer = 0

	def next_batch(self, l1):
		"""Output the next batch and corresponding frequencies."""
		timesteps = self.args.config.timesteps
		batch_size = self.args.config.batch_size
		vocab_size = self.vocab_size

		x = self.x_batches[self.pointer]
		y = self.y_batches[self.pointer]

		if abs(l1 - 1.0) <= 0.0001:
			freq = np.zeros((batch_size, timesteps, vocab_size))
		else:
			freq = self.get_freq()

		self.pointer += 1
		return x, y, freq

	def get_freq(self):
		"""Return a tensor having frequency data."""
		tr = self.data_loader.tr

		# Trial to check if trie implementation is correct
		# st = time.time()
		# tr.get_distro(contexts[0], np.zeros(self.vocab_size))
		# end = time.time()
		# print(end - st)

		# `tensor` will store the final batch to be sent to TensorFlow
		tensor = np.zeros([self.batch_size, self.timesteps, self.vocab_size])
		jobs = []
		for i in range(self.batch_size):
			for j in range(self.timesteps):
				print("Mapping %d %d"%(i,j))
				shared_tensor = sharedctypes.Array('d', np.ctypeslib.as_ctypes(tensor[i][j]), lock=False)
				ptr = i * self.timesteps * self.num_batches + self.pointer * self.timesteps + j
				p = mp.Process(target=tr.get_distro, args=(self.contexts[ptr], shared_tensor))
				jobs.append(p)
				p.start()

		for p in jobs:
			p.join()

		return np.ctypeslib.as_array(tensor)

	def reset_batch_pointer(self):
		"""Bring pointer back to first batch."""
		self.pointer = 0


def eval_loader(args, vocab, split):
	"""Convert raw evaluation data to correct format."""
	filename = "%s.%s.txt" % (args.dataset, split)
	input_path = os.path.join(args.data_dir, filename)
	with codecs.open(input_path, 'r', encoding='utf-8') as f:
		text = f.read()
	timesteps = args.config.timesteps
	# Fix data to add <s> and </s> characters
	tokens = text.replace('\n', ' </s> ').split()
	# Replacing all OOV with <unk>, converting to integers
	x = [vocab[c] for c in tokens]
	
	total_len = len(x)
	# pad ipa_x so the batch_size divides it exactly
	while len(x) % timesteps != 1:
		x.append(vocab['<unk>'])
	y = np.array(x[1:]).reshape((-1, timesteps))
	x = np.array(x[:-1]).reshape((-1, timesteps))
	return x, y, total_len
