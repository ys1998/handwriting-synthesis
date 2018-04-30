"""
Definitions for the layers of our model.
"""
import tensorflow as tf
import numpy as np

"""
Class description for attention window layer.
"""
class WindowLayer(object):
	def __init__(self, n_gaussians, n_chars, C):
		# Number of gaussians used in the mixture
		self.n_gaussians = K = n_gaussians
		# Number of distinct characters in dataset
		self.n_chars = n_chars
		# One-hot encoded matrix of input string
		self.encoded_string = C
		# Length of input string
		self.string_length = U = tf.shape(C).eval()[1]
		# Matrix to store numbers from 0 to string_length - 1
		# Create counter matrix such that it can be broadcasted later
		self.cntr_matrix = tf.reshape(tf.range(U, dtype=tf.float32), [1, 1, -1])

	def __call__(self, combined_x, prev_kappa, reuse=tf.AUTO_REUSE):
		"""
		Extracting parameters alpha, beta and kappa from combined input.
		NOTE: 
		- Author calculated them by slicing the inputs and operating on them;
		both slicing and later operations can be combined into a single matrix
		multiplication, and hence represented by a single FC layer.
		- Weights learn both these tasks themselves during training.
		"""
		with tf.variable_scope(name='WindowLayer', reuse=reuse):
			alpha = tf.layers.dense(combined_x, 
									activation = tf.exp,
									units = self.n_gaussians, 
									kernel_initializer = tf.random_normal_initializer(stddev=1e-3),
									name = 'window_alpha'
									)
			beta = tf.layers.dense(combined_x,
									activation=tf.exp,
									units=self.n_gaussians,
									kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
									name='window_beta'
									)
			kappa = prev_kappa + tf.layers.dense(combined_x,
												activation=tf.exp,
												units=self.n_gaussians,
												kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
												name='window_kappa'
												)
			# All three variables defined above have dimension [batch_size, n_gaussians]
			# Convert them to dimension [batch_size, n_gaussian, 1] by introducing a new axis
			# This is done so that they can be broadcasted over all characters of string
			expd_alpha = tf.expand_dims(alpha, axis=2)
			expd_beta = tf.expand_dims(beta, axis=2)
			expd_kappa = tf.expand_dims(kappa, axis=2)

			# Calculate 'phi' using converted variables
			# Dimension of 'phi' will be [batch_size, n_gaussians, string_length]
			# It becomes [batch_size, 1, string_length] after summing along axis=1
			phi = tf.reduce_sum(expd_alpha * tf.exp(-expd_beta * tf.square(expd_kappa - self.cntr_matrix)), axis=1, keep_dims=True)
			# Obtain window output of dimension [batch_size, 1, string_length] * [batch_size, string_length, n_chars]
			# Converted to [batch_size, n_chars] from [batch_size, 1, n_chars] after squeezing
			w = tf.squeeze(tf.matmul(phi, self.encoded_string), axis=1)

			# Return computed value(s)
			return w, expd_kappa, tf.squeeze(phi, axis=1)
		
		def output_size(self):
			return [self.n_chars, self.n_gaussian]

"""
Class description for Mixture Density Network applied to the outputs 
of final layer.
"""
class MDNLayer(object):
	def __init__(self, n_gaussians):
		# Number of gaussians used in MDN
		self.n_gaussians = n_gaussians

	def __call__(self, combined_x, bias=0.0, reuse=tf.AUTO_REUSE):
		"""
		Extracting parameters e, mu_x, mu_y, sigma_x, sigma_y and rho from combined input.
		NOTE: 
		- Author calculated them by slicing the inputs and operating on them;
		both slicing and later operations can be combined into a single matrix
		multiplication, and hence represented by a single FC layer.
		- Weights learn both these tasks themselves during training.
		"""
		with tf.variable_scope(name="MDNLayer", reuse=reuse):
			# Express desired variables in terms of combined input
			e = tf.layers.dense(combined_x, units=1, kernel_initializer=tf.random_normal_initializer(stddev=1e-3), name="e")
			pi = tf.layers.dense(combined_x, units=self.n_gaussians, kernel_initializer=tf.random_normal_initializer(stddev=1e-3), name="pi")
			mu_x = tf.layers.dense(combined_x, units=self.n_gaussians, kernel_initializer=tf.random_normal_initializer(stddev=1e-3), name="mu_x")
			mu_y = tf.layers.dense(combined_x, units=self.n_gaussians, kernel_initializer=tf.random_normal_initializer(stddev=1e-3), name="mu_y")
			sigma_x = tf.layers.dense(combined_x, units=self.n_gaussians, kernel_initializer=tf.random_normal_initializer(stddev=1e-3), name="sigma_x")
			sigma_y = tf.layers.dense(combined_x, units=self.n_gaussians, kernel_initializer=tf.random_normal_initializer(stddev=1e-3), name="sigma_y")
			rho = tf.layers.dense(combined_x, units=self.n_gaussians, kernel_initializer=tf.random_normal_initializer(stddev=1e-3), name="rho")
			
			# Apply transformation functions on each variable
			e = tf.nn.sigmoid(e)
			rho = tf.tanh(rho)
			# Apply bias to variables in their transformation functions for biased sampling
			pi = tf.nn.softmax(pi * (1 + bias))
			sigma_x = tf.exp(sigma_x - bias)
			sigma_y = tf.exp(sigma_y - bias)

			# Return computed values
			return e, pi, mu_x, mu_y, sigma_x, sigma_y, rho

"""
Class description for the LSTM network used in hidden layers.
"""
class HiddenLayers(tf.nn.rnn_cell.RNNCell):
	def __init__(self, n_layers, n_units, batch_size, state_size, window_layer):
		super(RNNCell, self).__init__()
		# Number of layers in LSTM network
		self.n_layers = n_layers
		# Number of LSTM cells in each layer
		self.n_units = n_units
		# Store the window layer
		self.window_layer = window_layer
		# Size of batch and state for each LSTM cell
		self.batch_size = batch_size
		self.state_size = state_size

		with tf.variable_scope(name="LSTMNetwork", reuse=None):
			# Define LSTM nodes
			self.lstm_nodes = [tf.nn.rnn_cell.LSTMCell(num_units=n_units, state_is_tuple=True) for _ in range(n_layers)]
			# Initialize states (cell, hidden) for these nodes
			self.states = [x for _ in range(n_layers) 
								for x in 
									(tf.Variable(tf.zeros([batch_size, n_units]), trainable=False),
									tf.Variable(tf.zeros([batch_size, n_units]), trainable=False))]
			# Initialize states for window layer output and previous value of kappa
			self.states += [tf.Variable(tf.zeros([batch_size, s]), trainable=False) for s in self.window_layer.output_size()]
		
	def __call__(self, x, prev_states):
		# Extract previous output/value of window layer and kappa
		prev_window, kappa = prev_states[-2:]
		# List for storing current states 
		curr_states = []
		# List for storing output of previous *layer*
		prev_output = []
		for n in range(self.n_layers):
			with tf.variable_scope("lstm_layer_%d"%(n+1), reuse=tf.AUTO_REUSE):
				"""
				Get combined input to lstm by concatenating all components along a single column
				for each batch. Dimension of 'combined_x' is [batch_size, ]
				NOTE: When this LSTM gets unrolled over desired number of timesteps, previous *timestep*
				outputs will be automatically used.
				"""
				combined_x = tf.concat([x, prev_window] + prev_output, axis=1)
				# Pass combined input to lstm layer and store the output and new states
				output, new_state_tuple = self.lstm_nodes[n](combined_x, (prev_states[2*n], prev_states[2*n+1]))
				prev_output = list(output)
				curr_states.extend(list(new_state_tuple))
			# Update 'prev_window' and 'kappa' using the output of first layer (i.e. n=0)
			if n == 0:
				prev_window, kappa, _ = self.window_layer(output, kappa) 
		# Add window layer output and kappa to current states list
		curr_states.extend([prev_window, kappa])
		return output, curr_states

	# Override properties
	@property
	def state_size(self):
		return [self.n_units] * self.n_layers * 2 + self.window_layer.output_size()
