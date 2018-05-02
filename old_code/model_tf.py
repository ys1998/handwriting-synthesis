import time
import tensorflow as tf
import numpy as np
from six.moves import cPickle

# def var(t, requires_grad=False):
# 	if cuda.device_count() > 0:
# 		return Variable(t.cuda(), requires_grad=requires_grad)
# 	else:
# 		return Variable(t, requires_grad=requires_grad)

def Gaussian2D(x, mu, sigma, rho):
	eps = 1e-10
	factor = 1.0 / (2 * np.pi * sigma[0] * sigma[1] * tf.sqrt(1 - rho**2) + eps)
	# print(factor)
	norm_x = (x - mu)/(sigma + eps)
	# print(norm_x)
	Z = tf.reduce_sum(norm_x**2, axis=0) - 2 * rho * norm_x[0] * norm_x[1]
	# print(Z)
	val = factor * tf.exp(-Z/(2 * (1 - rho**2) + eps))
	# print(val)
	return val

def generate_char_encoding(string, char_mapping):
	n_chars = len(char_mapping)
	indices = []
	for idx, c in enumerate(string):
		assert c in char_mapping, "Unknown character encountered."
		indices.append(char_mapping[c])
	return tf.transpose(tf.one_hot(indices, n_chars))

def map_strings(lst_str):
	mapping = {}
	cntr = 0
	for s in lst_str:
		for c in s:
			if c not in mapping:
				mapping[c] = cntr
				cntr += 1
	return mapping

def load_data(string_file, points_file):
	a = np.load(points_file, encoding='bytes')
	pts = [x.reshape(3, -1) for x in a]
	strings = []
	with open(string_file, 'r') as f:
		texts = f.readlines()
	strings = [s.strip('\n') for s in texts]
	return (strings, pts)

class SynthesisModel(object):
	def __init__(self, n_layers = 1, lstm_size = 400, K = 10, M = 20, n_chars = 80):

		assert n_layers >=1, "Atleast one hidden layer is required."
		assert lstm_size > 0, "Negative lstm_size not allowed."
		assert K > 0, "Number of gaussians in window layer must be positive."
		assert M > 0, "Number of gaussians in MDN layer must be positive."
		assert n_chars > 0, "Atleast one character is required."

		# Store parameters
		self.n_layers = n_layers
		self.lstm_size = lstm_size
		self.n_gaussians_window = K
		self.n_gaussians_mdn = M
		self.n_chars = n_chars

		"""
		Initialize all parameters
		"""
		# Dictionary to store parameters
		self.params = {}

		# Initialize first hidden layer
		self.params["Wi1"] = tf.Variable(tf.random_normal([self.lstm_size, 3], 0.0, 1e-3))
		self.params["W11"] = tf.Variable(tf.random_normal([self.lstm_size, self.lstm_size], 0.0, 1e-3))
		self.params["Ww1"] = tf.Variable(tf.random_normal([self.lstm_size, self.n_chars], 0.0, 1e-3))
		self.params["b1"] = tf.Variable(tf.zeros([self.lstm_size, 1]))

		# Initialize connections to window layer
		self.params["W1w"] = tf.Variable(tf.random_normal([3*self.n_gaussians_window, self.lstm_size], 0.0, 1e-3))
		self.params["bw"] = tf.Variable(tf.zeros([3*self.n_gaussians_window, 1]))

		# Initialize LSTM nodes
		self.lstm_nodes = []
		for i in range(n_layers):
			cell = tf.contrib.rnn.LSTMCell(1, state_is_tuple=True, reuse=tf.AUTO_REUSE)
			self.lstm_nodes.append(cell)
		
		# Initialize remaining hidden layers
		for i in range(n_layers - 1):
			self.params["Wi" + str(i+2)] = tf.Variable(tf.random_normal([self.lstm_size, 3], 0.0, 1e-3))
			self.params["W" + str(i+2) + str(i+2)] = tf.Variable(tf.random_normal([self.lstm_size, self.lstm_size], 0.0, 1e-3))
			self.params["Ww" + str(i+2)] = tf.Variable(tf.random_normal([self.lstm_size, self.n_chars], 0.0, 1e-3))
			self.params["W" + str(i+1) + str(i+2)] = tf.Variable(tf.random_normal([self.lstm_size, self.lstm_size], 0.0, 1e-3))
			self.params["b" + str(i+2)] = tf.Variable(tf.zeros([self.lstm_size, 1]))

		# Mixture Density Network (MDN) layer
		for i in range(n_layers + 1):
			self.params["W" + str(i+1) + "y"] = tf.Variable(tf.random_normal([6*self.n_gaussians_mdn + 1, self.lstm_size], 0.0, 1e-3))
		self.params["by"] = tf.Variable(tf.zeros([6*self.n_gaussians_mdn + 1, 1]))


	"""
	Function to define computational graph for model.
	"""
	def define_graph(self, T, U):
		# Hidden state for LSTM of each layer
		self.hidden_states = []
		# Activations of hidden layers
		self.activations = [[]]
		# Window layer output
		self.window_output = [tf.Variable(tf.zeros([self.n_chars, 1]))]

		for cell in self.lstm_nodes:
			self.hidden_states.append((tf.Variable(tf.zeros([self.lstm_size, 1])), tf.Variable(tf.zeros([self.lstm_size, 1]))))
			self.activations[0].append(tf.Variable(tf.zeros([self.lstm_size, 1])))		

		# Initialize window layer parameter 'kappa'
		self.kappa = tf.Variable(tf.zeros([self.n_gaussians_window, 1]))

		# Define placeholders
		self.input = tf.placeholder(tf.float32, shape=(3,T))
		self.C = tf.placeholder(tf.float32, shape=(self.n_chars, U))
		self.output = tf.placeholder(tf.float32, shape=(3,T))
		
		# Define computational graph for one datapoint over all timesteps
		loss_values = []
		for t in range(1, T+1):
			st = time.time()
			xt = tf.expand_dims(self.input[:,t-1], axis=1)
			yt = tf.expand_dims(self.output[:,t-1], axis=1)
			self.activations.append([])
			self.forward_lstm_1(xt, t)
			w_t = self._window_layer(self.C, U, t)
			self.window_output.append(w_t)
			for i in range(1, self.n_layers):
				self.forward_lstm_n(xt, t, i+1)
			loss_values.append(self.loss_fn(yt, t))
			print("Timestep %d, time %.4f"%(t+1, time.time() - st))

		self.loss = tf.add_n(loss_values)


	def forward_lstm_1(self, xt_, t_):
		# The outout of the Tth time is stored in at the Tth index in self.activations
		# print(self.params['Wi1'].shape, xt_.shape)
		xtr = tf.matmul(self.params['Wi1'], xt_)
		htr = tf.matmul(self.params['W11'], self.activations[t_ - 1][0])
		wtr = tf.matmul(self.params['Ww1'], self.window_output[t_ - 1])
		input_lstm = xtr + htr + wtr + self.params['b1']
		out, state = self.lstm_nodes[0](input_lstm, self.hidden_states[0])
		self.hidden_states[0] = state
		self.activations[t_].append(out)

	def forward_lstm_n(self, xt_, t_, n_):
		# The n for the 1st hidden node is assumed to be 1 and so on
		xtr = tf.matmul(self.params['Wi' + str(n_)], xt_)
		htr = tf.matmul(self.params['W' + str(n_) + str(n_)], self.activations[t_-1][n_ - 1])
		wtr = tf.matmul(self.params['Ww' + str(n_)], self.window_output[t_])
		h_prevtr = tf.matmul(self.params['W' + str(n_ - 1) + str(n_)], self.activations[t_][n_ - 2])
		input_lstm = xtr + htr + h_prevtr + wtr + self.params['b' + str(n_)]
		out, state = self.lstm_nodes[n_ - 1](input_lstm, self.hidden_states[n_ - 1])
		self.hidden_states[n_ - 1] = state
		self.activations[t_].append(out)

	def _extract_MDN_params(self, t):
		# Obtain input for MDN layer

		y_hat = self.params["by"]
		for i in range(self.n_layers):
			y_hat = y_hat + tf.matmul(self.params["W" + str(i+1) + "y"], self.activations[t][i])

		e = 1.0 / (1.0 + tf.exp(y_hat[0]))
		# pi = []; mu = []; sigma = []; rho = []
		
		# for i in range(self.n_gaussians_mdn):
		# 	# chunk = y_hat[1 + i*6 : 1 + (i+1)*6]
		# 	pi.append(y_hat[1+i*6])
		# 	mu.append(y_hat[2+i*6:4+i*6])
		# 	sigma.append(tf.exp(y_hat[4+i*6:6+i*6]))
		# 	rho.append(tf.tanh(y_hat[6+i*6]))

		# pi_sftmx = tf.nn.softmax(pi)
		
		temp = tf.reshape(y_hat[1:], [6, self.n_gaussians_mdn])
		pi_sftmx = tf.nn.softmax(temp[0])
		mu = temp[1:3]
		sigma = tf.exp(temp[3:5])
		rho = tf.tanh(temp[5])

		return e, pi_sftmx, mu, sigma, rho

	def _window_layer(self, C, U, t):
		h = self.activations[t][0]
		A_t = tf.matmul(self.params["W1w"], h) + self.params["bw"]
		K = self.n_gaussians_window
		A = tf.exp(tf.reshape(A_t, [3, K]))
		
		alpha = tf.expand_dims(A[0,:], axis=1); beta = tf.expand_dims(A[1,:], axis=1)
		self.kappa.assign_add(tf.expand_dims(A[2,:], axis=1))
		cntr_U = tf.reshape(tf.tile(tf.range(U, dtype=tf.float32), [K]), [K, U])
		phi = tf.reduce_sum(alpha * tf.exp(-beta * (self.kappa - cntr_U)**2), axis=0, keep_dims=True)
		return tf.reduce_sum(phi * C, axis=1, keep_dims=True)

	def loss_fn(self, Y, t):
		prob = 0.0
		eps = 1e-10
		e, pi, mu, sigma, rho = self._extract_MDN_params(t)
		# for i in range(self.n_gaussians_mdn):
		# 	temp = pi[i] * Gaussian2D(Y[:-1], mu[i], sigma[i], rho[i])
		# 	prob = prob + tf.squeeze(temp * (e * Y[2] + (1 - e) * (1 - Y[2])))

		temp = pi * Gaussian2D(tf.expand_dims(Y[:-1], axis=1), mu, sigma, rho)
		prob = tf.reduce_sum(temp) * (e * Y[2] + (1 - e) * (1 - Y[2]))

		return -tf.log(prob + eps)

	# def save(self, file="params.pkl"):
		# with open(file, 'wb') as f:
		# 	cPickle.dump(self, f)

	def train(self, strings, char_mapping, points, n_epochs=50, lr=1e-4, beta1=0.9, beta2=0.99, eps=1e-8, lmbda=0.0):

		min_loss = 10.0

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for n in range(n_epochs):
				for i in range(len(strings)):
					C = generate_char_encoding(strings[i], char_mapping)
					U = tf.shape(C).eval()[1]
					X = points[i][:-1]
					Y = points[i][1:]
					T = X.shape[1]
					print("Creating computational graph ...")
					self.define_graph(T, U)
					# Initialize optimizer
					print("Done.")
					optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=eps).minimize(self.loss)
					feed_dict = {self.input:X, self.output:Y, self.C:C}
					_, loss = sess.run([optimizer, self.loss], feed_dict=feed_dict)

					print("Epoch %d batch %d loss %.4f" % (n+1, i+1, loss))

					# if loss.data[0] < min_loss:
					# 	min_loss = loss.data[0]
					# 	self.save()

if __name__ == '__main__':

	# Load data
	strings, points = load_data("sentences.txt", "strokes.npy")

	# Generate mapping
	char_mapping = map_strings(strings)
	n_chars = len(char_mapping)

	# Create model
	model = SynthesisModel(n_layers=2, lstm_size=400, K=10, M=20, n_chars=n_chars)
	model.train(strings, char_mapping, points)

	print("Model trained.")
