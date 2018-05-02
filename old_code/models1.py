import torch
from torch.autograd import Variable as Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.cuda as cuda
from six.moves import cPickle

# Set default tensor type
if cuda.device_count() > 0:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

def var(t, requires_grad=False):
	if cuda.device_count() > 0:
		return Variable(t.cuda(), requires_grad=requires_grad)
	else:
		return Variable(t, requires_grad=requires_grad)

def Gaussian2D(x, mu, sigma, rho):
	eps = 1e-10
	factor = 1.0 / (2 * np.pi * sigma[0] * sigma[1] * torch.sqrt(1 - rho**2) + eps)
	# print(factor)
	norm_x = (x - mu)/(sigma + eps)
	# print(norm_x)
	Z = torch.sum(norm_x**2) - 2 * rho * norm_x[0] * norm_x[1]
	# print(Z)
	val = factor * torch.exp(-Z/(2 * (1 - rho**2) + eps))
	# print(val)
	return val

def generate_char_encoding(string, char_mapping):
	n_chars = len(char_mapping)
	C = var(torch.zeros(n_chars, len(string)), requires_grad=False)
	for idx, c in enumerate(string):
		assert c in char_mapping, "Unknown character encountered."
		C[char_mapping[c], idx] = 1
	return C

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
	pts = [var(torch.from_numpy(x).contiguous().t(), requires_grad=False) for x in a]
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

		# Dictionary to store parameters
		self.params = {}
		# Cell state for LSTM of each layer
		self.cell_states = []
		# Hidden state for LSTM of each layer
		self.hidden_states = []
		# Activations of hidden layers
		self.activations = [[]]
		# Outputs of the window layer
		self.windows = [var(torch.zeros(self.n_chars, 1), requires_grad=True)]

		# Initialize window layer parameters
		self.kappa = [var(torch.zeros(1, self.n_gaussians_window), requires_grad=True)]

		# Initialize first hidden layer
		self.params["Wi1"] = var(torch.randn(self.lstm_size, 3)/1e3, requires_grad=True)
		self.params["W11"] = var(torch.randn(self.lstm_size, self.lstm_size)/1e3, requires_grad=True)
		self.params["Ww1"] = var(torch.randn(self.lstm_size, self.n_chars)/1e3, requires_grad=True)
		self.params["b1"] = var(torch.zeros(self.lstm_size, 1), requires_grad=True)
		# Initialize connections to window layer
		self.params["W1w"] = var(torch.randn(3*self.n_gaussians_window, self.lstm_size)/1e3, requires_grad=True)
		self.params["bw"] = var(torch.zeros(3*self.n_gaussians_window, 1), requires_grad=True)

		self.cell_states.append(var(torch.zeros(1, 1, self.lstm_size)))
		self.hidden_states.append(var(torch.zeros(1, 1, self.lstm_size)))

		# Initialize LSTM nodes
		self.lstm_nodes = []
		for i in range(n_layers):
			self.lstm_nodes.append(nn.LSTM(self.lstm_size, self.lstm_size, 1))
		
		# Initialize remaining hidden layers
		for i in range(n_layers - 1):
			self.params["Wi" + str(i+2)] = var(torch.randn(self.lstm_size, 3)/1e3, requires_grad=True)
			self.params["W" + str(i+2) + str(i+2)] = var(torch.randn(self.lstm_size, self.lstm_size)/1e3, requires_grad=True)
			self.params["Ww" + str(i+2)] = var(torch.randn(self.lstm_size, self.n_chars)/1e3, requires_grad=True)
			self.params["W" + str(i+1) + str(i+2)] = var(torch.randn(self.lstm_size, self.lstm_size)/1e3, requires_grad=True)
			self.params["b" + str(i+2)] = var(torch.zeros(self.lstm_size, 1)/1e3, requires_grad=True)
			self.cell_states.append(var(torch.zeros(1, 1, self.lstm_size)))
			self.hidden_states.append(var(torch.zeros(1, 1, self.lstm_size)))
			self.activations[0].append(var(torch.zeros(self.lstm_size, 1), requires_grad=False))

		# Mixture Density Network (MDN) layer
		for i in range(n_layers + 1):
			self.params["W" + str(i+1) + "y"] = var(torch.randn(6*self.n_gaussians_mdn + 1, self.lstm_size)/1e3, requires_grad=True)
		self.params["by"] = var(torch.zeros(6*self.n_gaussians_mdn + 1, 1), requires_grad=True)

	def forward_lstm_1(self, xt_, t_):
		# The outout of the Tth time is stored in at the Tth index in self.activations
		# print(self.params['Wi1'].shape, xt_.shape)
		xtr = torch.mm(self.params['Wi1'], xt_)
		htr = torch.mm(self.params['W11'], self.activations[t_ - 1][0])
		wtr = torch.mm(self.params['Ww1'], self.windows[t_-1])
		input_lstm = xtr + htr + wtr + self.params['b1']
		out, states = self.lstm_nodes[0](input_lstm.contiguous().view(1, 1, 400), (self.hidden_states[0], self.cell_states[0]))
		self.hidden_states[0] = states[0]
		self.cell_states[0] = states[1]	
		self.activations[t_].append(out.contiguous().view(-1,1))
		# return out

	def forward_lstm_n(self, xt_, t_, n_):
		# The n for the 1st hidden node is assumed to be 1 and so on
		xtr = torch.mm(self.params['Wi' + str(n_)], xt_)
		htr = torch.mm(self.params['W' + str(n_) + str(n_)], self.activations[t_-1][n_ - 1])
		wtr = torch.mm(self.params['Ww' + str(n_)], self.windows[t_])
		h_prevtr = torch.mm(self.params['W' + str(n_ - 1) + str(n_)], self.activations[t_][n_ - 2])
		input_lstm = xtr + htr + h_prevtr + wtr + self.params['b' + str(n_)]
		out, states = self.lstm_nodes[n_ - 1](input_lstm.contiguous().view(1, 1, 400), (self.hidden_states[n_ - 1], self.cell_states[n_ - 1]))
		self.hidden_states[n_ - 1] = states[0]
		self.cell_states[n_ - 1] = states[1]
		self.activations[t_].append(out.contiguous().view(-1,1))
		# return out

	def forward(self, xt_, t_, C):
		self.activations.append([])	# for the Tth time
		self.forward_lstm_1(xt_, t_)
		w_t = self._window_layer(C, t_)
		self.windows.append(w_t)
		for i in range(1, self.n_layers):
			self.forward_lstm_n(xt_, t_, i+1)

	def _extract_MDN_params(self, t):
		# Obtain input for MDN layer
		# y_hat = self.params["by"]
		# for i in range(self.n_layers + 1):
		# 	y_hat += torch.matmul(self.params["W" + str(i+1) + "y"], self.activations[t][i])
		sum_terms = [self.params["by"]]
		for i in range(self.n_layers):
			sum_terms.append(torch.matmul(self.params["W" + str(i+1) + "y"], self.activations[t][i]))

		y_hat = torch.sum(torch.stack(sum_terms), dim=0)

		e = 1.0 / (1.0 + torch.exp(y_hat[0]))
		pi = []; mu = []; sigma = []; rho = []
		
		for i in range(self.n_gaussians_mdn):
			chunk = y_hat[1 + i*6 : 1 + (i+1)*6]
			pi.append(chunk[0])
			mu.append(chunk[1:3])
			sigma.append(torch.exp(chunk[3:5]))
			rho.append(torch.tanh(chunk[5]))

		# print(pi)
		pi_sftmx = F.softmax(torch.stack(pi), dim=0)
		# print(pi_sftmx)
		return e, pi_sftmx, mu, sigma, rho

	def _window_layer(self, C, t):
		h = self.activations[t][0]
		A_t = torch.mm(self.params["W1w"], h) + self.params["bw"]
		K = self.n_gaussians_window
		A = torch.exp(A_t.contiguous().view(3, K))
		self.alpha = A[0,:]; self.beta = A[1,:]; self.kappa.append(A[2,:].contiguous().view(1, -1))
		U = C.shape[1]
		cntr_U = var(torch.arange(U).repeat(K).contiguous().view(K, U).t(), requires_grad=False)
		phi = torch.sum(self.alpha * torch.exp(-self.beta * (self.kappa[-1] - cntr_U)**2), dim=1)
		w_t = torch.sum(phi * C, dim=1)
		return w_t.contiguous().view(-1,1)

	def loss_fn(self, Y, t):
		prob = 0.0
		eps = 1e-10
		e, pi, mu, sigma, rho = self._extract_MDN_params(t)
		for i in range(self.n_gaussians_mdn):
			temp = pi[i] * Gaussian2D(Y[:-1], mu[i], sigma[i], rho[i])
			prob += temp * e if Y[2].data[0] == 1 else temp * (1 - e)
		return -torch.log(prob + eps)

	def init_states(self):
		# Cell state for LSTM of each layer
		self.cell_states = []
		self.hidden_states = []
		# Activations of hidden layers
		self.activations = [[]]

		self.windows = [var(torch.zeros(self.n_chars, 1), requires_grad=True)]
		self.kappa = [var(torch.zeros(1, self.n_gaussians_window), requires_grad=True)]

		for i in range(self.n_layers):
			self.cell_states.append(var(torch.zeros(1, 1, self.lstm_size)))
			self.hidden_states.append(var(torch.zeros(1, 1, self.lstm_size)))
			self.activations[0].append(var(torch.zeros(self.lstm_size, 1), requires_grad=True))

	def save(self, file="params.pkl"):
		with open(file, 'wb') as f:
			cPickle.dump(self, f)

	def train(self, S, char_mapping, P, n_epochs=300, lr=1e-4, beta1=0.9, beta2=0.99, eps=1e-8, lmbda=0.0):
		# Collect all parameters
		parameters = []
		for k,v in self.params.items():
			parameters.append(v)
		for i in range(self.n_layers):
			for p in self.lstm_nodes[i].parameters():
				parameters.append(p)

		# Shift to GPU by calling cuda()
		# To be tried

		# Initialize optimizer
		optimizer = optim.SGD(parameters, lr, weight_decay=0.1) #, (beta1, beta2), eps, lmbda)
		min_loss = 10.0

		for n in range(n_epochs):
			for i in range(len(S)):
				self.init_states()
				C = generate_char_encoding(S[i], char_mapping)
				X = P[i][:,:-1]
				# print(X.shape)
				Y = P[i][:,1:]
				timesteps = X.shape[1]
				loss_values = []
				for t in range(timesteps):
					self.forward(X[:,t].contiguous().view(-1,1), t+1, C)
					loss_values.append(self.loss_fn(Y[:,t].contiguous().view(-1,1), t+1))
				
				optimizer.zero_grad()
				loss = torch.sum(torch.stack(loss_values), dim=0)/timesteps
				loss.backward()
				optimizer.step()
				print("Epoch %d batch %d loss %.4f" % (n+1, i+1, loss.data[0]))

				if loss.data[0] < min_loss:
					min_loss = loss.data[0]
					self.save()

if __name__ == '__main__':

	# Load data
	strings, points = load_data("sentences.txt", "strokes.npy")

	# Generate mapping
	char_mapping = map_strings(strings)
	n_chars = len(char_mapping)

	# Create model
	model = SynthesisModel(n_layers=2, lstm_size=400, K=10, M=20, n_chars=n_chars)
	model.init_states()
	model.train(strings, char_mapping, points)

	print("Model trained.")
