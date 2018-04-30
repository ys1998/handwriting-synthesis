import torch
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

def Gaussian2D(x, mu, sigma, rho):
	factor = 1.0 / (2 * np.pi * sigma[0] * sigma[1] * torch.sqrt(1 - rho**2))
	norm_x = (x - mu)/sigma
	Z = torch.sum(norm_x**2) - 2 * rho * norm_x[0] * norm_x[1]
	val = factor * torch.exp(-Z/(2 * (1 - rho**2)))
	return val

def SynthesisModel(object):
	def __init__(self, n_layers = 1, lstm_size = 400, K = 10, M = 20, n_chars = 80):

		assert n_layers >=1, "Atleast one layer is required."
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
		self.activations = []
		# Window layer parameters
		self.alpha = None
		self.beta = None
		self.kappa = torch.zeros(K)
		
		# Initialize first hidden layer
		self.params["Wi1"] = var(torch.randn(self.lstm_size, 3))
		self.params["W11"] = var(torch.randn(self.lstm_size, self.lstm_size))
		self.params["Ww1"] = var(torch.randn(self.lstm_size, self.n_chars))
		self.params["b1"] = var(torch.zeros(self.lstm_size, 1))
		self.cell_states.append(torch.zeros(1, 1, self.lstm_size))
		self.hidden_states.append(torch.zeros(1, 1, self.lstm_size))

		# Initialize remaining hidden layers
		for i in range(n_layers - 1):
			self.params["Wi" + str(i+2)] = var(torch.randn(self.lstm_size, 3))
			self.params["W" + str(i+2) + str(i+2)] = var(torch.randn(self.lstm_size, self.lstm_size))
			self.params["Ww" + str(i+2)] = var(torch.randn(self.lstm_size, self.n_chars))
			self.params["W" + str(i+1) + str(i+2)] = var(torch.randn(self.lstm_size, self.lstm_size))
			self.params["b" + str(i+2)] = var(torch.zeros(self.lstm_size, 1))
			self.cell_states.append(torch.zeros(1, 1, self.lstm_size))
			self.hidden_states.append(torch.zeros(1, 1, self.lstm_size))

		# Mixture Density Network (MDN) layer
		for i in range(n_layers):
			self.params["W" + str(i+1) + "y"] = var(torch.randn(6*self.n_gaussians_mdn + 1, self.lstm_size))
		self.params["by"] = var(torch.zeros(6*self.n_gaussians_mdn + 1, 1))

		def _extract_MDN_params(self):
			# Obtain input for MDN layer
			y_hat = self.params["by"]
			for i in range(self.n_layers + 1):
				y_hat += torch.matmul(self.params["W" + str(i+1) + "y"], self.activations[i])

			e = 1.0 / (1.0 + torch.exp(y_hat[0]))
			pi = []; mu = []; sigma = []; rho = []
			
			for i in range(self.n_gaussians_mdn):
				chunk = y_hat[1 + i*6 : 1 + (i+1)*6]
				pi.append(chunk[0])
				mu.append(tuple([chunk[1], chunk[2]]))
				sigma.append(tuple(torch.exp(chunk[3]), torch.exp(chunk[4])))
				rho.append(torch.tanh(chunk[5]))

			pi = F.softmax(torch.Tensor(pi))
			return e, pi, mu, sigma, rho

		def _window_layer(self, C, t):
			h = self.activations[t][0]
			A = torch.mm(self.params["W1w"], h) + self.params["bw"]
			K = self.n_gaussians_window
			A = torch.exp(A.view(3, K))
			self.alpha = A[0,:]; self.beta = A[1,:]; self.kappa += A[2,:]
			U = C.shape[1]
			cntr_U = torch.arange(U).repeat(K).view(K, U).t()
			phi = torch.sum(self.alpha * torch.exp(-self.beta * (self.kappa - cntr_U)**2), axis=1)
			w_t = torch.sum(phi * C, axis=1)
			return w_t

		def loss(self, Y):
			loss = 0.0
			e, pi, mu, sigma, rho = self._extract_MDN_params()
			for i in range(self.n_gaussians_mdn):
				temp = Gaussian2D(Y, mu, sigma, rho)
				loss += temp * e if Y[2] == 1 else temp * (1 - e)
			return -torch.log(loss)


