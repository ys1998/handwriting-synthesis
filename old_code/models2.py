import torch
from torch.autograd import Variable as var
import torch.nn as nn
import torch.nn.functional as F

def SynthesisModel(object):
	def init_states(self):
		# Cell state for LSTM of each layer
		self.cell_statM of each layer
		self.hidden_states = []
		# Activations of hidden layers
		self.activations = [[]]
		self.windows = []
		self.windows.append(torch.zeros(self.n_chars, 1))
		for i in range(self.n_layers):
			self.cell_states.append(torch.zeros(1, 1, self.lstm_size))
			self.hidden_states.append(torch.zeros(1, 1, self.lstm_size))
			self.activations[0].append(torch.zeros(1, 1, self.lstm_size))

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
		self.windows = []
		self.windows.append(torch.zeros(self.n_chars, 1))
		# Initialize first hidden layer
		self.params["Wi1"] = var(torch.randn(self.lstm_size, 3))
		self.params["W11"] = var(torch.randn(self.lstm_size, self.lstm_size))
		self.params["Ww1"] = var(torch.randn(self.lstm_size, self.n_chars))
		self.params["b1"] = var(torch.zeros(self.lstm_size, 1))
		self.cell_states.append(torch.zeros(1, 1, self.lstm_size))
		self.hidden_states.append(torch.zeros(1, 1, self.lstm_size))
		self.lstm_nodes = []
		for i in range(n_layers):
			self.lstm_nodes.append(nn.LSTM(self.lstm_size, self.lstm_size, 1))
		# Initialize remaining hidden layers
		for i in range(n_layers - 1):
			self.params["Wi" + str(i+2)] = var(torch.randn(self.lstm_size, 3))
			self.params["W" + str(i+2) + str(i+2)] = var(torch.randn(self.lstm_size, self.lstm_size))
			self.params["Ww" + str(i+2)] = var(torch.randn(self.lstm_size, self.n_chars))
			self.params["W" + str(i+1) + str(i+2)] = var(torch.randn(self.lstm_size, self.lstm_size))
			self.params["b" + str(i+2)] = var(torch.zeros(self.lstm_size, 1))
			self.cell_states.append(torch.zeros(1, 1, self.lstm_size))
			self.hidden_states.append(torch.zeros(1, 1, self.lstm_size))
			self.activations[0].append(torch.zeros(1, 1, self.lstm_size))

		# Mixture Density Network (MDN) layer
		for i in range(n_layers + 1):
			self.params["W" + str(i+1) + "y"] = var(torch.randn(6*self.n_gaussians_mdn + 1, self.lstm_size))
		self.params["by"] = var(torch.zeros(6*self.n_gaussians_mdn + 1, 1))

    def forward_lstm_1(self, xt_, t_):
        	#	The outout of the Tth time is stored in at the Tth index in self.activations
        xtr = torch.mm(self.params['Wi1'], xt_)
        htr = torch.mm(self.params['W11'], self.activations[t_ - 1][0])
        wtr = torch.mm(self.params['Ww1'], self.windows[t_-1])
       	input_lstm = xtr + htr + wtr + self.params['b1']
      	out, states = self.lstm_nodes[0](input_lstm.view(1, 1, 400), (self.hidden_states[0], self.cell_states[0]))
       	self.hidden_states[0] = states[0]
        self.cell_states[0] = states[1]	
        self.activations[t_].append(out)
                #return out

    def forward_lstm_n(self, xt_, t_, n_):
        	# The n for the 1st hidden node is assumed to be 1 and so on
        xtr = torch.mm(self.params['Wi' + str(n_)], xt_)
     	htr = torch.mm(self.params['W' + str(n_) + str(n_)], self.activations[t_-1][n_ - 1])
       	wtr = torch.mm(self.params['Ww' + str(n_)], self.windows[t_])
       	h_prevtr = torch.mm(self.params['W' + str(n_ - 1) + str(n_)], self.activations[t_][n_ - 2])
        input_lstm = xtr + htr + h_prevtr + wtr + slef.params['b' + str(n_)]
        out, states = self.lstm_nodes[n_ - 1](input_lstm.view(1, 1, 400), (self.hidden_states[0], self.cell_states[0]))
        self.hidden_states[n_ - 1] = states[0]
        self.cell_states[n_ - 1] = states[1]
        self.activations[t_].append(out)
                #return out

    def forward(self, input_string, xt_, t_):
    	self.activations.append([])	# for the Tth time
    	self.forward_lstm_1(xt_, t_)
    	self._window_layer(t_)
    	for i in range(1, n_layers):
    		self.forward_lstm_n(xt_, t_, i)
    	_extract_MDN_params()

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

	def _window_layer(self, t):
		# Adds the output of the window layer at the self.windows[t_]
		h = self.activations[t][0]
		A = torch.mm(self.params["W"])


