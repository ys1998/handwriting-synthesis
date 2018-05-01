"""
Model definition.
"""
import tensorflow as tf
import numpy as np
import time
from layers import WindowLayer, HiddenLayers, MDNLayer

class SynthesisModel(object):
    def __init__(self, n_layers=1, batch_size=100, lstm_size=400, num_units=100, K=10, M=20, n_chars=80, str_len=0, sampling_bias = 0.0):
        # Store parameters
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_chars = n_chars
        self.lstm_size = lstm_size
        self.n_units = num_units
        self.n_gaussians_window = K
        self.n_gaussians_mdn = M
        self.bias = sampling_bias

        # Define the computational graph here
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create placeholder for input of dimension [batch_size, timesteps, 3]
            inputs = tf.placeholder(tf.float32, [None, None, 3])
            # Create placeholder for expected output of dimension [batch_size, timesteps, 3]
            output = tf.placeholder(tf.float32, [None, None, 3])
            # Create placeholder for one-hot encoded string of dimension [batch_size, seq_length, n_chars]
            C = tf.placeholder(tf.float32, [None, None, self.n_chars])

            # Create layers for the model
            attention_window = WindowLayer(n_gaussians=self.n_gaussians_window, n_chars=self.n_chars, str_len=str_len, C=C)
            mdn_unit = MDNLayer(n_gaussians=self.n_gaussians_mdn)
            
            lstm_network = HiddenLayers(n_layers=self.n_layers, n_units=self.n_units, batch_size=self.batch_size, window_layer=attention_window)
            # Unroll lstm network over timesteps
            obtained_output, final_states = tf.nn.dynamic_rnn(lstm_network, inputs, initial_state=lstm_network.states)

            # Pass output of LSTM network to MDN unit
            e, pi, mu_x, mu_y, sigma_x, sigma_y, rho = mdn_unit(obtained_output, self.bias)
            
            # Calculate loss using above parameters and correct output
            # Add a small quantity to make computations stable
            epsilon = 1e-8 
            corr_x, corr_y, corr_e = tf.unstack(tf.expand_dims(output, axis=3), axis=2)
            norm_x = (corr_x - mu_x)/(sigma_x + epsilon)
            norm_y = (corr_y - mu_y)/(sigma_y + epsilon)
            mrho = 1 - tf.square(rho)
            Z = norm_x**2 + norm_y**2 - 2*norm_x*norm_y*rho
            factor = 1.0/(2*np.pi*sigma_x*sigma_y*tf.sqrt(mrho) + epsilon)
            val = pi*factor*tf.exp(-Z/(2*mrho + epsilon))
            val = val * (corr_e*e + (1-corr_e)*(1-e))
            # Take sum over all timesteps and average over all batch members, 'M' gaussians
            # Dimension of 'val' is [batch_size, seq_len, n_gaussians_mdn]
            loss = tf.reduce_mean(-tf.log(epsilon + tf.reduce_sum(val, axis=2)))

            # Save these parameters/variables for accessing later
            self.params = {'loss':loss, 'input':inputs, 'output':output, 'C':C}
    
    def train(self, points, C, restore=False, n_epochs=50, learning_rate=1e-3, path_to_model=None):
        if restore and path_to_model is None:
            print("Path to saved model not specified.")
            exit()
        else:
            # Split points into training data
            tr_d = [(pts[:,:-1,:], pts[:,1:,:]) for pts in points]
            
            with tf.Session(graph=self.graph) as sess:
                # Create optimizer
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.params['loss'])
                sess.run(tf.global_variables_initializer())
                for n in range(n_epochs):
                    batch_cntr = 1
                    for dp, c in zip(tr_d, C):
                        delta = time.time()
                        _, loss = sess.run([optimizer, self.params['loss']], feed_dict={self.params['input']:dp[0], self.params['output']:dp[1], self.params['C']:c})
                        delta = time.time() - delta
                        print("Epoch %d/%d, batch %d/%d, loss %.4f, time %.3f sec" % (n+1, n_epochs, batch_cntr, len(C), loss, delta))
                        batch_cntr += 1
