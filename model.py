"""
Model definition.
"""
import tensorflow as tf
import numpy as np
import time
from layers import WindowLayer, HiddenLayers, MDNLayer
from utilities import *
import matplotlib.pyplot as plt

def gaussian_sample(e, mu_x, mu_y, sg_x, sg_y, rho):
	cov = np.array([ [sg_x*sg_x, sg_x*sg_y*rho],
					 [sg_x*sg_y*rho, sg_y*sg_y]])
	mean = np.array([mu_x, mu_y])
	x,y = np.random.multivariate_normal(mean, cov)
	end = np.random.binomial(1, e)
	return (x, y, end)

class SynthesisModel(object):
	def __init__(self, n_layers=1, batch_size=100, num_units=100, K=10, M=20, n_chars=80, str_len=0, sampling_bias = 0.0):
		# Store parameters
		self.n_layers = n_layers
		self.batch_size = batch_size
		self.n_chars = n_chars
		self.n_units = num_units
		self.n_gaussians_window = K
		self.n_gaussians_mdn = M
		self.bias = sampling_bias

		# Define the computational graph here
		self.graph = tf.Graph()
		with self.graph.as_default():
			# Create placeholder for input of dimension [batch_size, timesteps, 3]
			inputs = tf.placeholder(tf.float32, [None, None, 3], name='input')
			# Create placeholder for expected output of dimension [batch_size, timesteps, 3]
			output = tf.placeholder(tf.float32, [None, None, 3], name='expected_output')
			# Create placeholder for one-hot encoded string of dimension [batch_size, seq_length, n_chars]
			C = tf.placeholder(tf.float32, [None, None, self.n_chars], name='C')
			# Placeholder for reset values
			reset = tf.placeholder(tf.float32, [None, 1], name='reset')

			# Declare variable to count number of steps
			self.step_cntr = tf.Variable(0)

			# Create layers for the model
			attention_window = WindowLayer(n_gaussians=self.n_gaussians_window, n_chars=self.n_chars, str_len=str_len, C=C)
			mdn_unit = MDNLayer(n_gaussians=self.n_gaussians_mdn)
			
			lstm_network = HiddenLayers(n_layers=self.n_layers, n_units=self.n_units, batch_size=self.batch_size, window_layer=attention_window)
			
			reset_states = tf.group(*[org.assign(org*reset) for org in lstm_network.states])

			# Unroll lstm network over timesteps
			obtained_output, final_states = tf.nn.dynamic_rnn(lstm_network, inputs, initial_state=lstm_network.states)

			# Pass output of LSTM network to MDN unit after reshaping
			outs = tf.reshape(obtained_output, [-1, self.n_units])
			e, pi, mu_x, mu_y, sigma_x, sigma_y, rho = mdn_unit(outs, self.bias)
			
			# Calculate loss using above parameters and correct output
			reshaped_output = tf.reshape(output, [-1, 3])
			corr_x, corr_y, corr_e = tf.unstack(tf.expand_dims(reshaped_output, axis=2), axis=1)
			norm_x = (corr_x - mu_x)/sigma_x
			norm_y = (corr_y - mu_y)/sigma_y
			mrho = 1.0 - tf.square(rho)
			Z = tf.square(norm_x) + tf.square(norm_y) - 2.0 * rho * norm_x * norm_y
			val = tf.exp(-Z / (2.0 * mrho)) / (2.0 * np.pi * sigma_x * sigma_y * tf.sqrt(mrho))
			prob = corr_e * e + (1.0 - corr_e) * (1.0 - e)
			rval = tf.reduce_sum(pi * val, axis=1)

			# Take sum over all timesteps and average over all batch members, 'M' gaussians
			# Dimension of 'val' is [batch_size, seq_len, n_gaussians_mdn]
			# Averaging step wise to prevent overflow
			# Add a small quantity to make computations stable
			loss = -tf.reduce_mean(tf.log(rval + 1e-8) + tf.log(prob + 1e-8))

			with tf.variable_scope('Optimizer'):
				# Create optimizer
				learning_rate = tf.train.exponential_decay(1e-3, global_step=self.step_cntr, decay_steps=1000, decay_rate=0.5)
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
				
				# Clip gradients to prevent 'nan' values
				grads, variables = map(list, zip(*optimizer.compute_gradients(loss)))
				grads, _ = tf.clip_by_global_norm(grads, 5.0)
				backprop = optimizer.apply_gradients(zip(grads, variables))

			# Save these parameters/variables for accessing later
			self.params = {
				'loss':loss,
				'step':backprop, 
				'input':inputs, 
				'output':output, 
				'C':C,
				'reset':reset,
				'reset_states':reset_states,
				'step_cntr':self.step_cntr, 
				'phi':final_states[-3],
				'e':e, 'pi':pi, 'mu_x':mu_x, 'mu_y':mu_y, 
				'sigma_x':sigma_x, 'sigma_y':sigma_y, 'rho':rho,
				}
	
	def train(self, batch_loader, restore=False, n_epochs=50, load_path=None, save_path=None, log_path=None):
		# Create session using generated graph 
		with tf.Session(graph=self.graph) as sess:
			saver = tf.train.Saver(max_to_keep=2)
			writer = tf.summary.FileWriter(log_path, sess.graph)
			loss_summary = tf.summary.scalar('Loss', self.params['loss'])

			sess.run(tf.global_variables_initializer())
			
			# Restore saved model if required
			if restore:
				saver.restore(sess, tf.train.latest_checkpoint(load_path))
				st_epoch = sess.run(self.params['step_cntr']) // len(C)
				print('Restored variables ->')
				for i in tf.global_variables():
					print(i)
				print("Loaded trained model, starting from epoch %d."%(st_epoch+1))
			else:
				st_epoch = 0

			for n in range(st_epoch, n_epochs):
				for batch_cntr in range(self.batch_size):
					# Get batch of data
					tr_d, C, reset, reset_reqd = batch_loader.get_batch()
					if reset_reqd:
						sess.run(self.params['reset_states'], feed_dict={self.params['reset']:reset})

					delta = time.time()
					_, loss, merged = \
					sess.run([
						self.params['step'], 
						self.params['loss'], 
						tf.summary.merge_all()
						], 
						feed_dict={
							self.params['input']:tr_d[:,:-1,:], 
							self.params['output']:tr_d[:,1:,:], 
							self.params['C']:C
							})
					delta = time.time() - delta
					print("Epoch %d/%d, batch %d/%d, loss %.4f, time %.3f sec" % (n+1, n_epochs, batch_cntr+1, len(C), loss, delta))

					# Add loss summary
					writer.add_summary(merged, global_step=self.params['step_cntr'].eval())

				# Save model after every epoch
				saver.save(sess, save_path, global_step=self.params['step_cntr'].eval())

	def generate(self, C, load_path=None):
		inputs = prediction_input(np.array([0., 0., 1.]), self.batch_size)
		phi = []
		coordinates = []
		coordinates.append(np.array([0., 0., 1.]))
		writing_flag = True
		str_len = C.shape[1]
		counter = 0
		phi_data = []

		# Creating the session
		with tf.Session(graph=self.graph) as sess:
			# Loading the trained params
			saver = tf.train.Saver(max_to_keep = 2)
			print(load_path)
			saver.restore(sess, tf.train.latest_checkpoint(load_path))
			print("Restored variables ->")
			for i in tf.global_variables():
				print(i)
				
			while(writing_flag and counter < str_len*30):
				# e, pi, mu_x, mu_y, sigma_x, sigma_y, rho, phi = \
				e, pi, mu_x, mu_y, sigma_x, sigma_y, rho = \
				sess.run([
					self.params['e'], 
					self.params['pi'], self.params['mu_x'],
					self.params['mu_y'], self.params['sigma_x'],
					self.params['sigma_y'], self.params['rho'],
					self.params['phi']
					],
					feed_dict = {
						self.params['input']:inputs,
						self.params['C']:C
						}
					)
                # Randomly choose a gaussian to sample from
				# Use its contribution to the mixture as its 'picking' probability
				g = np.random.choice(np.arange(pi.shape[2]), p=pi[0,0])
				# Sample point from 2D gaussian
				point = gaussian_sample(e[0,0], mu_x[0, g], mu_y[0, g],
								sigma_x[0, g], sigma_y[0, g], rho[0, g])
				coordinates.append(point)
				
				# Check if the model has finished sampling
				finish = phi[0, -1] > np.max(phi[0, :-1])
				if not finish:
					inputs = prediction_input(np.array(point), self.batch_size)
					counter += 1
				else:
					writing_flag = False
					print('\nSampling done.')

		# Plotting routine
		sum_x = sum_y = 0
		points = [[[],[]]]
		for pts in coordinates:
			sum_x += pts[0]
			sum_y += pts[1]
			points[-1][0].append(sum_x)
			points[-1][1].append(sum_y)
			if pts[2] == 1.:
				points.append([[],[]])

		plt.figure()
		for pts in points:
			plt.plot(pts[0], pts[1])
		plt.show()
