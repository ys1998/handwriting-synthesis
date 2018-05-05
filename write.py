import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple

def sample(e, mu1, mu2, std1, std2, rho):
	cov = np.array([[std1 * std1, std1 * std2 * rho],
					[std1 * std2 * rho, std2 * std2]])
	mean = np.array([mu1, mu2])

	x, y = np.random.multivariate_normal(mean, cov)
	end = np.random.binomial(1, e)
	return np.array([x, y, end])


def split_strokes(points):
	points = np.array(points)
	strokes = []
	b = 0
	for e in range(len(points)):
		if points[e, 2] == 1.:
			strokes += [points[b: e + 1, :2].copy()]
			b = e + 1
	return strokes


def cumsum(points):
	sums = np.cumsum(points[:, :2], axis=0)
	return np.concatenate([sums, points[:, 2:]], axis=1)

def sample_text(sess, args_text, translation):
	fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
			  'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
	params = namedtuple('Params', fields)(
		*[tf.get_collection(name)[0] for name in fields]
	)

	text = np.array([translation.get(c, 0) for c in args_text])
	coord = np.array([0., 0., 1.])
	coords = [coord]

	sequence = np.eye(len(translation), dtype=np.float32)[text]
	sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

	stroke_data = []
	sess.run(params.zero_states)
	sequence_len = len(args_text)
	for s in range(1, 60 * sequence_len + 1):
		print('\rSampled {0} points ...'.format(s), end='')

		e, pi, mu1, mu2, std1, std2, rho, \
		finish, phi, window, kappa = sess.run([params.e, params.pi, params.mu1, params.mu2,
											   params.std1, params.std2, params.rho, params.finish,
											   params.phi, params.window, params.kappa],
											  feed_dict={
												  params.coordinates: coord[None, None, ...],
												  params.sequence: sequence,
												  params.bias: 1.0
											  })

		g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
		coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
					   std1[0, g], std2[0, g], rho[0, g])
		coords += [coord]
		stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

		if finish[0, 0] > 0.8:
			print('\nSampling done.')
			break

	coords = np.array(coords)
	coords[-1, 2] = 1.

	return stroke_data, coords


def create_handwriting(line, model_path):
	with open(os.path.join(model_path, 'mapping.pkl'), 'rb') as file:
		mapping = pickle.load(file)
	rev_mapping = {v: k for k, v in mapping.items()}
	charset = [rev_mapping[i] for i in range(len(rev_mapping))]
	charset[0] = ''

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		saver = tf.train.import_meta_graph(model_path + '.meta')
		saver.restore(sess, model_path)

		stroke_data, coords = sample_text(sess, line, mapping)
		
		fig, ax = plt.subplots(1, 1)
		for stroke in split_strokes(cumsum(np.array(coords))):
			plt.plot(stroke[:, 0], -stroke[:, 1])
		ax.set_title('Handwriting')
		ax.set_aspect('equal')
		plt.show()