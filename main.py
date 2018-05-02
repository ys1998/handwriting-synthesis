import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import sys
from model import SynthesisModel
from utilities import load_data
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mode', action='store', choices=['train','generate'], help='how the model is to be used', required=True)
parser.add_argument('--load_dir', action='store', help='directory where pretrained model exists', default=None)
parser.add_argument('--save_dir', action='store', help='directory to save trained model', default='save/model')
parser.add_argument('--strokes', action='store', help='path to handwriting strokes data file', default='data/strokes.npy')
parser.add_argument('--sentences', action='store', help='path to training sentences file', default='data/sentences.txt')
parser.add_argument('--n_layers', action='store', help='number of layers in LSTM network', type=int, default=2)
parser.add_argument('--K', action='store', help='number of gaussians in attention window', type=int, default=10)
parser.add_argument('--M', action='store', help='number of gaussians in MDN layer', type=int, default=20)
parser.add_argument('--lstm_size', action='store', help='number of LSTM cells in each layer', type=int, default=200)
parser.add_argument('--bias', action='store', help='bias towards choosing high probability output', type=float, default=0.5)
parser.add_argument('--batch_size', action='store', help='batch size for training', type=int, default=50)
parser.add_argument('--lr', action='store', help='initial learning rate', type=float, default=1e-6)
parser.add_argument('--n_epochs', action='store', help='number of epochs of training', type=int, default=50)
args = parser.parse_args()

def main(args):
	# Load data
	if args.mode == 'train':
		cs, pts = load_data(args.sentences, args.strokes, batch_size=args.batch_size)
		model = SynthesisModel(
			n_layers=args.n_layers, 
			batch_size=args.batch_size, 
			num_units=args.lstm_size, 
			K=args.K,
			M=args.M,
			n_chars=cs[0].shape[2], 
			str_len=cs[0].shape[1],  
			sampling_bias=args.bias
			)
		if args.load_dir is not None:
			model.train(
				pts, cs, 
				n_epochs=args.n_epochs, 
				initial_learning_rate=args.lr, 
				restore=True, 
				load_path=args.load_dir,
				save_path=args.save_dir
				)
		else:
			model.train(
				pts, cs, 
				n_epochs=args.n_epochs, 
				initial_learning_rate=args.lr, 
				save_path=args.save_dir
				)
	elif args.mode == 'generate':
		pass

if __name__=='__main__':
	main(args)