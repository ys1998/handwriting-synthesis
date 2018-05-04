import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import sys
from model import SynthesisModel
from utilities import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mode', action='store', choices=['train','generate'], help='how the model is to be used', required=True)
parser.add_argument('--load_dir', action='store', help='directory where pretrained model exists', default=None)
parser.add_argument('--save_dir', action='store', help='directory to save trained model', default='save/model')
parser.add_argument('--log_dir', action='store', help='directory to save model summaries', default='logs/')
parser.add_argument('--strokes', action='store', help='path to handwriting strokes data file', default='data/strokes.npy')
parser.add_argument('--sentences', action='store', help='path to training sentences file', default='data/sentences.txt')
parser.add_argument('--n_layers', action='store', help='number of layers in LSTM network', type=int, default=3)
parser.add_argument('--K', action='store', help='number of gaussians in attention window', type=int, default=10)
parser.add_argument('--M', action='store', help='number of gaussians in MDN layer', type=int, default=20)
parser.add_argument('--lstm_size', action='store', help='number of LSTM cells in each layer', type=int, default=400)
parser.add_argument('--bias', action='store', help='bias towards choosing high probability output', type=float, default=0.5)
parser.add_argument('--batch_size', action='store', help='batch size for training', type=int, default=64)
parser.add_argument('--n_epochs', action='store', help='number of epochs of training', type=int, default=30)
parser.add_argument('--max_str_len', action='store', help='max characters in one section of input string', type=int, default=64)
parser.add_argument('--seq_len', action='store', help='max timesteps per batch', type=int, default=128)
parser.add_argument('--line', action='store', help='input string for handwriting synthesis', type=str, default='I am cool')
args = parser.parse_args()

def main(args):
	# Make directories if they don't exist
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)
	
	# Load data
	if args.mode == 'train':
		# cs, pts = load_data(args.sentences, args.strokes, args.save_dir, batch_size=args.batch_size, max_str_len=args.max_str_len)
		batch_loader = BatchLoader(args.sentences, args.strokes, args.save_dir, args.batch_size, args.seq_len, args.max_str_len)
		model = SynthesisModel(
			n_layers=args.n_layers, 
			batch_size=args.batch_size, 
			num_units=args.lstm_size, 
			K=args.K,
			M=args.M,
			n_chars=batch_loader.n_chars, 
			str_len=args.max_str_len,  
			sampling_bias=args.bias
			)
		if args.load_dir is not None:
			model.train(
				batch_loader, 
				n_epochs=args.n_epochs,
				restore=True, 
				load_path=args.load_dir,
				save_path=args.save_dir,
				log_path=args.log_dir
				)
		else:
			model.train(
				batch_loader, 
				n_epochs=args.n_epochs, 
				save_path=args.save_dir,
				log_path=args.log_dir
				)
	elif args.mode == 'generate':
		# Hack to get number of characters
		char_mapping = map_strings([], os.path.join(args.save_dir, 'mapping'))
		# Generate attention window tensor 'C'
		one_hot = generate_char_encoding(args.line, char_mapping, args.max_str_len)
		C = np.stack([one_hot]+[np.zeros([args.max_str_len, len(char_mapping)])]*(args.batch_size - 1))
		model = SynthesisModel(
			n_layers=args.n_layers, 
			batch_size=args.batch_size, 
			num_units=args.lstm_size, 
			K=args.K,
			M=args.M,
			n_chars=len(char_mapping), 
			str_len=args.max_str_len,  
			sampling_bias=args.bias
			)
		assert args.load_dir is not None, "Invalid load_dir."
		model.generate(C, args.load_dir)

if __name__=='__main__':
	main(args)
