import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import sys, subprocess
from model import SynthesisModel
from utilities import *
from argparse import ArgumentParser
from write import create_handwriting

parser = ArgumentParser()
parser.add_argument('--mode', action='store', choices=['train','generate', 'predict'], help='how the model is to be used', required=True)
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
parser.add_argument('--prior', action='store', help='prior string for text prediction', type=str, default='I am cool')
parser.add_argument('--words', action='store', help='number of words to be predicted', type=int, default=0)
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
		create_handwriting(args.line, 'save/model-29')

	elif args.mode == 'predict':		
		assert len(args.prior) < args.max_str_len, "Prior length should be less than %d"%args.max_str_len
		# Predict sentence
		os.chdir('language_model')
		command = "python3 generate.py --mode generate --line \"{0}\" --words {1} --data_dir ptb/ --dataset ptb --lm ngram-lm --job_id job_0".format(args.prior, args.words)
		os.system(command + "> temp")
		os.chdir('..')
		with open('language_model/temp', 'r') as f:
			line = f.readlines()[-2].strip('\n')

		print('Predicted text: ', line.strip('\n'))
		create_handwriting(line, 'save/model-29')

if __name__=='__main__':
	main(args)
