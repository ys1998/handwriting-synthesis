import numpy as np
import sys
from model import SynthesisModel
from utilities import load_data

def main(args=None):
    # Load data
    cs, pts = load_data('../sentences.txt', '../strokes.npy', batch_size=50)
    model = SynthesisModel(n_layers=2, batch_size=50, lstm_size=400, n_chars=cs[0].shape[2], str_len=cs[0].shape[1],  sampling_bias=0.5)
    model.train(pts, cs)

if __name__=='__main__':
    main()