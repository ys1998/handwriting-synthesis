import numpy as np

def load_data(string_file, points_file):
        a = np.load(points_file, encoding='bytes')
        pts = [np.transpose(x) for x in a]
        strings = []
        with open(string_file, 'r') as f:
                texts = f.readlines()
        strings = [s.strip('\n') for s in texts]
        return (strings, pts)
