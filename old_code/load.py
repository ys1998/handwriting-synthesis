import numpy as np

def load_data(self, filename):
        a = np.load(filename, encoding='bytes')
        b = [np.transpose(x) for x in a]
        return b
