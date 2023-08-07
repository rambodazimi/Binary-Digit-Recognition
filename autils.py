import numpy as np
import os


def load_data():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the new path by concatenating the script directory and the relative path
    relative_path_X = os.path.join(script_directory, 'data/X.npy')
    relative_path_y = os.path.join(script_directory, 'data/y.npy')

    X = np.load(relative_path_X)
    y = np.load(relative_path_y)
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def load_weights():
    w1 = np.load("data/w1.npy")
    b1 = np.load("data/b1.npy")
    w2 = np.load("data/w2.npy")
    b2 = np.load("data/b2.npy")
    return w1, b1, w2, b2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
