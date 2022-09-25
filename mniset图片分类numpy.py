import numpy as np
import matplotlib.pyplot as plt
import pickle
import traceback
import  activation_function as f

class MLP():
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [np.random.randn()]