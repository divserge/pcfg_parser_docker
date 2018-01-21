import numpy as np
import scipy.io as sio
from tt import tensor
from copy import deepcopy
import time

class BaseTensor:
    """
    A base class for multilinear function
    """
    def __init__(self, data, rank = None):
        """
        Args:
            data: np.ndarray : the underlying multi-dimensional array
            rank: 
        Returns:
            BaseTensor object
        """
        self.data = data
        self.shape = data.shape

    def dot(self, vectors, modes):
        """
        Args:
            vectors: list of two np.ndarrat vectors - the vectors to map
            modes: list of two integers - modes to contract
        Returns:
            np.ndarray - the result of the mapping
        """
        return np.einsum(self.data, [0, 1, 2], vectors[0], [modes[0]], vectors[1], [modes[1]])
        

def TensorTrain(data, rank=6):
    """
    Performs a tensor train decomposition with a given fixed maximal rank and returns the factors as a list
    """
    return tensor.to_list(tensor(data, rmax=rank))