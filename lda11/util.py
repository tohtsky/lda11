import numpy as np
from scipy import sparse as sps

from ._lda import train_test_split
from .lda import IntegerType, RealType


def rowwise_train_test_split(X, random_seed=None, test_ratio=0.5):
    """
    split matrix randomly
    """
    if random_seed is None:
        random_seed = np.random.randint(-(2 ** 63), 2 ** 63 - 1)
    X = sps.csr_matrix(X, dtype=IntegerType)
    return train_test_split(X, test_ratio, random_seed)
