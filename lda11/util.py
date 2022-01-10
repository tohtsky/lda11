from typing import Optional, Tuple

import numpy as np
from scipy import sparse as sps

from ._lda import train_test_split
from .lda import IntegerType, RealType, ValidXType


def rowwise_train_test_split(
    X: ValidXType, random_seed: Optional[int] = None, test_ratio: float = 0.5
) -> Tuple[sps.csr_matrix, sps.csr_matrix]:
    """
    split matrix randomly
    """
    if random_seed is None:
        random_seed = np.random.randint(-(2 ** 31), 2 ** 31 - 1)
    X = sps.csr_matrix(X, dtype=IntegerType)
    return train_test_split(X, test_ratio, random_seed)
