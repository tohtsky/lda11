m: int
n: int
from numpy import float32

"""Backend C++ inplementation for lda11."""
from __future__ import annotations
import lda11._lda
import typing
import numpy
import scipy.sparse

_Shape = typing.Tuple[int, ...]

__all__ = [
    "LDATrainer",
    "LabelledLDATrainer",
    "Predictor",
    "learn_dirichlet",
    "learn_dirichlet_symmetric",
    "log_likelihood_doc_topic",
    "train_test_split",
]

class LDATrainer:
    def __init__(
        self,
        arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, 1]],
        arg2: numpy.ndarray[numpy.uint64, _Shape[m, 1]],
        arg3: numpy.ndarray[numpy.uint64, _Shape[m, 1]],
        arg4: int,
        arg5: int,
        arg6: int,
    ) -> None: ...
    def initialize(
        self,
        arg0: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg2: numpy.ndarray[numpy.int32, _Shape[m, 1]],
    ) -> None: ...
    def iterate_gibbs(
        self,
        arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg2: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg3: numpy.ndarray[numpy.int32, _Shape[m, 1]],
    ) -> None: ...
    def log_likelihood(
        self,
        arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
    ) -> float: ...
    def obtain_phi(
        self,
        arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg2: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg3: numpy.ndarray[numpy.int32, _Shape[m, 1]],
    ) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def set_doc_topic_prior(
        self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]
    ) -> None: ...
    pass

class LabelledLDATrainer:
    def __init__(
        self,
        arg0: float,
        arg1: float,
        arg2: scipy.sparse.csr_matrix[numpy.int32],
        arg3: numpy.ndarray[numpy.int32, _Shape[m, 1]],
        arg4: numpy.ndarray[numpy.uint64, _Shape[m, 1]],
        arg5: numpy.ndarray[numpy.uint64, _Shape[m, 1]],
        arg6: int,
        arg7: int,
        arg8: int,
    ) -> None: ...
    def initialize(
        self,
        arg0: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg2: numpy.ndarray[numpy.int32, _Shape[m, 1]],
    ) -> None: ...
    def iterate_gibbs(
        self,
        arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg2: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg3: numpy.ndarray[numpy.int32, _Shape[m, 1]],
    ) -> None: ...
    def log_likelihood(
        self,
        arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
    ) -> float: ...
    def obtain_phi(
        self,
        arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]],
        arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg2: numpy.ndarray[numpy.int32, _Shape[m, n]],
        arg3: numpy.ndarray[numpy.int32, _Shape[m, 1]],
    ) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    pass

class Predictor:
    def __getstate__(self) -> tuple: ...
    def __init__(
        self, arg0: int, arg1: numpy.ndarray[numpy.float64, _Shape[m, 1]], arg2: int
    ) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def add_beta(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def predict_gibbs(
        self,
        arg0: typing.List[numpy.ndarray[numpy.int32, _Shape[m, 1]]],
        arg1: typing.List[numpy.ndarray[numpy.int32, _Shape[m, 1]]],
        arg2: int,
        arg3: int,
        arg4: int,
        arg5: bool,
    ) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def predict_gibbs_batch(
        self,
        arg0: typing.List[scipy.sparse.csr_matrix[numpy.int32]],
        arg1: int,
        arg2: int,
        arg3: int,
        arg4: bool,
        arg5: int,
    ) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def predict_gibbs_with_word_assignment(
        self,
        arg0: typing.List[numpy.ndarray[numpy.int32, _Shape[m, 1]]],
        arg1: typing.List[numpy.ndarray[numpy.int32, _Shape[m, 1]]],
        arg2: int,
        arg3: int,
        arg4: int,
        arg5: bool,
    ) -> typing.Tuple[
        numpy.ndarray[numpy.float64, _Shape[m, 1]],
        typing.List[typing.Dict[int, numpy.ndarray[numpy.int32, _Shape[m, 1]]]],
    ]: ...
    def predict_mf(
        self,
        arg0: typing.List[numpy.ndarray[numpy.int32, _Shape[m, 1]]],
        arg1: typing.List[numpy.ndarray[numpy.int32, _Shape[m, 1]]],
        arg2: int,
        arg3: float,
    ) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def predict_mf_batch(
        self,
        arg0: typing.List[scipy.sparse.csr_matrix[numpy.int32]],
        arg1: int,
        arg2: float,
        arg3: int,
    ) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    @property
    def phis(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]
        """
    pass

def learn_dirichlet(
    arg0: numpy.ndarray[numpy.int32, _Shape[m, n]],
    arg1: numpy.ndarray[numpy.float64, _Shape[m, 1]],
    arg2: float,
    arg3: float,
    arg4: int,
) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
    pass

def learn_dirichlet_symmetric(
    arg0: numpy.ndarray[numpy.int32, _Shape[m, n]],
    arg1: float,
    arg2: float,
    arg3: float,
    arg4: int,
) -> float:
    pass

def log_likelihood_doc_topic(
    arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]],
    arg1: numpy.ndarray[numpy.int32, _Shape[m, n]],
    arg2: numpy.ndarray[numpy.int32, _Shape[m, 1]],
) -> float:
    pass

def train_test_split(
    arg0: scipy.sparse.csr_matrix[numpy.int32], arg1: float, arg2: int
) -> typing.Tuple[
    scipy.sparse.csr_matrix[numpy.int32], scipy.sparse.csr_matrix[numpy.int32]
]:
    pass
