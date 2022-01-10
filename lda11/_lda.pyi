"""Backend C++ inplementation for lda11."""
from __future__ import annotations
import lda11._lda
import typing
import numpy
import numpy.typing as npt
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
        arg0: npt.NDArray[numpy.float64],
        arg1: npt.NDArray[numpy.int32],
        arg2: npt.NDArray[numpy.uint64],
        arg3: npt.NDArray[numpy.uint64],
        arg4: int,
        arg5: int,
        arg6: int,
    ) -> None: ...
    def initialize(
        self,
        arg0: npt.NDArray[numpy.int32],
        arg1: npt.NDArray[numpy.int32],
        arg2: npt.NDArray[numpy.int32],
    ) -> None: ...
    def iterate_gibbs(
        self,
        arg0: npt.NDArray[numpy.float64],
        arg1: npt.NDArray[numpy.int32],
        arg2: npt.NDArray[numpy.int32],
        arg3: npt.NDArray[numpy.int32],
    ) -> None: ...
    def log_likelihood(
        self,
        arg0: npt.NDArray[numpy.float64],
        arg1: npt.NDArray[numpy.int32],
    ) -> float: ...
    def obtain_phi(
        self,
        arg0: npt.NDArray[numpy.float64],
        arg1: npt.NDArray[numpy.int32],
        arg2: npt.NDArray[numpy.int32],
        arg3: npt.NDArray[numpy.int32],
    ) -> npt.NDArray[numpy.float64]: ...
    def set_doc_topic_prior(self, arg0: npt.NDArray[numpy.float64]) -> None: ...
    pass

class LabelledLDATrainer:
    def __init__(
        self,
        arg0: float,
        arg1: float,
        arg2: scipy.sparse.csr_matrix[numpy.int32],
        arg3: npt.NDArray[numpy.int32],
        arg4: npt.NDArray[numpy.uint64],
        arg5: npt.NDArray[numpy.uint64],
        arg6: int,
        arg7: int,
        arg8: int,
    ) -> None: ...
    def initialize(
        self,
        arg0: npt.NDArray[numpy.int32],
        arg1: npt.NDArray[numpy.int32],
        arg2: npt.NDArray[numpy.int32],
    ) -> None: ...
    def iterate_gibbs(
        self,
        arg0: npt.NDArray[numpy.float64],
        arg1: npt.NDArray[numpy.int32],
        arg2: npt.NDArray[numpy.int32],
        arg3: npt.NDArray[numpy.int32],
    ) -> None: ...
    def log_likelihood(
        self,
        arg0: npt.NDArray[numpy.float64],
        arg1: npt.NDArray[numpy.int32],
    ) -> float: ...
    def obtain_phi(
        self,
        arg0: npt.NDArray[numpy.float64],
        arg1: npt.NDArray[numpy.int32],
        arg2: npt.NDArray[numpy.int32],
        arg3: npt.NDArray[numpy.int32],
    ) -> npt.NDArray[numpy.float64]: ...
    pass

class Predictor:
    def __getstate__(self) -> tuple: ...
    def __init__(
        self, arg0: int, arg1: npt.NDArray[numpy.float64], arg2: int
    ) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def add_beta(self, arg0: npt.NDArray[numpy.float64]) -> None: ...
    def predict_gibbs(
        self,
        arg0: typing.List[npt.NDArray[numpy.int32]],
        arg1: typing.List[npt.NDArray[numpy.int32]],
        arg2: int,
        arg3: int,
        arg4: int,
        arg5: bool,
    ) -> npt.NDArray[numpy.float64]: ...
    def predict_gibbs_batch(
        self,
        arg0: typing.List[scipy.sparse.csr_matrix[numpy.int32]],
        arg1: int,
        arg2: int,
        arg3: int,
        arg4: bool,
        arg5: int,
    ) -> npt.NDArray[numpy.float64]: ...
    def predict_gibbs_with_word_assignment(
        self,
        arg0: typing.List[npt.NDArray[numpy.int32]],
        arg1: typing.List[npt.NDArray[numpy.int32]],
        arg2: int,
        arg3: int,
        arg4: int,
        arg5: bool,
    ) -> typing.Tuple[
        npt.NDArray[numpy.float64],
        typing.List[typing.Dict[int, npt.NDArray[numpy.int32]]],
    ]: ...
    def predict_mf_batch(
        self,
        arg0: typing.List[scipy.sparse.csr_matrix[numpy.int32]],
        arg1: int,
        arg2: float,
        arg3: int,
    ) -> npt.NDArray[numpy.float64]: ...
    @property
    def phis(self) -> typing.List[npt.NDArray[numpy.float64]]:
        """
        :type: typing.List[npt.NDArray[numpy.float64]]
        """
    pass

def learn_dirichlet(
    arg0: npt.NDArray[numpy.int32],
    arg1: npt.NDArray[numpy.float64],
    arg2: float,
    arg3: float,
    arg4: int,
) -> npt.NDArray[numpy.float64]:
    pass

def learn_dirichlet_symmetric(
    arg0: npt.NDArray[numpy.int32],
    arg1: float,
    arg2: float,
    arg3: float,
    arg4: int,
) -> float:
    pass

def log_likelihood_doc_topic(
    arg0: npt.NDArray[numpy.float64],
    arg1: npt.NDArray[numpy.int32],
    arg2: npt.NDArray[numpy.int32],
) -> float:
    pass

def train_test_split(
    arg0: scipy.sparse.csr_matrix[numpy.int32], arg1: float, arg2: int
) -> typing.Tuple[
    scipy.sparse.csr_matrix[numpy.int32], scipy.sparse.csr_matrix[numpy.int32]
]:
    pass
