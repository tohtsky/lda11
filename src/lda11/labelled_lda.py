from typing import Optional

import numpy as np
from numpy import typing as npt
from scipy import sparse as sps
from tqdm import tqdm

from ._lda import LabelledLDATrainer
from ._lda import Predictor as CorePredictor
from .lda import (
    IntegerType,
    LDAPredictorMixin,
    RealType,
    ValidXType,
    check_array,
    number_to_array,
)


class LabelledLDA(LDAPredictorMixin):
    def __init__(
        self,
        alpha: float = 1e-2,
        epsilon: float = 1e-30,
        n_iter: int = 1000,
        n_workers: int = 1,
        use_cgs_p: bool = True,
    ):
        self.n_components_: Optional[int] = None
        self.alpha = alpha
        self.epsilon = 1e-20
        self.n_vocabs = None
        self.docstate_ = None
        self.components_: Optional[npt.NDArray[np.int32]] = None
        self.predictor = None
        self.n_workers = n_workers
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.use_cgs_p = use_cgs_p

    def fit(self, X: ValidXType, Y: ValidXType) -> "LabelledLDA":
        self._fit_llda(X, Y)
        return self

    def _fit_llda(
        self,
        X: ValidXType,
        Y: ValidXType,
    ) -> npt.NDArray[np.int32]:
        if not sps.issparse(Y):
            Y = sps.csr_matrix(Y).astype(IntegerType)
        else:
            Y = Y.astype(IntegerType)

        self.n_components = int(Y.shape[1])
        self.topic_word_priors_ = [
            number_to_array(X.shape[1], 1 / float(self.n_components), None)
        ]

        try:
            count, dix, wix = check_array(X)
        except:
            print("Check for X failed.")
            raise

        doc_topic = np.zeros((X.shape[0], self.n_components), dtype=IntegerType)
        word_topic = np.zeros((X.shape[1], self.n_components), dtype=IntegerType)
        topic_counts = np.zeros(self.n_components, dtype=IntegerType)

        docstate = LabelledLDATrainer(
            self.alpha,
            self.epsilon,
            Y,
            count,
            dix,
            wix,
            self.n_components,
            42,
            self.n_workers,
        )
        docstate.initialize(word_topic, doc_topic, topic_counts)

        with tqdm(range(self.n_iter)) as pbar:
            for _ in pbar:
                docstate.iterate_gibbs(
                    self.topic_word_priors_[0], doc_topic, word_topic, topic_counts
                )

        doc_topic_prior = self.alpha * np.ones(self.n_components, dtype=RealType)

        self.components_ = word_topic.transpose()
        predictor = CorePredictor(self.n_components, doc_topic_prior, 42)
        if self.use_cgs_p:
            phi = docstate.obtain_phi(
                self.topic_word_priors_[0], doc_topic, word_topic, topic_counts
            )
        else:
            phi = word_topic + self.topic_word_priors_[0][:, np.newaxis]
            phi /= phi.sum(axis=0)[np.newaxis, :]
            phi = phi.transpose()
        predictor.add_beta(phi.transpose())
        self.predictor = predictor

        return doc_topic

    @property
    def phi(self) -> npt.NDArray[np.float64]:
        assert self.predictor is not None
        return self.predictor.phis[0]
