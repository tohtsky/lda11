import random
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy import typing as npt
from scipy import sparse as sps
from tqdm import tqdm
from typing_extensions import Literal

from ._lda import LDATrainer
from ._lda import Predictor as CorePredictor
from ._lda import learn_dirichlet, learn_dirichlet_symmetric, log_likelihood_doc_topic

RealType = np.float64

IntegerType = np.int32
IndexType = np.uint64


ValidXType = Union[sps.spmatrix, npt.NDArray[np.int32], npt.NDArray[np.int64]]
PriorType = Union[np.ndarray, float, None]


class LDAInput(NamedTuple):
    counts: np.ndarray
    dix: np.ndarray
    wis: np.ndarray


def number_to_array(
    n_components: int,
    default: float,
    arg_: Union[float, None, np.ndarray] = None,
    ensure_symmetry: bool = False,
) -> npt.NDArray[np.float64]:
    if arg_ is None or isinstance(arg_, float):
        value_ = default if arg_ is None else float(arg_)
        return np.ones(n_components, dtype=RealType) * value_
    if isinstance(arg_, np.ndarray):
        assert arg_.shape[0] == n_components
        if ensure_symmetry and np.unique(arg_).shape[0] > 1:
            raise ValueError("Symmetric array required.")
        return arg_.astype(RealType)
    raise ValueError("Number of ndarray is required.")


def check_array(X: ValidXType) -> LDAInput:
    assert X.dtype == np.int32 or X.dtype == np.int64
    if isinstance(X, np.ndarray):
        assert len(X.shape) == 2
        dix, wix = X.nonzero()
        counts: np.ndarray = X[dix, wix]
    elif sps.issparse(X):
        # if X is either types of, scipy.sparse X has this attribute.
        X = sps.csr_matrix(X)
        X.sort_indices()
        dix, wix = X.nonzero()
        counts = X.data.astype(np.int32)
    else:
        raise ValueError("The input must be either np.ndarray or sparse array.")
    return LDAInput(
        counts.astype(IntegerType), dix.astype(IndexType), wix.astype(IndexType)
    )


def bow_row_to_counts(X: ValidXType, i: int) -> Tuple[np.ndarray, np.ndarray]:
    wix: np.ndarray
    if isinstance(X, np.ndarray):
        assert len(X.shape) == 2
        assert X.dtype == np.int32 or X.dtype == np.int64
        (wix,) = X[i].nonzero()
        counts: np.ndarray = X[i, wix]
    else:
        _, wix = X[i].nonzero()
        counts = X[i, wix].toarray().ravel()

    return counts.astype(IntegerType), wix.astype(IndexType)


def to_valid_csr(X: ValidXType) -> sps.csr_matrix:
    result = sps.csr_matrix(X)
    result.data = result.data.astype(IntegerType)
    return result


class LDAPredictorMixin:
    topic_word_priors_: Optional[List[np.ndarray]]
    predictor: Optional[CorePredictor]

    def transform(
        self,
        *Xs: Union[ValidXType, None],
        n_iter: int = 100,
        random_seed: int = 0,
        mode: Literal["gibbs", "mf"] = "gibbs",
        mf_tolerance: float = 1e-10,
        gibbs_burn_in: int = 10,
        use_cgs_p: bool = True,
        n_workers: int = 1
    ) -> npt.NDArray[RealType]:
        assert self.topic_word_priors_ is not None
        assert self.predictor is not None
        shapes = set({int(X.shape[0]) for X in Xs if X is not None})
        if len(shapes) != 1:
            raise ValueError("Got different shape for Xs.")
        shape = list(shapes)[0]

        Xs_csr: List[sps.csr_matrix] = []
        for i, X in enumerate(Xs):
            if X is None:
                Xs_csr.append(
                    sps.csr_matrix(
                        (shape, self.topic_word_priors_[i].shape[0]),
                        dtype=IntegerType,
                    )
                )
            else:
                Xs_csr.append(to_valid_csr(X))

        if mode == "gibbs":
            return self.predictor.predict_gibbs_batch(
                Xs_csr, n_iter, gibbs_burn_in, random_seed, use_cgs_p, n_workers
            )
        elif mode == "mf":
            return self.predictor.predict_mf_batch(
                Xs_csr, n_iter, mf_tolerance, n_workers
            )
        else:
            raise ValueError('"mode" argument must be either "gibbs" for "mf".')

    def word_topic_assignment(
        self,
        *Xs: Union[ValidXType, None],
        n_iter: int = 100,
        random_seed: int = 0,
        gibbs_burn_in: int = 10,
        use_cgs_p: bool = True
    ) -> List[Tuple[np.ndarray, List[Dict[int, np.ndarray]]]]:
        assert self.topic_word_priors_ is not None
        assert self.predictor is not None
        n_domains = len(Xs)
        shapes = set({X.shape[0] for X in Xs if X is not None})
        if len(shapes) != 1:
            raise ValueError("Got different shape for Xs.")

        shape = list(shapes)[0]
        Xs_csr = []
        for i, X in enumerate(Xs):
            if X is None:
                Xs_csr.append(
                    sps.csr_matrix(
                        (shape, self.topic_word_priors_[i].shape[0]), dtype=IntegerType
                    )
                )
        results = []
        for i in range(shape):
            counts = []
            wixs = []
            for n in range(n_domains):
                count, wix = bow_row_to_counts(Xs[n], i)
                counts.append(count)
                wixs.append(wix)

            results.append(
                self.predictor.predict_gibbs_with_word_assignment(
                    wixs, counts, n_iter, gibbs_burn_in, random_seed, use_cgs_p
                )
            )
        return results

    @property
    def phis(self) -> List[npt.NDArray[RealType]]:
        assert self.predictor is not None
        return self.predictor.phis


class LDABase(LDAPredictorMixin):
    def __init__(
        self,
        n_components: int = 100,
        doc_topic_prior: PriorType = None,
        n_iter: int = 1000,
        optimize_interval: Optional[int] = None,
        optimize_burn_in: Optional[int] = None,
        n_workers: int = 1,
        use_cgs_p: bool = True,
        is_phi_symmetric: bool = True,
        random_seed: Optional[int] = 0,
    ):
        n_components = int(n_components)
        assert n_iter >= 1
        assert n_components >= 1
        self.n_components = n_components

        self.doc_topic_prior = number_to_array(
            self.n_components, 1 / float(self.n_components), doc_topic_prior
        )
        self.topic_word_priors_ = None
        self.is_phi_symmetric = is_phi_symmetric
        self.n_vocabs: Optional[List[int]] = None
        self.docstate_ = None
        self.components_: Optional[int] = None
        self.n_modals: Optional[int] = None

        self.predictor: Optional[CorePredictor] = None
        self.use_cgs_p: bool = use_cgs_p

        self.n_iter = n_iter
        self.optimize_interval = optimize_interval
        if optimize_interval is not None:
            if optimize_burn_in is None:
                optimize_burn_in = n_iter // 2
            else:
                optimize_burn_in = optimize_burn_in
        self.optimize_burn_in = optimize_burn_in

        self.n_workers = n_workers
        if random_seed is None:
            random_seed = random.randint(-(2 ** 31), 2 ** 31 - 1)
        self.random_seed = random_seed

    def _fit(self, *Xs: ValidXType, ll_freq: int = 10) -> npt.NDArray[IntegerType]:
        """
        Xs should be a list of contents.
        All entries must have the same shape[0].
        """

        self.modality = len(Xs)

        topic_word_priors_canonical: List[npt.NDArray[RealType]] = []

        doc_tuples: List[LDAInput] = []

        n_rows: Optional[int] = None
        for X in Xs:
            doc_tuples.append(check_array(X))
            if n_rows is None:
                n_rows = X.shape[0]
            else:
                assert n_rows == X.shape[0]
            topic_word_priors_canonical.append(
                number_to_array(
                    X.shape[1],
                    1 / float(self.n_components),
                    ensure_symmetry=True,
                )
            )
        if n_rows is None:
            raise ValueError("At least one doc-term matrix must be given.")

        doc_topic: npt.NDArray[IntegerType] = np.zeros(
            (n_rows, self.n_components), dtype=IntegerType
        )

        topic_counts: npt.NDArray[IntegerType] = np.zeros(
            self.n_components, dtype=IntegerType
        )

        word_topics: List[npt.NDArray[IntegerType]] = [
            np.zeros((X.shape[1], self.n_components), dtype=IntegerType) for X in Xs
        ]

        docstates: List[LDATrainer] = []
        for (count, dix, wix), word_topic in zip(doc_tuples, word_topics):
            docstate = LDATrainer(
                self.doc_topic_prior,
                count,
                dix,
                wix,
                self.n_components,
                self.random_seed,
                self.n_workers,
            )
            docstates.append(docstate)
            docstate.initialize(word_topic, doc_topic, topic_counts)
        doc_length: npt.NDArray[IntegerType] = doc_topic.sum(axis=1).astype(IntegerType)

        ll = log_likelihood_doc_topic(self.doc_topic_prior, doc_topic, doc_length)
        for topic_word_prior, word_topic, docstate in zip(
            topic_word_priors_canonical, word_topics, docstates
        ):
            ll += docstate.log_likelihood(topic_word_prior, word_topic)

        with tqdm(range(self.n_iter)) as pbar:
            pbar.set_description("Log Likelihood = {0:.2f}".format(ll))
            for i in pbar:
                for topic_word_prior, word_topic, docstate in zip(
                    topic_word_priors_canonical, word_topics, docstates
                ):
                    docstate.iterate_gibbs(
                        topic_word_prior, doc_topic, word_topic, topic_counts
                    )
                if (i + 1) % ll_freq == 0:
                    ll = log_likelihood_doc_topic(
                        self.doc_topic_prior, doc_topic, doc_length
                    )

                    for topic_word_prior, word_topic, docstate in zip(
                        topic_word_priors_canonical, word_topics, docstates
                    ):
                        ll += docstate.log_likelihood(topic_word_prior, word_topic)
                    pbar.set_description("Log Likelihood = {0:.2f}".format(ll))

                if (
                    (self.optimize_interval is not None)
                    and (i >= self.optimize_burn_in)
                    and (i % self.optimize_interval == 0)
                ):
                    doc_topic_prior_new = learn_dirichlet(
                        doc_topic,
                        self.doc_topic_prior,
                        0.1,
                        1 / float(self.n_components),
                        100,
                    )
                    for topic_word_prior, word_topic, docstate in zip(
                        topic_word_priors_canonical, word_topics, docstates
                    ):
                        if self.is_phi_symmetric:
                            topic_word_prior_new = np.ones_like(
                                topic_word_prior
                            ) * learn_dirichlet_symmetric(
                                word_topic.transpose().copy(),
                                topic_word_prior.mean(),
                                0.1,
                                1 / float(self.n_components),
                                100,
                            )
                        else:
                            topic_word_prior_new = learn_dirichlet(
                                word_topic.transpose().copy(),
                                topic_word_prior,
                                0.1,
                                1 / float(self.n_components),
                                100,
                            )
                        topic_word_prior[:] = topic_word_prior_new
                        self.doc_topic_prior = doc_topic_prior_new
                        docstate.set_doc_topic_prior(doc_topic_prior_new)
        self.topic_word_priors_ = topic_word_priors_canonical

        predictor = CorePredictor(
            self.n_components, self.doc_topic_prior, self.random_seed
        )

        for i, (twp, wt, docstate) in enumerate(
            zip(self.topic_word_priors_, word_topics, docstates)
        ):
            if self.use_cgs_p:
                phi = docstate.obtain_phi(twp, doc_topic, wt, topic_counts)
            else:
                phi = wt + twp[:, np.newaxis]
                phi /= phi.sum(axis=0)[np.newaxis, :]
                phi = phi.transpose()
            predictor.add_beta(phi.transpose())

        self.predictor = predictor

        return doc_topic


class MultilingualLDA(LDABase):
    def fit(self, *X: ValidXType, ll_freq: int = 10) -> "MultilingualLDA":
        self._fit(*X, ll_freq=ll_freq)
        return self


class LDA(LDABase):
    def fit(self, X: ValidXType, ll_freq: int = 10) -> "LDA":
        self._fit(X, ll_freq=ll_freq)
        return self

    @property
    def phi(self) -> np.ndarray:
        assert self.predictor is not None
        return self.predictor.phis[0]
