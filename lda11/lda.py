from gc import collect
from numbers import Number

import numpy as np
from scipy import sparse as sps
from scipy.special import digamma
from tqdm import tqdm

from ._lda import (
    LDATrainer,
    Predictor,
    learn_dirichlet,
    learn_dirichlet_symmetric,
    log_likelihood_doc_topic,
)

RealType = np.float64
IntegerType = np.int32
IndexType = np.uint64


def number_to_array(n_components, default, arg=None, ensure_symmetry=False):
    if arg is None:
        arg = default
    if isinstance(arg, Number):
        return np.ones(n_components, dtype=RealType) * RealType(arg)
    elif isinstance(arg, np.ndarray):
        assert arg.shape[0] == n_components
        if ensure_symmetry and np.unique(arg).shape[0] > 1:
            raise ValueError("Symmetric array required.")
        return arg.astype(RealType)
    return None


def check_array(X):
    if isinstance(X, np.ndarray):
        assert len(X.shape) == 2
        assert X.dtype == np.int32 or X.dtype == np.int64

        dix, wix = X.nonzero()
        counts = X[dix, wix]
    elif sps.issparse(X):
        # if X is either types of, scipy.sparse X has this attribute.
        X = X.tocsr()
        X.sort_indices()
        dix, wix = X.nonzero()
        counts = X.data
    else:
        raise ValueError("The input must be either np.ndarray or sparse array.")
    return counts.astype(IntegerType), dix.astype(IndexType), wix.astype(IndexType)


def bow_row_to_counts(X, i):
    if isinstance(X, np.ndarray):
        assert len(X.shape) == 2
        assert X.dtype == np.int32 or X.dtype == np.int64
        (wix,) = X[i].nonzero()
        counts = X[i, wix]
    else:
        _, wix = X[i].nonzero()
        counts = X[i, wix].toarray().ravel()

    return counts.astype(IntegerType), wix.astype(IndexType)


def to_sparse(X):
    result = sps.csr_matrix(X)
    result.data = result.data.astype(IntegerType)
    return result


class LDAPredictorMixin:
    """
    self.components_
    self.n_components
    self.predictor
    are needed
    """

    def transform(
        self,
        *Xs,
        n_iter=100,
        random_seed=42,
        mode="gibbs",
        mf_tolerance=1e-10,
        gibbs_burn_in=10,
        use_cgs_p=True,
        n_workers=1
    ):
        shapes = set({X.shape[0] for X in Xs})
        if len(shapes) != 1:
            raise ValueError("Got different shape for Xs.")
        shape = list(shapes)[0]

        Xs_csr = []
        for i, X in enumerate(Xs):
            if X is None:
                Xs_csr.append(
                    sps.csr_matrix(
                        ([], ([], [])),
                        shape=(shape, self.topic_word_priors[i].shape[0]),
                    )
                )
            else:
                Xs_csr.append(to_sparse(X))

        if mode == "gibbs":
            return self.predictor.predict_gibbs_batch(
                Xs_csr, n_iter, gibbs_burn_in, random_seed, use_cgs_p, n_workers
            )
        else:
            return self.predictor.predict_mf_batch(
                Xs_csr, n_iter, mf_tolerance, n_workers
            )

    def word_topic_assignment(
        self, *Xs, n_iter=100, random_seed=42, gibbs_burn_in=10, use_cgs_p=True
    ):
        n_domains = len(Xs)
        shapes = set({X.shape[0] for X in Xs})
        if len(shapes) != 1:
            raise ValueError("Got different shape for Xs.")

        shape = list(shapes)[0]
        Xs_csr = []
        for i, X in enumerate(Xs):
            if X is None:
                Xs_csr.append(
                    sps.csr_matrix(
                        ([], ([], [])),
                        shape=(shape, self.topic_word_priors[i].shape[0]),
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
    def phis(self):
        return self.predictor.phis


class MultipleContextLDA(LDAPredictorMixin):
    def __init__(
        self,
        n_components=100,
        doc_topic_prior=None,
        topic_word_priors=None,
        n_iter=1000,
        optimize_interval=None,
        optimize_burn_in=None,
        n_workers=1,
        use_cgs_p=True,
        is_phi_symmetric=True,
    ):
        n_components = int(n_components)
        assert n_iter >= 1
        assert n_components >= 1
        self.n_components = n_components

        self.doc_topic_prior = doc_topic_prior
        self.topic_word_priors = topic_word_priors
        self.is_phi_symmetric = is_phi_symmetric
        self.n_vocabs = None
        self.docstate_ = None
        self.components_ = None
        self.n_modals = None

        self.predictor = None
        self.use_cgs_p = use_cgs_p

        self.n_iter = n_iter
        self.optimize_interval = optimize_interval
        if optimize_interval is not None:
            if optimize_burn_in is None:
                optimize_burn_in = n_iter // 2
            else:
                optimize_burn_in = optimize_burn_in
        self.optimize_burn_in = optimize_burn_in

        self.n_workers = n_workers

    def fit(self, *X, **kwargs):
        self._fit(*X, **kwargs)
        return self

    def _fit(self, *Xs, ll_freq=10):
        """
        Xs should be a list of contents.
        All entries must have the same shape[0].
        """
        n_vocabs = []

        self.modality = len(Xs)
        self.doc_topic_prior = number_to_array(
            self.n_components, 1 / float(self.n_components), self.doc_topic_prior
        )

        if self.topic_word_priors is None:
            self.topic_word_priors = [None for i in range(self.modality)]

        self.topic_word_priors = [
            number_to_array(
                X.shape[1],
                1 / float(self.n_components),
                ensure_symmetry=self.is_phi_symmetric,
            )
            for X, val in zip(Xs, self.topic_word_priors)
        ]

        doc_tuples = []
        for X in Xs:
            doc_tuples.append((check_array(X)))

        doc_topic = np.zeros((X.shape[0], self.n_components), dtype=IntegerType)

        topic_counts = np.zeros(self.n_components, dtype=IntegerType)

        word_topics = [
            np.zeros((X.shape[1], self.n_components), dtype=IntegerType) for X in Xs
        ]

        docstates = []
        for (count, dix, wix), word_topic in zip(doc_tuples, word_topics):
            docstate = LDATrainer(
                self.doc_topic_prior,
                count,
                dix,
                wix,
                self.n_components,
                42,
                self.n_workers,
            )
            docstates.append(docstate)
            docstate.initialize(word_topic, doc_topic, topic_counts)
        doc_length = doc_topic.sum(axis=1).astype(IntegerType)

        ll = log_likelihood_doc_topic(self.doc_topic_prior, doc_topic, doc_length)
        for topic_word_prior, word_topic, docstate in zip(
            self.topic_word_priors, word_topics, docstates
        ):
            ll += docstate.log_likelihood(topic_word_prior, word_topic)

        with tqdm(range(self.n_iter)) as pbar:
            pbar.set_description("Log Likelihood = {0:.2f}".format(ll))
            for i in pbar:
                for topic_word_prior, word_topic, docstate in zip(
                    self.topic_word_priors, word_topics, docstates
                ):
                    docstate.iterate_gibbs(
                        topic_word_prior, doc_topic, word_topic, topic_counts
                    )
                if (i + 1) % ll_freq == 0:
                    ll = log_likelihood_doc_topic(
                        self.doc_topic_prior, doc_topic, doc_length
                    )

                    for topic_word_prior, word_topic, docstate in zip(
                        self.topic_word_priors, word_topics, docstates
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
                        self.topic_word_priors, word_topics, docstates
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

        predictor = Predictor(self.n_components, self.doc_topic_prior, 42)

        for i, (twp, wt, docstate) in enumerate(
            zip(self.topic_word_priors, word_topics, docstates)
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


class LDA(MultipleContextLDA):
    pass

    def __init__(
        self,
        n_components=100,
        doc_topic_prior=None,
        topic_word_prior=None,
        n_iter=1000,
        optimize_burn_in=None,
        optimize_interval=None,
        n_workers=1,
        use_cgs_p=True,
        is_phi_symmetric=True,
    ):
        if topic_word_prior is not None:
            topic_word_priors = [topic_word_prior]
        else:
            topic_word_priors = None

        super(LDA, self).__init__(
            n_components=n_components,
            doc_topic_prior=doc_topic_prior,
            topic_word_priors=topic_word_priors,
            n_iter=n_iter,
            optimize_burn_in=optimize_burn_in,
            optimize_interval=optimize_interval,
            n_workers=n_workers,
            use_cgs_p=use_cgs_p,
            is_phi_symmetric=is_phi_symmetric,
        )

    def fit(self, X, **kwargs):
        super(LDA, self).fit(X, **kwargs)
        return self

    @property
    def phi(self):
        return self.predictor.phis[0]
