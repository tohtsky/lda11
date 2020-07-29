import numpy as np
from scipy import sparse as sps
from tqdm import tqdm
from ._lda import LabelledLDATrainer
from .lda import (
    Predictor,
    RealType, IntegerType, IndexType,
    number_to_array, check_array,
    LDAPredictorMixin
)


class LabelledLDA(LDAPredictorMixin):
    def __init__(self,
                 alpha=1e-2, epsilon=1e-30, topic_word_prior=None, add_dummy_topic=False,
                 n_iter=1000,
                 n_workers=1,
                 use_cgs_p=True
                 ):
        self.n_components = None
        self.topic_word_prior = topic_word_prior
        self.alpha = alpha
        self.epsilon = 1e-20
        self.n_vocabs = None
        self.docstate_ = None
        self.components_ = None
        self.predictor = None
        self.n_workers = n_workers
        self.epsilon = epsilon
        self.add_dummy_topic = add_dummy_topic
        self.n_iter = n_iter
        self.use_cgs_p = use_cgs_p

    def fit(self, X, Y):
        self._fit(X, Y)
        return self

    def fit_transform(self, X, Y, **kwargs):
        result = self._fit(X, **kwargs) + self.doc_topic_prior[np.newaxis, :]
        result /= result.sum(axis=1)[:, np.newaxis]
        return result

    def _fit(self, X, Y, ll_freq=10):
        if not sps.issparse(Y):
            Y = sps.csr_matrix(Y).astype(IntegerType)
        else:
            Y = Y.astype(IntegerType)

        self.n_components = Y.shape[1]
        ones_topic = np.ones(self.n_components, dtype=RealType)
        self.topic_word_prior = number_to_array(
            X.shape[1], 1 / float(self.n_components),
            self.topic_word_prior
        )

        try:
            count, dix, wix = check_array(X)
        except:
            print('Check for X failed.')
            raise

        doc_topic = np.zeros(
            (X.shape[0], self.n_components), dtype=IntegerType)
        word_topic = np.zeros(
            (X.shape[1], self.n_components), dtype=IntegerType)
        topic_counts = np.zeros(self.n_components, dtype=IntegerType)

        docstate = LabelledLDATrainer(
            self.alpha,
            self.epsilon,
            Y,
            count, dix, wix, self.n_components, 42,
            self.n_workers
        )
        docstate.initialize(word_topic, doc_topic, topic_counts)

        with tqdm(range(self.n_iter)) as pbar:
            for _ in pbar:
                docstate.iterate_gibbs(
                    self.topic_word_prior,
                    doc_topic,
                    word_topic,
                    topic_counts
                )

        doc_topic_prior = (
            self.alpha * np.ones(self.n_components, dtype=RealType))

        self.components_ = word_topic.transpose()
        predictor = Predictor(self.n_components, doc_topic_prior, 42)
        if self.use_cgs_p:
            phi = docstate.obtain_phi(
                self.topic_word_prior,
                doc_topic,
                word_topic,
                topic_counts
            )
        else:
            phi = word_topic + self.topic_word_prior[:, np.newaxis]
            phi /= phi.sum(axis=0)[np.newaxis, :]
            phi = phi.transpose()
        predictor.add_beta(phi.transpose())
        self.predictor = predictor

        return doc_topic

    @property
    def phi(self):
        return self.predictor.phis[0]
