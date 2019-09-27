import numpy as np
from numbers import Number
from gc import collect
from ._lda import DocState
from tqdm import tqdm

RealType = np.float64
IntegerType = np.int32

def number_to_array(n_components, default, arg=None):
    if arg is None:
        arg = default
    if isinstance(arg, Number):
        return np.ones(
            n_components, dtype=RealType
        ) * float(arg)
    elif isinstance(arg, np.ndarray):
        assert(arg.shape[0] == n_components)
        return arg.astype(RealType)
    return None

def check_array(X):
    if isinstance(X, np.ndarray):
        assert(len(X.shape) == 2)
        assert(X.dtype == np.int32 or X.dtype == np.int64)

        dix, wix = X.nonzero()
        counts = X[dix, wix]
    else:
        X = X.tocsr() # if X is either types of, scipy.sparse X has this attribute.
        X.sort_indices()
        dix, wix = X.nonzero()
        counts = X.data
    return counts, dix, wix

class LDA(object):
    def __init__(self, n_components=10, doc_topic_prior=None, topic_word_prior=None):
        n_components = int(n_components)
        self.n_components = n_components

        self.doc_topic_prior = doc_topic_prior 
        self.topic_word_prior = topic_word_prior
        self.n_vocabs = None
        self.docstate_ = None

    def fit(self, X, n_iter=1000):

        return self._fit(X, n_iter=n_iter)

    def fit_transform(self, X, **kwargs):
        result = self._fit(X, **kwargs) + self.doc_topic_prior[np.newaxis, :]
        result /= result.sum(axis=1)[:, np.newaxis]
        return result 

    def _fit(self, X, n_iter=1000, ll_freq=5):
        self.doc_topic_prior = number_to_array(
            self.n_components, 1 / float(self.n_components), 
            self.doc_topic_prior
        )
        self.topic_word_prior = number_to_array(
            X.shape[1], 1 / float(self.n_components),
            self.topic_word_prior
        )

        try:
            count, dix, wix = check_array(X)
        except:
            print('Check for X failed.')
            raise
        count = count.astype(IntegerType)
        dix = dix.astype(np.uint64)
        wix = wix.astype(np.uint64)

        doc_topic = np.zeros( (X.shape[0], self.n_components), dtype=IntegerType)
        word_topic = np.zeros( (X.shape[1], self.n_components), dtype=IntegerType) 

        docstate = DocState(count, dix, wix, self.n_components, 42)
        docstate.initialize( doc_topic, word_topic )

        topic_counts = doc_topic.sum(axis=0).astype(IntegerType)
        self.components = word_topic.transpose()

        ll = docstate.log_likelihood(
            self.doc_topic_prior,
            self.topic_word_prior,
            doc_topic,
            word_topic,
            topic_counts
        )
        for i in tqdm(range(n_iter)):
            docstate.iterate_gibbs(
                self.doc_topic_prior,
                self.topic_word_prior,
                doc_topic,
                word_topic,
                topic_counts
            )
            if ( i + 1) % ll_freq == 0:
                ll = docstate.log_likelihood(
                    self.doc_topic_prior,
                    self.topic_word_prior,
                    doc_topic,
                    word_topic,
                    topic_counts
                )
                print(ll)

        return doc_topic
