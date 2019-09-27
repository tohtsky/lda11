import numpy as np
from numbers import Number

def number_to_array(n_components, arg=None):
    if arg is None:
        arg = 1 / float(n_components)
    if isinstance(arg, Number):
        return np.ones(
            n_components, dtype=np.float64
        ) * float(arg)
    elif isinstance(arg, np.ndarray):
        assert(arg.shape[0] == n_components)
        return arg.astype(np.float64)
    return None

def check_array(X):
    if isinstance(X, np.ndarray):
        assert(len(X.shape) == 2)
        assert(X.dtype == np.int32 or X.dtype == np.int64)

        dix, wix = X.nonzero()
        counts = X[dix, wix]
    else:
        X = X.tolil() # if X is either types of, scipy.sparse X has this attribute.
        dix, wix = X.nonzero()
        counts = X.data
    return counts, dix, wix



class LDA(object):
    def __init__(self, n_components=10, doc_topic_prior=None, topic_word_prior=None):
        n_components = int(n_components)
        self.n_components = n_components
        self.alpha = number_to_array(n_components, doc_topic_prior)
        self.eta = number_to_array(n_components, topic_word_prior)
        self.n_vocabs = None

    def fit(self, X):
        try:
            count, dix, wix = check_array(X)
        except:
            print('Check for X failed.')
            raise
        print(count, dix, wix)
        return self
