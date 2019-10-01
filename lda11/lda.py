import numpy as np
from numbers import Number
from gc import collect
from ._lda import LDATrainer, log_likelihood_doc_topic, Predictor, LabelledLDATrainer
from tqdm import tqdm

RealType = np.float64
IntegerType = np.int32
IndexType = np.uint64

def number_to_array(n_components, default, arg=None):
    if arg is None:
        arg = default
    if isinstance(arg, Number):
        return np.ones(
            n_components, dtype=RealType
        ) * RealType(arg)
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
    return counts.astype(IntegerType), dix.astype(IndexType), wix.astype(IndexType)

def bow_row_to_counts(X, i):
    if isinstance(X, np.ndarray):
        assert(len(X.shape) == 2)
        assert(X.dtype == np.int32 or X.dtype == np.int64) 
        wix = X[i].nonzero()
        counts = X[i, wix] 
    else:
        _, wix = X[i].nonzero()
        counts = X[i, wix].toarray().ravel()

    return counts.astype(IntegerType), wix.astype(IndexType)


class LDA(object):
    def __init__(self, n_components=10, doc_topic_prior=None, topic_word_prior=None):
        n_components = int(n_components)
        self.n_components = n_components

        self.doc_topic_prior = doc_topic_prior 
        self.topic_word_prior = topic_word_prior
        self.n_vocabs = None
        self.docstate_ = None
        self.components_ = None

    def fit(self, X, n_iter=1000): 
        self._fit(X, n_iter=n_iter)
        return self

    def fit_transform(self, X, **kwargs):
        result = self._fit(X, **kwargs) + self.doc_topic_prior[np.newaxis, :]
        result /= result.sum(axis=1)[:, np.newaxis]
        return result 

    def _fit(self, X, n_iter=1000, ll_freq=10):
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

        doc_topic = np.zeros( (X.shape[0], self.n_components), dtype=IntegerType)
        word_topic = np.zeros( (X.shape[1], self.n_components), dtype=IntegerType) 

        docstate = LDATrainer(self.topic_word_prior, count, dix, wix, self.n_components, 42)
        docstate.initialize( doc_topic, word_topic )

        topic_counts = doc_topic.sum(axis=0).astype(IntegerType)
        self.components_ = word_topic.transpose()

        ll = docstate.log_likelihood(
            self.topic_word_prior,
            word_topic,
        ) + log_likelihood_doc_topic( 
            self.doc_topic_prior, doc_topic
        )

        with tqdm(range(n_iter)) as pbar:
            pbar.set_description("Log Likelihood = {0:.2f}".format(ll))
            for i in pbar:
                docstate.iterate_gibbs(
                    self.topic_word_prior,
                    doc_topic,
                    word_topic,
                    topic_counts
                )
                if ( i + 1) % ll_freq == 0:
                    ll = docstate.log_likelihood(
                        self.topic_word_prior,
                        word_topic,
                    ) + log_likelihood_doc_topic( 
                        self.doc_topic_prior, doc_topic
                    ) 

                    pbar.set_description("Log Likelihood = {0:.2f}".format(ll)) 

        return doc_topic

class LabelledLDA(object):
    def __init__(self, alpha=1e-2, epsilon=1e-30, topic_word_prior=None):
        self.n_components = None
        self.topic_word_prior = topic_word_prior
        self.alpha = alpha 
        self.epsilon = 1e-20
        self.n_vocabs = None
        self.docstate_ = None
        self.components_ = None

    def fit(self, X, Y, n_iter=1000): 
        self._fit(X, Y, n_iter=n_iter)
        return self

    def fit_transform(self, X, Y, **kwargs):
        result = self._fit(X, **kwargs) + self.doc_topic_prior[np.newaxis, :]
        result /= result.sum(axis=1)[:, np.newaxis]
        return result 

    def _fit(self, X, Y, n_iter=1000, ll_freq=10):
        if isinstance(Y, np.ndarray):
            Y = Y.astype(IntegerType) 
        else:
            Y = Y.toarray().astype(IntegerType)

        self.n_components = Y.shape[1]

        self.topic_word_prior = number_to_array(
            X.shape[1], 1 / float(self.n_components),
            self.topic_word_prior
        )

        try:
            count, dix, wix = check_array(X)
        except:
            print('Check for X failed.')
            raise

        doc_topic = np.zeros( (X.shape[0], self.n_components), dtype=IntegerType)
        word_topic = np.zeros( (X.shape[1], self.n_components), dtype=IntegerType) 

        docstate = LabelledLDATrainer(
            self.alpha,
            self.epsilon,
            Y,
            count, dix, wix, self.n_components, 42
        )
        docstate.initialize( doc_topic, word_topic )

        topic_counts = doc_topic.sum(axis=0).astype(IntegerType)
        self.components_ = word_topic.transpose()

        with tqdm(range(n_iter)) as pbar:
            for i in pbar:
                docstate.iterate_gibbs(
                    self.topic_word_prior,
                    doc_topic,
                    word_topic,
                    topic_counts
                )
        return doc_topic


class MultipleContextLDA(object):
    def __init__(
        self, n_components=100, 
        doc_topic_prior=None, topic_word_priors=None
    ):
        n_components = int(n_components)
        self.n_components = n_components

        self.doc_topic_prior = doc_topic_prior 
        self.topic_word_priors = topic_word_priors
        self.n_vocabs = None
        self.docstate_ = None
        self.components_ = None
        self.n_modals = None

        self.predictor = None

    def fit(self, *X, n_iter=1000): 
        self._fit(*X, n_iter=n_iter)
        return self

    def fit_transform(self, *X, **kwargs):
        result = self._fit(*X, **kwargs) + self.doc_topic_prior[np.newaxis, :]
        result /= result.sum(axis=1)[:, np.newaxis]
        return result 

    def _fit(self, *Xs, n_iter=1000, ll_freq=10):
        """
        Xs should be a list of contents.
        All entries must have the same shape[0].
        """
        n_vocabs = []

        self.modality = len(Xs)
        self.doc_topic_prior = number_to_array(
            self.n_components, 1 / float(self.n_components), 
            self.doc_topic_prior
        )

        if self.topic_word_priors is None:
            self.topic_word_priors = [ None for i in range(self.modality) ]

        self.topic_word_priors = [
            number_to_array(
                X.shape[1], 1 / float(self.n_components), 
            )
            for X, val in zip(Xs, self.topic_word_priors)
        ]

        doc_tuples = []
        for X in Xs:
            try:
                doc_tuples.append(
                    (check_array(X))
                )
            except:
                print('Check for X failed.')
                raise

        doc_topic = np.zeros( (X.shape[0], self.n_components), dtype=IntegerType)
        word_topics = [
            np.zeros( (X.shape[1], self.n_components), dtype=IntegerType) 
            for X in Xs
        ]

        docstates = []
        for (count, dix, wix), word_topic in zip(doc_tuples, word_topics):
            docstate = LDATrainer(
                self.doc_topic_prior,
                count, dix, wix, self.n_components, 42
            )
            docstates.append(docstate)
            docstate.initialize(doc_topic, word_topic)

        topic_counts = doc_topic.sum(axis=0).astype(IntegerType)
        ll = log_likelihood_doc_topic( 
            self.doc_topic_prior, doc_topic
        )
        for topic_word_prior, word_topic, docstate in zip(
                    self.topic_word_priors, word_topics, docstates
        ): 
            ll += docstate.log_likelihood(
                topic_word_prior, word_topic
            )

        with tqdm(range(n_iter)) as pbar: 
            pbar.set_description("Log Likelihood = {0:.2f}".format(ll))
            for i in pbar:
                for topic_word_prior, word_topic, docstate in zip(
                    self.topic_word_priors, word_topics, docstates
                ): 
                    docstate.iterate_gibbs(
                        topic_word_prior,
                        doc_topic,
                        word_topic,
                        topic_counts
                    )
                if ( i + 1) % ll_freq == 0:
                    ll = log_likelihood_doc_topic( 
                        self.doc_topic_prior, doc_topic
                    )
                    for topic_word_prior, word_topic, docstate in zip(
                                self.topic_word_priors, word_topics, docstates
                    ): 
                        ll += docstate.log_likelihood(
                            topic_word_prior, word_topic
                        )
                    pbar.set_description("Log Likelihood = {0:.2f}".format(ll)) 

        self.components_ = [
            word_topic.transpose()
            for word_topic in word_topics
        ]
        self.word_topics = word_topics

        predictor = Predictor(self.n_components, self.doc_topic_prior, 42)

        for i, wt in enumerate(word_topics):
            wt = wt + self.topic_word_priors[i][:, np.newaxis]
            wt /= wt.sum(axis=0)[np.newaxis, :]
            wt = wt.astype(RealType)
            predictor.add_beta(wt)
        self.predictor = predictor

        return doc_topic

    def transform(self, *Xs, n_iter=100, random_seed=42):
        n_domains = len(Xs)
        shapes =  set({X.shape[0] for X in Xs})
        assert(len(shapes) == 1)
        for shape in shapes: break

        results = np.zeros((shape, self.n_components), dtype=RealType)
        for i in range(shape):
            counts = []
            wixs = []
            for n in range(n_domains):
                count, wix = bow_row_to_counts(Xs[n], i)
                counts.append(count)
                wixs.append(wix)

            m = self.predictor.predict_gibbs(
                wixs, counts, n_iter, random_seed
            )
            m = m + self.doc_topic_prior
            results[i] = m / m.sum()
        return results

