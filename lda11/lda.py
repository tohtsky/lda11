import numpy as np
from numbers import Number
from gc import collect
from ._lda import LDATrainer, log_likelihood_doc_topic, Predictor
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
        # if X is either types of, scipy.sparse X has this attribute.
        X = X.tocsr()
        X.sort_indices()
        dix, wix = X.nonzero()
        counts = X.data
    return counts.astype(IntegerType), dix.astype(IndexType), wix.astype(IndexType)


def bow_row_to_counts(X, i):
    if isinstance(X, np.ndarray):
        assert(len(X.shape) == 2)
        assert(X.dtype == np.int32 or X.dtype == np.int64)
        wix, = X[i].nonzero()
        counts = X[i, wix]
    else:
        _, wix = X[i].nonzero()
        counts = X[i, wix].toarray().ravel()

    return counts.astype(IntegerType), wix.astype(IndexType)

class LDAPredictorMixin:
    """
    self.components_
    self.n_components
    self.predictor
    are needed
    """
    def transform(
            self, *Xs,
            n_iter=100, random_seed=42, mode="gibbs", mf_tolerance=1e-10
        ):
        
        n_domains = len(Xs)
        if len(self.components_) != n_domains:
            raise ValueError(f"Got {n_domains} input, while training was {len(self.components_)} domaints.")
        shapes =  set({X.shape[0] for X in Xs if X is not None})
        assert(len(shapes) == 1)
        for shape in shapes: break

        Xs_completed = []
        for i, X in enumerate(Xs):
            if X is not None:
                Xs_completed.append(X)
                continue

            try:
                from scipy import sparse as sps
            except: 
                raise RuntimeError('In ordere to use None as the default input for transform, scipy is needed.')


            # X is None
            X_zero = sps.csr_matrix((shape, self.components_[i].shape[1]))
            Xs_completed.append(X_zero)
        Xs = Xs_completed

        results = np.zeros((shape, self.n_components), dtype=RealType)
        for i in range(shape):
            counts = []
            wixs = []
            for n in range(n_domains):
                count, wix = bow_row_to_counts(Xs[n], i)
                counts.append(count)
                wixs.append(wix)
            if mode == "gibbs": 
                m = self.predictor.predict_gibbs(
                    wixs, counts, n_iter, random_seed
                )
                m = m + self.doc_topic_prior
                results[i] = m / m.sum()
            else:
                results[i] = self.predictor.predict_mf(
                    wixs, counts, n_iter, mf_tolerance
                )
        return results


class LDA(object):
    def __init__(self, n_components=10, doc_topic_prior=None, topic_word_prior=None):
        n_components = int(n_components)
        self.n_components = n_components

        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.n_vocabs = None
        self.docstate_ = None
        self.components_ = None

    def fit(self, X, n_iter=500, ll_freq=10):
        self._fit(X, n_iter=n_iter, ll_freq=ll_freq)
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

        doc_topic = np.zeros(
            (X.shape[0], self.n_components), dtype=IntegerType)
        word_topic = np.zeros(
            (X.shape[1], self.n_components), dtype=IntegerType)

        docstate = LDATrainer(self.doc_topic_prior, count,
                              dix, wix, self.n_components, 42)
        docstate.initialize(doc_topic, word_topic)

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
                def log(ll):
                    pbar.set_description("Log Likelihood = {0:.2f}".format(ll)) 

                docstate.iterate_gibbs(
                    self.topic_word_prior,
                    doc_topic,
                    word_topic,
                    topic_counts
                )
                if (i + 1) % ll_freq == 0:
                    ll = docstate.log_likelihood(
                        self.topic_word_prior,
                        word_topic,
                    ) + log_likelihood_doc_topic(
                        self.doc_topic_prior, doc_topic
                    )

                    pbar.set_description("Log Likelihood = {0:.2f}".format(ll))

        phi = docstate.obtain_phi(
            self.topic_word_prior,
            doc_topic,
            word_topic,
            topic_counts
        )
 
        self.predictor = Predictor(self.n_components, self.doc_topic_prior, 42)

        self.predictor.add_beta(phi.transpose())
        return doc_topic

    @property
    def phi(self):
        return self.predictor.phis[0]

    def transform(self, X, n_iter=100, random_seed=42, mode="gibbs", mf_tolerance=1e-10, gibbs_burn_in=10):
        shape = X.shape[0]
        results = np.zeros((shape, self.n_components), dtype=RealType)
        for i in range(shape):
            count, wix = bow_row_to_counts(X, i)
            if mode == "gibbs":
                m = self.predictor.predict_gibbs(
                    [wix], [count], n_iter, gibbs_burn_in, random_seed
                )
                results[i] = m
            else:
                results[i] = self.predictor.predict_mf(
                    [wix], [count], n_iter, mf_tolerance
                )
        return results



class MultipleContextLDA(LDAPredictorMixin):
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
            self.topic_word_priors = [None for i in range(self.modality)]

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

        doc_topic = np.zeros(
            (X.shape[0], self.n_components), dtype=IntegerType)
        word_topics = [
            np.zeros((X.shape[1], self.n_components), dtype=IntegerType)
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
                        topic_counts,
                        log_function
                    )
                if (i + 1) % ll_freq == 0:
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

        predictor = Predictor(self.n_components, self.doc_topic_prior, 42)

        for i, (twp, wt, docstate) in enumerate(zip(self.topic_word_priors, word_topics, docstates)):
            phi = docstate.obtain_phi(
                twp,
                doc_topic,
                wt,
                topic_counts
            )
            predictor.add_beta(phi.transpose())

        self.predictor = predictor

        return doc_topic
    
    @property
    def phis(self):
        return self.predictor.phis

    def transform(self, *Xs, n_iter=100, random_seed=42, mode="gibbs", mf_tolerance=1e-10, gibbs_burn_in=10):
        n_domains = len(Xs)
        shapes = set({X.shape[0] for X in Xs})
        assert(len(shapes) == 1)
        for shape in shapes:
            break

        results = np.zeros((shape, self.n_components), dtype=RealType)
        for i in range(shape):
            counts = []
            wixs = []
            for n in range(n_domains):
                count, wix = bow_row_to_counts(Xs[n], i)
                counts.append(count)
                wixs.append(wix)
            if mode == "gibbs":
                m = self.predictor.predict_gibbs(
                    wixs, counts, n_iter, gibbs_burn_in, random_seed
                )
                results[i] = m
            else:
                results[i] = self.predictor.predict_mf(
                    wixs, counts, n_iter, mf_tolerance
                )
        return results

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

        doc_topic = np.zeros(
            (X.shape[0], self.n_components), dtype=IntegerType)
        word_topic = np.zeros(
            (X.shape[1], self.n_components), dtype=IntegerType)

        docstate = LabelledLDATrainer(
            self.alpha,
            self.epsilon,
            Y,
            count, dix, wix, self.n_components, 42
        )
        docstate.initialize(doc_topic, word_topic)

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

        self.component_ = word_topic.transpose()

        predictor = Predictor(self.n_components, self.doc_topic_prior, 42)

        word_topic = word_topic + self.topic_word_priors[:, np.newaxis]
        word_topic = word_topic.astype(RealType)
        word_topic /= word_topic.sum(axis=0)[np.newaxis, :]
        predictor.add_beta(word_topic)
        self.predictor = predictor


        return doc_topic
