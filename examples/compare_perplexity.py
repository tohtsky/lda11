# This example requires scikit-leran.
from time import time

import numpy as np
from scipy import sparse as sps
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation as LDA_vb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from lda11 import LDA as LDA_cgs_p
from lda11.util import rowwise_train_test_split

N_TOPICS = 16
print("reading data...")
dataset = fetch_20newsgroups(shuffle=False, remove=("headers", "footers", "quotes"))
data_samples = dataset.data
train_docs, test_docs = train_test_split(data_samples, random_state=42)

print("priparing Count Vectorizer")
tf_vectorizer = CountVectorizer(max_df=1.0, stop_words="english")

X_train = tf_vectorizer.fit_transform(train_docs)
X_test = tf_vectorizer.transform(test_docs)

feature_names = tf_vectorizer.get_feature_names()

tf_vectorizer.get_stop_words()

print("Splitting test documents...")
X_test_train, X_test_test = rowwise_train_test_split(X_test, random_seed=114514)


print("Start fitting sk-learn model...")
start = time()
vb_model = LDA_vb(n_components=N_TOPICS)
vb_model.fit(X_train)

phi_vb = vb_model.components_ / vb_model.components_.sum(axis=1)[:, np.newaxis]
end = time()
print("done in {:.2f} seconds".format((end - start)))

print("Start fitting our lda model...")
start = time()
cgs_p_model = LDA_cgs_p(n_components=N_TOPICS, n_iter=500)
cgs_p_model.fit(X_train)
phi_cgs_p = cgs_p_model.phi.transpose()
end = time()
print("done in {:.2f} seconds".format((end - start)))

print("Start fitting paralleized CGS sampler with hyper-parameter optimization...")
start = time()
parallel_cgs_model = LDA_cgs_p(
    n_components=N_TOPICS, n_iter=500, n_workers=2, optimize_interval=50
)
parallel_cgs_model.fit(X_train)
phi_parallel_cgs = parallel_cgs_model.phi.transpose()
end = time()
print("done in {:.2f} seconds".format((end - start)))


def test_perplexity(model, phi, **kwargs):
    theta = model.transform(X_test_train, **kwargs)
    log_ps = np.log(theta.dot(phi))
    coo = X_test_test.tocoo()
    # perplexity
    return np.exp(-(log_ps[coo.row, coo.col] * coo.data).sum() / coo.data.sum())


print("Start testing vb model")
start = time()
ll_vb = test_perplexity(vb_model, phi_vb)
end = time()
print("Done in {:.2f} seconds, test perplexity = {:.2f}".format(end - start, ll_vb))


print("Start testing cgs_p model")
start = time()
ll_cgs_p = test_perplexity(cgs_p_model, phi_cgs_p)
end = time()
print("Done in {:.2f} seconds, test perplexity = {:.2f}".format(end - start, ll_cgs_p))

print("Start testing parallelized + optimized cgs model")
start = time()
ll_cgs_p = test_perplexity(parallel_cgs_model, phi_parallel_cgs, n_workers=4)
end = time()
print("Done in {:.2f} seconds, test perplexity = {:.2f}".format(end - start, ll_cgs_p))
