import numpy as np

from lda11 import MultilingualLDA

from .conftest import Docs


def test_mlda(docs_gen: Docs) -> None:
    (X1, X2), true_theta = docs_gen.gen_doc(1000)
    lda = MultilingualLDA(2, n_iter=50, optimize_interval=1, optimize_burn_in=25)
    lda.fit(X1, X2)
    phi1, phi2 = lda.phis

    # determin which is TOPIC1

    lang1_topic1_strong_index = np.where(docs_gen.languages[0].topic_1 > 0.1)[0]
    lang1_topic2_strong_index = np.where(docs_gen.languages[0].topic_1 < 0.1)[0]
    if (
        phi1[lang1_topic1_strong_index, 0].mean()
        > phi1[lang1_topic2_strong_index, 0].mean()
    ):
        topic1_index = 0
        topic2_index = 1
    else:
        topic1_index = 1
        topic2_index = 0
    for i in lang1_topic1_strong_index:
        for j in lang1_topic2_strong_index:
            assert phi1[i, topic1_index] > phi1[j, topic1_index]
            assert phi1[i, topic2_index] < phi1[j, topic2_index]

    lang2_topic1_strong_index = np.where(docs_gen.languages[1].topic_1 > 0.1)[0]
    lang2_topic2_strong_index = np.where(docs_gen.languages[1].topic_1 < 0.1)[0]
    for i in lang2_topic1_strong_index:
        for j in lang2_topic2_strong_index:
            assert phi2[i, topic1_index] > phi2[j, topic1_index]
            assert phi2[i, topic2_index] < phi2[j, topic2_index]

    # just check it works.
    for algo in ["mf", "gibbs"]:
        checked_cnt = 0
        theta_inferred = lda.transform(X1, X2, mode=algo)  # type: ignore
        for i in range(X1.shape[0]):
            if (true_theta[i, 0] / true_theta[i, 1]) > 10:
                checked_cnt += 1
                assert theta_inferred[i, topic1_index] > theta_inferred[i, topic2_index]
        assert checked_cnt > 0
