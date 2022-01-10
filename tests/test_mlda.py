import numpy as np
from scipy import sparse as sps

from lda11 import MultilingualLDA

from .conftest import Docs


def test_mlda(docs_gen: Docs) -> None:
    (X1, X2), true_theta = docs_gen.gen_doc(1000)
    X2 = sps.lil_matrix(X2)
    lda = MultilingualLDA(
        2, n_iter=50, optimize_interval=1, optimize_burn_in=25, use_cgs_p=False
    )
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
            if (true_theta[i, 0] / true_theta[i, 1]) > 10 and (X1[i].sum() > 5):
                checked_cnt += 1
                assert theta_inferred[i, topic1_index] > theta_inferred[i, topic2_index]
        assert checked_cnt > 0

    wdt = lda.word_topic_assignment(X1, X2)
    assert len(wdt) == 1000
    for i, wdt_result_doc in enumerate(wdt):
        theta = wdt_result_doc[0]
        if (true_theta[i, 0] / true_theta[i, 1]) > 10:
            assert theta[topic1_index] > theta[topic2_index]
        m = wdt_result_doc[1]
        assert len(m) == 2
        # lang 1
        lang1_assignment = m[0]
        for word, topic in lang1_assignment.items():
            if (topic[topic1_index] / (1e-10 + topic[topic2_index])) > 10:
                assert word in lang1_topic1_strong_index
            if (topic[topic2_index] / (1e-10 + topic[topic1_index])) > 10:
                assert word in lang1_topic2_strong_index

        lang2_assignment = m[1]
        for word, topic in lang2_assignment.items():
            if (topic[topic1_index] / (1e-10 + topic[topic2_index])) > 10:
                assert word in lang2_topic1_strong_index
            if (topic[topic2_index] / (1e-10 + topic[topic1_index])) > 10:
                assert word in lang2_topic2_strong_index
