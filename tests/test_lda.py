import numpy as np

from lda11 import LDA

from .conftest import Docs


def test_lda(docs_gen: Docs) -> None:
    (X1, _), true_theta = docs_gen.gen_doc(1000)
    lda = LDA(
        2,
        n_iter=50,
        optimize_interval=1,
        optimize_burn_in=25,
        use_cgs_p=True,
        n_workers=4,
    )
    lda.fit(X1)
    phi1 = lda.phi

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
