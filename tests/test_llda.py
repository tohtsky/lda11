import pickle
import sys
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import numpy.typing as npt

from lda11 import LabelledLDA, labelled_lda


class LabelledLanguage:
    def __init__(
        self, TOPIC1: npt.NDArray[np.float64], TOPIC2: npt.NDArray[np.float64]
    ):
        self.topic_1: npt.NDArray[np.float64] = TOPIC1 / TOPIC1.sum()
        self.topic_2: npt.NDArray[np.float64] = TOPIC2 / TOPIC2.sum()
        self.common = np.ones_like(TOPIC1) / TOPIC1.shape[0]

    def gen_doc(
        self, n_docs: int
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        rns = np.random.RandomState(0)
        Xs = []
        labels = []
        for i in range(n_docs):
            cnt = rns.poisson(10)
            label = np.asfarray([1, rns.binomial(1, 0.5), rns.binomial(1, 0.5)])
            p = (
                label[0] * self.common
                + label[1] * self.topic_1
                + label[2] * self.topic_2
            )
            words = rns.multinomial(cnt, p / p.sum())
            Xs.append(words)
            labels.append(label)
        return np.vstack(Xs), np.vstack(labels)


def test_llda() -> None:
    TOPIC_A = np.asfarray([0.01, 1, 0.01, 1])
    TOPIC_B = np.asfarray([1, 0.01, 1, 0.01])
    A_word_index = np.where(TOPIC_A > 0.1)[0]
    B_word_index = np.where(TOPIC_A < 0.1)[0]

    for A_index, cgs_p in zip([1, 2], [False, True]):
        if A_index == 1:
            language = LabelledLanguage(TOPIC_A, TOPIC_B)
            B_index = 2
        else:
            language = LabelledLanguage(TOPIC_B, TOPIC_A)
            B_index = 1

        X, Y = language.gen_doc(1000)

        llda = LabelledLDA(use_cgs_p=cgs_p, n_workers=1, n_iter=100).fit(X, Y)
        if sys.platform.startswith("linux"):
            with NamedTemporaryFile() as temp_fs:
                pickle.dump(llda, temp_fs)
                temp_fs.seek(0)
                del llda
                llda_new: LabelledLDA = pickle.load(temp_fs)
        else:
            llda_new = llda

        for a_word in A_word_index:
            for b_word in B_word_index:
                assert llda_new.phi[a_word, A_index] > llda_new.phi[b_word, A_index]
                assert llda_new.phi[a_word, B_index] < llda_new.phi[b_word, B_index]

        A_DOC = np.asarray(([0, 10, 0, 10]), dtype=np.int32)
        for mode in ["mf", "gibbs"]:
            theta = llda_new.transform(A_DOC, mode=mode, n_workers=1)[0]  # type: ignore
            if A_index == 1:
                assert (theta[1] / theta[2]) > 5
            else:
                assert (theta[2] / theta[1]) > 5
