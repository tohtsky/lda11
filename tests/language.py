from typing import List, Tuple

import numpy as np
import numpy.typing as npt

N_DOCS = 1000


class Language:
    def __init__(
        self, TOPIC1: npt.NDArray[np.float64], TOPIC2: npt.NDArray[np.float64]
    ):
        self.topic_1: npt.NDArray[np.float64] = TOPIC1 / TOPIC1.sum()
        self.topic_2: npt.NDArray[np.float64] = TOPIC2 / TOPIC2.sum()


class Docs:
    def __init__(self, languages: List[Language]):
        self.languages = languages

    def gen_doc(
        self, n_docs: int
    ) -> Tuple[List[npt.NDArray[np.int32]], npt.NDArray[np.float64]]:
        rns = np.random.RandomState(0)
        words: List[List[npt.NDArray[np.int64]]] = [
            [] for _ in range(len(self.languages))
        ]
        thetas: List[np.ndarray] = []
        for _ in range(n_docs):
            theta = rns.dirichlet(np.asfarray([0.2, 0.2]))
            thetas.append(theta)
            for lind, language in enumerate(self.languages):
                cnt = rns.poisson(5)
                wdist = (
                    float(theta[0]) * language.topic_1
                    + float(theta[1]) * language.topic_2
                )
                words[lind].append(rns.multinomial(cnt, wdist))

        return [np.vstack(x) for x in words], np.vstack(thetas)
