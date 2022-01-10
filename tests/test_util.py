import numpy as np
from scipy import sparse as sps

from lda11 import rowwise_train_test_split

from .conftest import Docs


def test_split(docs_gen: Docs) -> None:
    (X1, X2), _ = docs_gen.gen_doc(1000)
    X2_sp = sps.lil_matrix(X2)
    X1_tr, X1_te = rowwise_train_test_split(X1)
    assert np.all(np.asarray(X1 - X1_tr - X1_te) == 0)
    X2_tr, X2_te = rowwise_train_test_split(X2_sp, random_seed=0)
    # raise RuntimeError((X2.tocsr() - X2_tr - X2_te))
    v = np.abs(X2 - X2_tr.toarray() - X2_te.toarray()).sum()
    assert v == 0
