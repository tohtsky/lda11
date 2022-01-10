from pkg_resources import DistributionNotFound, get_distribution  # type: ignore

from .labelled_lda import LabelledLDA
from .lda import LDA, MultilingualLDA
from .util import rowwise_train_test_split

try:
    __version__ = get_distribution("lda11").version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "__version__",
    "LDA",
    "LabelledLDA",
    "MultilingualLDA",
    "rowwise_train_test_split",
]
