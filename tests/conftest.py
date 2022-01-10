import numpy as np
import pytest

from .language import Docs, Language


@pytest.fixture
def docs_gen() -> Docs:
    language_1 = Language(
        np.asfarray([1, 1, 1, 0.01, 0.01, 0.01]),
        np.asfarray([0.01, 0.01, 0.01, 1, 1, 1]),
    )
    language_2 = Language(
        np.asfarray([1, 0.01, 1, 0.01]), np.asfarray([0.01, 1, 0.01, 1])
    )
    return Docs([language_1, language_2])
