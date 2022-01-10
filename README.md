# LDA11 - yet another collapsed gibbs sampler for python.

## Features

- Support parallelized sampler proposed in [Distributed Inference for Latent Dirichlet Allocation](https://dl.acm.org/doi/abs/10.5555/2981562.2981698).
- Implement [CGS_p estimator](http://www.jmlr.org/papers/volume18/16-526/16-526.pdf) for more precise point estimate of topic-word distribution.
- Implement [Labelled LDA](https://www-nlp.stanford.edu/cmanning/papers/llda-emnlp09.pdf)
- Able to obtain per-word topic frequency.

The implementaion relies on [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for faster array multiplication and  [pybind11](https://github.com/pybind/pybind11) for simple binding.


## Installation

You can install the wheel from pypi:

```
pip install lda11
```

For x64 architecture, the above wheel is built using AVX.
If it is not convenient for you, try e.g.

```
CFLAGS="-march=native" pip install git+https://github.com/tohtsky/lda11
```
