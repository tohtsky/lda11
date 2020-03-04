# LDA11 - yet another collapsed gibbs sampler for python.

## Features

- Use [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for faster array multiplication.
- Use [pybind11](https://github.com/pybind/pybind11) to bind the code into python.
- Support parallelized sampler proposed in [Distributed Inference for Latent Dirichlet Allocation](https://dl.acm.org/doi/abs/10.5555/2981562.2981698).
- Implement [CGS_p estimator](http://www.jmlr.org/papers/volume18/16-526/16-526.pdf) for more precise point estimate of topic-word distribution.


## Installation

```
pip install git+https://github.com/tohtsky/lda11
```

The above command will automatically download Eigen (ver 3.3.7).
If you want to use an existing version of Eigen (located on `path/to/eigen`),
type

```
EIGEN3_INCLUDE_DIR=/path/to/eigen pip install git+https://github.com/tohtsky/lda11
```

## Todos

- Add Labelled LDA
