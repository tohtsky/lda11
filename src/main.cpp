#include <vector> 
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> 
#include "Eigen/Core"
#include "./defs.hpp" 

struct DocState {
    DocState (
        Eigen::Ref<IntegerVector> counts,
        Eigen::Ref<IndexVector> dixs,
        Eigen::Ref<IndexVector> wixs,
        size_t n_topics,
        int random_seed=32 
    ): word_states(), n_topics_(n_topics), random_state_(random_seed), udist_(0.0, 1.0) {
        if (counts.rows() != dixs.rows())
            throw std::runtime_error("");
    }

    std::vector<WordState> word_states;
    const std::size_t n_topics_;
    std::mt19937 random_state_;
    std::uniform_real_distribution<Real> udist_;
};

std::vector<WordState> initialize_doc(
    Eigen::Ref<IntegerVector> counts,
    Eigen::Ref<IndexVector> dixs,
    Eigen::Ref<IndexVector> wixs
);

namespace py = pybind11;

PYBIND11_MODULE(_lda, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    py::class_<DocState>(m, "DocState")
        .def(py::init< 
                Eigen::Ref<IntegerVector>, 
                Eigen::Ref<IndexVector>,
                Eigen::Ref<IndexVector>,
                const size_t,
                int 
              >());

}
