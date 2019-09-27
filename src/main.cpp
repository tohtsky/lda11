#include <vector> 
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> 
#include "Eigen/Core"
#include "./defs.hpp" 
#include "./state.hpp"


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
              >())
        .def("initialize", &DocState::initialize_count)
        .def("iterate_gibbs", &DocState::iterate_gibbs) 
        .def("log_likelihood", &DocState::log_likelihood) 
    ;

}
