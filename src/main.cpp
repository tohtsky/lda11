#include <vector> 
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> 
#include <pybind11/stl.h>
#include "Eigen/Core"
#include "./defs.hpp" 
#include "./state.hpp"


namespace py = pybind11;

PYBIND11_MODULE(_lda, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    py::class_<LDATrainer>(m, "LDATrainer")
        .def(py::init< 
                const RealVector &, 
                Eigen::Ref<IntegerVector>, 
                Eigen::Ref<IndexVector>,
                Eigen::Ref<IndexVector>,
                const size_t,
                int 
              >())
        .def("initialize", &LDATrainer::initialize_count)
        .def("iterate_gibbs", &LDATrainer::iterate_gibbs) 
        .def("log_likelihood", &LDATrainer::log_likelihood) 
    ;

    py::class_<LabelledLDATrainer>(m, "LabelledLDATrainer")
        .def(py::init<
                Real,
                Real, 
                const IntegerMatrix &, 
                Eigen::Ref<IntegerVector>, 
                Eigen::Ref<IndexVector>,
                Eigen::Ref<IndexVector>,
                const size_t,
                int 
              >())
        .def("initialize", &LDATrainer::initialize_count)
        .def("iterate_gibbs", &LDATrainer::iterate_gibbs) 
        .def("log_likelihood", &LDATrainer::log_likelihood) 
    ;
    m.def("log_likelihood_doc_topic", &log_likelihood_doc_topic);

    py::class_<Predictor>(m, "Predictor")
        .def(py::init<
            size_t,
            const RealVector &,
            int > ()
        )
        .def("add_beta", &Predictor::add_beta)
        .def("predict_gibbs", &Predictor::predict_gibbs) 
        .def("__getstate__", [](const Predictor &p) {
            std::vector<RealMatrix> betas;
            for (auto b = p.beta_begin(); b!=p.beta_end(); b++) {
              betas.push_back(*b);
            }
            return py::make_tuple(p.n_topics(), p.doc_topic_prior(), betas);
        })
        .def("__setstate__", [](Predictor &p, py::tuple t) {
            if (t.size() != 3)
            throw std::runtime_error("Invalid state!");
            new (&p) Predictor(t[0].cast<size_t>(), t[1].cast<RealVector>());
            py::list betas_ = t[2].cast<py::list>();
            for (auto item: betas_) {
              p.add_beta(item.cast<RealMatrix>());
            }
        });
}
