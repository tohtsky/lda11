#include "defs.hpp"
#include "predictor.hpp"
#include "state.hpp"
#include "trainer.hpp"

#include "Eigen/Core"
#include "util.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

namespace py = pybind11;

RealVector learn_dirichlet(const Eigen::Ref<IntegerMatrix> &counts,
                           const Eigen::Ref<RealVector> &alpha_start,
                           Real alpha_prior_scale, Real alpha_prior_exponent,
                           size_t iteration) {
  using Eigen::numext::digamma;
  using std::vector;
  const int n_topic = alpha_start.rows();
  const int n_docs = counts.rows();
  if (counts.cols() != n_topic) {
    throw std::invalid_argument("count array and alpha have different sizes.");
  }
  RealVector alpha_current(alpha_start);
  RealVector numerator(n_topic);

  vector<Real> doc_length;
  vector<Real> doc_length_freq;

  vector<vector<Real>> topic_cnt(n_topic);
  vector<vector<Real>> topic_cnt_freq(n_topic);

  {
    using Map = std::unordered_map<Integer, Real>; // cnt, freq
    vector<Map> component_wise_hist(n_topic);
    Map doc_length_hist;

    for (int dix = 0; dix < n_docs; dix++) {
      int length = 0;
      for (int topic = 0; topic < n_topic; topic++) {
        Integer cnt = counts(dix, topic);
        if (cnt == 0) {
          continue;
        }
        component_wise_hist[topic][cnt]++;
        length += cnt;
      }
      doc_length_hist[length]++;
    }

    size_t topic_index = 0;
    for (auto &map : component_wise_hist) {
      for (auto &cnt_freq : map) {
        topic_cnt[topic_index].push_back(cnt_freq.first);
        topic_cnt_freq[topic_index].push_back(cnt_freq.second);
      }
      topic_index++;
    }

    for (auto &iter : doc_length_hist) {
      doc_length.push_back(iter.first);
      doc_length_freq.push_back(iter.second);
    }
  }
  for (size_t it = 0; it < iteration; it++) {
    Real alpha_sum = alpha_current.sum();
    numerator.array() = 0;
    Real denominator =
        ((vector_to_eigen(doc_length).array() + alpha_sum).digamma() -
         digamma(alpha_sum))
            .matrix()
            .transpose() *
        vector_to_eigen(doc_length_freq);
    for (int topic_index = 0; topic_index < n_topic; topic_index++) {
      Real numerator = ((vector_to_eigen(topic_cnt[topic_index]).array() +
                         alpha_current(topic_index))
                            .digamma() -
                        digamma(alpha_current(topic_index)))
                           .matrix()
                           .transpose() *
                       vector_to_eigen(topic_cnt_freq[topic_index]);
      alpha_current(topic_index) =
          (alpha_current(topic_index) * numerator + alpha_prior_exponent) /
          (denominator + alpha_prior_scale);
    }
  }

  return alpha_current;
}

Real log_likelihood_doc_topic(Eigen::Ref<RealVector> doc_topic_prior,
                              Eigen::Ref<IntegerMatrix> doc_topic) {
  size_t n_doc = doc_topic.rows();
  size_t n_topics = doc_topic.cols();
  Real ll = n_doc * (std::lgamma(doc_topic_prior.sum()) -
                     doc_topic_prior.array().lgamma().sum());

  RealArray exponent(n_topics);
  for (size_t i = 0; i < n_doc; i++) {
    exponent = doc_topic.row(i).cast<Real>().transpose().array() +
               doc_topic_prior.array();
    ll += exponent.lgamma().sum();
    ll -= std::lgamma(exponent.sum());
  }
  return ll;
}

PYBIND11_MODULE(_lda, m) {
  m.doc() = "Backend C++ inplementation for lda11.";
  py::class_<LDATrainer>(m, "LDATrainer")
      .def(py::init<const RealVector &, Eigen::Ref<IntegerVector>,
                    Eigen::Ref<IndexVector>, Eigen::Ref<IndexVector>,
                    const size_t, int, size_t>())
      .def("set_doc_topic_prior", &LDATrainer::set_doc_topic_prior)
      .def("initialize", &LDATrainer::initialize_count)
      .def("iterate_gibbs", &LDATrainer::iterate_gibbs)
      .def("obtain_phi", &LDATrainer::obtain_phi)
      .def("log_likelihood", &LDATrainer::log_likelihood);

  m.def("log_likelihood_doc_topic", &log_likelihood_doc_topic);
  m.def("learn_dirichlet", &learn_dirichlet);

  py::class_<Predictor>(m, "Predictor")
      .def(py::init<size_t, const RealVector &, int>())
      .def("add_beta", &Predictor::add_beta)
      .def("predict_gibbs", &Predictor::predict_gibbs)
      .def("predict_mf", &Predictor::predict_mf)
      .def_readonly("phis", &Predictor::betas_)
      .def("__getstate__",
           [](const Predictor &p) {
             std::vector<RealMatrix> betas;
             for (auto b = p.beta_begin(); b != p.beta_end(); b++) {
               betas.push_back(*b);
             }
             return py::make_tuple(p.n_topics(), p.doc_topic_prior(), betas);
           })
      .def("__setstate__", [](Predictor &p, py::tuple t) {
        if (t.size() != 3)
          throw std::runtime_error("Invalid state!");
        new (&p) Predictor(t[0].cast<size_t>(), t[1].cast<RealVector>());
        py::list betas_ = t[2].cast<py::list>();
        for (auto item : betas_) {
          p.add_beta(item.cast<RealMatrix>());
        }
      });
}
