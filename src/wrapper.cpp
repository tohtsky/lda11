#include "defs.hpp"
#include "predictor.hpp"
#include "pybind11/attr.h"
#include "state.hpp"
#include "trainer.hpp"

#include "unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h"
#include "util.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;
std::pair<SparseIntegerMatrix, SparseIntegerMatrix>
train_test_split(const SparseIntegerMatrix &X, const double test_ratio,
                 std::int64_t random_seed) {
  using Triplet = Eigen::Triplet<Integer>;
  std::mt19937 random_state(random_seed);
  if (test_ratio > 1.0 || test_ratio < 0)
    throw std::invalid_argument("test_ratio must be within [0, 1]");
  std::vector<Integer> buffer;
  std::vector<Triplet> train_data, test_data;
  for (int row = 0; row < X.outerSize(); ++row) {
    buffer.clear(); // does not change capacity
    Integer cnt = 0;
    for (SparseIntegerMatrix::InnerIterator it(X, row); it; ++it) {
      cnt += it.value();
      for (int i = 0; i < it.value(); i++) {
        buffer.push_back(it.col());
      }
    }
    std::shuffle(buffer.begin(), buffer.end(), random_state);
    size_t n_test = static_cast<Integer>(std::floor(cnt * test_ratio));
    for (size_t i = 0; i < n_test; i++) {
      test_data.emplace_back(row, buffer[i], 1);
    }
    for (size_t i = n_test; i < buffer.size(); i++) {
      train_data.emplace_back(row, buffer[i], 1);
    }
  }
  SparseIntegerMatrix X_train(X.rows(), X.cols()), X_test(X.rows(), X.cols());
  auto dupfunction = [](const Integer &a, const Integer &b) { return a + b; };
  X_train.setFromTriplets(train_data.begin(), train_data.end(), dupfunction);
  X_test.setFromTriplets(test_data.begin(), test_data.end(), dupfunction);
  return {X_train, X_test};
}

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

Real learn_dirichlet_symmetric(const Eigen::Ref<IntegerMatrix> &counts,
                               Real alpha_start, Real alpha_prior_scale,
                               Real alpha_prior_exponent, size_t iteration) {
  using Eigen::numext::digamma;
  using std::vector;
  const int n_topic = counts.cols();
  const int n_docs = counts.rows();
  if (counts.cols() != n_topic) {
    throw std::invalid_argument("count array and alpha have different sizes.");
  }
  Real alpha_current(alpha_start);
  Real numerator;

  vector<Real> doc_length;
  vector<Real> doc_length_freq;

  vector<Real> topic_cnt;
  vector<Real> topic_cnt_freq;

  {
    using Map = std::unordered_map<Integer, Real>; // cnt, freq
    Map doc_length_hist;
    Map topic_cnt_hist;

    for (int dix = 0; dix < n_docs; dix++) {
      int length = 0;
      for (int topic = 0; topic < n_topic; topic++) {
        Integer cnt = counts(dix, topic);
        if (cnt == 0) {
          continue;
        }
        topic_cnt_hist[cnt]++;
        length += cnt;
      }
      doc_length_hist[length]++;
    }

    for (auto &cnt_freq : topic_cnt_hist) {
      topic_cnt.push_back(cnt_freq.first);
      topic_cnt_freq.push_back(cnt_freq.second);
    }

    for (auto &iter : doc_length_hist) {
      doc_length.push_back(iter.first);
      doc_length_freq.push_back(iter.second);
    }
  }
  for (size_t it = 0; it < iteration; it++) {
    Real alpha_sum = n_topic * alpha_current;
    numerator = 0;
    Real denominator =
        ((vector_to_eigen(doc_length).array() + alpha_sum).digamma() -
         digamma(alpha_sum))
            .matrix()
            .transpose() *
        vector_to_eigen(doc_length_freq);
    Real numerator =
        ((vector_to_eigen(topic_cnt).array() + alpha_current).digamma() -
         digamma(alpha_current))
            .matrix()
            .transpose() *
        vector_to_eigen(topic_cnt_freq);
    alpha_current = (alpha_current * numerator + alpha_prior_exponent) /
                    (denominator + alpha_prior_scale) / n_topic;
  }

  return alpha_current;
}

Real log_likelihood_doc_topic(const Eigen::Ref<RealVector> &doc_topic_prior,
                              const Eigen::Ref<IntegerMatrix> &doc_topic,
                              const Eigen::Ref<IntegerVector> &doc_length) {
  size_t n_doc = doc_topic.rows();
  Real ll = n_doc * (std::lgamma(doc_topic_prior.sum()) -
                     doc_topic_prior.array().lgamma().sum());
  Real doc_topic_sum = doc_topic_prior.array().sum();
  ll += (doc_topic.array().cast<Real>().rowwise() +
         doc_topic_prior.transpose().array())
            .lgamma()
            .sum();
  ll -= (doc_length.array().cast<Real>() + doc_topic_sum).lgamma().sum();
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
  m.def("learn_dirichlet_symmetric", &learn_dirichlet_symmetric);

  m.def("train_test_split", &train_test_split);

  py::class_<Predictor>(m, "Predictor")
      .def(py::init<size_t, const RealVector &, int>())
      .def("add_beta", &Predictor::add_beta)
      .def("predict_gibbs", &Predictor::predict_gibbs)
      .def("predict_gibbs_batch", &Predictor::predict_gibbs_batch)
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
