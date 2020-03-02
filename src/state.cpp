#include "state.hpp"
#include "iostream"
#include <stdexcept>

LabelledLDATrainer::LabelledLDATrainer(Real alpha, Real epsilon,
                                       const IntegerMatrix &labels,
                                       Eigen::Ref<IntegerVector> counts,
                                       Eigen::Ref<IndexVector> dixs,
                                       Eigen::Ref<IndexVector> wixs,
                                       size_t n_topics, int random_seed)
    : LDATrainerBase(counts, dixs, wixs, n_topics, random_seed), alpha_(alpha),
      epsilon_(epsilon), labels_(labels) {}

const RealVector &LabelledLDATrainer::doc_topic_prior(size_t doc_index) {
  return (labels_.row(doc_index).cast<Real>().array() * alpha_ + epsilon_)
      .transpose();
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
