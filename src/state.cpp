#include "state.hpp"
#include "defs.hpp"
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

const RealVector & LabelledLDATrainer::obtain_doc_topic_prior(size_t doc_index) {
  return (labels_.row(doc_index).cast<Real>().array() * alpha_ + epsilon_)
               .transpose();
}
