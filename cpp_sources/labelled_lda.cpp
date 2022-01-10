#include "iostream"
#include <stdexcept>

#include "defs.hpp"
#include "labelled_lda.hpp"

LabelledLDATrainer::LabelledLDATrainer(Real alpha, Real epsilon,
                                       const SparseIntegerMatrix &labels,
                                       Eigen::Ref<IntegerVector> counts,
                                       Eigen::Ref<IndexVector> dixs,
                                       Eigen::Ref<IndexVector> wixs,
                                       size_t n_topics, int random_seed, size_t n_workers)
    : LDATrainerBase(counts, dixs, wixs, n_topics, random_seed, n_workers), alpha_(alpha),
      epsilon_(epsilon), labels_(labels), alpha_hat( labels.cast<Real>().transpose() ) {
        alpha_hat.array() *= alpha_;
        alpha_hat.array() += epsilon_;
      }

Eigen::Ref<RealVector>
LabelledLDATrainer::obtain_doc_topic_prior(size_t doc_index) {
  return alpha_hat.col(doc_index);
}
