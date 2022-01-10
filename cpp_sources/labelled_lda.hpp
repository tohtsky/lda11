#pragma once
#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <vector>

#include "defs.hpp"
#include "trainer_base.hpp"

struct LabelledLDATrainer : LDATrainerBase {
  LabelledLDATrainer(Real alpha, Real epsilon, const SparseIntegerMatrix &labels,
                     Eigen::Ref<IntegerVector> counts,
                     Eigen::Ref<IndexVector> dixs, Eigen::Ref<IndexVector> wixs,
                     size_t n_topics, int random_seed = 42, size_t n_workers=1);

  virtual Eigen::Ref<RealVector> obtain_doc_topic_prior(size_t doc_index) override;

private:
  Real alpha_, epsilon_;
  SparseIntegerMatrix labels_;
  Eigen::Matrix<Real, -1, -1, Eigen::ColMajor> alpha_hat;
};
