#pragma once
#include "defs.hpp"

struct Predictor {
  Predictor(size_t n_topics, const RealVector &doc_topic_prior,
            int random_seed = 42);

  void add_beta(const RealMatrix &beta);

  RealVector predict_mf(std::vector<IntegerVector> nonzeros,
                        std::vector<IntegerVector> counts, std::size_t iter,
                        Real delta);

  RealVector predict_gibbs(std::vector<IntegerVector> nonzeros,
                           std::vector<IntegerVector> counts, std::size_t iter,
                           std::size_t burn_in, int random_seed = 42,
                           bool use_cgs_p = true);

  RealMatrix predict_gibbs_batch(std::vector<SparseIntegerMatrix> Xs,
                                 std::size_t iter, std::size_t burn_in,
                                 int random_seed = 42, bool use_cgs_p = true,
                                 size_t n_workers = 1);

  const RealVector &doc_topic_prior() const { return doc_topic_prior_; }

  inline std::vector<RealMatrix>::const_iterator beta_begin() const {
    return betas_.cbegin();
  }

  inline std::vector<RealMatrix>::const_iterator beta_end() const {
    return betas_.cend();
  }

  size_t n_topics() const { return n_topics_; }

  std::vector<RealMatrix> betas_;

private:
  const std::size_t n_topics_;
  RealVector doc_topic_prior_;
  std::size_t n_domains_;
};
