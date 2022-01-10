#pragma once
#include "defs.hpp"
#include <tuple>

struct Predictor {
  Predictor(size_t n_topics, const RealVector &doc_topic_prior,
            int random_seed = 42);

  void add_beta(const RealMatrix &beta);

  RealMatrix predict_mf_batch(std::vector<SparseIntegerMatrix> Xs,
                              std::size_t iter, Real delta,
                              size_t n_workers) const;

  std::pair<RealVector, std::vector<std::map<size_t, IntegerVector>>>
  predict_gibbs_with_word_assignment(std::vector<IntegerVector> nonzeros,
                                     std::vector<IntegerVector> counts,
                                     std::size_t iter, std::size_t burn_in,
                                     int random_seed = 42,
                                     bool use_cgs_p = true);

  RealVector predict_gibbs(const std::vector<IntegerVector> &nonzeros,
                           const std::vector<IntegerVector> &counts,
                           std::size_t iter, std::size_t burn_in,
                           int random_seed = 42, bool use_cgs_p = true);

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
  RealVector predict_gibbs_write_assignment(
      const std::vector<IntegerVector> &nonzeros,
      const std::vector<IntegerVector> &counts, std::size_t iter,
      std::size_t burn_in, int random_seed = 42, bool use_cgs_p = true,
      std::vector<std::map<size_t, IntegerVector>> *cnt_target = nullptr);
  const std::size_t n_topics_;
  RealVector doc_topic_prior_;
  std::size_t n_domains_;
};
