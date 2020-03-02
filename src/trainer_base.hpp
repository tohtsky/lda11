#pragma once
#include "defs.hpp"

struct LDATrainerBase {
  LDATrainerBase(Eigen::Ref<IntegerVector> counts, Eigen::Ref<IndexVector> dixs,
                 Eigen::Ref<IndexVector> wixs, size_t n_topics,
                 int random_seed = 42);

  void initialize_count(Eigen::Ref<IntegerMatrix> doc_topic,
                        Eigen::Ref<IntegerMatrix> word_topic);

  void iterate_gibbs(Eigen::Ref<RealVector> topic_word_prior,
                     Eigen::Ref<IntegerMatrix> doc_topic,
                     Eigen::Ref<IntegerMatrix> word_topic,
                     Eigen::Ref<IntegerVector> topic_counts);

  virtual const RealVector &doc_topic_prior(size_t doc_index) = 0;

  Real log_likelihood(Eigen::Ref<RealVector> topic_word_prior,
                      Eigen::Ref<IntegerMatrix> word_topic);

  RealMatrix obtain_phi(Eigen::Ref<RealVector> topic_word_prior,
                        Eigen::Ref<IntegerMatrix> doc_topic,
                        Eigen::Ref<IntegerMatrix> word_topic,
                        Eigen::Ref<IntegerVector> topic_counts);

protected:
  std::vector<WordState> word_states;
  const std::size_t n_topics_;

  std::mt19937 random_state_;
  UrandDevice urand_;
};
