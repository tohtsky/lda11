#pragma once
#include "defs.hpp"
#include "trainer_base.hpp"

struct LDATrainer : LDATrainerBase {
  LDATrainer(const RealVector &doc_topic_prior,
             Eigen::Ref<IntegerVector> counts, Eigen::Ref<IndexVector> dixs,
             Eigen::Ref<IndexVector> wixs, size_t n_topics,
             int random_seed = 42, size_t n_workers = 1);

  void set_doc_topic_prior(const Eigen::Ref<RealVector> &new_dtp);

private:
  struct ChildWorker {
    ChildWorker(LDATrainer *parent, int random_seed);
    void set_word_topic(Eigen::Ref<IntegerMatrix> word_topic_global);

    std::vector<size_t> original_incides;
    IntegerMatrix doc_topic;
    IntegerMatrix word_topic_local;

    int random_seed;
  };

  virtual const RealVector & obtain_doc_topic_prior(size_t doc_index) override;
  std::vector<std::unique_ptr<ChildWorker>> children;

private:
  RealVector doc_topic_prior_;
};
