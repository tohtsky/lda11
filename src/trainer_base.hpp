#pragma once
#include "defs.hpp"
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

struct LDATrainerBase {
  LDATrainerBase(Eigen::Ref<IntegerVector> counts, Eigen::Ref<IndexVector> dixs,
                 Eigen::Ref<IndexVector> wixs, size_t n_topics,
                 int random_seed = 42, size_t n_workers = 1);
  virtual ~LDATrainerBase();

  void initialize_count(Eigen::Ref<IntegerMatrix> word_topic,
                        Eigen::Ref<IntegerMatrix> doc_topic,
                        Eigen::Ref<IntegerVector> topic_counts);

  void iterate_gibbs(Eigen::Ref<RealVector> topic_word_prior,
                     Eigen::Ref<IntegerMatrix> doc_topic,
                     Eigen::Ref<IntegerMatrix> word_topic,
                     Eigen::Ref<IntegerVector> topic_counts);

  virtual const RealVector &obtain_doc_topic_prior(size_t doc_index) = 0;

  Real log_likelihood(Eigen::Ref<RealVector> topic_word_prior,
                      Eigen::Ref<IntegerMatrix> word_topic);

  RealMatrix obtain_phi(const Eigen::Ref<RealVector> &topic_word_prior,
                        Eigen::Ref<IntegerMatrix> doc_topic,
                        Eigen::Ref<IntegerMatrix> word_topic,
                        Eigen::Ref<IntegerVector> topic_counts);
  struct ChildWorker {
    ChildWorker(LDATrainerBase *parent, size_t n_topics, int random_seed);
    void add_doc(size_t dix_original);
    void add_word(size_t global_dix, size_t wix, size_t count);
    void initialize_count(size_t n_words);
    void sync_topic(const Eigen::Ref<IntegerMatrix> &word_topic_global,
                    const Eigen::Ref<IntegerMatrix> &doc_topic_global,
                    const Eigen::Ref<IntegerVector> &topic_counts);
    void decr_count(Eigen::Ref<IntegerMatrix> word_topic_global,
                    Eigen::Ref<IntegerMatrix> doc_topic_global,
                    Eigen::Ref<IntegerVector> topic_counts);
    void add_count(Eigen::Ref<IntegerMatrix> word_topic_global,
                   Eigen::Ref<IntegerMatrix> doc_topic_global,
                   Eigen::Ref<IntegerVector> topic_counts);

    void do_work(Eigen::Ref<IntegerMatrix> word_topic,
                 Eigen::Ref<IntegerMatrix> doc_topic,
                 Eigen::Ref<IntegerVector> topic_counts,
                 const Eigen::Ref<RealVector> &topic_word_prior);
    RealMatrix obtain_phi(const Eigen::Ref<RealVector> &topic_word_prior);
    LDATrainerBase *parent_;
    size_t n_topics_;
    UrandDevice urand_;

    std::vector<WordState> word_states_local;
    IntegerMatrix doc_topic_local;
    IntegerMatrix word_topic_local;
    IntegerVector topic_counts_local;

    std::vector<size_t> global_indices;
    std::unordered_map<size_t, size_t> dix_to_internal_index;
  };

protected:
  std::vector<std::unique_ptr<ChildWorker>> children;
  std::vector<WordState> word_states;
  const std::size_t n_topics_;

  std::mt19937 random_state_;
  UrandDevice urand_;
  std::mutex mutex_;
};
