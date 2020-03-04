#include "trainer.hpp"

LDATrainer::LDATrainer(const RealVector &doc_topic_prior,
                       Eigen::Ref<IntegerVector> counts,
                       Eigen::Ref<IndexVector> dixs,
                       Eigen::Ref<IndexVector> wixs, size_t n_topics,
                       int random_seed, size_t n_workers)
    : LDATrainerBase(counts, dixs, wixs, n_topics, random_seed, n_workers),
      doc_topic_prior_(doc_topic_prior) {}

const RealVector & LDATrainer::obtain_doc_topic_prior(std::size_t doc_index) {
  return doc_topic_prior_;
}

void LDATrainer::set_doc_topic_prior(const Eigen::Ref<RealVector> &new_dtp) {
  if (static_cast<size_t>(new_dtp.rows()) != n_topics_) {
    throw std::invalid_argument("Topic size mismatch.");
  }
  doc_topic_prior_ = new_dtp;
}
