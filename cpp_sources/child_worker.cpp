#include "defs.hpp"
#include "trainer_base.hpp"
#include "util.hpp"
#include <cstddef>
#include <mutex>

LDATrainerBase::ChildWorker::ChildWorker(LDATrainerBase *parent,
                                         size_t n_topics, int random_seed)
    : parent_(parent), n_topics_(n_topics), urand_(random_seed) {}

void LDATrainerBase::ChildWorker::add_doc(size_t dix) {
  dix_to_internal_index.insert({dix, global_indices.size()});
  global_indices.push_back(dix);
}

void LDATrainerBase::ChildWorker::add_word(size_t global_dix, size_t wix,
                                           size_t count) {
  size_t local_index = dix_to_internal_index.at(global_dix);
  for (size_t _ = 0; _ < count; _++) {
    word_states_local.emplace_back(local_index, wix, 0);
  }
}

void LDATrainerBase::ChildWorker::initialize_count(size_t n_words) {
  RealVector temp_p(n_topics_);
  doc_topic_local = IntegerMatrix{global_indices.size(), n_topics_};
  word_topic_local = IntegerMatrix{n_words, n_topics_};
  topic_counts_local = IntegerVector{n_topics_};

  doc_topic_local.array() = 0;
  word_topic_local.array() = 0;
  topic_counts_local.array() = 0;
  for (auto &ws : word_states_local) {
    temp_p = parent_->obtain_doc_topic_prior(ws.doc_id);
    ws.topic_id = draw_from_p(temp_p, urand_);
    doc_topic_local(ws.doc_id, ws.topic_id)++;
    word_topic_local(ws.word_id, ws.topic_id)++;
    topic_counts_local(ws.topic_id)++;
  }
}

void LDATrainerBase::ChildWorker::decr_count(
    Eigen::Ref<IntegerMatrix> word_topic_global,
    Eigen::Ref<IntegerMatrix> doc_topic_global,
    Eigen::Ref<IntegerVector> topic_counts) {
  {
    std::lock_guard<std::mutex> lock(parent_->mutex_);
    word_topic_global -= word_topic_local;
    topic_counts -= topic_counts_local;
  }
  for (auto &ws : word_states_local) {
    size_t dix_global = global_indices[ws.doc_id];
    doc_topic_global(dix_global, ws.topic_id)--;
  }
}

void LDATrainerBase::ChildWorker::add_count(
    Eigen::Ref<IntegerMatrix> word_topic_global,
    Eigen::Ref<IntegerMatrix> doc_topic_global,
    Eigen::Ref<IntegerVector> topic_counts) {
  {
    std::lock_guard<std::mutex> lock(parent_->mutex_);
    word_topic_global += word_topic_local;
    topic_counts += topic_counts_local;
  }
  for (auto &ws : word_states_local) {
    size_t dix_global = global_indices[ws.doc_id];
    doc_topic_global(dix_global, ws.topic_id)++;
  }
}

void LDATrainerBase::ChildWorker::sync_topic(
    const Eigen::Ref<IntegerMatrix> &word_topic_global,
    const Eigen::Ref<IntegerMatrix> &doc_topic_global,
    const Eigen::Ref<IntegerVector> &topic_counts) {
  word_topic_local = word_topic_global;
  topic_counts_local = topic_counts;
  for (size_t internal_dix = 0; internal_dix < global_indices.size();
       internal_dix++) {
    size_t global_dix = global_indices[internal_dix];
    doc_topic_local.row(internal_dix) = doc_topic_global.row(global_dix);
  }
}

void LDATrainerBase::ChildWorker::do_work(
    Eigen::Ref<IntegerMatrix> word_topic, Eigen::Ref<IntegerMatrix> doc_topic,
    Eigen::Ref<IntegerVector> topic_counts,
    const Eigen::Ref<RealVector> &topic_word_prior) {
  this->sync_topic(word_topic, doc_topic, topic_counts);
  this->decr_count(word_topic, doc_topic, topic_counts);
  Real eta_sum = topic_word_prior.sum();
  RealVector p_(n_topics_);

  for (auto &ws : word_states_local) {
    doc_topic_local(ws.doc_id, ws.topic_id)--;
    word_topic_local(ws.word_id, ws.topic_id)--;
    topic_counts_local(ws.topic_id)--;
    size_t global_dix = global_indices[ws.doc_id];

    p_ = (word_topic_local.row(ws.word_id).cast<Real>().transpose().array() +
          topic_word_prior.array())
             .array() /
         (topic_counts_local.cast<Real>().array() + eta_sum) *
         (doc_topic_local.row(ws.doc_id).cast<Real>().transpose().array() +
          parent_->obtain_doc_topic_prior(global_dix).array());

    ws.topic_id = draw_from_p(p_, urand_);

    doc_topic_local(ws.doc_id, ws.topic_id)++;
    word_topic_local(ws.word_id, ws.topic_id)++;
    topic_counts_local(ws.topic_id)++;
  }
  this->add_count(word_topic, doc_topic, topic_counts);
}

RealMatrix LDATrainerBase::ChildWorker::obtain_phi(
    const Eigen::Ref<RealVector> &topic_word_prior) {
  Real eta_sum = topic_word_prior.sum();
  RealVector p_(n_topics_);

  RealMatrix result(n_topics_, topic_word_prior.rows());
  result.array() = 0;
  for (auto &ws : word_states_local) {
    doc_topic_local(ws.doc_id, ws.topic_id)--;
    word_topic_local(ws.word_id, ws.topic_id)--;
    topic_counts_local(ws.topic_id)--;
    size_t global_dix = global_indices[ws.doc_id];
    p_ = (word_topic_local.row(ws.word_id).cast<Real>().transpose().array() +
          topic_word_prior.array())
             .array() /
         (topic_counts_local.cast<Real>().array() + eta_sum) *
         (doc_topic_local.row(ws.doc_id).cast<Real>().transpose().array() +
          parent_->obtain_doc_topic_prior(global_dix).array());

    p_.array() /= p_.sum();
    doc_topic_local(ws.doc_id, ws.topic_id)++;
    word_topic_local(ws.word_id, ws.topic_id)++;
    topic_counts_local(ws.topic_id)++;
    result.col(ws.word_id) += p_;
  }
  return result;
}
