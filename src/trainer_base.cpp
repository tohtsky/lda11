#include <cstddef>
#include <future>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "defs.hpp"
#include "trainer_base.hpp"
#include "util.hpp"

LDATrainerBase::LDATrainerBase(Eigen::Ref<IntegerVector> counts,
                               Eigen::Ref<IndexVector> dixs,
                               Eigen::Ref<IndexVector> wixs, size_t n_topics,
                               int random_seed, size_t n_workers)
    : word_states(), n_topics_(n_topics), urand_(random_seed) {

  if (counts.rows() != dixs.rows() || dixs.cols() != wixs.cols()) {
    throw std::runtime_error("Shape Mismatch of indices");
  }
  size_t n_ = counts.rows();

  if (n_workers == 0) {
    throw std::invalid_argument("n_workers has to be strictly positive.");
  } else if (n_workers == 1) {
    for (size_t i = 0; i < n_; i++) {
      size_t c = counts(i);
      size_t dix = dixs(i);
      size_t wix = wixs(i);

      for (size_t j = 0; j < c; j++) {
        word_states.emplace_back(dix, wix, 0);
      }
    }
  } else {
    std::unordered_map<size_t, size_t> dix_seen;
    for (size_t w = 0; w < n_workers; w++) {
      children.emplace_back(new ChildWorker(this, n_topics, random_seed + w));
    }
    for (size_t i = 0; i < n_; i++) {
      size_t c = counts(i);
      size_t dix = dixs(i);
      size_t wix = wixs(i);
      size_t assigned;
      if (dix_seen.find(dix) == dix_seen.end()) {
        assigned = std::floor(urand_.rand() * n_workers);
        assigned = assigned < n_workers - 1 ? assigned : n_workers - 1;
        dix_seen.insert({dix, assigned});
        children[assigned]->add_doc(dix);
      } else {
        assigned = dix_seen[dix];
      }
      children[assigned]->add_word(dix, wix, c);
    }
  }
}

LDATrainerBase::~LDATrainerBase() {}

void LDATrainerBase::initialize_count(Eigen::Ref<IntegerMatrix> word_topic,
                                      Eigen::Ref<IntegerMatrix> doc_topic,
                                      Eigen::Ref<IntegerVector> topic_counts) {

  if (children.empty()) {
    RealVector temp_p(n_topics_);
    for (auto &ws : word_states) {
      temp_p = obtain_doc_topic_prior(ws.doc_id);
      ws.topic_id = draw_from_p(temp_p, urand_);
      doc_topic(ws.doc_id, ws.topic_id)++;
      word_topic(ws.word_id, ws.topic_id)++;
      topic_counts(ws.topic_id)++;
    }
  } else {
    for (auto &child : children) {
      child->initialize_count(word_topic.rows());
      child->add_count(word_topic, doc_topic, topic_counts);
    }
  }
}

void LDATrainerBase::iterate_gibbs(Eigen::Ref<RealVector> topic_word_prior,
                                   Eigen::Ref<IntegerMatrix> doc_topic,
                                   Eigen::Ref<IntegerMatrix> word_topic,
                                   Eigen::Ref<IntegerVector> topic_counts) {

  if (children.empty()) {
    Real eta_sum = topic_word_prior.sum();
    RealVector p_(n_topics_);

    for (auto &ws : word_states) {
      doc_topic(ws.doc_id, ws.topic_id)--;
      word_topic(ws.word_id, ws.topic_id)--;
      topic_counts(ws.topic_id)--;

      p_ = (word_topic.row(ws.word_id).cast<Real>().transpose().array() +
            topic_word_prior.array())
               .array() /
           (topic_counts.cast<Real>().array() + eta_sum) *
           (doc_topic.row(ws.doc_id).cast<Real>().transpose().array() +
            obtain_doc_topic_prior(ws.doc_id).array());

      ws.topic_id = draw_from_p(p_, urand_);

      doc_topic(ws.doc_id, ws.topic_id)++;
      word_topic(ws.word_id, ws.topic_id)++;
      topic_counts(ws.topic_id)++;
    }
  } else {
    std::vector<std::thread> workers;
    for (auto &child : children) {
      workers.emplace_back([&child, &word_topic, &doc_topic, &topic_counts,
                            &topic_word_prior] {
        child->do_work(word_topic, doc_topic, topic_counts, topic_word_prior);
      });
    }
    for (auto &th : workers) {
      th.join();
    }
  }
}

RealMatrix
LDATrainerBase::obtain_phi(const Eigen::Ref<RealVector> &topic_word_prior,
                           Eigen::Ref<IntegerMatrix> doc_topic,
                           Eigen::Ref<IntegerMatrix> word_topic,
                           Eigen::Ref<IntegerVector> topic_counts) {

  RealMatrix result(n_topics_, word_topic.rows());
  result.array() = 0;
  if (children.empty()) {
    Real eta_sum = topic_word_prior.sum();
    RealVector p_(n_topics_);

    for (auto &ws : word_states) {
      doc_topic(ws.doc_id, ws.topic_id)--;
      word_topic(ws.word_id, ws.topic_id)--;
      topic_counts(ws.topic_id)--;
      p_ = (word_topic.row(ws.word_id).cast<Real>().transpose().array() +
            topic_word_prior.array())
               .array() /
           (topic_counts.cast<Real>().array() + eta_sum) *
           (doc_topic.row(ws.doc_id).cast<Real>().transpose().array() +
            obtain_doc_topic_prior(ws.doc_id).array());

      p_.array() /= p_.sum();
      doc_topic(ws.doc_id, ws.topic_id)++;
      word_topic(ws.word_id, ws.topic_id)++;
      topic_counts(ws.topic_id)++;
      result.col(ws.word_id) += p_;
    }
  } else {
    std::vector<std::future<RealMatrix>> phi_locals;
    for (auto &child : children) {
      child->sync_topic(word_topic, doc_topic, topic_counts);
      phi_locals.emplace_back(
          std::async(std::launch::async, [&child, &topic_word_prior] {
            return child->obtain_phi(topic_word_prior);
          }));
    }
    for (auto &phi_local_future : phi_locals) {
      result += phi_local_future.get();
    }
  }

  result.rowwise() += topic_word_prior.transpose();
  result.array().colwise() /= result.array().rowwise().sum();
  return result;
}

Real LDATrainerBase::log_likelihood(Eigen::Ref<RealVector> topic_word_prior,
                                    Eigen::Ref<IntegerMatrix> word_topic) {
  size_t n_words = word_topic.rows();
  Real ll = n_topics_ * (std::lgamma(topic_word_prior.sum()) -
                         topic_word_prior.array().lgamma().sum());

  RealArray exponent(n_words);
  for (size_t j = 0; j < n_topics_; j++) {
    exponent =
        word_topic.col(j).cast<Real>().array() + topic_word_prior.array();
    ll += exponent.lgamma().sum();
    ll -= std::lgamma(exponent.sum());
  }

  return ll;
}
