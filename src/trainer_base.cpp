#include "trainer_base.hpp"
#include "util.hpp"

LDATrainerBase::LDATrainerBase(Eigen::Ref<IntegerVector> counts,
                               Eigen::Ref<IndexVector> dixs,
                               Eigen::Ref<IndexVector> wixs, size_t n_topics,
                               int random_seed)
    : word_states(), n_topics_(n_topics), urand_(random_seed) {
  if (counts.rows() != dixs.rows() || dixs.cols() != wixs.cols())
    throw std::runtime_error("");
  size_t n_ = counts.rows();
  for (size_t i = 0; i < n_; i++) {
    size_t c = counts(i);
    size_t dix = dixs(i);
    size_t wix = wixs(i);

    for (size_t j = 0; j < c; j++) {
      word_states.emplace_back(dix, wix, 0);
    }
  }
}

void LDATrainerBase::initialize_count(Eigen::Ref<IntegerMatrix> doc_topic,
                                      Eigen::Ref<IntegerMatrix> word_topic) {
  RealVector temp_p(n_topics_);
  for (auto &ws : word_states) {
    temp_p = doc_topic_prior(ws.doc_id);
    ws.topic_id = draw_from_p(temp_p, urand_);
    doc_topic(ws.doc_id, ws.topic_id)++;
    word_topic(ws.word_id, ws.topic_id)++;
  }
}

void LDATrainerBase::iterate_gibbs(Eigen::Ref<RealVector> topic_word_prior,
                                   Eigen::Ref<IntegerMatrix> doc_topic,
                                   Eigen::Ref<IntegerMatrix> word_topic,
                                   Eigen::Ref<IntegerVector> topic_counts) {

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
          doc_topic_prior(ws.doc_id).array());

    ws.topic_id = draw_from_p(p_, urand_);

    doc_topic(ws.doc_id, ws.topic_id)++;
    word_topic(ws.word_id, ws.topic_id)++;
    topic_counts(ws.topic_id)++;
  }
}

RealMatrix LDATrainerBase::obtain_phi(Eigen::Ref<RealVector> topic_word_prior,
                                      Eigen::Ref<IntegerMatrix> doc_topic,
                                      Eigen::Ref<IntegerMatrix> word_topic,
                                      Eigen::Ref<IntegerVector> topic_counts) {

  RealMatrix result(n_topics_, word_topic.rows());
  for (size_t i = 0; i <= n_topics_; i++) {
    result.row(i) = topic_word_prior.transpose();
  }
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
          doc_topic_prior(ws.doc_id).array());
    p_.array() /= p_.sum();
    doc_topic(ws.doc_id, ws.topic_id)++;
    word_topic(ws.word_id, ws.topic_id)++;
    topic_counts(ws.topic_id)++;
    result.col(ws.word_id) += p_;
  }
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
