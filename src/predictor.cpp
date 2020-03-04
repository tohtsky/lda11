#include "predictor.hpp"
#include "util.hpp"
Predictor::Predictor(size_t n_topics, const RealVector &doc_topic_prior,
                     int random_seed)
    : betas_(), n_topics_(n_topics), doc_topic_prior_(doc_topic_prior),
      n_domains_(0) {}

void Predictor::add_beta(const RealMatrix &beta) {
  betas_.push_back(beta);
  n_domains_++;
}

RealVector Predictor::predict_mf(std::vector<IntegerVector> nonzeros,
                                 std::vector<IntegerVector> counts,
                                 std::size_t iter, Real delta) {
  size_t dim_buffer = 0;
  for (size_t n = 0; n < n_domains_; n++) {
    dim_buffer += counts[n].sum();
  }
  if (dim_buffer == 0) {
    return doc_topic_prior_ / doc_topic_prior_.sum();
  }
  RealMatrix current_prob(dim_buffer, n_topics_);
  current_prob.array() = 0;
  RealMatrix new_prob(dim_buffer, n_topics_);
  RealMatrix beta_rel(dim_buffer, n_topics_);

  size_t current_iter = 0;
  for (size_t n = 0; n < n_domains_; n++) {
    size_t n_unique_words = nonzeros[n].rows();
    for (size_t j = 0; j < n_unique_words; j++) {
      size_t wid = nonzeros[n](j);
      size_t count = counts[n][j];
      for (size_t k = 0; k < count; k++) {
        beta_rel.row(current_iter) = betas_[n].row(wid);
        current_iter++;
      }
    }
  }

  for (size_t i = 0; i <= iter; i++) {
    new_prob = -current_prob;
    new_prob.rowwise() += current_prob.colwise().sum();
    new_prob.rowwise() += doc_topic_prior_.transpose();
    new_prob.array() = new_prob.array() * beta_rel.array();
    new_prob.array().colwise() /= new_prob.array().rowwise().sum();
    double diff = (new_prob - current_prob).array().abs().sum();
    current_prob = new_prob;
    if (diff < delta)
      break;
  }
  RealVector theta = current_prob.array().colwise().sum().transpose();
  theta /= theta.sum();
  return theta;
}

RealVector Predictor::predict_gibbs(std::vector<IntegerVector> nonzeros,
                                    std::vector<IntegerVector> counts,
                                    std::size_t max_iter, std::size_t burn_in,
                                    int random_seed) {
  if (burn_in >= max_iter) {
    throw std::invalid_argument("max_iter must be larger than burn_in.");
  }
  // std::vector<Real> p_(n_topics_);
  IntegerVector current_state(n_topics_);
  current_state.array() = 0;
  RealVector result(n_topics_);
  result.array() = 0;

  UrandDevice urand_(random_seed);
  RealVector p_(n_topics_);
  RealVector p_temp(n_topics_);
  std::vector<size_t> topics;
  for (size_t n = 0; n < n_domains_; n++) {
    IntegerVector &count = counts[n];
    size_t n_unique_word = nonzeros[n].rows();
    for (size_t j = 0; j < n_unique_word; j++) {
      for (int k = 0; k < count(j); k++) {
        p_ = doc_topic_prior();
        size_t init_topic = draw_from_p(p_, urand_);
        topics.push_back(init_topic);
        current_state(init_topic)++;
      }
    }
  }
  for (size_t iter_ = 0; iter_ < max_iter; iter_++) {
    size_t current_iter = 0;
    for (size_t n = 0; n < n_domains_; n++) {
      size_t n_unique_word = nonzeros[n].rows();
      for (size_t j = 0; j < n_unique_word; j++) {
        size_t wid = nonzeros[n](j);
        size_t wcount = counts[n](j);
        for (size_t k = 0; k < wcount; k++) {
          size_t current_topic = topics[current_iter];

          current_state(current_topic)--;

          p_ =
              (betas_[n].row(wid).transpose().array() *
               (current_state.cast<Real>().array() + doc_topic_prior_.array()));
          if (iter_ >= burn_in) {
            p_temp.array() = p_.array() / p_.sum();
          }

          current_topic = draw_from_p(p_, urand_);

          current_state(current_topic)++;
          topics[current_iter] = current_topic;
          if (iter_ >= burn_in) {
            // result(current_topic) ++;
            result += p_temp;
          }

          current_iter++;
        }
      }
    }
  }
  result.array() /= (max_iter - burn_in);
  result += doc_topic_prior_;
  result.array() /= result.array().sum();
  return result;
}
