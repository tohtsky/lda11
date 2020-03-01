#include "./state.hpp"
#include "iostream"

inline size_t binary_search(const Real* array, size_t size, Real value ) {
    int lower = 0, upper = size -1; 
    int half = 0;
    int idx = -1;
    while (upper >= lower){
        half = lower + (upper - lower) / 2;
        double trial = array[half];
        if (value == trial) {
            idx = half;
            break;
        }
        else if ( value > trial ) {
            lower = half + 1;
        } else {
            upper = half - 1;
        }
    }
    if (idx == - 1) // Element not found, return where it should be
        return static_cast<size_t>(lower);

    return static_cast<size_t>(idx);

}

inline size_t binary_search(const std::vector<Real> & array, Real value) {
    return binary_search(array.data(), array.size(), value);
}

inline size_t binary_search(const RealVector & array, Real value) {
    return binary_search(array.data(), array.size(), value);
}

inline Real cumsum(std::vector<Real> & v) {
    auto n = v.size();
    Real q = 0;
    for(size_t i = 0; i < n ; i++ ) {
        q = (v[i] += q);
    };
    return q;
}

inline Real cumsum(RealVector& v) {
    auto n = v.rows();
    Real q = 0;
    for(int i = 0; i < n; i++ ) {
        q = (v(i) += q);
    };
    return q;
}

template<class ArrayType>
inline size_t draw_from_p(ArrayType& array, UrandDevice & rand) {
    Real max = cumsum(array);
    return binary_search(array, max * rand.rand());
}


LDATrainerBase::LDATrainerBase (
        Eigen::Ref<IntegerVector> counts,
        Eigen::Ref<IndexVector> dixs,
        Eigen::Ref<IndexVector> wixs,
        size_t n_topics,
        int random_seed
): word_states(), n_topics_(n_topics), urand_(random_seed){
    if (counts.rows() != dixs.rows() || dixs.cols() != wixs.cols() ) 
        throw std::runtime_error("");
    size_t n_ = counts.rows();
    for (size_t i = 0; i < n_; i++) {
        size_t c = counts(i);
        size_t dix = dixs(i);
        size_t wix = wixs(i);

        for (size_t j = 0; j < c; j++ ) {
            word_states.emplace_back(dix, wix, 0);
        }
    }
}

void LDATrainerBase::initialize_count(
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic
) {
    RealVector temp_p(n_topics_);
    for (auto & ws: word_states) {
        temp_p = doc_topic_prior(ws.doc_id);
        ws.topic_id = draw_from_p(temp_p, urand_);
        doc_topic(ws.doc_id, ws.topic_id)++;
        word_topic(ws.word_id, ws.topic_id)++;
    }
}

void LDATrainerBase::iterate_gibbs(
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic,
        Eigen::Ref<IntegerVector> topic_counts,
        std::function<void(double)> logger
        ) {

    Real eta_sum = topic_word_prior.sum();
    RealVector p_(n_topics_);

    for (auto & ws : word_states ) {
        doc_topic(ws.doc_id, ws.topic_id)--;
        word_topic(ws.word_id, ws.topic_id)--;
        topic_counts(ws.topic_id)--;

        p_ = (
            word_topic.row(ws.word_id).cast<Real>().transpose().array() + topic_word_prior.array()
        ).array() / ( topic_counts.cast<Real>().array() + eta_sum)  * (
            doc_topic.row(ws.doc_id).cast<Real>().transpose().array() +
            doc_topic_prior(ws.doc_id).array()
        );

        ws.topic_id = draw_from_p(p_, urand_);

        doc_topic(ws.doc_id, ws.topic_id)++;
        word_topic(ws.word_id, ws.topic_id)++; 
        topic_counts(ws.topic_id)++;
    }
    logger(urand_.rand()); 

}

RealMatrix LDATrainerBase::obtain_phi(
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic,
        Eigen::Ref<IntegerVector> topic_counts 
        ) {

    RealMatrix result(n_topics_, word_topic.rows());
    for(size_t i=0; i<=n_topics_;i++) {
        result.row(i) = topic_word_prior.transpose();
    }
    Real eta_sum = topic_word_prior.sum();
    RealVector p_(n_topics_);

    for (auto & ws : word_states ) {
        doc_topic(ws.doc_id, ws.topic_id)--;
        word_topic(ws.word_id, ws.topic_id)--;
        topic_counts(ws.topic_id)--;
        p_ = (
            word_topic.row(ws.word_id).cast<Real>().transpose().array() + topic_word_prior.array()
        ).array() / ( topic_counts.cast<Real>().array() + eta_sum)  * (
            doc_topic.row(ws.doc_id).cast<Real>().transpose().array() +
            doc_topic_prior(ws.doc_id).array()
        );
        p_.array() /= p_.sum(); 
        doc_topic(ws.doc_id, ws.topic_id)++;
        word_topic(ws.word_id, ws.topic_id)++; 
        topic_counts(ws.topic_id)++;
        result.col(ws.word_id) += p_;
    }
    result.array().colwise() /= result.array().rowwise().sum();
    return result;
}

Real LDATrainerBase::log_likelihood(
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> word_topic) { 
    size_t n_words = word_topic.rows();
    Real ll =  n_topics_ * (
        std::lgamma( topic_word_prior.sum() ) - topic_word_prior.array().lgamma().sum()
    );

    RealArray exponent(n_words); 
    for (size_t j = 0; j < n_topics_; j++) {
        exponent = word_topic.col(j).cast<Real>().array() + topic_word_prior.array();
        ll += exponent.lgamma().sum();
        ll -= std::lgamma(exponent.sum()); 
    }

    return ll;
}

LDATrainer::LDATrainer(
        const RealVector & doc_topic_prior,
        Eigen::Ref<IntegerVector> counts,
        Eigen::Ref<IndexVector> dixs,
        Eigen::Ref<IndexVector> wixs,
        size_t n_topics,
        int random_seed
) : LDATrainerBase(counts, dixs, wixs, n_topics, random_seed),
    doc_topic_prior_(doc_topic_prior) {
}

const RealVector & LDATrainer::doc_topic_prior(
    std::size_t doc_index
){ 
    return doc_topic_prior_;
}


LabelledLDATrainer::LabelledLDATrainer(
        Real alpha,
        Real epsilon,
        const IntegerMatrix & labels,
        Eigen::Ref<IntegerVector> counts,
        Eigen::Ref<IndexVector> dixs,
        Eigen::Ref<IndexVector> wixs,
        size_t n_topics,
        int random_seed
): LDATrainerBase(counts, dixs, wixs, n_topics, random_seed), alpha_(alpha),
    epsilon_(epsilon), labels_(labels){

}

const RealVector & LabelledLDATrainer::doc_topic_prior(size_t doc_index) {
    return (labels_.row(doc_index).cast<Real>().array() * alpha_ + epsilon_).transpose();
}

Predictor::Predictor(
        size_t n_topics,
        const RealVector & doc_topic_prior,
        int random_seed
): n_topics_(n_topics), doc_topic_prior_(doc_topic_prior), n_domains_(0), betas_() {
}

void Predictor::add_beta(const RealMatrix & beta) {
    betas_.push_back(beta);
    n_domains_++;
}

RealVector Predictor::predict_mf(
    std::vector<IntegerVector> nonzeros,
    std::vector<IntegerVector> counts,
    std::size_t iter,
    Real delta
) {
  size_t dim_buffer = 0;
  for (size_t n = 0; n < n_domains_; n++ ) {
    dim_buffer += counts[n].sum();
  }
  if (dim_buffer == 0 ) {
      return doc_topic_prior_ / doc_topic_prior_.sum();
  }
  RealMatrix current_prob (dim_buffer, n_topics_);
  current_prob.array() = 0;
  RealMatrix new_prob(dim_buffer, n_topics_);
  RealMatrix beta_rel (dim_buffer, n_topics_);

  size_t current_iter = 0;
  for (size_t n = 0; n < n_domains_; n++ ) {
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

  for (size_t i = 0; i <= iter; i ++ ){
    new_prob = - current_prob; 
    new_prob.rowwise() += current_prob.colwise().sum();
    new_prob.rowwise() += doc_topic_prior_.transpose();
    new_prob.array() = new_prob.array() * beta_rel.array();
    new_prob.array().colwise() /= new_prob.array().rowwise().sum();
    double diff = (new_prob - current_prob).array().abs().sum();
    current_prob = new_prob;
    if (diff < delta) break;
  }
  RealVector theta = current_prob.array().colwise().sum().transpose();
  theta /= theta.sum();
  return theta;
}

RealVector Predictor::predict_gibbs(
        std::vector<IntegerVector> nonzeros,
        std::vector<IntegerVector> counts,
        std::size_t max_iter,
        std::size_t burn_in,
        int random_seed
<<<<<<< HEAD
){ 
    IntegerVector result(n_topics_);
=======
){
    if (burn_in >= max_iter) {
        throw std::invalid_argument("max_iter must be larger than burn_in.");
    }
    //std::vector<Real> p_(n_topics_);
    IntegerVector current_state(n_topics_);
    current_state.array() = 0;
    RealVector result(n_topics_);
>>>>>>> master
    result.array() = 0;

    UrandDevice urand_(random_seed);
    RealVector p_(n_topics_);
    RealVector p_temp(n_topics_);
    std::vector<size_t> topics;
    for (size_t n = 0; n < n_domains_; n++) {
        IntegerVector & count = counts[n];
        size_t n_unique_word = nonzeros[n].rows();
        for ( size_t j = 0; j < n_unique_word; j++ ) {
            for (int k = 0; k < count(j); k++ ) {
                p_ = doc_topic_prior();
                size_t init_topic = draw_from_p(p_, urand_); 
                topics.push_back( init_topic );
                current_state(init_topic)++;
            }
        }
    }
    for (size_t iter_ = 0; iter_ < max_iter; iter_++) {
        size_t current_iter = 0;
        for (size_t n = 0; n < n_domains_; n++) {
            size_t n_unique_word = nonzeros[n].rows();
            for ( size_t j = 0; j < n_unique_word; j++ ) {
                size_t wid = nonzeros[n](j);
                size_t wcount = counts[n](j);
                for (size_t k = 0; k < wcount ; k++ ) {
                    size_t current_topic = topics[current_iter]; 

                    current_state(current_topic)--;

                    p_ = (
                        betas_[n].row(wid).transpose().array() 
                        * ( current_state.cast<Real>().array() + doc_topic_prior_.array() )
                    );
                    if (iter_ >= burn_in) {
                        p_temp.array() = p_.array() / p_.sum();
                    }


                    current_topic = draw_from_p(
                        p_, urand_
                    );

                    current_state(current_topic)++;
                    topics[current_iter] = current_topic;
                    if (iter_ >= burn_in) {
                        //result(current_topic) ++;
                        result += p_temp;
                    }

                    current_iter++;
                }
            }
        }
    }
<<<<<<< HEAD
    RealVector normalized_result = result.cast<Real>() + doc_topic_prior_;
    normalized_result /= normalized_result.sum();
    return normalized_result;
=======
    result.array() /= (max_iter - burn_in);
    result += doc_topic_prior_;
    result.array() /= result.array().sum();
    return result;
>>>>>>> master
}

Real log_likelihood_doc_topic (
    Eigen::Ref<RealVector> doc_topic_prior,
    Eigen::Ref<IntegerMatrix> doc_topic 
){
    size_t n_doc = doc_topic.rows();
    size_t n_topics = doc_topic.cols();
    Real ll = n_doc * (
            std::lgamma(
                doc_topic_prior.sum()
                )
            - doc_topic_prior.array().lgamma().sum()
            ); 

    RealArray exponent(n_topics);
    for (size_t i = 0; i < n_doc; i++) {
        exponent = doc_topic.row(i).cast<Real>().transpose().array() + doc_topic_prior.array();
        ll += exponent.lgamma().sum();
        ll -= std::lgamma(exponent.sum());
    }
    return ll;
}



