#include "./state.hpp"
#include "iostream"

inline size_t binary_search(const std::vector<Real> & array, Real value) {
    int lower = 0, upper = array.size() -1; 
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

DocState::DocState (
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
            std::size_t assign = std::floor(n_topics_ * urand_.rand()); 
            word_states.emplace_back(dix, wix, assign);
        }
    }
}

void DocState::initialize_count(
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic
) {
    for (auto & ws: word_states) {
        doc_topic(ws.doc_id, ws.topic_id)++;
        word_topic(ws.word_id, ws.topic_id)++;
    }
}

void DocState::iterate_gibbs(
        Eigen::Ref<RealVector> doc_topic_prior,
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic,
        Eigen::Ref<IntegerVector> topic_counts 
        ) {

    Real eta_sum = topic_word_prior.sum();
    std::vector<Real> p_(n_topics_);

    for (auto & ws : word_states ) {
        doc_topic(ws.doc_id, ws.topic_id)--;
        word_topic(ws.word_id, ws.topic_id)--;
        topic_counts(ws.topic_id)--;

        Eigen::Map<Eigen::Array<Real,  Eigen::Dynamic, 1>> (p_.data(), n_topics_) = (
            word_topic.row(ws.word_id).cast<Real>().transpose().array() + topic_word_prior.array()
        ).array() / ( topic_counts.cast<Real>().array() + eta_sum)  * (
            doc_topic.row(ws.doc_id).cast<Real>().transpose().array() + doc_topic_prior.array()
        );

        double q = 0;
        for(size_t i = 0; i < n_topics_; i++ ) {
            p_[i] += q;
            q = p_[i];
        }
        q *= urand_.rand();
        ws.topic_id = binary_search(p_, q);

        doc_topic(ws.doc_id, ws.topic_id)++;
        word_topic(ws.word_id, ws.topic_id)++; 
        topic_counts(ws.topic_id)++;
    }
}

Real DocState::log_likelihood(
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

Predictor::Predictor(
        size_t n_topics,
        int random_seed
): n_topics_(n_topics), n_domains_(0), betas_(), urand_(random_seed) {
}

void Predictor::add_beta(RealMatrix beta) {
    betas_.push_back(beta);
    n_domains_++;
}

void Predictor::predict(
        Eigen::Ref<IntegerVector> result,
        std::vector<IntegerVector> nonzeros,
        std::vector<IntegerVector> counts,
        std::size_t max_iter
) { 
    std::vector<Real> p_(n_topics_);
    Eigen::Map<Eigen::Array<Real, Eigen::Dynamic, 1>> p_mapped(p_.data(), n_topics_);
    std::vector<size_t> topics;
    for (size_t n = 0; n < n_domains_; n++) {
        IntegerVector & nonzero = nonzeros[n]; 
        IntegerVector & count = counts[n];
        size_t n_unique_word = nonzeros[n].rows();
        for ( size_t j = 0; j < n_unique_word; j++ ) {
            for (int k = 0; k < count(j); k++ ) {
                size_t init_topic = std::floor(n_topics_ * urand_.rand()); 
                topics.push_back( init_topic );
                result(init_topic)++;
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

                    result(current_topic)--;

                    p_mapped = (
                        betas_[n].row(wid).transpose().array() 
                        * result.cast<Real>().array()
                    );


                    double q = 0;
                    for(size_t i = 0; i < n_topics_; i++ ) {
                        p_[i] += q;
                        q = p_[i];
                    }
                    q *= urand_.rand();
                    current_topic = binary_search(p_, q);

                    result(current_topic)++;
                    topics[current_iter] = current_topic;

                    current_iter++;
                }
            }
        }
    }
    //return result;
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


