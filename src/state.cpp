#include "./state.hpp"
#include <unsupported/Eigen/SpecialFunctions>
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
): word_states(), n_topics_(n_topics), random_state_(random_seed), udist_(0.0, 1.0) {
    if (counts.rows() != dixs.rows() || dixs.cols() != wixs.cols() ) 
        throw std::runtime_error("");
    size_t n_ = counts.rows();
    for (size_t i = 0; i < n_; i++) {
        size_t c = counts(i);
        size_t dix = dixs(i);
        size_t wix = wixs(i);

        for (size_t j = 0; j < c; j++ ) {
            std::size_t assign = std::floor(n_topics_ * rand()); 
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
        q *= rand();
        ws.topic_id = binary_search(p_, q);

        doc_topic(ws.doc_id, ws.topic_id)++;
        word_topic(ws.word_id, ws.topic_id)++; 
        topic_counts(ws.topic_id)++;
    }
}

Real DocState::rand () {
        return udist_(random_state_);
}

Real DocState::log_likelihood(
        Eigen::Ref<RealVector> doc_topic_prior,
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic,
        Eigen::Ref<IntegerVector> topic_counts 
        ) {
    size_t n_doc = doc_topic.rows();
    size_t n_words = word_topic.rows(); 

    Real ll =  n_doc * (
        std::lgamma(
            doc_topic_prior.sum()
        )
        - doc_topic_prior.array().lgamma().sum()
    ) + n_topics_ * (
        std::lgamma(
            topic_word_prior.sum()
        ) - topic_word_prior.array().lgamma().sum()
    );

    {
        RealArray exponent(n_topics_);
        for (size_t i = 0; i < n_doc; i++) {
            exponent = doc_topic.row(i).cast<Real>().transpose().array() + doc_topic_prior.array();
            ll += exponent.lgamma().sum();
            ll -= std::lgamma(exponent.sum());
        }
    }
    std::cout << ll << std::endl;
    {
        RealArray exponent(n_words); 
        for (size_t j = 0; j < n_topics_; j++) {
            exponent = word_topic.col(j).cast<Real>().array() + topic_word_prior.array();
            ll += exponent.lgamma().sum();
            ll -= std::lgamma(exponent.sum()); 
        }
    }

    return ll;
}

/*
cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double eta) nogil:
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta, lgamma_alpha
    with nogil:
        lgamma_eta = lgamma(eta)
        lgamma_alpha = lgamma(alpha)

        ll += n_topics * lgamma(eta * vocab_size)
        for k in range(n_topics):
            ll -= lgamma(eta * vocab_size + nz[k])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += lgamma(eta + nzw[k, w]) - lgamma_eta

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += lgamma(alpha + ndz[d, k]) - lgamma_alpha
        return ll
*/
