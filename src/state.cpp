#include "./state.hpp"

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
                word_topic.row(ws.word_id).cast<Real>() + topic_word_prior
                ).array() / ( topic_counts.cast<Real>().array() + eta_sum)  * (
                    doc_topic.row(ws.doc_id).cast<Real>().array() + doc_topic_prior.array()
                    ); 

        double q = 0;
        for(size_t i = 0; i < n_topics_; i++ ) {
            q = (p_[i] += q);
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

