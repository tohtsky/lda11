#ifndef LDA_STATE_HPP
#define LDA_STATE_HPP
#include <vector> 
#include <random> 
#include <cmath>
#include "Eigen/Core" 
#include "./defs.hpp"


struct DocState {
    DocState (
        Eigen::Ref<IntegerVector> counts,
        Eigen::Ref<IndexVector> dixs,
        Eigen::Ref<IndexVector> wixs,
        size_t n_topics,
        int random_seed=42 
    );

    void initialize_count(
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic
    );

    void iterate_gibbs(
        Eigen::Ref<RealVector> doc_topic_prior,
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic,
        Eigen::Ref<IntegerVector> topic_counts 
    ); 

    Real log_likelihood (
        Eigen::Ref<RealVector> doc_topic_prior,
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic,
        Eigen::Ref<IntegerVector> topic_counts 
    );

    private:
    Real rand ();

    std::vector<WordState> word_states;
    const std::size_t n_topics_;
    std::mt19937 random_state_;
    std::uniform_real_distribution<Real> udist_; 

};

#endif
