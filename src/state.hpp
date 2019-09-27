#ifndef LDA_STATE_HPP
#define LDA_STATE_HPP
#include <vector> 
#include <random> 
#include "Eigen/Core" 
#include "./defs.hpp" 


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

    private:
    Real rand ();

    std::vector<WordState> word_states;
    const std::size_t n_topics_;
    std::mt19937 random_state_;
    std::uniform_real_distribution<Real> udist_; 

};

#endif
