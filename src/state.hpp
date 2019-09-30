#ifndef LDA_STATE_HPP
#define LDA_STATE_HPP
#include <vector> 
#include <random> 
#include <cmath>
#include <Eigen/Core> 
#include <unsupported/Eigen/SpecialFunctions>

#include "./defs.hpp"

struct UrandDevice {
    inline UrandDevice(int random_seed):
        random_state_(random_seed), udist_(0.0, 1.0){}

    inline Real rand() {
        return udist_(random_state_);
    }

    private:

    std::mt19937 random_state_;
    std::uniform_real_distribution<Real> udist_; 
};

struct LDATrainerBase {
    LDATrainerBase (
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
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> doc_topic, 
        Eigen::Ref<IntegerMatrix> word_topic,
        Eigen::Ref<IntegerVector> topic_counts 
    ); 

    virtual const RealVector & doc_topic_prior(
        size_t doc_index
    ) = 0;

    Real log_likelihood (
        Eigen::Ref<RealVector> topic_word_prior, 
        Eigen::Ref<IntegerMatrix> word_topic
    );

    private:
    std::vector<WordState> word_states;
    const std::size_t n_topics_;
    std::mt19937 random_state_;
    UrandDevice urand_;
};

struct LDATrainer: LDATrainerBase {
    LDATrainer (
        const RealVector & doc_topic_prior,
        Eigen::Ref<IntegerVector> counts,
        Eigen::Ref<IndexVector> dixs,
        Eigen::Ref<IndexVector> wixs,
        size_t n_topics,
        int random_seed=42 
    );

    virtual const RealVector & doc_topic_prior(
        size_t doc_index
    ) override;
    private:
    RealVector doc_topic_prior_;
};

struct LabelledLDATrainer :LDATrainerBase {
    LabelledLDATrainer(
        Real alpha,
        Real epsilon,
        const IntegerMatrix & Labels,
        Eigen::Ref<IntegerVector> counts,
        Eigen::Ref<IndexVector> dixs,
        Eigen::Ref<IndexVector> wixs,
        size_t n_topics,
        int random_seed=42 
    );

    virtual const RealVector & doc_topic_prior(
        size_t doc_index
    ) override;

    private:
    Real alpha_, epsilon_;
    IntegerMatrix labels_;
};

struct Predictor {
    Predictor(
        size_t n_topics,
        int random_seed = 42
    );

    void add_beta(RealMatrix beta);

    void predict(
        Eigen::Ref<IntegerVector> result,
        std::vector<IntegerVector> nonzeros,
        std::vector<IntegerVector> counts,
        std::size_t iter
    );

    private:

    const std::size_t n_topics_; 
    std::size_t n_domains_;
    std::vector<RealMatrix> betas_;
    UrandDevice urand_; 
};

Real log_likelihood_doc_topic(
    Eigen::Ref<RealVector> doc_topic_prior,
    Eigen::Ref<IntegerMatrix> doc_topic
);

#endif
