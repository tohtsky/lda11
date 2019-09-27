#ifndef LDA11_DEFS_HPP
#define LDA11_DEFS_HPP
#include "Eigen/Core"

using Real = double;
using Integer = int32_t;
using IntegerMatrix = Eigen::Matrix<
    Integer, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
>;
using IntegerVector = Eigen::Matrix<
    Integer, Eigen::Dynamic, 1
>;

using IndexVector = Eigen::Matrix<
    size_t, Eigen::Dynamic, 1
>;




struct WordState {
    inline WordState (size_t doc_id, size_t word_id, size_t topic_id):
        doc_id(doc_id), word_id(word_id), topic_id(topic_id)
    { }
    const size_t doc_id;
    const size_t word_id;
    size_t topic_id;
};

#endif
