#pragma once
#include <random>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SpecialFunctions>

using Real = double;
using Integer = int32_t;

using IntegerMatrix =
    Eigen::Matrix<Integer, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IntegerVector = Eigen::Matrix<Integer, Eigen::Dynamic, 1>;

using IndexVector = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
using SparseIntegerMatrix = Eigen::SparseMatrix<Integer, Eigen::RowMajor>;

using RealMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using RealVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using RealArray = Eigen::Array<Real, Eigen::Dynamic, 1>;

struct WordState {
  inline WordState(size_t doc_id, size_t word_id, size_t topic_id)
      : doc_id(doc_id), word_id(word_id), topic_id(topic_id) {}
  const size_t doc_id;
  const size_t word_id;
  size_t topic_id;
};

struct UrandDevice {
  inline UrandDevice(int random_seed)
      : random_state_(random_seed), udist_(0.0, 1.0) {}

  inline Real rand() { return udist_(random_state_); }

private:
  std::mt19937 random_state_;
  std::uniform_real_distribution<Real> udist_;
};