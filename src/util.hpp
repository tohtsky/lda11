#pragma once

#include "defs.hpp"
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unsupported/Eigen/SpecialFunctions>

inline size_t binary_search(const Real *array, size_t size, Real value) {
  int lower = 0, upper = size - 1;
  int half = 0;
  int idx = -1;
  while (upper >= lower) {
    half = lower + (upper - lower) / 2;
    double trial = array[half];
    if (value == trial) {
      idx = half;
      break;
    } else if (value > trial) {
      lower = half + 1;
    } else {
      upper = half - 1;
    }
  }
  if (idx == -1) // Element not found, return where it should be
    return static_cast<size_t>(lower);

  return static_cast<size_t>(idx);
}

inline size_t binary_search(const std::vector<Real> &array, Real value) {
  return binary_search(array.data(), array.size(), value);
}

inline size_t binary_search(const RealVector &array, Real value) {
  return binary_search(array.data(), array.size(), value);
}

inline Real cumsum(std::vector<Real> &v) {
  auto n = v.size();
  Real q = 0;
  for (size_t i = 0; i < n; i++) {
    q = (v[i] += q);
  };
  return q;
}

inline Real cumsum(RealVector &v) {
  auto n = v.rows();
  Real q = 0;
  for (int i = 0; i < n; i++) {
    q = (v(i) += q);
  };
  return q;
}

template <class ArrayType>
inline size_t draw_from_p(ArrayType &array, UrandDevice &rand) {
  Real max = cumsum(array);
  return binary_search(array, max * rand.rand());
}

inline Eigen::Map<RealVector> vector_to_eigen(std::vector<Real> &v) {
  return Eigen::Map<RealVector>(v.data(), v.size());
}
