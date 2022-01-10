cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RELS)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_INIT} -std=c++11 -fPIC -O0")
set(PYBIND11_PYTHON_VERSION 3.10)

add_subdirectory(pybind11)
include_directories(eigen-3.4.0)
pybind11_add_module(_lda cpp_sources/wrapper.cpp cpp_sources/labelled_lda.cpp cpp_sources/trainer_base.cpp cpp_sources/trainer.cpp cpp_sources/predictor.cpp cpp_sources/child_worker.cpp)
