cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_EXPORT_COMPILE_COMMANDS, True)
project(lda11)

add_subdirectory(pybind11)
include_directories(eigen-eigen-323c052e1731)
pybind11_add_module(lda11 src/wrapper.cpp src/labelled_lda.cpp src/trainer_base.cpp src/trainer.cpp src/predictor.cpp src/child_worker.cpp)