// bindings.cpp — Pybind11 module that exposes CUDA kernels to Python.
//
// This file is compiled by g++ (NOT nvcc). It can only see regular C++
// function declarations. The actual CUDA code lives in .cu files,
// which are compiled separately by nvcc and linked in.

#include <torch/extension.h>

#include <optional>

// Forward-declare the wrapper function defined in vector_add.cu
// (the linker will resolve this when combining the .o files)
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor rms_norm(torch::Tensor x, float eps, std::optional<torch::Tensor> weight);

PYBIND11_MODULE(gemma4_kernels, m) {
  m.def("vector_add", &vector_add, "Element-wise vector addition (CUDA)");
  m.def("rms_norm", &rms_norm, "RMS Norm (CUDA)",
        pybind11::arg("x"), pybind11::arg("eps"), pybind11::arg("weight") = std::nullopt);
}