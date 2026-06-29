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
torch::Tensor rms_norm(torch::Tensor x, float eps,
                       std::optional<torch::Tensor> weight);
torch::Tensor apply_rope(torch::Tensor &x, torch::Tensor &cos,
                         torch::Tensor &sin, torch::Tensor &pos_ids,
                         float partial_rotary_factor);

torch::Tensor elementwise(torch::Tensor combined_in, int intermediate_size);

torch::Tensor logit_soft_capping(torch::Tensor logits,
                                 float final_logit_softcapping);

PYBIND11_MODULE(gemma4_kernels, m) {
  m.def("vector_add", &vector_add, "Element-wise vector addition (CUDA)");
  m.def("rms_norm", &rms_norm, "RMS Norm (CUDA)", pybind11::arg("x"),
        pybind11::arg("eps"), pybind11::arg("weight") = std::nullopt);
  m.def("apply_rope", &apply_rope, "Apply Rotary Embeddings (CUDA)",
        pybind11::arg("x"), pybind11::arg("cos"), pybind11::arg("sin"),
        pybind11::arg("position_ids"), pybind11::arg("partial_rotary_factor"));
  m.def("elementwise", &elementwise, "Elementwise GEMLU operation (CUDA)",
        pybind11::arg("combined_in"), pybind11::arg("intermediate_size"));

  m.def("logit_soft_capping", &logit_soft_capping, "Logit soft capping (CUDA)",
        pybind11::arg("logits"), pybind11::arg("final_logit_softcapping"));
}