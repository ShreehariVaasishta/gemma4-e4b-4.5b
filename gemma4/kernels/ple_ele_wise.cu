#include "gelu_tanh.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void ele_wise_kernel(const scalar_t *__restrict__ x,
                                const scalar_t *__restrict__ per_layer_input,
                                scalar_t *__restrict__ out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;
  float val_x = static_cast<float>(x[idx]);
  float val_y = static_cast<float>(per_layer_input[idx]);
  out[idx] = gelu_tanh(val_x) * val_y;
}

torch::Tensor gemma4_decode_layer_ple_inj(torch::Tensor gate_pre,
                                          torch::Tensor per_layer_input) {
  TORCH_CHECK(gate_pre.is_cuda());
  auto out = torch::empty_like(gate_pre);
  int N = gate_pre.numel();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, out.scalar_type(),
      "gemma4_decode_layer_ple_inj", [&] {
        ele_wise_kernel<scalar_t><<<blocks, threads>>>(
            gate_pre.data_ptr<scalar_t>(), per_layer_input.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), N);
      });
  return out;
}