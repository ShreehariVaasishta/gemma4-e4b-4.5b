#include "torch/types.h"
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ float gelu_tanh(float x) {
  // Pre-calculated constant for sqrt(2/pi)
  const float sqrt_2_over_pi = 0.79788456f;
  const float coef = 0.044715f;

  return 0.5 * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x * x * x)));
}

__global__ void elementwise_kernel(const float *__restrict__ x,
                                   float *__restrict__ out,
                                   int intermediate_size, int N_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N_out)
    return;

  int row_idx = idx / intermediate_size;
  int col_idx = idx % intermediate_size;

  int gate_idx = (row_idx * 2 * intermediate_size) + col_idx;
  int up_idx = gate_idx + intermediate_size;

  float gate_val = x[gate_idx];
  float up_val = x[up_idx];

  float gate_act = gelu_tanh(gate_val);
  float activation = gate_act * up_val;
  out[idx] = activation;
}

torch::Tensor elementwise(torch::Tensor combined_in, int intermediate_size) {
  TORCH_CHECK(combined_in.is_cuda(), "combined_in must be cuda");

  auto combined_in_fp32 = combined_in.to(torch::kFloat32);

  // Define the correct output shape: same as input, except the last dimension
  // is halved
  auto sizes = combined_in.sizes().vec();
  sizes.back() = intermediate_size;

  auto out_fp32 = torch::empty(sizes, combined_in_fp32.options());

  // Number of output elements
  int N_out = out_fp32.numel();

  int threads = 256;
  int blocks = (N_out + threads - 1) / threads;

  elementwise_kernel<<<blocks, threads>>>(combined_in_fp32.data_ptr<float>(),
                                          out_fp32.data_ptr<float>(),
                                          intermediate_size, N_out);

  return out_fp32.to(combined_in.dtype());
}