#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void vector_add_kernel(const float *a, const float *b, float *out,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

// This is the C++ wrapper that Python will call.
// It handles tensor validation, output allocation, and kernel launch.
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

  auto out = torch::empty_like(a);
  int n = a.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  vector_add_kernel<<<blocks, threads>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);

  return out;
}
