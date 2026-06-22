#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void rms_norm_kernel(const float *__restrict__ x,
                                const float *__restrict__ weight,
                                float *__restrict__ y, int hidden_dim,
                                float eps) {
  int row = blockIdx.x;

  const float *row_ptr = x + row * hidden_dim;
  float *out_ptr = y + row * hidden_dim;

  //   Step1: each thread computes partial sum
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    float val = row_ptr[col];
    local_sum += val * val;
  }

  // Step 2:Put partial sums into shared memory
  __shared__ float smem[256];
  smem[threadIdx.x] = local_sum;
  __syncthreads();

  // Step 3: reduce to get total sum
  // >> - right shift bitwise operator. divide by 2. happens on every iteration
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Step4: Compute RMS scale
  __shared__ float scale;
  if (threadIdx.x == 0) {
    float mean_sq = smem[0] / hidden_dim;
    scale = rsqrt(mean_sq + eps);
  }
  __syncthreads();

  // Step5: Apply normalisation and weight
  for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    float x_val = row_ptr[col];
    if (weight != nullptr) {
      out_ptr[col] = x_val * scale * weight[col];
    } else {
      out_ptr[col] = x_val * scale;
    }
  }
}

torch::Tensor rms_norm(torch::Tensor x, float eps,
                       std::optional<torch::Tensor> weight) {
  TORCH_CHECK(x.is_cuda(), "tensor is on cpu");

  // Cast inputs to float32 for the kernel
  auto x_fp32 = x.to(torch::kFloat32);

  // Handle optional weight
  float *weight_ptr = nullptr;
  torch::Tensor weight_fp32; // keep in scope so pointer stays valid
  if (weight.has_value() && weight.value().defined()) {
    weight_fp32 = weight.value().to(torch::kFloat32);
    weight_ptr = weight_fp32.data_ptr<float>();
  }

  // Create float32 output tensor
  auto out_fp32 = torch::empty_like(x_fp32);

  // The kernel processes one row per block.
  int hidden_dim = x.size(-1);
  int rows = x.numel() / hidden_dim;

  int threads = 256;
  int blocks = rows;

  rms_norm_kernel<<<blocks, threads>>>(x_fp32.data_ptr<float>(), weight_ptr,
                                       out_fp32.data_ptr<float>(), hidden_dim,
                                       eps);

  // Cast output back to the original dtype (BFloat16)
  return out_fp32.to(x.dtype());
}
