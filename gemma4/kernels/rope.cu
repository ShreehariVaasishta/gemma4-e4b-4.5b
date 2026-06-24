#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void apply_rope_kernel(float *out, const float *x, const float *cos,
                                  const float *sin, const int *pos_ids,
                                  const float partial_rotary_factor,
                                  int batch_size, int num_heads, int head_dim,
                                  int seq_len, int rotary_dim) {
  int stride_seq = head_dim;
  int stride_head = seq_len * stride_seq;
  int stride_batch = num_heads * stride_head;
  int token_start_idx = (blockIdx.x * stride_batch) +
                        (blockIdx.y * stride_head) + (blockIdx.z * stride_seq);

  int i1 = token_start_idx + threadIdx.x;
  int i2 = token_start_idx + threadIdx.x + (rotary_dim / 2);

  float x1 = x[i1];
  float x2 = x[i2];
  int pos = pos_ids[blockIdx.x * seq_len + blockIdx.z];

  int theta_idx = pos * rotary_dim + threadIdx.x;

  float cos_theta = cos[theta_idx];
  float sin_theta = sin[theta_idx];

  out[i1] = x1 * cos_theta - x2 * sin_theta;
  out[i2] = x1 * sin_theta + x2 * cos_theta;
}

torch::Tensor apply_rope(torch::Tensor &x, torch::Tensor &cos,
                         torch::Tensor &sin, torch::Tensor &pos_ids,
                         float partial_rotary_factor) {
  int batch_size = x.size(0);
  int num_heads = x.size(1);
  int seq_len = x.size(2);
  int head_dim = x.size(3);

  // 1. Cast inputs to the correct C++ types (Float32 and Int32)
  auto x_fp32 = x.to(torch::kFloat32);
  auto cos_fp32 = cos.to(torch::kFloat32);
  auto sin_fp32 = sin.to(torch::kFloat32);
  auto pos_ids_int32 = pos_ids.to(torch::kInt32);

  // 2. Clone x_fp32 so the un-rotated dimensions are preserved
  torch::Tensor out_fp32 = x_fp32.clone();

  int rotary_dim = static_cast<int>(head_dim * partial_rotary_factor);

  dim3 grid(batch_size, num_heads, seq_len);
  dim3 block(rotary_dim / 2);

  apply_rope_kernel<<<grid, block>>>(
      out_fp32.data_ptr<float>(), x_fp32.data_ptr<float>(),
      cos_fp32.data_ptr<float>(), sin_fp32.data_ptr<float>(),
      pos_ids_int32.data_ptr<int>(), partial_rotary_factor, batch_size,
      num_heads, head_dim, seq_len, rotary_dim);

  // 3. Cast back to original dtype (BFloat16)
  return out_fp32.to(x.dtype());
}