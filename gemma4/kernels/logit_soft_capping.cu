#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void logit_soft_capping_kernel(const float *__restrict__ logits,
                                          float *__restrict__ out,
                                          const float final_logit_softcapping,
                                          const int numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel)
    return;

  float scaled_logit = logits[idx] / final_logit_softcapping;
  out[idx] = tanhf(scaled_logit) * final_logit_softcapping;
}

torch::Tensor logit_soft_capping(torch::Tensor logits,
                                 float final_logit_softcapping) {
  TORCH_CHECK(logits.is_cuda(), "logits must be cuda");
  int numel = logits.numel();

  auto logits_fp32 = logits.to(torch::kFloat32);
  auto out_fp32 = torch::empty_like(logits_fp32);

  int threads = 256;
  int blocks = (numel + threads - 1) / threads;

  logit_soft_capping_kernel<<<blocks, threads>>>(
      logits_fp32.data_ptr<float>(), out_fp32.data_ptr<float>(),
      final_logit_softcapping, numel);
  return out_fp32.to(logits.dtype());
}