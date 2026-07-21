#include <cuda_runtime.h>

__device__ inline float gelu_tanh(float x) {
  // Pre-calculated constant for sqrt(2/pi)
  const float sqrt_2_over_pi = 0.79788456f;
  const float coef = 0.044715f;

  return 0.5 * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x * x * x)));
}