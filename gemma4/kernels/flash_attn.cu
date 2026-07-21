#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>

// We define a maximum head dimension to avoid dynamic memory allocation in
// local memory (registers). Gemma 4 E4B uses up to 512 for the head dimension.
#define D_MAX 512

/**
 * Custom FlashAttention Kernel (Educational Implementation)
 *
 * This kernel implements the core ideas of FlashAttention:
 * 1. Tiling: We load blocks of K and V into fast Shared Memory (SRAM) to
 * minimize slow global memory reads.
 * 2. Online Softmax: We compute the softmax dynamically while iterating,
 * without needing an N x N matrix.
 */
__global__ void flash_attn_kernel(const float *__restrict__ Q,
                                  const float *__restrict__ K,
                                  const float *__restrict__ V,
                                  float *__restrict__ O, int batch_size,
                                  int num_heads, int q_len, int kv_len,
                                  int head_dim, int sliding_window_size) {

  // Grid setup:
  // z = batch_size, y = num_heads, x = seq_len / block_size
  int b = blockIdx.z;
  int h = blockIdx.y;

  // Each thread is responsible for computing the exact attention output for ONE
  // query token (one row of Q).
  int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool is_valid_q = (q_idx < q_len);

  // Clamp q_idx to 0 for out-of-bounds threads so they don't read/write bad
  // memory addresses
  int safe_q_idx = is_valid_q ? q_idx : 0;

  // Calculate memory offsets for this specific batch and head
  int head_offset_q =
      (b * num_heads * q_len * head_dim) + (h * q_len * head_dim);
  int head_offset_kv =
      (b * num_heads * kv_len * head_dim) + (h * kv_len * head_dim);

  const float *q_row = Q + head_offset_q + (safe_q_idx * head_dim);
  float *o_row = O + head_offset_q + (safe_q_idx * head_dim);

  // ==========================================
  // 1. ONLINE SOFTMAX INITIALIZATION
  // ==========================================
  // Instead of holding the entire N x N attention matrix, we just need to keep
  // track of:
  float m_i = -1e20f; // The maximum score seen so far (for numerical stability)
  float l_i = 0.0f;   // The running sum of exponentiated scores

  // This array will hold the running output vector for our query token.
  // It's stored in thread-local memory (registers or L1 cache).
  float out_val[D_MAX];
  for (int d = 0; d < head_dim; ++d) {
    out_val[d] = 0.0f;
  }

  // ==========================================
  // 2. TILING K AND V (SHARED MEMORY)
  // ==========================================
  // We process Keys and Values in blocks (tiles) of size B_c.
  int B_c = blockDim.x;

  // Dynamically allocated shared memory for our block
  extern __shared__ float sram[];
  float *k_tile = sram;                  // Size: B_c * head_dim
  float *v_tile = &sram[B_c * head_dim]; // Size: B_c * head_dim

  // Outer loop: Iterate over blocks of K and V
  for (int kv_start = 0; kv_start < kv_len; kv_start += B_c) {

    // --- COOPERATIVE LOADING ---
    // The threads in this block work together to load the K and V tiles from
    // slow global memory into fast shared memory.
    int num_elements = B_c * head_dim;
    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
      int k_idx = kv_start + (i / head_dim);
      int d_idx = i % head_dim;

      if (k_idx < kv_len) {
        k_tile[i] = K[head_offset_kv + k_idx * head_dim + d_idx];
        v_tile[i] = V[head_offset_kv + k_idx * head_dim + d_idx];
      } else {
        k_tile[i] = 0.0f; // Padding for out-of-bounds
        v_tile[i] = 0.0f;
      }
    }
    // Wait for all threads to finish loading the tiles before proceeding
    __syncthreads();

    // --- COMPUTE ATTENTION FOR THE TILE ---
    // Now, each valid thread loops over the keys in the loaded tile to update
    // its assigned query.
    if (is_valid_q) {
      for (int j = 0; j < B_c; ++j) {
        int kv_idx = kv_start + j;
        if (kv_idx >= kv_len)
          break;

        // Absolute position of this query token in the full sequence
        int abs_q_idx = kv_len - q_len + q_idx;

        // -- MASKING --
        // 1. Causal Mask: Queries can only attend to keys that come before or
        // at the same position.
        if (kv_idx > abs_q_idx)
          continue;

        // 2. Sliding Window Mask: Queries can only attend to a local
        // neighborhood of past keys. If sliding_window_size is -1, it means
        // global attention (no sliding window).
        if (sliding_window_size > 0 &&
            kv_idx <= abs_q_idx - sliding_window_size)
          continue;

        // Compute the dot product between our query row and the current key:
        // Q_i * K_j^T
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          score += q_row[d] * k_tile[j * head_dim + d];
        }
        // Note: Gemma 4 bakes the 1/sqrt(d) scaling factor into the Q/K weights
        // directly, so we DO NOT scale the score here! score /=
        // sqrtf((float)head_dim);

        // -- ONLINE SOFTMAX UPDATE --
        // This is the core magic! We update our running max and correct
        // previous calculations.

        // 1. Find the new maximum
        float m_new = fmaxf(m_i, score);

        // 2. Compute the exponent of the current score, stabilized by the new
        // max
        float exp_score = expf(score - m_new);

        // 3. Correction factor for our previous running sum and output.
        // If the max didn't change, this is e^0 = 1. If it did change, we scale
        // down old values.
        float correction = expf(m_i - m_new);

        // 4. Update the running sum
        l_i = l_i * correction + exp_score;

        // 5. Update the running output (multiply by Value and accumulate)
        for (int d = 0; d < head_dim; ++d) {
          out_val[d] =
              out_val[d] * correction + exp_score * v_tile[j * head_dim + d];
        }

        // 6. Save the new maximum
        m_i = m_new;
      }
    }

    // Wait for all threads to finish using the tile before the next iteration
    // overwrites it
    __syncthreads();
  }

  // ==========================================
  // 3. FINALIZE AND WRITE TO GLOBAL MEMORY
  // ==========================================
  // Divide by the total sum of exponentials to complete the softmax operation.
  if (is_valid_q) {
    for (int d = 0; d < head_dim; ++d) {
      o_row[d] = out_val[d] / l_i;
    }
  }
}

void GemmaAttnKernel(float const *const __restrict__ q,
                     float const *const __restrict__ k,
                     float const *const __restrict__ v,
                     float *const __restrict__ o, int const b, int const heads,
                     int const d_head, int const q_len, int const kv_len,
                     int sliding_window_size) {

  // We process sequences in blocks of 8 tokens to ensure we don't exceed the
  // 48KB default dynamic shared memory limit when head_dim is 512. (2 * 8 * 512
  // * 4 bytes = 32 KB)
  int B_r = 8;
  int B_c = 8;

  // Grid setup: (sequence blocks, number of heads, batch size)
  dim3 grid((q_len + B_r - 1) / B_r, heads, b);
  dim3 block(B_r);

  // Calculate the amount of shared memory needed for the K and V tiles.
  // We need space for B_c * d_head floats for K, and the same for V.
  size_t shared_mem_bytes = 2 * B_c * d_head * sizeof(float);

  // Launch the kernel
  flash_attn_kernel<<<grid, block, shared_mem_bytes>>>(
      q, k, v, o, b, heads, q_len, kv_len, d_head, sliding_window_size);

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error launching FlashAttention kernel: %s\n",
           cudaGetErrorString(err));
  }
}

// PyBind11 wrapper function to call the kernel from Python
torch::Tensor flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         int sliding_window_size) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32,
              "Only float32 is supported for this educational kernel");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "Inputs must be contiguous");

  // Extract dimensions [batch_size, num_heads, seq_len, head_dim]
  int b = q.size(0);
  int heads = q.size(1);
  int q_len = q.size(2);
  int d_head = q.size(3);

  int kv_len = k.size(2);

  // Create an empty tensor for the output
  auto o = torch::empty_like(q);

  GemmaAttnKernel(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                  o.data_ptr<float>(), b, heads, d_head, q_len, kv_len,
                  sliding_window_size);

  return o;
}