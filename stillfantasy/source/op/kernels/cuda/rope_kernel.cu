#include "rope_kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <base/base.h>
namespace kernel {

#if defined (LLAMA3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

template <typename T>
__global__ void rope_kernel_cu_T(int pos, int dim, int kv_dim, int head_size, T* input_q, T* input_k,
                                 const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) return;
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;
  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
  int rotn = i < kv_dim ? 2 : 1;
  for (int v = 0; v < rotn; v++) {
    T* vec = v == 0 ? input_q : input_k;
    float v0 = static_cast<float>(vec[v0_idx]);
    float v1 = static_cast<float>(vec[v1_idx]);
    vec[v0_idx] = static_cast<T>(fcr * v0 - fci * v1);
    vec[v1_idx] = static_cast<T>(fcr * v1 + fci * v0);
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#elif defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

template <typename T>
__global__ void rope_kernel_cu_T_qwen(int pos, int dim, int kv_dim, int head_size, T* input_q,
                                      T* input_k, const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) return;
  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;
  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;
  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];
  int rotn = i < kv_dim ? 2 : 1;
  for (int v = 0; v < rotn; v++) {
    T* vec = v == 0 ? input_q : input_k;
    float v0 = static_cast<float>(vec[v0_idx]);
    float v1 = static_cast<float>(vec[v1_idx]);
    vec[v0_idx] = static_cast<T>(fcr * v0 - fci * v1);
    vec[v1_idx] = static_cast<T>(fcr * v1 + fci * v0);
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#else
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) {
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;
  *vec_ptr =
      make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }

  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);

  rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

template <typename T>
__global__ void rope_kernel_cu_T_default(int pos, int dim, int kv_dim, int head_size, T* input_q,
                                         T* input_k, const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;
  if (idx >= dim) return;
  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);
  T* q = input_q;
  float v0 = static_cast<float>(q[idx]);
  float v1 = static_cast<float>(q[idx + 1]);
  q[idx] = static_cast<T>(fcr * v0 - fci * v1);
  q[idx + 1] = static_cast<T>(fcr * v1 + fci * v0);
  if (idx < kv_dim) {
    T* k = input_k;
    v0 = static_cast<float>(k[idx]);
    v1 = static_cast<float>(k[idx + 1]);
    k[idx] = static_cast<T>(fcr * v0 - fci * v1);
    k[idx + 1] = static_cast<T>(fcr * v1 + fci * v0);
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#endif

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                            const_cast<float*>(sin_cache.ptr<float>()),
                                            const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()));
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                    void* stream) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : 0;
  base::DataType dtype = input_q.data_type();

  auto launch_fp32 = [&]() {
    if (stream_)
      rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
          pos, dim, kv_dim, head_size, const_cast<float*>(input_q.ptr<float>()),
          const_cast<float*>(input_k.ptr<float>()), sin_cache.ptr<float>(), cos_cache.ptr<float>());
    else
      rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size,
                                               const_cast<float*>(input_q.ptr<float>()),
                                               const_cast<float*>(input_k.ptr<float>()),
                                               sin_cache.ptr<float>(), cos_cache.ptr<float>());
  };

#if defined(LLAMA3_SUPPORT)
  if (dtype == base::DataType::kDataTypeFp32) {
    launch_fp32();
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    rope_kernel_cu_T<__half><<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, const_cast<__half*>(input_q.ptr<__half>()),
        const_cast<__half*>(input_k.ptr<__half>()), sin_cache.ptr<float>(), cos_cache.ptr<float>());
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    rope_kernel_cu_T<__nv_bfloat16><<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, const_cast<__nv_bfloat16*>(input_q.ptr<__nv_bfloat16>()),
        const_cast<__nv_bfloat16*>(input_k.ptr<__nv_bfloat16>()), sin_cache.ptr<float>(),
        cos_cache.ptr<float>());
    return;
  }
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  if (dtype == base::DataType::kDataTypeFp32) {
    launch_fp32();
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    rope_kernel_cu_T_qwen<__half><<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, const_cast<__half*>(input_q.ptr<__half>()),
        const_cast<__half*>(input_k.ptr<__half>()), sin_cache.ptr<float>(), cos_cache.ptr<float>());
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    rope_kernel_cu_T_qwen<__nv_bfloat16><<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, const_cast<__nv_bfloat16*>(input_q.ptr<__nv_bfloat16>()),
        const_cast<__nv_bfloat16*>(input_k.ptr<__nv_bfloat16>()), sin_cache.ptr<float>(),
        cos_cache.ptr<float>());
    return;
  }
#else
  if (dtype == base::DataType::kDataTypeFp32) {
    launch_fp32();
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    rope_kernel_cu_T_default<__half><<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, const_cast<__half*>(input_q.ptr<__half>()),
        const_cast<__half*>(input_k.ptr<__half>()), sin_cache.ptr<float>(), cos_cache.ptr<float>());
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    rope_kernel_cu_T_default<__nv_bfloat16><<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, const_cast<__nv_bfloat16*>(input_q.ptr<__nv_bfloat16>()),
        const_cast<__nv_bfloat16*>(input_k.ptr<__nv_bfloat16>()), sin_cache.ptr<float>(),
        cos_cache.ptr<float>());
    return;
  }
#endif
  LOG(FATAL) << "rope_kernel_cu: unsupported dtype";
}
}  // namespace kernel