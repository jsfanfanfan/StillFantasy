#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "mha_kernel.cuh"
#include <base/base.h>
#include <base/tick.h>
namespace kernel {
constexpr static int thread_num = 256;
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

template <typename T>
__device__ void softmax_gpu_T(T* x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;
  float max_val = tid < size ? static_cast<float>(x[tid]) : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    float v = static_cast<float>(x[i]);
    if (v > max_val) max_val = v;
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) shared_val = max_val;
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    float v = expf(static_cast<float>(x[i]) - max_val);
    x[i] = static_cast<T>(v);
    sum += v;
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] = static_cast<T>(static_cast<float>(x[i]) / sum);
  }
}

template <typename T>
__global__ void multi_head_attention_kernel_T(int32_t pos, int32_t seq_len, T* query,
                                               T* score_ptr, T* output, T* key_cache,
                                               T* value_cache, int32_t kv_dim, int32_t kv_mul,
                                               int32_t head_num, int32_t head_size,
                                               int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) return;
  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  T* query_head = query + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = static_cast<float>(query_head[i]);
  }
  __syncthreads();

  T* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    T* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val;
      key_val.x = static_cast<float>(key_head[i]);
      key_val.y = static_cast<float>(key_head[i + 1]);
      key_val.z = static_cast<float>(key_head[i + 2]);
      key_val.w = static_cast<float>(key_head[i + 3]);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = static_cast<T>(score);
  }
  __syncthreads();
  softmax_gpu_T<T>(score_head, pos + 1);
  __syncthreads();

  T* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      T* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = static_cast<float>(score_head[t]);
      value += score * static_cast<float>(value_head[i]);
    }
    output_head[i] = static_cast<T>(value);
  }
}

__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // 预加载query到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  // head当前的注意力头索引，kv_mul用于gqa，head_size表示一个自注意力头的维度
  // kv_dim = head_size * head_num，多头自注意力情况下的key,value 维度
  // kv_dim = head_size * head_num / kv_num，GQA情况下的key,value 维度
  int head_offset = (head / kv_mul) * head_size;
  // 计算自注意力分数
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  // 使用自注意力分数对value矩阵加权
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  cudaStream_t stream = config->stream;
  base::DataType dtype = query_tensor.data_type();

  if (dtype == base::DataType::kDataTypeFp32) {
    float* query = const_cast<float*>(query_tensor.ptr<float>());
    float* score = const_cast<float*>(score_tensor.ptr<float>());
    float* output = const_cast<float*>(mha_out.ptr<float>());
    float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
    float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());
    multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
        pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
        head_size, layer_offset);
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    __half* query = const_cast<__half*>(query_tensor.ptr<__half>());
    __half* score = const_cast<__half*>(score_tensor.ptr<__half>());
    __half* output = const_cast<__half*>(mha_out.ptr<__half>());
    __half* key_cache = const_cast<__half*>(key_cache_tensor.ptr<__half>());
    __half* value_cache = const_cast<__half*>(value_cache_tensor.ptr<__half>());
    multi_head_attention_kernel_T<__half><<<head_num, thread_num, head_size * sizeof(float),
                                            stream>>>(pos, seq_len, query, score, output,
                                                      key_cache, value_cache, kv_dim, kv_mul,
                                                      head_num, head_size, layer_offset);
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    __nv_bfloat16* query = const_cast<__nv_bfloat16*>(query_tensor.ptr<__nv_bfloat16>());
    __nv_bfloat16* score = const_cast<__nv_bfloat16*>(score_tensor.ptr<__nv_bfloat16>());
    __nv_bfloat16* output = const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());
    __nv_bfloat16* key_cache = const_cast<__nv_bfloat16*>(key_cache_tensor.ptr<__nv_bfloat16>());
    __nv_bfloat16* value_cache =
        const_cast<__nv_bfloat16*>(value_cache_tensor.ptr<__nv_bfloat16>());
    multi_head_attention_kernel_T<__nv_bfloat16><<<head_num, thread_num,
                                                     head_size * sizeof(float), stream>>>(
        pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
        head_size, layer_offset);
    return;
  }
  LOG(FATAL) << "mha_kernel_cu: unsupported dtype";
}

}  // namespace kernel