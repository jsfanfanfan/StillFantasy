#include <tensor/tensor.h>
#include <base/base.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <typename T, int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_Tfloat(const T* input, const float* weight, T* output, int M,
                                        int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) return;

  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      sdata[tid] += static_cast<float>(input[i]) * weight[row_offset + i];
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = static_cast<T>(part_sum);
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <typename T, int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_Tint8(const T* input, const int8_t* weight,
                                       const float* scales, const int32_t group_size,
                                       T* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      float x = static_cast<float>(input[i]);
      sdata[tid] += x * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = static_cast<T>(part_sum);
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M, input.get_dim(0));

  cudaStream_t stream = config && config->stream ? config->stream : nullptr;
  base::DataType act_dtype = input.data_type();

  if (act_dtype == base::DataType::kDataTypeFp32) {
    if (stream) {
      matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, stream>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                                const_cast<float*>(output.ptr<float>()), M, K);
    }
    return;
  }
  if (act_dtype == base::DataType::kDataTypeFp16) {
    matmul_kernel_cu_Tfloat<__half, 128, 1><<<K, 128, 0, stream>>>(
        input.ptr<__half>(), weight.ptr<float>(), const_cast<__half*>(output.ptr<__half>()), M, K);
    return;
  }
  if (act_dtype == base::DataType::kDataTypeBf16) {
    matmul_kernel_cu_Tfloat<__nv_bfloat16, 128, 1><<<K, 128, 0, stream>>>(
        input.ptr<__nv_bfloat16>(), weight.ptr<float>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()), M, K);
    return;
  }
  LOG(FATAL) << "matmul_kernel_cu: unsupported activation dtype";
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  CHECK_EQ(M, input.get_dim(0));

  base::DataType act_dtype = input.data_type();
  cudaStream_t stream = config->stream ? config->stream : 0;

  if (act_dtype == base::DataType::kDataTypeFp32) {
    if (config->stream) {
      matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
          input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
          const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                    scale.ptr<float>(), group_size,
                                                    const_cast<float*>(output.ptr<float>()), M, K);
    }
    return;
  }
  if (act_dtype == base::DataType::kDataTypeFp16) {
    matmul_kernel_cu_Tint8<__half, 128, 1><<<K, 128, 0, stream>>>(
        input.ptr<__half>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<__half*>(output.ptr<__half>()), M, K);
    return;
  }
  if (act_dtype == base::DataType::kDataTypeBf16) {
    matmul_kernel_cu_Tint8<__nv_bfloat16, 128, 1><<<K, 128, 0, stream>>>(
        input.ptr<__nv_bfloat16>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()), M, K);
    return;
  }
  LOG(FATAL) << "matmul_kernel_cu_qint8: unsupported activation dtype";
}
}  // namespace kernel