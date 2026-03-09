#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "rmsnorm_kernel.cuh"
namespace kernel {
/**
 * 计算多维输入 in = (dim1, dim2), 计算在dim2维度上的rmsnorm
 */
static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size,
                                           int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

template <typename T, int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_T(const T* in, const float* wei, T* out, int size, float eps) {
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    float x = static_cast<float>(in[i]);
    sum += x * x;
  }
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  sum = BlockReduce(temp).Sum(sum);
  __shared__ float shared_val;
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  for (int i = tid; i < size; i += blockDim.x) {
    out[i] = static_cast<T>(wei[i] * static_cast<float>(in[i]) * scale);
  }
}

template <typename T>
static __global__ void row_rmsnorm_T_dim(const T* in, const float* wei, T* out, int dim_size,
                                        int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) return;
  const T* block_in = in + bid * size;
  T* block_out = out + bid * size;
  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    float x = static_cast<float>(block_in[i]);
    sum += x * x;
  }
  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  sum = BlockReduce(temp).Sum(sum);
  __shared__ float shared_val;
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  for (int i = tid; i < size; i += blockDim.x) {
    block_out[i] = static_cast<T>(wei[i] * static_cast<float>(block_in[i]) * scale);
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  constexpr int threads_num = 128;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : 0;

  base::DataType dtype = input.data_type();
  if (dtype == base::DataType::kDataTypeFp32) {
    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    if (stream_) {
      row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    __half* in_ptr = const_cast<__half*>(input.ptr<__half>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    __half* out_ptr = const_cast<__half*>(output.ptr<__half>());
    row_rmsnorm_T<__half, 128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    __nv_bfloat16* in_ptr = const_cast<__nv_bfloat16*>(input.ptr<__nv_bfloat16>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());
    row_rmsnorm_T<__nv_bfloat16, 128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size,
                                                                     eps);
    return;
  }
  LOG(FATAL) << "rmsnorm_kernel_cu: unsupported dtype";
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;
  constexpr int threads_num = 128;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : 0;

  base::DataType dtype = input.data_type();
  if (dtype == base::DataType::kDataTypeFp32) {
    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    if (stream_) {
      row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                                 size, eps);
    } else {
      row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    __half* in_ptr = const_cast<__half*>(input.ptr<__half>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    __half* out_ptr = const_cast<__half*>(output.ptr<__half>());
    row_rmsnorm_T_dim<__half><<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr,
                                                                      dim_size, size, eps);
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    __nv_bfloat16* in_ptr = const_cast<__nv_bfloat16*>(input.ptr<__nv_bfloat16>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());
    row_rmsnorm_T_dim<__nv_bfloat16><<<dim_size, threads_num, 0, stream_>>>(
        in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    return;
  }
  LOG(FATAL) << "rmsnorm_kernel_cu_dim: unsupported dtype";
}
}  // namespace kernel