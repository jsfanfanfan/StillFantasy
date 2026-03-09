#include <tensor/tensor.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <base/base.h>
#include "swiglu_kernel.cuh"
namespace kernel {
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  out[idx] = smem1[tid] * smem2[tid];
}

template <typename T>
__global__ void swiglu_kernel_cu_T(int size, const T* in1, const T* in2, T* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;
  smem1[tid] = static_cast<float>(in1[idx]);
  smem2[tid] = static_cast<float>(in2[idx]);
  __syncthreads();
  float value = 1.0f / (1.0f + expf(-smem1[tid]));
  out[idx] = static_cast<T>(smem1[tid] * value * smem2[tid]);
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  const size_t shmem = threads * sizeof(float) * 2;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : 0;
  base::DataType dtype = input1.data_type();

  if (dtype == base::DataType::kDataTypeFp32) {
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    swiglu_kernel_cu_T<__half><<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<__half>(), input2.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()));
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    swiglu_kernel_cu_T<__nv_bfloat16><<<blocks, threads, shmem, stream_>>>(size,
        input1.ptr<__nv_bfloat16>(), input2.ptr<__nv_bfloat16>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
    return;
  }
  LOG(FATAL) << "swiglu_kernel_cu: unsupported dtype";
}
}  // namespace kernel