#include "add_kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <base/base.h>

namespace kernel {
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) return;
  out[tid] = in1[tid] + in2[tid];
}

template <typename T>
__global__ void add_kernel_cu_T(int32_t size, const T* in1, const T* in2, T* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) return;
  out[tid] = static_cast<T>(static_cast<float>(in1[tid]) + static_cast<float>(in2[tid]));
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : 0;
  base::DataType dtype = input1.data_type();

  if (dtype == base::DataType::kDataTypeFp32) {
    if (stream_)
      add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
          size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
    else
      add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                    const_cast<float*>(output.ptr<float>()));
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    add_kernel_cu_T<__half><<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<__half>(), input2.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()));
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    add_kernel_cu_T<__nv_bfloat16><<<block_num, thread_num, 0, stream_>>>(size,
        input1.ptr<__nv_bfloat16>(), input2.ptr<__nv_bfloat16>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
    return;
  }
  LOG(FATAL) << "add_kernel_cu: unsupported dtype";
}
}  // namespace kernel
