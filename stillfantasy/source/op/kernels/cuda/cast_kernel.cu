#include "cast_kernel.cuh"
#include <base/base.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace kernel {

__global__ void cast_fp16_to_fp32_kernel(const __half* src, float* dst, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void cast_bf16_to_fp32_kernel(const __nv_bfloat16* src, float* dst, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __bfloat162float(src[i]);
}

void cast_logits_to_float_cu(const tensor::Tensor& src, tensor::Tensor& dst, void* stream) {
  CHECK(!src.is_empty() && !dst.is_empty());
  CHECK(src.device_type() == base::DeviceType::kDeviceCUDA &&
        dst.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(src.size(), dst.size());
  CHECK(dst.data_type() == base::DataType::kDataTypeFp32);

  int64_t n = static_cast<int64_t>(src.size());
  if (n == 0) return;

  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : 0;
  constexpr int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);

  base::DataType dtype = src.data_type();
  if (dtype == base::DataType::kDataTypeFp32) {
    if (src.get_buffer() && dst.get_buffer() && src.get_buffer()->ptr() != dst.get_buffer()->ptr()) {
      size_t bytes = n * sizeof(float);
      cudaMemcpyAsync(dst.get_buffer()->ptr(), src.get_buffer()->ptr(), bytes,
                      cudaMemcpyDeviceToDevice, stream_);
    }
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    const __half* s = src.ptr<__half>();
    float* d = const_cast<float*>(dst.ptr<float>());
    cast_fp16_to_fp32_kernel<<<grid, block, 0, stream_>>>(s, d, n);
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    const __nv_bfloat16* s = src.ptr<__nv_bfloat16>();
    float* d = const_cast<float*>(dst.ptr<float>());
    cast_bf16_to_fp32_kernel<<<grid, block, 0, stream_>>>(s, d, n);
    return;
  }
  LOG(FATAL) << "cast_logits_to_float_cu: unsupported src dtype";
}

}  // namespace kernel
