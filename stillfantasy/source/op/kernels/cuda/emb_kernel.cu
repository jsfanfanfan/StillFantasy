#include "emb_kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <base/base.h>
namespace kernel {
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) return;
  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

template <typename T>
__global__ void emb_kernel_cu_T(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                const int32_t* input_ptr, const float* weight_ptr, T* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) return;
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) return;
  T* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = static_cast<T>(weight_ptr_start[i]);
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_cu;
  const tensor::Tensor* in_tensor = &input;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
    in_tensor = &input_cu;
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : 0;
  int32_t* in_ptr = const_cast<int32_t*>(in_tensor->ptr<int32_t>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  base::DataType dtype = output.data_type();

  if (dtype == base::DataType::kDataTypeFp32) {
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    if (stream_)
      emb_kernel_cu_fp32<<<max_seq_len, thread_num, 0, stream_>>>(vocab_size, input_num, weight_dim,
                                                                 in_ptr, wei_ptr, out_ptr);
    else
      emb_kernel_cu_fp32<<<max_seq_len, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
                                                      wei_ptr, out_ptr);
    return;
  }
  if (dtype == base::DataType::kDataTypeFp16) {
    __half* out_ptr = const_cast<__half*>(output.ptr<__half>());
    emb_kernel_cu_T<__half><<<max_seq_len, thread_num, 0, stream_>>>(vocab_size, input_num,
                                                                    weight_dim, in_ptr, wei_ptr,
                                                                    out_ptr);
    return;
  }
  if (dtype == base::DataType::kDataTypeBf16) {
    __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());
    emb_kernel_cu_T<__nv_bfloat16><<<max_seq_len, thread_num, 0, stream_>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    return;
  }
  LOG(FATAL) << "emb_kernel_cu: unsupported dtype";
}
}  // namespace kernel