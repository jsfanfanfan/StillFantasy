#ifndef CAST_KERNEL_CU_CUH
#define CAST_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {
void cast_logits_to_float_cu(const tensor::Tensor& src, tensor::Tensor& dst, void* stream);
}
#endif
