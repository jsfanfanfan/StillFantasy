#include "sampler/argmax_sampler.h"
#include <algorithm>
#include <cmath>
#include <random>
#include "../op/kernels/cuda/argmax_kernel.cuh"
#include <cuda_runtime.h>

namespace sampler {

namespace {
static std::mt19937& get_rng() {
  static std::mt19937 rng(std::random_device{}());
  return rng;
}

size_t sample_from_softmax(float* probs, size_t size) {
  std::discrete_distribution<size_t> dist(probs, probs + size);
  return dist(get_rng());
}
}  // namespace

size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  const bool use_temperature = temperature_ > 1e-6f && std::fabs(temperature_ - 1.0f) > 1e-6f;

  if (!use_temperature) {
    if (device_type_ == base::DeviceType::kDeviceCPU) {
      return static_cast<size_t>(std::distance(
          logits, std::max_element(logits, logits + size)));
    }
    return kernel::argmax_kernel_cu(logits, size, stream);
  }

  logits_cpu_.resize(size);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;
    cudaMemcpyAsync(logits_cpu_.data(), logits, size * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
  } else {
    std::copy(logits, logits + size, logits_cpu_.begin());
  }

  float* probs = logits_cpu_.data();
  float max_logit = *std::max_element(logits_cpu_.begin(), logits_cpu_.end());
  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    probs[i] = std::exp((probs[i] - max_logit) / temperature_);
    sum += probs[i];
  }
  for (size_t i = 0; i < size; ++i) {
    probs[i] /= sum;
  }
  return sample_from_softmax(probs, size);
}
}  // namespace sampler