#ifndef LLAMA_INFER_SAMPLER_H
#define LLAMA_INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>
namespace sampler {
class Sampler {
 public:
  explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

  virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

  void set_temperature(float t) { temperature_ = t; }
  float temperature() const { return temperature_; }

 protected:
  base::DeviceType device_type_;
  float temperature_ = 1.0f;
};
}  // namespace sampler
#endif  // LLAMA_INFER_SAMPLER_H
