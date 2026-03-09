#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <string>
#include <cuda_runtime_api.h>
#include "model/llama3.h"

static bool get_gpu_mem_mb(double* used_mb, double* total_mb) {
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return false;
  if (total_mb) *total_mb = total_bytes / (1024.0 * 1024.0);
  if (used_mb) *used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0);
  return true;
}

static void update_peak_gpu_mb(double* peak_mb) {
  if (!peak_mb) return;
  double used = 0, total = 0;
  if (!get_gpu_mem_mb(&used, &total)) return;
  if (used > *peak_mb) *peak_mb = used;
}

int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false, double* peak_gpu_mb = nullptr) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    update_peak_gpu_mb(peak_gpu_mb);
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
    }

    pos += 1;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}


int main(int argc, char* argv[]) {
  if (argc < 3) {
    LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path [--fp16|--bf16] [--temperature T]";
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];
  base::DataType activation_dtype = base::DataType::kDataTypeFp32;
  float temperature = 1.0f;
  for (int i = 3; i < argc; i++) {
    if (std::string(argv[i]) == "--fp16") activation_dtype = base::DataType::kDataTypeFp16;
    else if (std::string(argv[i]) == "--bf16") activation_dtype = base::DataType::kDataTypeBf16;
    else if (std::string(argv[i]) == "--temperature" && i + 1 < argc) {
      temperature = std::stof(argv[++i]);
    }
  }

  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
    checkpoint_path, true);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA, activation_dtype);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  model.set_temperature(temperature);

  double used_after_init = 0, total_mb = 0;
  if (get_gpu_mem_mb(&used_after_init, &total_mb)) {
    printf("[GPU] After init: %.2f MB used / %.2f MB total\n", used_after_init, total_mb);
    fflush(stdout);
  }

  const std::string& sentence = "hello";
  double peak_mb = used_after_init;
  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, 128, true, &peak_mb);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  if (get_gpu_mem_mb(nullptr, &total_mb)) {
    printf("[GPU] Peak during run: %.2f MB used / %.2f MB total\n", peak_mb, total_mb);
    fflush(stdout);
  }
  return 0;
}
