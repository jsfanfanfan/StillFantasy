/**
 * demo/main_engine.cpp - 多序列 Batch + Prefix Caching 推理测试
 *
 * 借鉴 nano-vllm / StillFantasyInfer：Prefix Caching（块级 hash 复用）+ Continuous Batching。
 * 用法：./main_engine checkpoint_path tokenizer_path [num_sequences]
 * 需在 CMake 中开启 LLAMA3_SUPPORT、QWEN2_SUPPORT 或 QWEN3_SUPPORT 之一才会编译本目标。
 */
#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "engine/scheduler.h"
#if defined(LLAMA3_SUPPORT)
#include "model/llama3.h"
#elif defined(QWEN2_SUPPORT)
#include "model/qwen2.h"
#elif defined(QWEN3_SUPPORT)
#include "model/qwen3.h"
#else
#error "main_engine requires one of LLAMA3_SUPPORT, QWEN2_SUPPORT, QWEN3_SUPPORT"
#endif

static int get_num_sequences(int argc, char* argv[]) {
  for (int i = 3; i < argc; i++) {
    int n = std::atoi(argv[i]);
    if (n > 0) return std::min(n, 32);
  }
  return 6;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    LOG(INFO) << "Usage: ./main_engine checkpoint_path tokenizer_path [num_sequences]";
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];
  int num_sequences = get_num_sequences(argc, argv);

  std::unique_ptr<model::Model> model;
#if defined(LLAMA3_SUPPORT)
  model = std::make_unique<model::LLama2Model>(base::TokenizerType::kEncodeSpe,
                                               tokenizer_path, checkpoint_path, false);
#elif defined(QWEN2_SUPPORT)
  model = std::make_unique<model::Qwen2Model>(base::TokenizerType::kEncodeBpe,
                                              tokenizer_path, checkpoint_path, false);
#elif defined(QWEN3_SUPPORT)
  model = std::make_unique<model::Qwen3Model>(base::TokenizerType::kEncodeBpe,
                                              tokenizer_path, checkpoint_path, false);
#endif

  auto init_status = model->init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "Model init failed: " << init_status.get_err_code();
  }

  engine::Scheduler scheduler(model.get(), 512);
  scheduler.set_max_running(num_sequences);
  engine::SamplingParams params;
  params.temperature = 0.6f;
  params.max_tokens = 32;

  std::vector<std::string> prompts = {
      "The quick brown fox jumps over the lazy dog.",
      "The quick brown fox runs in the forest.",
      "Hello, world. Today we",
      "The quick brown fox and the lazy dog",
      "Once upon a time there was",
      "In machine learning, we often",
  };
  while (static_cast<int>(prompts.size()) < num_sequences) {
    prompts.push_back("Prompt " + std::to_string(prompts.size() + 1) + ":");
  }

  std::cout << "Submitting " << num_sequences << " sequences (Prefix Cache + Continuous Batching) ..." << std::endl;
  std::vector<int64_t> ids;
  for (int i = 0; i < num_sequences; ++i) {
    ids.push_back(scheduler.add_request(model->encode(prompts[i]), params));
    std::cout << "  seq " << ids.back() << ": \"" << prompts[i].substr(0, 40)
              << (prompts[i].size() > 40 ? "..." : "") << "\"" << std::endl;
  }

  auto start = std::chrono::steady_clock::now();
  int steps = 0;
  int total_finished = 0;
  int total_output_tokens = 0;
  std::cout << "\n--- Running batch (multi-sequence) ---" << std::endl;
  while (scheduler.has_pending()) {
    auto done_list = scheduler.schedule_step();
    steps++;
    auto running = scheduler.running_sequences();
    for (const auto& fs : done_list) {
      total_finished++;
      total_output_tokens += fs.num_output_tokens;
      std::vector<int32_t> generated(fs.token_ids.begin() + fs.num_prompt_tokens,
                                     fs.token_ids.end());
      std::string text = generated.empty() ? "" : model->decode(generated);
      std::cout << "  [seq " << fs.seq_id << "] " << generated.size() << " tokens"
                << " | TTFT=" << fs.ttft_ms << " ms, TPOT=" << fs.tpot_ms << " ms, "
                << fs.throughput_tokens_per_sec << " tok/s"
                << "\n    \"" << text << "\"" << std::endl;
    }
    if (steps <= 3 || !running.empty()) {
      std::cout << "step " << steps << ": running=" << running.size()
                << ", finished_this_step=" << done_list.size() << std::endl;
    }
  }

  auto end = std::chrono::steady_clock::now();
  double wall_sec = std::chrono::duration<double>(end - start).count();
  double throughput_global = (wall_sec > 0 && total_output_tokens > 0)
                                 ? total_output_tokens / wall_sec
                                 : 0;
  std::cout << "\n--- Summary ---" << std::endl;
  std::cout << "  Sequences: " << total_finished << ", Output tokens: " << total_output_tokens
            << ", Wall time: " << wall_sec << " s" << std::endl;
  std::cout << "  Throughput (global): " << throughput_global << " tokens/s" << std::endl;
  return 0;
}
