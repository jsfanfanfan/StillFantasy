#ifndef STILLFANTASY_INCLUDE_ENGINE_SEQUENCE_H_
#define STILLFANTASY_INCLUDE_ENGINE_SEQUENCE_H_
/**
 * sequence.h - 请求序列抽象（借鉴 nano-vllm / StillFantasyInfer）
 *
 * 每个推理请求对应一个 Sequence：token 序列、块表、状态、采样参数等。
 * 含时间戳用于统计 TTFT / TPOT / 吞吐量。
 */
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>
#include "base/base.h"

namespace engine {

enum class SequenceStatus {
  WAITING = 0,   // 等待 prefill
  RUNNING = 1,   // 正在 decode
  FINISHED = 2,  // 已结束（EOS 或达到 max_tokens）
};

struct SamplingParams {
  float temperature = 0.6f;
  int32_t max_tokens = 256;
  bool ignore_eos = false;
};

class Sequence {
 public:
  Sequence(std::vector<int32_t> token_ids, const SamplingParams& params);

  int64_t seq_id() const { return seq_id_; }
  SequenceStatus status() const { return status_; }
  void set_status(SequenceStatus s) { status_ = s; }

  const std::vector<int32_t>& token_ids() const { return token_ids_; }
  void append_token(int32_t token) { token_ids_.push_back(token); }
  int32_t last_token() const { return token_ids_.empty() ? -1 : token_ids_.back(); }
  int32_t num_tokens() const { return static_cast<int32_t>(token_ids_.size()); }
  int32_t num_prompt_tokens() const { return num_prompt_tokens_; }

  /** 已缓存的 token 数（前缀命中 + 已 prefill 的长度） */
  int32_t num_cached_tokens() const { return num_cached_tokens_; }
  void set_num_cached_tokens(int32_t n) { num_cached_tokens_ = n; }

  /** 块表：逻辑块索引 -> 物理 block_id */
  std::vector<int32_t>& block_table() { return block_table_; }
  const std::vector<int32_t>& block_table() const { return block_table_; }

  const SamplingParams& sampling_params() const { return sampling_params_; }
  void set_sampling_params(const SamplingParams& p) { sampling_params_ = p; }

  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  void set_request_time(TimePoint t) { request_time_ = t; }
  void set_prefill_finished_time(TimePoint t) { prefill_finished_time_ = t; }
  void set_first_token_time(TimePoint t) { first_token_time_ = t; }
  void set_finished_time(TimePoint t) { finished_time_ = t; }
  TimePoint request_time() const { return request_time_; }
  TimePoint prefill_finished_time() const { return prefill_finished_time_; }
  TimePoint first_token_time() const { return first_token_time_; }
  TimePoint finished_time() const { return finished_time_; }

  static int64_t next_seq_id();

 private:
  static std::atomic<int64_t> seq_id_counter_;
  int64_t seq_id_;
  SequenceStatus status_;
  std::vector<int32_t> token_ids_;
  int32_t num_prompt_tokens_;
  int32_t num_cached_tokens_;
  std::vector<int32_t> block_table_;
  SamplingParams sampling_params_;
  TimePoint request_time_{};
  TimePoint prefill_finished_time_{};
  TimePoint first_token_time_{};
  TimePoint finished_time_{};
};

}  // namespace engine

#endif  // STILLFANTASY_INCLUDE_ENGINE_SEQUENCE_H_
