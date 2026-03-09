/**
 * sequence.cpp - Sequence 实现
 */
#include "engine/sequence.h"
#include <atomic>

namespace engine {

std::atomic<int64_t> Sequence::seq_id_counter_{0};

int64_t Sequence::next_seq_id() {
  return seq_id_counter_.fetch_add(1);
}

Sequence::Sequence(std::vector<int32_t> token_ids, const SamplingParams& params)
    : seq_id_(next_seq_id()),
      status_(SequenceStatus::WAITING),
      token_ids_(std::move(token_ids)),
      num_cached_tokens_(0),
      sampling_params_(params) {
  num_prompt_tokens_ = static_cast<int32_t>(token_ids_.size());
}

}  // namespace engine
