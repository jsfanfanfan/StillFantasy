/**
 * scheduler.cpp - 调度器实现：块与 model cache 拷贝、Prefix Cache prefill、Continuous Batching 步进
 */
#include "engine/scheduler.h"
#include "model/model.h"
#include "base/alloc.h"
#include <glog/logging.h>
#include <chrono>

namespace engine {

namespace {

base::MemcpyKind get_memcpy_kind(base::DeviceType device) {
  return device == base::DeviceType::kDeviceCUDA
             ? base::MemcpyKind::kMemcpyCUDA2CUDA
             : base::MemcpyKind::kMemcpyCPU2CPU;
}

void copy_kv_region(const tensor::Tensor& src, int64_t src_offset,
                   tensor::Tensor& dst, int64_t dst_offset,
                   int64_t num_elems, base::DeviceType device_type) {
  size_t elem_size = base::DataTypeSize(src.data_type());
  size_t byte_size = num_elems * elem_size;
  if (byte_size == 0 || !src.get_buffer() || !src.get_buffer()->ptr() ||
      !dst.get_buffer() || !dst.get_buffer()->ptr()) return;
  const char* src_ptr =
      static_cast<const char*>(src.get_buffer()->ptr()) + src_offset * elem_size;
  char* dst_ptr =
      static_cast<char*>(const_cast<void*>(dst.get_buffer()->ptr())) + dst_offset * elem_size;
  std::shared_ptr<base::DeviceAllocator> alloc =
      device_type == base::DeviceType::kDeviceCUDA
          ? std::shared_ptr<base::DeviceAllocator>(base::CUDADeviceAllocatorFactory::get_instance())
          : std::shared_ptr<base::DeviceAllocator>(base::CPUDeviceAllocatorFactory::get_instance());
  base::MemcpyKind kind = get_memcpy_kind(device_type);
  alloc->memcpy(src_ptr, dst_ptr, byte_size, kind, nullptr, true);
}

}  // namespace

void copy_block_to_model_cache(const BlockManager& block_mgr, int32_t block_id,
                               model::Model* model, int32_t cache_start) {
  if (!model) return;
  tensor::Tensor block_key = block_mgr.get_block_key(block_id);
  tensor::Tensor block_val = block_mgr.get_block_value(block_id);
  tensor::Tensor& model_key = model->get_buffer(model::ModelBufferType::kKeyCache);
  tensor::Tensor& model_val = model->get_buffer(model::ModelBufferType::kValueCache);
  int32_t num_layers = model->get_layer_num();
  int32_t seq_len = model->get_seq_len();
  int32_t block_size = block_mgr.block_size();
  int32_t kv_dim = block_mgr.kv_dim();
  base::DeviceType device = model->get_device_type();

  for (int32_t layer = 0; layer < num_layers; ++layer) {
    int64_t block_off = static_cast<int64_t>(layer) * block_size * kv_dim;
    int64_t cache_off = static_cast<int64_t>(layer) * seq_len * kv_dim +
                       static_cast<int64_t>(cache_start) * kv_dim;
    copy_kv_region(block_key, block_off, model_key, cache_off,
                  static_cast<int64_t>(block_size) * kv_dim, device);
    copy_kv_region(block_val, block_off, model_val, cache_off,
                  static_cast<int64_t>(block_size) * kv_dim, device);
  }
}

void copy_model_cache_to_block(model::Model* model, int32_t cache_start, int32_t length,
                               BlockManager* block_mgr, int32_t block_id) {
  if (!model || !block_mgr) return;
  tensor::Tensor& model_key = model->get_buffer(model::ModelBufferType::kKeyCache);
  tensor::Tensor& model_val = model->get_buffer(model::ModelBufferType::kValueCache);
  tensor::Tensor block_key = block_mgr->get_block_key(block_id);
  tensor::Tensor block_val = block_mgr->get_block_value(block_id);
  int32_t num_layers = model->get_layer_num();
  int32_t seq_len = model->get_seq_len();
  int32_t kv_dim = block_mgr->kv_dim();
  base::DeviceType device = model->get_device_type();

  for (int32_t layer = 0; layer < num_layers; ++layer) {
    int64_t cache_off = static_cast<int64_t>(layer) * seq_len * kv_dim +
                        static_cast<int64_t>(cache_start) * kv_dim;
    int64_t block_off = static_cast<int64_t>(layer) * length * kv_dim;
    copy_kv_region(model_key, cache_off, block_key, block_off,
                  static_cast<int64_t>(length) * kv_dim, device);
    copy_kv_region(model_val, cache_off, block_val, block_off,
                  static_cast<int64_t>(length) * kv_dim, device);
  }
}

void copy_one_token_model_cache_to_block(model::Model* model, int32_t token_pos,
                                         BlockManager* block_mgr, int32_t block_id,
                                         int32_t offset_in_block) {
  if (!model || !block_mgr) return;
  tensor::Tensor& model_key = model->get_buffer(model::ModelBufferType::kKeyCache);
  tensor::Tensor& model_val = model->get_buffer(model::ModelBufferType::kValueCache);
  tensor::Tensor block_key = block_mgr->get_block_key(block_id);
  tensor::Tensor block_val = block_mgr->get_block_value(block_id);
  int32_t num_layers = model->get_layer_num();
  int32_t seq_len = model->get_seq_len();
  int32_t block_size = block_mgr->block_size();
  int32_t kv_dim = block_mgr->kv_dim();
  base::DeviceType device = model->get_device_type();

  for (int32_t layer = 0; layer < num_layers; ++layer) {
    int64_t cache_off = static_cast<int64_t>(layer) * seq_len * kv_dim +
                        static_cast<int64_t>(token_pos) * kv_dim;
    int64_t block_off = static_cast<int64_t>(layer) * block_size * kv_dim +
                        static_cast<int64_t>(offset_in_block) * kv_dim;
    copy_kv_region(model_key, cache_off, block_key, block_off,
                  static_cast<int64_t>(kv_dim), device);
    copy_kv_region(model_val, cache_off, block_val, block_off,
                  static_cast<int64_t>(kv_dim), device);
  }
}

Scheduler::Scheduler(model::Model* model, int32_t max_num_blocks)
    : model_(model) {
  int32_t block_size = kDefaultBlockSize;
  int32_t num_layers = model_ ? model_->get_layer_num() : 0;
  int32_t kv_dim = model_ ? model_->get_kv_dim() : 0;
  base::DataType dtype = model_ ? model_->get_activation_dtype() : base::DataType::kDataTypeFp32;
  block_manager_ = std::make_unique<BlockManager>(
      num_layers, block_size, kv_dim, dtype,
      model_ ? model_->get_device_type() : base::DeviceType::kDeviceCPU,
      max_num_blocks);
}

int64_t Scheduler::add_request(std::vector<int32_t> token_ids,
                               const SamplingParams& params) {
  auto seq = std::make_unique<Sequence>(std::move(token_ids), params);
  int64_t sid = seq->seq_id();
  seq->set_request_time(Sequence::Clock::now());
  waiting_.push(std::move(seq));
  return sid;
}

base::Status Scheduler::prefill_with_prefix_cache(Sequence* seq) {
  if (!model_ || !seq) return base::error::InvalidArgument("model or sequence is null");
  const std::vector<int32_t>& token_ids = seq->token_ids();
  int32_t prompt_len = static_cast<int32_t>(token_ids.size());
  if (prompt_len == 0) return base::error::InvalidArgument("empty prompt");

  int32_t block_size = block_manager_->block_size();
  int32_t num_layers = model_->get_layer_num();
  int32_t seq_len = model_->get_seq_len();
  tensor::Tensor pos_tensor = model_->get_buffer(model::ModelBufferType::kInputPos);
  uint64_t prev_hash = 0;
  int32_t filled = 0;

  op::EmbeddingOutput prompt_embedding = model_->embedding(token_ids);
  bool is_prompt = true;
  int next = -1;

  for (int32_t start = 0; start < prompt_len; ) {
    int32_t chunk_len = std::min(block_size, prompt_len - start);
    std::vector<int32_t> chunk(token_ids.begin() + start, token_ids.begin() + start + chunk_len);
    uint64_t hash = BlockManager::compute_block_hash(prev_hash, chunk);

    int32_t cached_id = block_manager_->get_cached_block(hash);
    if (cached_id >= 0) {
      block_manager_->ref_block(cached_id);
      seq->block_table().push_back(cached_id);
      copy_block_to_model_cache(*block_manager_, cached_id, model_, filled);
      filled += chunk_len;
      prev_hash = hash;
      start += chunk_len;
      continue;
    }

    int32_t new_block_id = block_manager_->allocate_block();
    if (new_block_id < 0) {
      return base::error::InternalError("BlockManager: no free block");
    }
    int32_t cache_start = 0;
    for (int32_t bid : seq->block_table()) {
      copy_block_to_model_cache(*block_manager_, bid, model_, cache_start);
      cache_start += block_size;
    }
    for (int32_t pos = start; pos < start + chunk_len; ++pos) {
      pos_tensor.index<int32_t>(0) = pos;
      tensor::Tensor input = model_->fill_input(pos_tensor, prompt_embedding, is_prompt);
      model_->forward(input, pos_tensor, next);
    }
    copy_model_cache_to_block(model_, start, chunk_len, block_manager_.get(), new_block_id);
    block_manager_->add_cached_block(hash, new_block_id);
    seq->block_table().push_back(new_block_id);
    filled += chunk_len;
    prev_hash = hash;
    start += chunk_len;
  }

  seq->set_num_cached_tokens(filled);
  seq->set_prefill_finished_time(Sequence::Clock::now());
  seq->set_status(SequenceStatus::RUNNING);
  return base::error::Success();
}

base::Status Scheduler::decode_step(Sequence* seq) {
  if (!model_ || !seq) return base::error::InvalidArgument("model or sequence is null");
  int32_t pos = seq->num_tokens();
  int32_t last_tok = seq->last_token();
  if (seq->num_tokens() == 0) return base::error::InvalidArgument("sequence has no tokens");

  int next = -1;
  tensor::Tensor pos_tensor = model_->get_buffer(model::ModelBufferType::kInputPos);
  pos_tensor.index<int32_t>(0) = pos;
  std::vector<int32_t> single_token = {last_tok};
  op::EmbeddingOutput token_emb = model_->embedding(single_token);
  tensor::Tensor input = model_->fill_input(pos_tensor, token_emb, false);
  model_->predict(input, pos_tensor, false, next);
  seq->append_token(next);

  if (model_->is_sentence_ending(next) && !seq->sampling_params().ignore_eos) {
    seq->set_status(SequenceStatus::FINISHED);
  } else if (seq->num_tokens() - seq->num_prompt_tokens() >= seq->sampling_params().max_tokens) {
    seq->set_status(SequenceStatus::FINISHED);
  }
  return base::error::Success();
}

std::vector<Scheduler::FinishedSequence> Scheduler::schedule_step() {
  std::vector<FinishedSequence> finished;
  int32_t block_size = block_manager_->block_size();
  const bool limit_running = (max_running_ > 0);

  while (!waiting_.empty()) {
    if (limit_running && static_cast<int32_t>(running_.size()) >= max_running_) {
      break;
    }
    std::unique_ptr<Sequence> seq = std::move(waiting_.front());
    waiting_.pop();
    base::Status st = prefill_with_prefix_cache(seq.get());
    if (!st) {
      continue;
    }
    running_.push_back(std::move(seq));
  }

  for (size_t i = 0; i < running_.size(); ) {
    Sequence* seq = running_[i].get();
    if (seq->status() == SequenceStatus::FINISHED) {
      seq->set_finished_time(Sequence::Clock::now());
      FinishedSequence fs;
      fs.seq_id = seq->seq_id();
      fs.token_ids = seq->token_ids();
      fs.num_prompt_tokens = seq->num_prompt_tokens();
      fs.num_output_tokens = static_cast<int32_t>(fs.token_ids.size()) - fs.num_prompt_tokens;
      if (fs.num_output_tokens < 0) fs.num_output_tokens = 0;
      using namespace std::chrono;
      if (fs.num_output_tokens >= 1 && seq->first_token_time() >= seq->request_time()) {
        fs.ttft_ms = duration_cast<nanoseconds>(seq->first_token_time() - seq->request_time()).count() / 1e6;
      }
      auto decode_ns = duration_cast<nanoseconds>(seq->finished_time() - seq->prefill_finished_time()).count();
      double decode_sec = decode_ns / 1e9;
      if (fs.num_output_tokens > 0 && decode_sec > 0) {
        fs.tpot_ms = (decode_sec * 1000.0) / fs.num_output_tokens;
        fs.throughput_tokens_per_sec = fs.num_output_tokens / decode_sec;
      }
      finished.push_back(std::move(fs));
      for (int32_t bid : seq->block_table()) {
        block_manager_->free_block(bid);
      }
      running_.erase(running_.begin() + static_cast<std::ptrdiff_t>(i));
      continue;
    }

    int32_t cache_start = 0;
    for (int32_t bid : seq->block_table()) {
      copy_block_to_model_cache(*block_manager_, bid, model_, cache_start);
      cache_start += block_size;
    }
    base::Status st = decode_step(seq);
    if (!st) {
      ++i;
      continue;
    }
    if (seq->num_tokens() - seq->num_prompt_tokens() == 1) {
      seq->set_first_token_time(Sequence::Clock::now());
    }

    int32_t pos = seq->num_tokens() - 1;
    int32_t block_index = pos / block_size;
    int32_t offset_in_block = pos % block_size;
    while (static_cast<int32_t>(seq->block_table().size()) <= block_index) {
      int32_t new_bid = block_manager_->allocate_block();
      if (new_bid < 0) {
        ++i;
        continue;
      }
      seq->block_table().push_back(new_bid);
    }
    int32_t bid = seq->block_table()[block_index];
    copy_one_token_model_cache_to_block(model_, pos, block_manager_.get(), bid, offset_in_block);
    ++i;
  }

  return finished;
}

std::vector<Sequence*> Scheduler::running_sequences() {
  std::vector<Sequence*> out;
  for (auto& u : running_) out.push_back(u.get());
  return out;
}

}  // namespace engine
