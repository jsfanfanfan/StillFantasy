#ifndef STILLFANTASY_INCLUDE_ENGINE_SCHEDULER_H_
#define STILLFANTASY_INCLUDE_ENGINE_SCHEDULER_H_
/**
 * scheduler.h - 调度器：Prefix Caching + Continuous Batching（借鉴 nano-vllm）
 *
 * - 维护等待 prefill 的队列与正在 decode 的序列；支持多序列同时运行（batch）。
 * - 每轮：对等待队列做 prefill（含前缀块复用），对每个运行中序列各做一步 decode，
 *   decode 前将该序列的 blocks 载入 model cache，decode 后将新 token 的 KV 写回 block。
 */
#include <memory>
#include <queue>
#include <vector>
#include "engine/block.h"
#include "engine/sequence.h"

namespace model {
class Model;
}

namespace engine {

/**
 * 将 BlockManager 中某块 KV 拷贝到 Model 的 KV cache 的指定偏移处。
 * cache_start：在 model cache 中的起始 token 位置（即写入 [cache_start, cache_start+block_size)）。
 */
void copy_block_to_model_cache(const BlockManager& block_mgr, int32_t block_id,
                               model::Model* model, int32_t cache_start);

/**
 * 将 Model 的 KV cache 的 [cache_start, cache_start+length) 拷贝到 BlockManager 的某块。
 */
void copy_model_cache_to_block(model::Model* model, int32_t cache_start, int32_t length,
                               BlockManager* block_mgr, int32_t block_id);

/**
 * 将 Model 的 KV cache 中单个 token 位置（token_pos）拷贝到某块的 offset_in_block 处。
 * 用于多序列 decode 后把新生成 token 的 KV 写回该序列的 block。
 */
void copy_one_token_model_cache_to_block(model::Model* model, int32_t token_pos,
                                         BlockManager* block_mgr, int32_t block_id,
                                         int32_t offset_in_block);

/**
 * 调度器：管理请求队列、Prefix Cache 预填、Continuous Batching 解码。
 */
class Scheduler {
 public:
  Scheduler(model::Model* model, int32_t max_num_blocks = 256);

  model::Model* model() const { return model_; }
  BlockManager* block_manager() const { return block_manager_.get(); }

  /** 提交一个请求（token_ids 为 prompt），返回 seq_id */
  int64_t add_request(std::vector<int32_t> token_ids,
                      const SamplingParams& params = SamplingParams());

  /** 本轮结束的序列信息，供调用方解码打印与指标统计 */
  struct FinishedSequence {
    int64_t seq_id = 0;
    std::vector<int32_t> token_ids;
    int32_t num_prompt_tokens = 0;
    int32_t num_output_tokens = 0;
    double ttft_ms = 0;           /** Time To First Token (ms) */
    double tpot_ms = 0;           /** Time Per Output Token (ms) */
    double throughput_tokens_per_sec = 0; /** 该序列吞吐 tokens/s */
  };

  /**
   * 执行一轮调度：先尝试对等待中的请求做 prefill（含前缀复用），再对就绪序列各做一步 decode。
   * 返回本轮已结束的序列（含 token_ids，便于 decode 打印）。
   */
  std::vector<FinishedSequence> schedule_step();

  /** 带 Prefix Caching 的 prefill：对单个 sequence 做预填，尽量复用已有块 */
  base::Status prefill_with_prefix_cache(Sequence* seq);

  /** 单序列 decode 一步：用当前 sequence 的 pos，执行 predict，更新 token 与状态 */
  base::Status decode_step(Sequence* seq);

  /** 获取当前正在运行的序列（返回指针列表，便于调用方遍历） */
  std::vector<Sequence*> running_sequences();
  /** 是否还有未完成或等待中的请求 */
  bool has_pending() const { return !waiting_.empty() || !running_.empty(); }

  /** 最大同时运行的序列数（0 表示不限制，仅受块数约束） */
  void set_max_running(int32_t n) { max_running_ = n; }
  int32_t max_running() const { return max_running_; }

 private:
  model::Model* model_;
  std::unique_ptr<BlockManager> block_manager_;
  std::queue<std::unique_ptr<Sequence>> waiting_;
  std::vector<std::unique_ptr<Sequence>> running_;
  int32_t max_running_ = 0;
};

}  // namespace engine

#endif  // STILLFANTASY_INCLUDE_ENGINE_SCHEDULER_H_
