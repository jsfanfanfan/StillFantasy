#ifndef STILLFANTASY_INCLUDE_ENGINE_BLOCK_H_
#define STILLFANTASY_INCLUDE_ENGINE_BLOCK_H_
/**
 * block.h - Prefix Cache 块与块管理器（借鉴 nano-vllm / StillFantasyInfer）
 *
 * Block：固定大小（如 256 token）的 KV 缓存块，含 block_id、ref_count、hash，用于前缀复用。
 * BlockManager：块池、分配/释放、按 hash 查找复用。
 */
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace engine {

constexpr int32_t kDefaultBlockSize = 256;

struct Block {
  int32_t block_id = -1;
  int32_t ref_count = 0;
  uint64_t hash = 0;
  std::vector<int32_t> token_ids;  // 本块对应的 token id（用于 hash 链）
};

/**
 * 块管理器：维护物理块池与 hash -> block_id 映射，支持前缀复用。
 * 每块存一层内 block_size 个 token 的 KV，所有层连续存（或按层分块）。
 */
class BlockManager {
 public:
  BlockManager(int32_t num_layers, int32_t block_size, int32_t kv_dim,
               base::DataType dtype, base::DeviceType device_type,
               int32_t max_num_blocks);

  /** 分配一块空闲块，返回 block_id；无可用块返回 -1 */
  int32_t allocate_block();
  /** 释放块，ref_count 减一，为 0 时回收到空闲池并从 hash 表移除 */
  void free_block(int32_t block_id);
  /** 增加引用计数（用于前缀命中时复用） */
  void ref_block(int32_t block_id);

  /** 根据前缀 hash 查找已缓存块，命中返回 block_id，否则返回 -1 */
  int32_t get_cached_block(uint64_t hash) const;
  /** 将块注册到 hash 表（预填完成后调用） */
  void add_cached_block(uint64_t hash, int32_t block_id);

  /** 取某块的 Key/Value 张量视图 [layer_num, block_size, kv_dim] */
  tensor::Tensor get_block_key(int32_t block_id) const;
  tensor::Tensor get_block_value(int32_t block_id) const;

  int32_t block_size() const { return block_size_; }
  int32_t num_layers() const { return num_layers_; }
  int32_t kv_dim() const { return kv_dim_; }
  int32_t max_num_blocks() const { return max_num_blocks_; }

  /** 计算块 hash：H(prev_hash, token_ids...) */
  static uint64_t compute_block_hash(uint64_t prev_hash,
                                     const std::vector<int32_t>& token_ids);

 private:
  int32_t num_layers_;
  int32_t block_size_;
  int32_t kv_dim_;
  base::DataType dtype_;
  base::DeviceType device_type_;
  int32_t max_num_blocks_;

  tensor::Tensor key_pool_;
  tensor::Tensor value_pool_;
  std::vector<int32_t> free_block_ids_;
  std::vector<Block> blocks_;
  std::vector<tensor::Tensor> block_key_tensors_;
  std::vector<tensor::Tensor> block_value_tensors_;
  std::unordered_map<uint64_t, int32_t> hash_to_block_;
};

}  // namespace engine

#endif  // STILLFANTASY_INCLUDE_ENGINE_BLOCK_H_
