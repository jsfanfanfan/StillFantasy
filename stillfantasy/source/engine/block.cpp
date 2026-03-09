/**
 * block.cpp - BlockManager 实现：块池、hash 查找、分配/释放
 */
#include "engine/block.h"
#include "base/alloc.h"
#include <glog/logging.h>
#include <cstring>

namespace engine {

uint64_t BlockManager::compute_block_hash(uint64_t prev_hash,
                                         const std::vector<int32_t>& token_ids) {
  uint64_t h = prev_hash;
  for (int32_t t : token_ids) {
    h = (h * 31) + static_cast<uint64_t>(static_cast<uint32_t>(t));
  }
  return h;
}

BlockManager::BlockManager(int32_t num_layers, int32_t block_size, int32_t kv_dim,
                           base::DataType dtype, base::DeviceType device_type,
                           int32_t max_num_blocks)
    : num_layers_(num_layers),
      block_size_(block_size),
      kv_dim_(kv_dim),
      dtype_(dtype),
      device_type_(device_type),
      max_num_blocks_(max_num_blocks) {
  std::shared_ptr<base::DeviceAllocator> alloc =
      (device_type_ == base::DeviceType::kDeviceCUDA)
          ? std::shared_ptr<base::DeviceAllocator>(base::CUDADeviceAllocatorFactory::get_instance())
          : std::shared_ptr<base::DeviceAllocator>(base::CPUDeviceAllocatorFactory::get_instance());

  size_t elem_size = base::DataTypeSize(dtype_);
  size_t block_elems = static_cast<size_t>(num_layers_) * block_size_ * kv_dim_;
  size_t pool_elems = block_elems * max_num_blocks_;
  size_t pool_bytes = pool_elems * elem_size;

  key_pool_ = tensor::Tensor(dtype_, static_cast<int32_t>(pool_elems), true, alloc);
  value_pool_ = tensor::Tensor(dtype_, static_cast<int32_t>(pool_elems), true, alloc);
  key_pool_.set_device_type(device_type_);
  value_pool_.set_device_type(device_type_);

  block_key_tensors_.reserve(max_num_blocks_);
  block_value_tensors_.reserve(max_num_blocks_);
  blocks_.resize(max_num_blocks_);
  free_block_ids_.reserve(max_num_blocks_);

  for (int32_t i = 0; i < max_num_blocks_; ++i) {
    blocks_[i].block_id = i;
    blocks_[i].ref_count = 0;
    blocks_[i].hash = 0;

    int64_t offset_elems = static_cast<int64_t>(i) * block_elems;
    int64_t offset_bytes = offset_elems * static_cast<int64_t>(elem_size);
    void* key_ptr = key_pool_.get_buffer()->ptr()
        ? static_cast<char*>(key_pool_.get_buffer()->ptr()) + offset_bytes
        : nullptr;
    void* val_ptr = value_pool_.get_buffer()->ptr()
        ? static_cast<char*>(value_pool_.get_buffer()->ptr()) + offset_bytes
        : nullptr;

    std::shared_ptr<base::Buffer> kb = std::make_shared<base::Buffer>(
        block_elems * elem_size, nullptr, key_ptr, true);
    std::shared_ptr<base::Buffer> vb = std::make_shared<base::Buffer>(
        block_elems * elem_size, nullptr, val_ptr, true);
    tensor::Tensor kview(dtype_, num_layers_, block_size_, kv_dim_, false, nullptr, key_ptr);
    tensor::Tensor vview(dtype_, num_layers_, block_size_, kv_dim_, false, nullptr, val_ptr);
    kview.assign(kb);
    vview.assign(vb);
    kview.set_device_type(device_type_);
    vview.set_device_type(device_type_);
    block_key_tensors_.push_back(kview);
    block_value_tensors_.push_back(vview);
    free_block_ids_.push_back(i);
  }
}

int32_t BlockManager::allocate_block() {
  if (free_block_ids_.empty()) {
    return -1;
  }
  int32_t id = free_block_ids_.back();
  free_block_ids_.pop_back();
  blocks_[id].ref_count = 1;
  blocks_[id].hash = 0;
  blocks_[id].token_ids.clear();
  return id;
}

void BlockManager::free_block(int32_t block_id) {
  if (block_id < 0 || block_id >= max_num_blocks_) return;
  Block& b = blocks_[block_id];
  if (b.ref_count > 0) {
    b.ref_count--;
    if (b.ref_count == 0) {
      if (b.hash != 0) {
        auto it = hash_to_block_.find(b.hash);
        if (it != hash_to_block_.end() && it->second == block_id) {
          hash_to_block_.erase(it);
        }
      }
      b.hash = 0;
      b.token_ids.clear();
      free_block_ids_.push_back(block_id);
    }
  }
}

void BlockManager::ref_block(int32_t block_id) {
  if (block_id >= 0 && block_id < max_num_blocks_) {
    blocks_[block_id].ref_count++;
  }
}

int32_t BlockManager::get_cached_block(uint64_t hash) const {
  auto it = hash_to_block_.find(hash);
  if (it == hash_to_block_.end()) return -1;
  return it->second;
}

void BlockManager::add_cached_block(uint64_t hash, int32_t block_id) {
  if (block_id >= 0 && block_id < max_num_blocks_) {
    blocks_[block_id].hash = hash;
    hash_to_block_[hash] = block_id;
  }
}

tensor::Tensor BlockManager::get_block_key(int32_t block_id) const {
  if (block_id < 0 || block_id >= static_cast<int32_t>(block_key_tensors_.size())) {
    return tensor::Tensor();
  }
  return block_key_tensors_[block_id];
}

tensor::Tensor BlockManager::get_block_value(int32_t block_id) const {
  if (block_id < 0 || block_id >= static_cast<int32_t>(block_value_tensors_.size())) {
    return tensor::Tensor();
  }
  return block_value_tensors_[block_id];
}

}  // namespace engine
