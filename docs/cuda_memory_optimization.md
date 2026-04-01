# CUDA 显存分配优化说明

本文档说明 StillFantasy 框架对 CUDA 显存分配的优化策略，对应实现位于 `stillfantasy/include/base/alloc.h` 与 `stillfantasy/source/base/alloc_cu.cpp`。

---

## 1. 优化目标与思路

- **减少 cudaMalloc / cudaFree 调用**：显存分配/释放是相对昂贵的操作，频繁调用会带来延迟和碎片。
- **复用已释放的显存**：用户侧调用 `release(ptr)` 时，不立即 `cudaFree`，而是将块收回池中标记为“空闲”，后续 `allocate` 时优先从池中复用。
- **控制碎片与占用**：按块大小分层管理，并对小块在“空闲总量”超过阈值时做批量释放，在复用与峰值占用之间取得平衡。
- **多卡隔离**：按 `cudaGetDevice()` 得到的设备 ID 分别维护池，每个 GPU 独立池化。

---

## 2. 核心数据结构

### 2.1 CudaMemoryBuffer（`include/base/alloc.h`）

```cpp
struct CudaMemoryBuffer {
  void* data;       // 显存指针
  size_t byte_size; // 块大小（字节）
  bool busy;        // 是否正在被使用（true=已分配出去，false=在池中可复用）
};
```

池中每个块都由该结构描述：指针、大小、以及是否已被某次 `allocate` 分配出去。

### 2.2 CUDADeviceAllocator 内部成员（`include/base/alloc.h`）

```cpp
mutable std::map<int, size_t> no_busy_cnt_;
mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
```

| 成员 | 含义 |
|------|------|
| **big_buffers_map_[device_id]** | 该 GPU 上的**大块池**：单块大小 > 1MB，用于大块请求的复用。 |
| **cuda_buffers_map_[device_id]** | 该 GPU 上的**小块池**：单块大小 ≤ 1MB，用于小块请求的复用。 |
| **no_busy_cnt_[device_id]** | 该 GPU 上当前**处于“未 busy”（已 release、在池中空闲）的小块的总字节数**，用于触发批量释放。 |

大块与小块分开管理，便于采用不同的复用与释放策略。

---

## 3. 分配流程（allocate）

请求大小为 `byte_size` 时，逻辑如下。

### 3.1 大块请求（byte_size > 1MB）

1. 取当前设备 ID：`cudaGetDevice(&id)`。
2. 在 `big_buffers_map_[id]` 中寻找可复用块，条件：
   - `block.byte_size >= byte_size`（容量足够）；
   - `!block.busy`（当前空闲）；
   - `block.byte_size - byte_size < 1 * 1024 * 1024`（浪费小于 1MB，避免大块被拆成很多“零头”）。
3. 若有多个满足条件的块，选择 **byte_size 最小** 的那块（最佳适配，减少浪费）。
4. 若找到：
   - 将该块标记为 `busy = true`；
   - 返回 `block.data`，**不再调用 cudaMalloc**。
5. 若未找到：
   - 调用 `cudaMalloc(&ptr, byte_size)`；
   - 将 `(ptr, byte_size, true)` 加入 `big_buffers_map_[id]`；
   - 返回 `ptr`。

大块策略要点：**只复用“浪费 < 1MB”的块，并优先用最小的可满足块**，在复用与碎片之间折中。

### 3.2 小块请求（byte_size ≤ 1MB）

1. 在 `cuda_buffers_map_[id]` 中做 **首次适配**：找第一个满足  
   `block.byte_size >= byte_size && !block.busy` 的块。
2. 若找到：
   - 将该块标记为 `busy = true`；
   - `no_busy_cnt_[id] -= block.byte_size`（从“空闲小块总大小”中扣掉这块）；
   - 返回 `block.data`，**不调用 cudaMalloc**。
3. 若未找到：
   - 调用 `cudaMalloc(&ptr, byte_size)`；
   - 将 `(ptr, byte_size, true)` 加入 `cuda_buffers_map_[id]`；
   - 返回 `ptr`。

小块策略要点：**首次适配、快速复用**，不追求最小浪费，以降低分配延迟为主。

---

## 4. 释放流程（release）

传入要释放的 `ptr` 时，逻辑如下。

### 4.1 小块池批量释放（触发条件）

在根据 `ptr` 查找所属块**之前**，先检查是否需要对小块池做“瘦身”：

- 对每个设备 ID，若 `no_busy_cnt_[id] > 1024 * 1024 * 1024`（即该设备上当前空闲小块总大小 > 1GB）：
  - 遍历 `cuda_buffers_map_[id]`；
  - 对每个 **!busy** 的块调用 `cudaFree(block.data)`；
  - 只保留 **busy** 的块，得到新的 `cuda_buffers` 列表并写回；
  - 将 `no_busy_cnt_[id]` 置为 0。

这样可以在“空闲小块”积压过多时，一次性把未使用的显存还给驱动，**控制峰值占用和碎片**，同时不影响仍在使用的块。

### 4.2 将 ptr 归还到池中（不立即 cudaFree）

- 在 **cuda_buffers_map_** 中查找 `block.data == ptr`：
  - 若找到：  
    - `no_busy_cnt_[id] += block.byte_size`；  
    - `block.busy = false`；  
    - **不调用 cudaFree**，直接 return。
- 若未在小块池中找到，则在 **big_buffers_map_** 中查找 `block.data == ptr`：
  - 若找到：  
    - `block.busy = false`；  
    - **不调用 cudaFree**，直接 return。
- 若两个池中都没找到（例如非本分配器分配的指针）：
  - 调用 `cudaFree(ptr)`，交给驱动释放。

因此，**凡是从本分配器池中分配出去的 ptr，在 release 时都是“归还到池、标记空闲”**，只有触发批量释放或未入池的指针才会真正 `cudaFree`。

---

## 5. 策略小结

| 方面 | 实现方式 |
|------|----------|
| **复用** | release 不立即 cudaFree，块标记为 !busy，allocate 时优先从池中取。 |
| **大小分层** | >1MB 用 big_buffers_map_，≤1MB 用 cuda_buffers_map_，不同策略。 |
| **大块复用条件** | 仅当“浪费 < 1MB”且容量足够时复用，并选最小满足块，减少碎片。 |
| **小块复用** | 首次适配，优先降低分配延迟。 |
| **峰值与碎片控制** | 小块空闲总量超过 1GB 时，批量 cudaFree 所有空闲小块，再置 no_busy_cnt_ 为 0。 |
| **多 GPU** | 按 device id 分池，各卡独立池化与释放。 |

---

## 6. 与上层的关系

- **Buffer**（`base/buffer.h`）在析构或需要释放时调用 `allocator_->release(ptr_)`，对 CUDADeviceAllocator 而言就是“归还到池”。
- **Tensor** 通过 `allocate(allocator)` 创建 Buffer 时，若使用 CUDADeviceAllocator，则其显存来自上述池化逻辑；Tensor/Buffer 析构或替换时，会 release 对应指针，从而重新进入池中供后续 allocate 复用。

因此，框架对 CUDA 显存的优化集中在 **CUDADeviceAllocator** 的池化与释放策略上，上层无需改动即可受益于更少的 cudaMalloc/cudaFree 和更好的显存复用。
