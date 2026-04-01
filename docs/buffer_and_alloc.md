# Buffer 与 Alloc 关系说明

本文档基于 `stillfantasy/include` 与 `stillfantasy/source` 下的代码，说明 **Buffer** 与 **Alloc（分配器）** 的设计与关系。

---

## 1. 概念总览

| 概念 | 所在头文件 | 职责简述 |
|------|------------|----------|
| **DeviceAllocator（alloc）** | `base/alloc.h` | 设备内存的分配/释放策略，不持有具体内存块 |
| **Buffer** | `base/buffer.h` | 一块具有大小和裸指针的内存块，**通过 Allocator 分配与释放** |
| **Tensor** | `tensor/tensor.h` | 逻辑上的多维数组，**内部持有一个 `Buffer`**，可选用 Allocator 创建或共享 Buffer |
| **Model 的 buffers_** | `model/model.h` | 模型内按名称管理的 **Tensor 槽位**（每个 Tensor 内部再有自己的 Buffer） |

关系链可以概括为：

```
DeviceAllocator (alloc)  →  被 Buffer 使用，用于 allocate/release/memcpy
       ↓
Buffer  →  持有 ptr_ + byte_size_，生命周期内通过 allocator_ 分配/释放
       ↓
Tensor  →  持有 std::shared_ptr<base::Buffer> buffer_，可共享或新建 Buffer
       ↓
Model.buffers_  →  std::map<ModelBufferType, tensor::Tensor>，按类型取放 Tensor
```

---

## 2. Alloc：DeviceAllocator（分配器）

### 2.1 接口定义（`include/base/alloc.h`）

```cpp
class DeviceAllocator {
  virtual void* allocate(size_t byte_size) const = 0;
  virtual void release(void* ptr) const = 0;
  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind, void* stream, bool need_sync) const;
  virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync);
  // ...
  DeviceType device_type_;  // kDeviceCPU / kDeviceCUDA
};
```

- **allocate(byte_size)**：在对应设备上分配 `byte_size` 字节，返回裸指针。
- **release(ptr)**：释放之前由本分配器分配的指针。
- **memcpy / memset_zero**：在 CPU/GPU 间或同设备内做拷贝或清零，由 `MemcpyKind` 区分方向。

分配器只提供“如何分配/释放/拷贝”，不保存具体内存块；谁调用 `allocate`，谁负责在合适时机调用 `release`。

### 2.2 两种具体分配器

- **CPUDeviceAllocator**（`source/base/alloc_cpu.cpp`）  
  - `allocate`：有 `posix_memalign` 时用对齐分配，否则 `malloc`。  
  - `release`：`free(ptr)`。

- **CUDADeviceAllocator**（`source/base/alloc_cu.cpp`）  
  - 内部用 `CudaMemoryBuffer`（data, byte_size, busy）做池化。  
  - 大块（>1MB）走 `big_buffers_map_`，小块走 `cuda_buffers_map_`。  
  - `allocate`：优先从池中找足够大且未 busy 的块，否则 `cudaMalloc` 并加入池。  
  - `release`：不立刻 `cudaFree`，只把对应块标为 non-busy，供后续复用；在特定条件下会批量释放以控制占用。

### 2.3 获取分配器实例

通过工厂获取单例，在模型初始化、Tensor 分配时传入：

```cpp
auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
auto alloc_cu  = base::CUDADeviceAllocatorFactory::get_instance();
```

例如在 `source/model/qwen3.cpp` 的 `init_mem()` 中，根据 `device_type_` 选择 `alloc`，并用其创建各种 Tensor（见下节）。

---

## 3. Buffer：内存块与 Allocator 的绑定

### 3.1 定义（`include/base/buffer.h`）

```cpp
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
  size_t byte_size_ = 0;
  void* ptr_ = nullptr;
  bool use_external_ = false;           // 是否使用外部传入的 ptr，不交给 allocator 释放
  DeviceType device_type_ = ...;
  std::shared_ptr<DeviceAllocator> allocator_;
  // ...
};
```

- **byte_size_**：本块大小（字节）。  
- **ptr_**：实际数据指针，要么由 **allocator_** 分配，要么由外部传入（此时 **use_external_ = true**）。  
- **allocator_**：用于分配、释放以及跨设备/同设备 **memcpy**。

### 3.2 构造与分配（`source/base/buffer.cpp`）

```cpp
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr, bool use_external)
  : byte_size_(byte_size), allocator_(allocator), ptr_(ptr), use_external_(use_external) {
  if (!ptr_ && allocator_) {
    device_type_ = allocator_->device_type();
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size);   // 通过 alloc 分配
  }
}
```

- 若未传 `ptr` 且传了 `allocator`：在构造时就用 **allocator->allocate(byte_size)** 得到 **ptr_**，并记下 **device_type_**。  
- 若传了外部 `ptr` 且 `use_external == true`：不调用分配器，析构时也不释放（见下）。

### 3.3 析构与释放

```cpp
Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->release(ptr_);
      ptr_ = nullptr;
    }
  }
}
```

只有 **非外部指针** 时，Buffer 才在析构时用 **allocator_->release(ptr_)** 释放内存。  
因此：**Buffer 是“持有了一块内存”的抽象，而这块内存的分配/释放策略完全由 Allocator 决定。**

### 3.4 再次分配：allocate()

```cpp
bool Buffer::allocate() {
  if (allocator_ && byte_size_ != 0) {
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size_);
    return (ptr_ != nullptr);
  }
  return false;
}
```

在已有 **byte_size_** 和 **allocator_** 的情况下，可延迟分配或重新分配，同样通过 **allocator_** 完成。

### 3.5 拷贝：copy_from

`copy_from(const Buffer&)` / `copy_from(const Buffer*)` 中，使用 **this->allocator_** 的 **memcpy**，根据源/目标 **device_type** 选择 `MemcpyKind`（CPU2CPU、CPU2CUDA、CUDA2CPU、CUDA2CUDA）。  
因此跨设备或同设备拷贝都统一通过当前 Buffer 所绑定的 Allocator 完成。

### 3.6 小结：Buffer 与 Alloc 的关系

- **Buffer** 持有：大小、指针、设备类型、以及一个 **DeviceAllocator**。  
- **分配**：在构造或 **allocate()** 时调用 **allocator_->allocate(byte_size)**。  
- **释放**：在析构时（且非外部指针）调用 **allocator_->release(ptr_)**。  
- **拷贝**：通过 **allocator_->memcpy** 实现。  

所以：**Buffer 是“一块由某 Allocator 管理生命周期的内存”的封装；Allocator 是策略，Buffer 是使用该策略的具体内存块。**

---

## 4. Tensor 对 Buffer 与 Allocator 的使用

### 4.1 Tensor 与 Buffer

Tensor 内部持有一块存储，由 **std::shared_ptr<base::Buffer> buffer_** 表示（`tensor/tensor.h`）。  
所有元素访问（如 **ptr<T>()**、**ptr<T>(index)**、**index<T>(offset)**）都基于 **buffer_->ptr()**。

### 4.2 通过 Allocator 创建 Buffer（需要分配时）

构造时若 **need_alloc == true** 且传入了 **alloc**，会调用：

```cpp
bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
  // ...
  buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  // Buffer 构造时内部会调用 allocator->allocate(byte_size)
  return (buffer_->ptr() != nullptr);
}
```

即：Tensor 根据自己算出的 **byte_size** 和传入的 **allocator** 创建一个新的 **Buffer**，该 Buffer 在构造时就会用 **alloc** 分配内存。  
例如模型 init 时：

```cpp
tensor::Tensor input_embeddings(activation_dtype_, 1, config_->hidden_dim_, true, alloc);
tensor::Tensor sin_cache(...);
CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));
CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
```

这里的 **alloc** 来自 **CPUDeviceAllocatorFactory** 或 **CUDADeviceAllocatorFactory**，决定这些 Tensor 的 Buffer 在 CPU 还是 CUDA 上。

### 4.3 使用外部指针（不分配，仅包装）

当 **need_alloc == false** 且传入外部 **ptr** 时，走 **init_buffer**：

```cpp
void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                         bool need_alloc, void* ptr) {
  if (!alloc && !need_alloc) {
    buffer_ = std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
    // Buffer(byte_size, nullptr, ptr, use_external=true) → 不分配、不释放
  } else {
    allocate(alloc, true);
  }
}
```

典型用法：权重从文件 mmap 或某块内存映射而来，Tensor 只包装这块内存（**Buffer 用 use_external_=true**），不通过 Allocator 分配/释放。  
例如 `source/op/layer.cpp` 中：

```cpp
std::shared_ptr<base::Buffer> buffer =
    std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
// ...
CHECK(weight.assign(buffer));
```

这里 Buffer 不绑定 allocator，不负责释放 **weight_ptr**。

### 4.4 共享已有 Buffer：assign

```cpp
bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  // 检查 device_type、byte_size 等
  buffer_ = buffer;
  return true;
}
```

多个 Tensor 可以共享同一个 **Buffer**（例如模型里复用一个中间结果 Tensor 的 buffer）。

### 4.5 设备迁移：to_cpu / to_cuda

在 `tensor.cpp` 中：

- **to_cuda**：用 **CUDADeviceAllocator** 分配一块新 Buffer，用 **alloc->memcpy(..., kMemcpyCPU2CUDA)** 拷过去，然后 **this->buffer_ = cu_buffer**。  
- **to_cpu**：用 **CPUDeviceAllocator** 分配新 Buffer，**memcpy(..., kMemcpyCUDA2CPU)**，再替换 **buffer_**。

这里再次体现：**谁要分配内存，就传对应的 Allocator；Buffer 负责持有这块内存并在析构时通过自己的 allocator_ 释放（若非外部指针）。**

### 4.6 clone、reshape

- **clone()**：用 **buffer_->allocator()** 创建新 Buffer，再 **copy_from**，得到同设备、同大小的一份拷贝。  
- **reshape()**：若新 size 更大，会 **make_shared<Buffer>(new_byte_size, buffer_->allocator())**，然后 **allocate()** 和 **copy_from**，再替换 **buffer_**。

两者都延续“用当前 Buffer 的 Allocator 分配新 Buffer”的模式。

---

## 5. Model 中的 “buffer”（ModelBufferType）

Model 里还有一层“buffer”概念，与 **base::Buffer** 不同：

- **buffers_** 类型为 **std::map<ModelBufferType, tensor::Tensor>**（`model/model.h`）。  
- **ModelBufferType** 是枚举：如 kInputTokens、kSinCache、kKeyCache、kForwardOutput 等（`base/base.h`）。  
- **get_buffer(ModelBufferType)** 返回 **tensor::Tensor&**；**insert_buffer(ModelBufferType, tensor::Tensor)** 把 Tensor 放进 map。

因此：

- **Model 的 buffer** = 按名称（ModelBufferType）管理的一组 **Tensor 槽位**。  
- 每个 **Tensor** 内部再有一个 **base::Buffer**（即 **buffer_**），那块才是真正由 **DeviceAllocator** 分配或外部挂接的内存。

关系可以写成：

```
Model.buffers_[ModelBufferType::kSinCache]  →  tensor::Tensor
                                                    ↓
                                            tensor.buffer_  →  base::Buffer
                                                                    ↓
                                                            allocator_->allocate/release
```

---

## 6. 总结关系图

```
                    DeviceAllocator (alloc)
                    - CPUDeviceAllocator / CUDADeviceAllocator
                    - allocate(byte_size) / release(ptr) / memcpy
                                    │
                    ┌───────────────┴───────────────┐
                    │ 被使用于                       │
                    ▼                               ▼
              Buffer 构造/allocate()           Buffer::copy_from
              Buffer 析构 release              (memcpy 方向由 device_type 决定)
                    │
                    │ 持有 ptr_, byte_size_, allocator_
                    ▼
    Tensor.buffer_ (std::shared_ptr<base::Buffer>)
    - 通过 Tensor::allocate(allocator) 创建 Buffer（内部用 alloc 分配）
    - 或 Tensor::assign(shared_buffer) 共享 Buffer
    - 或 init_buffer(..., ptr) 包装外部指针（Buffer 不分配不释放）
                    │
                    │ 多个 Tensor 可被模型按“名字”管理
                    ▼
    Model.buffers_ = map<ModelBufferType, tensor::Tensor>
    （这里的“buffer”是 Tensor 槽位，不是 base::Buffer 本身）
```

**一句话**：**Alloc（DeviceAllocator）定义“怎么分配/释放/拷贝”；Buffer 使用 Allocator 持有并管理一块具体内存；Tensor 持有或共享 Buffer，并在需要时通过传入的 Allocator 创建新 Buffer；Model 的 buffers_ 是按类型索引的 Tensor 容器，每个 Tensor 内部再带一个 base::Buffer。**
