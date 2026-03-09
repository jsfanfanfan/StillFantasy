# StillFantasy 在本仓库上的改进说明

本文档说明在**原课程项目《动手自制大模型推理框架》**基础上的改进内容，并致谢原项目。

---

## 致谢原项目

本仓库基于 **《动手自制大模型推理框架》** 课程项目进行扩展。原项目由 KuiperInfer 作者等人推出，支持：

- **Llama2 / Llama3.x** 与 **Qwen2.5** 等大模型
- **C++20**、CMake 工程化、CPU/CUDA 双后端
- **Int8 量化**（Q8_0 风格分组量化）
- 全手写 CUDA 算子与完整推理链路

感谢原课程与开源社区提供的优秀基础框架与教程。

- 课程介绍与目录见原 [readme.md](readme.md)
- 模型导出、编译、单序列推理等用法仍以原 readme 为准

---

## 本仓库主要改进

在保留原项目模型实现与算子的前提下，本仓库增加了 **推理引擎层**，用于多请求调度与 KV 缓存管理，主要包含以下部分。

### 1. 块式 KV 缓存与块管理器（Block / BlockManager）

- **位置**：`stillfantasy/include/engine/block.h`、`stillfantasy/source/engine/block.cpp`
- **思路**：借鉴 **nano-vllm / StillFantasyInfer** 的块式设计。
- **内容**：
  - **Block**：固定大小（默认 256 token）的 KV 缓存块，包含 `block_id`、`ref_count`、`hash`，用于前缀复用。
  - **BlockManager**：块池的分配/释放、按 hash 查找已缓存块，支持多序列共享物理块。
- **作用**：为 Prefix Caching 和连续 batching 提供底层存储与复用能力。

### 2. 序列与调度器（Sequence / Scheduler）

- **位置**：`stillfantasy/include/engine/sequence.h`、`stillfantasy/include/engine/scheduler.h`、`stillfantasy/source/engine/scheduler.cpp`
- **Sequence**：表示一条请求/序列，维护 `block_table`（逻辑块到物理 block_id 的映射）、token 列表、采样参数、状态（等待 / 运行 / 结束）等。
- **Scheduler**：
  - 维护**等待 prefill 的队列**与**正在 decode 的序列**，支持多序列同时运行（调度层面）。
  - 每轮：对等待队列中的请求做 **prefill**（含前缀块复用），对每条运行中序列各执行一步 **decode**。
  - decode 前将该序列的 blocks 拷贝到 Model 的 KV cache，decode 后将新 token 的 KV 写回对应 block。

### 3. Prefix Caching（前缀缓存）

- 在 prefill 阶段按**块**计算内容 hash，相同前缀命中已有块则直接复用，避免重复计算。
- 减少相同或相似 prompt 的重复编码，提升多请求、重复前缀场景下的效率。

### 4. Continuous Batching（连续批调度）

- 多条请求可同时处于「等待」或「运行」状态；每轮对多条运行中序列**轮流**各执行一步 decode。
- 当前实现为**调度层多序列**，单次 model forward 仍为单序列（逐条拷贝 block → model cache 再推理）。
- 通过 `set_max_running(n)` 限制最大同时运行序列数，块数由 `max_num_blocks` 控制。

### 5. 多序列 + 前缀缓存 Demo（main_engine）

- **位置**：`demo/main_engine.cpp`
- **用法**：`./main_engine checkpoint_path tokenizer_path [num_sequences]`
- **功能**：一次性提交多条 prompt，演示 Prefix Caching + Continuous Batching；输出每条序列的 TTFT、TPOT、吞吐（tokens/s）及汇总信息。
- **编译**：需在 CMake 中开启 `LLAMA3_SUPPORT`、`QWEN2_SUPPORT` 或 `QWEN3_SUPPORT` 之一，才会编译 `main_engine` 目标。

---

## 模块结构概览

```
stillfantasy/
├── include/engine/
│   ├── block.h      # Block / BlockManager
│   ├── scheduler.h  # Scheduler、block ↔ model cache 接口
│   └── sequence.h   # Sequence、SamplingParams 等
├── source/engine/
│   ├── block.cpp
│   └── scheduler.cpp
demo/
└── main_engine.cpp   # 多序列 + Prefix Cache 示例
```

---

## W8A16（FP16 / BF16）激活量化

在现有 **W8A32**（int8 权重量化 + fp32 激活）基础上，支持 **W8A16**：权重大矩阵仍为 int8 + 每 group 一个 fp32 scale，**激活**（embedding 输出、各层输入/输出、KV cache、logits 等）统一为 16 位，可选 **FP16** 或 **BF16**。

- **使用方式**：构造/初始化模型时指定激活 dtype；Demo 支持可选参数：
  - `./still_fantasy checkpoint_path tokenizer_path [--fp16]` 或 `[--bf16]`
  - `./main_engine checkpoint_path tokenizer_path [num_sequences] [--fp16|--bf16]`
- **约束**：W8A16 仅支持 **CUDA**；CPU 下若指定 `--fp16`/`--bf16` 会报错。可与量化模型（int8 权重）搭配使用。
- **实现要点**：`Model::init(device_type, activation_dtype)`；Engine 的 BlockManager KV 池与 model 的 `activation_dtype` 一致；采样前将 A16 logits 转为 float 再调用现有 Sampler。

---

## 使用与阅读建议

- **单序列推理、模型导出、量化、编译**：以原 [readme.md](readme.md) 为准。
- **多序列与前缀缓存**：参考 `demo/main_engine.cpp` 的调用方式及 `engine/scheduler.h` 的接口说明。
- **实现细节**：可结合 `engine/block.h`、`engine/scheduler.cpp` 中的注释理解块管理与调度流程。

再次感谢原《动手自制大模型推理框架》课程与开源贡献。
