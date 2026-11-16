# Megatron-FSDP

> 来源：[NVIDIA Megatron-Core 开发者指南 — Megatron-FSDP](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/megatron_fsdp.html)

## 目录

- [✨ 概述（Overview）](#-概述overview)
  - [🧩 兼容性（Compatibility）](#-兼容性compatibility)
  - [💡 特性（Features）](#-特性features)
  - [⏱️ 优化项（Optimizations）](#️-优化项optimizations)
- [🚀 快速开始（Quick Start）](#-快速开始quick-start)
  - [📦 安装（Installation）](#-安装installation)
  - [🎛️ Megatron-FSDP `fully_shard`](#️-megatron-fsdp-fully_shard)
  - [🤖 Megatron-LM](#-megatron-lm)
  - [🤖 Megatron-Core](#-megatron-core)
  - [🔁 检查点转换（Checkpoint Conversion）](#-检查点转换checkpoint-conversion)
- [Megatron-FSDP 特性指南与 API](#megatron-fsdp-特性指南与-api)
  - [FSDP 基础（FSDP Fundamentals）](#fsdp-基础fsdp-fundamentals)
  - [FSDP 单元模块（FSDP Unit Modules）](#fsdp-单元模块fsdp-unit-modules)
  - [数据并行分片策略（Data-Parallel Sharding Strategies）](#数据并行分片策略data-parallel-sharding-strategies)
  - [混合精度与量化（Mixed-Precision & Quantization）](#混合精度与量化mixed-precision--quantization)
  - [NCCL](#nccl)

---

## ✨ 概述（Overview）

Megatron-FSDP 是 NVIDIA 开发的分布式并行库，以原生 PyTorch 编写，提供了全分片数据并行（Fully Sharded Data Parallelism，FSDP）的高性能实现。它与多种深度学习框架及并行库（如 Megatron-Core）无缝交叉兼容，并针对性能进行了优化，可在 NVIDIA GPU 上以数据中心规模支持超大规模 PyTorch 模型的训练与推理。

- PyPI: https://pypi.org/project/megatron-fsdp/
- 源代码：https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src

### 🧩 兼容性（Compatibility）

- PyTorch DeviceMesh、DTensor 以及分布式检查点（Distributed Checkpoint，DCP）
- Megatron Core
- TransformerEngine
- NVIDIA NeMo Framework 容器

### 💡 特性（Features）

- **高性能与可扩展（Performant & Scalable）**：针对 NVIDIA CUDA 优化，具备高效的内存管理与性能表现。从单计算节点到整个数据中心都能实现近线性扩展。
- **多算法一体（Multiple Algorithms in One）**：支持按需对优化器状态、梯度和模型参数进行分片（FSDP），包括分层数据并行策略，例如混合分片数据并行（Hybrid-Sharded Data Parallelism，HSDP）和混合 FSDP（HFSDP / 全分片优化器状态），用以优化节点内与节点间的内存、通信与性能。
- **“自带并行”（Bring Your Own Parallelism）**：与 PyTorch、Megatron-LM、Megatron-Bridge 和 TransformerEngine 无缝协作，也可插入 HuggingFace Transformers、TorchTitan 等其他框架。
- **简洁而强大（Simple & Powerful）**：与 PyTorch FSDP 类似，`fully_shard` API 不依赖任何复杂的训练框架或分布式环境。

### ⏱️ 优化项（Optimizations）

- **TransformerEngine 混合精度与融合核（Mixed-Precision & Fused Kernels）**：对 TransformerEngine 提供的 MXFP8、NVFP4 及其他多种量化配方与融合核具备原生的性能与内存最优兼容性。
- **高级分桶（Advanced Bucketing）**：可按 `dtype` 定制、精度感知的分桶系统，用以调节集合通信的内存开销、数值精度与延迟。避免集合通信前后的冗余 `COPY` 操作，同时保持与 DTensor 特性（如 Torch 分布式检查点 DCP）的兼容。
- **缓冲区管理（Buffer Management）**：高效利用存储空间，并通过 NCCL 用户缓冲区注册（User Buffer Registration）实现直接向 NCCL 托管内存通信，达成真正的零 `COPY` 数据搬运。自 NCCL `v2.27` 引入的 NCCL 对称内存（Symmetric Memory）通信采用对称核，大幅降低 SM 占用，并包含诸如线路上高精度（`FP32`）规约等网络优化。
- **借助 SHARP 优化通信与 SM 占用（Optimized Communication & SM Utilization via SHARP）**：利用 SHARP（可扩展分层聚合与规约协议，Scalable Hierarchical Aggregation and Reduction Protocol）将 FSDP 集合通信卸载到网络交换机（InfiniBand 或 NVLink-Switch），在多节点 NVLink（MNNVL）系统（Grace-Blackwell、Vera-Rubin 等）上将 GPU 流多处理器（SM）的占用从 16-32 显著降低到 1-6，从而在大规模横向扩展负载中降低通信延迟，并释放 GPU 处理器用于重叠计算（GEMM）核。当 FSDP 分片域同时跨越 NVLink 与 InfiniBand 时，分层 SHARP 集合通信（NVL-SHARP 与 IB-SHARP）会在整个系统拓扑上优化通信路径。
- **混合 FSDP（HFSDP）**：这是混合分片数据并行（HSDP）的一种变体，它进一步将优化器状态在节点内与节点间的数据并行 rank 之间分片，弥合了 HSDP 与 FSDP 之间的内存-通信权衡，以极小的性能代价换取内存效率。

## 🚀 快速开始（Quick Start）

### 📦 安装（Installation）

#### NeMo Framework 容器

Megatron-FSDP 已随 Megatron-Core 预装于 [NVIDIA NeMo Framework 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) 中。

#### Megatron-Core

Megatron-FSDP 与 Megatron-Core 捆绑发布，可通过 `pip` 安装：

```bash
# 通过 PyPI 安装
pip install --no-build-isolation megatron-core[mlm,dev]

# 从源码安装
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation .[mlm,dev]
```

在 Python 中导入 Megatron-FSDP：

```bash
import megatron.core.distributed.fsdp.src.megatron_fsdp
```

#### PyPI

将 Megatron-FSDP 作为独立包安装，以使用 `fully_shard` API：

```bash
pip install megatron-fsdp
```

在 Python 中导入 Megatron-FSDP：

```bash
import megatron_fsdp
```

### 🎛️ Megatron-FSDP `fully_shard`

Megatron-FSDP 提供简洁的 `fully_shard` API，仅需极少的代码即可无缝启用 FSDP。

```python
import torch
from megatron_fsdp import (
    fully_shard_model,
    fully_shard_optimizer,
)

# 初始化 Torch 分布式。
torch.distributed.init_process_group()
torch.cuda.set_device(torch.distributed.get_rank())

# 全分片模型。
model = torch.nn.Transformer()
fsdp_model = fully_shard_model(
    module=model,
    fsdp_unit_modules=[
        torch.nn.TransformerEncoder,
        torch.nn.TransformerDecoder
    ]
)

# 全分片优化器。
toy_adam = torch.optim.AdamW(params=fsdp_model.parameters(), lr=0.01)
optimizer = fully_shard_optimizer(optimizer=toy_adam)

# 前向传播。
inp = torch.randn(1, 512, 512).to("cuda")
tgt = torch.randn(1, 512, 512).to("cuda")
output = fsdp_model(inp, inp)

# 反向传播。
torch.nn.functional.mse_loss(output, tgt).backward()

# 优化器步进。
optimizer.step()
optimizer.zero_grad()

# 检查点保存模型与优化器。
torch.distributed.checkpoint.save({
    "model": fsdp_model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, checkpoint_id="ckpt/")

# 加载保存的检查点。
ckpt = {
    "model": fsdp_model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.distributed.checkpoint.load(state_dict=ckpt, checkpoint_id="ckpt/")
fsdp_model.load_state_dict(ckpt["model"], strict=False)
optimizer.load_state_dict(ckpt["optimizer"])
```

> ℹ️ `fully_shard` 是实验性 API。随着我们持续打磨用户体验，请留意后续更新！关于使用 `fully_shard` 的更多 Megatron-FSDP 示例，请参阅我们的单元测试套件：[`tests/unit_tests/distributed/mfsdp_v1/test_mfsdp_fully_shard.py`](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/unit_tests/distributed/mfsdp_v1/test_mfsdp_fully_shard.py)

### 🤖 Megatron-LM

Megatron-FSDP 已深度集成到 Megatron-Core 中。要在 Megatron 中启用 FSDP（对优化器状态、梯度与计算参数进行分片），使用以下参数：

```bash
# 在 Megatron-LM 中使用 Megatron-FSDP 训练模型。
--use-megatron-fsdp
--data-parallel-sharding-strategy {no_shard, optim, optim_grads, optim_grads_params}
--ckpt-format fsdp_dtensor
```

> ℹ️ 加载 `fsdp_dtensor` 检查点时，如果检查点无法提供模型所需的每一个模型权重，就会报错。与 `torch_dist` 不同，对于该格式，默认的 `--dist-ckpt-strictness assume_ok_unexpected` 行为等同于 `raise_unexpected`，因为 Torch DCP 会静默跳过缺失的权重，使其保持初始化值不变——这会表现为收敛性退化而非崩溃。检查点中持有、但某个 rank 的模型并不需要的权重（例如分片的专家）则始终没有问题。可传入 `--dist-ckpt-strictness ignore_all` 以关闭该检查，或在只需加载模型一部分时使用 `--finetune`。

使用 Megatron-FSDP 并采用推荐设置的完整 Llama-8B 与 DeepSeek-V3 训练脚本可在 [Megatron-LM/examples/megatron_fsdp](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/megatron_fsdp) 中找到。

#### Megatron-LM 推荐配置

与 Megatron-FSDP 搭配使用的常用选项包括：

```bash
# 取消设置 CUDA_DEVICE_MAX_CONNECTIONS，以确保流独立 / FSDP 计算与通信的充分并行化。不过这可能会略微影响 TP 和 CP 的性能。
unset CUDA_DEVICE_MAX_CONNECTIONS

# Meta 设备初始化 —— 以分片方式将大模型加载到 CUDA 设备上，避免 OOM。
--init-model-with-meta-device

# 按 token 计算损失 / 不做梯度缩放 —— 在梯度规约期间关闭 DP 缩放，它可能会消耗 SM 资源。
--calculate-per-token-loss

# 根据模型参数的精度（通常为 BF16）将梯度规约与累加精度降低到推荐数据类型。减少反向传播期间的通信量。可通过 `--megatron-fsdp-main-grads-dtype` 和 `--megatron-fsdp-grad-comm-dtype` 进一步定制，这两者由该参数启用。
--grad-reduce-in-bf16

# 注册 NCCL 用户缓冲区与 Megatron-FSDP 双缓冲区，以启用零拷贝对称核及经由 SHARP 的低 SM 占用。可提升整体性能，但由于双缓冲会增加内存开销，且与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 不兼容。
--use-nccl-ub
--fsdp-double-buffer
--fsdp-manual-registration
```

### 🤖 Megatron-Core

Megatron-FSDP 提供了更底层的 `FullyShardedDataParallel` 类 API，可配合 Megatron-LM 训练循环的简化版本使用。

```python
# 初始化模型与优化器。
ddp_config.use_megatron_fsdp = True
# Megatron-FSDP 基础分片策略：
# no_shard, optim, optim_grads, optim_grads_params
ddp_config.data_parallel_sharding_strategy = "optim_grads_params"
model = GPTModel(transformer_config)
model = FullyShardedDataParallel(
    transformer_config,
    model,
    ddp_config,
    fsdp_unit_modules = [TransformerLayer, LanguageModelEmbedding],
)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer = DistributedOptimizer(optimizer, [model], [model.param_and_grad_buffer])

# 训练循环
def train_step(inputs, labels):
    optimizer.zero_grad()
    for mbs_input, mbs_label in zip(inputs, labels):
        outputs = model(mbs_input)
        loss = loss_fn(outputs, mbs_label)
        loss.backward()
    optimizer.step()

# 保存与加载模型及优化器 state dict
def model_and_optimizer_state_dict():
    state_dict = {
        "model": model.sharded_state_dict(),
        "optimizer": optimizer.sharded_state_dict(),
    }
    return state_dict

def load_model_and_optimizer_state_dict(state_dict):
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
```

### 🔁 检查点转换（Checkpoint Conversion）

Megatron-FSDP 的检查点机制支持 [PyTorch 分布式检查点（DCP）](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)。在 Megatron-LM 中，这对应 `--ckpt-format fsdp_dtensor` 检查点格式。

#### 将 Torch DCP 转换为 Torch Save（非分布式）检查点

PyTorch 提供了在 Torch DCP 检查点与普通 Torch 检查点之间互相转换的工具：

```bash
python -m torch.distributed.checkpoint.format_utils --help
usage: format_utils.py [-h] {torch_to_dcp,dcp_to_torch} src dst

positional arguments:
  {torch_to_dcp,dcp_to_torch}
                        Conversion mode
  src                   Path to the source model
  dst                   Path to the destination model

options:
  -h, --help            show this help message and exit
```

例如：

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch dcp_ckpt/ torch_ckpt.pt
```

或者：

```python
from torch.distributed.checkpoint.format_utils import (
    dcp_to_torch_save,
    torch_save_to_dcp,
)

# 将 DCP 模型检查点转换为 torch.save 格式。
dcp_to_torch_save(CHECKPOINT_DIR, TORCH_SAVE_CHECKPOINT_PATH)

# 将 torch.save 模型检查点转换回 DCP 格式。
torch_save_to_dcp(TORCH_SAVE_CHECKPOINT_PATH, f"{CHECKPOINT_DIR}_new")
```

随后可将 Torch Save 检查点转换为 HuggingFace SafeTensors 或其他检查点格式以便分发。

> ℹ️ Megatron-FSDP 的检查点会在 state dictionary 中所有模型参数名前加上 `module.` 前缀；将 Torch Save 检查点转换为 Megatron-FSDP 的 Torch DCP 检查点仍需进一步测试，正在开发中（Work-in-progress）！

#### 将 N 维并行（`torch_dist`）转换为 Megatron-FSDP（`fsdp_dtensor`）检查点

作为检查点转换的前置条件，在使用 3D 并行（DDP、TP、PP）和/或 EP 训练时，需转储参数组映射（parameter group mapping）：

```bash
--dump-param-to-param-group-map /path/to/param_to_param_group_map
```

并在 `/path/to/param_to_param_group_map` 目录中将该映射转换为 `param_to_param_group_map.json` JSON 文件：

```bash
python tools/checkpoint/checkpoint_inspector.py print-torch-dcp-in-json /path/to/param_to_param_group_map
```

> ℹ️ 如果你已经拥有 `torch_dist` 检查点，只需指定 `--dump-param-to-param-group-map /path/to/param_to_param_group_map` 标志，并运行一次简单的训练或检查点实验，即可在无需完整预训练的情况下创建所需的 `param_to_param_group_map`。

最后，使用 `param_to_param_group_map.json` 将你的 `torch_dist` 检查点转换为 `fsdp_dtensor` 格式：

```bash
torchrun --nproc_per_node=8 --nnodes=1 \
    tools/checkpoint/checkpoint_inspector.py \
    convert-torch-dist-to-fsdp-dtensor (--swiglu) \     # --swiglu 用于特定模型。
    /path/to/input_torch_dist_checkpoint/ \
    /path/to/output_fsdp_dtensor_checkpoint/ \
    --param-to-param-group-map-json /path/to/param_to_param_group_map.json
```

> ℹ️ 关于多节点转换任务，请参阅 [Megatron-LM/examples/megatron_fsdp](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/megatron_fsdp) 中的 DeepSeek-V3 示例脚本（`sbatch_checkpoint_convert.sh`）。

## Megatron-FSDP 特性指南与 API

| 优化项 | 说明 | `Megatron-Core` 配置 | `fully_shard` 配置 |
| --- | --- | --- | --- |
| Megatron-FSDP | 在 Megatron-LM 中使用 Megatron-FSDP。 | `--use-megatron-fsdp` | `fully_shard_model(module)` |
| Megatron-FSDP 检查点 | 使用 [Torch 分布式检查点（DCP）](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) 保存与加载非均匀的 DTensor 检查点。 | `--ckpt-format fsdp_dtensor` | `preproc_state_dict_for_dcp_ckpt=True` |
| Meta 设备初始化 | Megatron-FSDP 以分片方式将 meta 初始化的模型加载到 CUDA 设备，避免大模型 OOM。需要实现 `Module.reset_parameters()` 以进行逐模块的分片初始化。 | `--init-model-with-meta-device` | `init_model_with_meta_device=True` |
| 分布式优化器 | Megatron-FSDP 使用 Megatron-Core 的 `DistributedOptimizer`。使用 Megatron-FSDP 时自动设置。 | `--use-distributed-optimizer` | `fully_shard_optimizer(optimizer)` |

### FSDP 基础（FSDP Fundamentals）

![FSDP Pipeline](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/DDP_vs_FSDP.png)

*分布式数据并行（DDP）与全分片数据并行（FSDP）的对比。在 DDP 中梯度被all-reduce（all-reduce），而在 FSDP 中梯度被分片并进行reduce-scatte（reduce-scatter）。*

> 来源：Meta AI, Ott, Myle, et al. “Fully Sharded Data Parallel: Faster AI Training with Fewer GPUs.” Facebook Engineering, 15 July 2021, https://engineering.fb.com/2021/07/15/open-source/fsdp/.

全分片数据并行（FSDP）是一种分布式数据并行（DDP），它将优化器状态、权重梯度（`wgrad`）以及模型权重在那些摄取数据并行样本进行数据并行训练或推理的设备间进行分片。激活（`fprop`）与数据梯度（`dgrad`）不会被分片或分发，而是为反向传播保留，但它们可以在反向传播期间重新计算、卸载到 CPU，或使用张量并行（TP）、上下文并行（CP）、专家并行（EP）等其他并行方式进行分片/路由。

![ZeRO-3 Model State](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/zero3_model_state.png)

*ZeRO-1（优化器状态）、ZeRO-2（优化器状态与梯度）和 ZeRO-3（优化器状态、梯度与参数）的分片内存分布。*

> 来源：Zero-Redundancy Optimizer Model State Partition Diagram. From The Ultra-Scale Playbook: Training LLMs on GPU Clusters by Tazi, Nouamane, et al. HuggingFace, 2025, https://huggingface.co/spaces/nanotron/ultrascale-playbook.

FSDP 的核心原则是：

- 任意时刻只有模型状态中一个沿深度方向的小片段可以处于未分片（un-sharded）状态。
- 通信应与计算重叠。

由这些核心原则可以推导出软件层面的要求：

- 被 FSDP 分片的模型状态以分片的形式直接在各设备上初始化。
- 模型参数会在前向前（pre-forward）与反向前（pre-backward）按预先指定的分组或模块进行全收集（all-gather，AG），从而在训练或推理期间的任意时刻只解除模型状态中一个小片段的分片。在 `fprop` 与 `dgrad` 计算完成后，这些未分片的权重会被立即释放。
- `wgrad` 会在反向后（post-backward）立即按预先指定的分组或模块进行reduce-scatte（reduce-scatter，RS）并累加，以限制任意时刻未分片梯度的数量。
- 分布式优化器（即针对分片的模型状态初始化、并支持分布式机制的优化器）使用已规约的梯度分片来更新分片的模型状态，从而实现数据并行（DP）。
- 计算与通信在多个 CUDA 流上重叠，消耗多个流多处理器（SM）。后续分组或模块的权重会被预取（pre-fetch），理想情况下可将 FSDP 所需的通信延迟隐藏在模型计算核（GEMM）之后。

FSDP 还可以被视为将 DDP 中使用的all-reduce（all-reduce）集合通信分解为梯度reduce-scatte、分布式优化步骤以及参数全收集。

![FSDP RS & AG](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/FSDP_Allreduce.png)

> 来源：Feng, Wei, Will Constable, and Yifan Mao. “Getting Started with Fully Sharded Data Parallel (FSDP2).” PyTorch Tutorials, 17 Mar. 2022, https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html.

### FSDP 单元模块（FSDP Unit Modules）

| 优化项 | 说明 | `Megatron-Core` 配置 | `fully_shard` 配置 |
| --- | --- | --- | --- |
| FSDP 单元模块 | 一组 `str` 或 `class` 导入路径，指向被视为 FSDP 单元模块、并由 Megatron-FSDP 分片的 `torch.nn.Module`。不属于任何 FSDP 单元的参数与子模块不会被分片。在 Megatron-LM 中默认采用受支持的 Megatron-Core 模块（`TransformerLayer` 等）。 | — | `fsdp_unit_modules=[...]` |
| FSDP 双缓冲分配器 | Megatron-FSDP 使用双缓冲分配器（double-buffer allocator），它会持久分配一对缓冲区，分配给交替的 FSDP 单元，用于临时存放参数与梯度。配合 NCCL 用户缓冲区注册时自动启用。 | `--fsdp-double-buffer` | `fsdp_double_buffer=True` |
| FSDP Max Pool 分配器 | Megatron-FSDP 使用 `MaxPoolAllocator`，通过取所有层的最大值来支持对混合/非对称模型架构进行双缓冲。会自动设置 `--fsdp-double-buffer`。 | `--megatron-fsdp-max-pool-double-buffer` | `maxpool_double_buffer=True` |
| 参数全收集重叠 | 是否将参数全收集与计算重叠。在 ZeRO-3 分片策略下自动激活。 | `--overlap-param-gather` | `overlap_param_gather=True` |
| 梯度reduce-scatte重叠 | 是否将梯度reduce-scatte或all-reduce与计算重叠。在 ZeRO-2 与 ZeRO-3 分片策略下自动激活。 | `--overlap-grad-reduce` | `overlap_grad_reduce=True` |
| FSDP 通信大小 | 定制 Megatron-FSDP 中 AG 与 RS 通信的大小（以 `numel()` 元素个数计），通过限制 AG 与 RS 并发预取或规约的元素数量来实现。实际上建议了并发处理多少个 FSDP 单元，可更早发起集合通信并提升性能。可按系统内存与性能需求选择性调优该值。 | `--suggested-communication-unit-size` | N/A（仅 Megatron-Core） |

> **任意时刻只有模型状态中一个沿深度方向的小片段可以处于未分片状态。**

FSDP 单元模块代表模型状态中被作为一个（合并的）分组进行计算与通信的片段：需要计算时解除分片，计算后重新分片以释放内存供后续模型状态使用。隐含地，FSDP 单元模块同时也是一种建模契约，要求被 FSDP 管理的单元模块参数不得在前向传播、反向传播或优化步骤的范围之外被访问或修改。

Megatron-FSDP 通过 `fsdp_unit_modules` 参数接受一组表示 FSDP 单元模块的 `str` 或 `class` 路径，该参数目前在 Megatron-Core 中硬编码为受支持的模型类（如 `TransformerLayer`）。它对模型执行深度优先遍历（通过 `torch.nn.Module.named_modules()`），并将每个匹配模块的参数分组以进行分片与合并通信。嵌套单元按优先级解析：如果某个模块匹配某个 FSDP 单元类，但它已是先前注册的某个 FSDP 单元的子模块，则会被跳过——因此任何模块子树中最外层（也必然是最大）的 FSDP 单元类成为有效的 FSDP 单元模块。

> **通信应与计算重叠。**

一旦模型被划分为单元模块，计算便会基于 FSDP 单元模块的粒度与通信重叠。根据计算核与通信核的大小，微调单元模块大小与分组配置会影响性能，并在使用 FSDP 时引出重叠与内存之间的权衡。

![FSDP Streams](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/fsdp_streams.png)

*在 SM 资源发生争用这一简化（且最坏情况）的场景下，计算流与通信流中每个被合并并归类的彩色编码块，对应一个 FSDP 单元模块。*

计算-通信重叠通过 CUDA 流来编排，流会捕获并并行化串行操作。`{DP-Inner, DP-Outer}` 与 `{AG, RS}` 的所有组合所关联的全部集合通信都由各自独立的流以及通信器/`ProcessGroup` 来调度与跟踪。

- 参数在 `fprop` 与 `dgrad` 计算之前解除分片。为使预取全收集与计算重叠，任意时刻至少需要两个 FSDP 单元规模的未分片权重内存。
- 梯度在 `wgrad` 计算之后被规约并分片。为使梯度reduce-scatte与 `wgrad` 计算重叠，任意时刻至少需要两个 FSDP 单元规模的未分片梯度内存。

#### FSDP 模块钩子（FSDP Module Hooks）

为实现这些“以单元为周期”的机制，Megatron-FSDP 使用 `Module` 钩子来安装各类（前置与后置的）前向与反向操作：

- **Pre-Forward（前向前）**

  解除当前 FSDP 单元模块、以及（通过预取）前向后续 FSDP 单元模块的模型参数分片。

  - 当 `MegatronFSDP.forward()` 被调用时，Megatron-FSDP 会将所有参数引用切换为指向用于前向与反向传播的未分片 `Tensor` 计算权重。

- **Post-Forward（前向后）**

  若该模块是 FSDP 单元，则在前向传播后重新分片模型权重。非单元模块保持持久未分片。

  当在反向传播期间使用激活重计算时，计算 `fprop` 与 `dgrad` 都需要这些参数，因此参数会在 Post-Backward 期间重新分片。

  - 针对 `TransformerEngine` 中特定的量化配方，释放量化参数（在 FSDP / ZeRO-3 中）的转置缓存。

- **Pre-Backward（反向前）**

  解除当前 FSDP 单元模块、以及（通过预取）反向后续 FSDP 单元模块的模型参数分片。

  通过由输出 `dgrad` 触发的 `torch.autograd.graph.register_multi_grad_hook` 实现，并经由 `Module` 的前向后（post-forward）钩子安装。

- **Post-Backward（反向后）**

  若该模块是 FSDP 单元，则在反向传播后重新分片模型权重。非单元模块保持持久未分片。

  通过注入一个在 `Module` 前向前（pre-forward）钩子期间安装的 Autograd 函数（`RegisterFSDPBackwardFunction`）实现。

  - 在反向传播后规约梯度。

  通过由 `param.grad` 触发的 `Tensor.register_post_accumulate_grad_hook`，以及一个在 Pre-Backward 期间安装的根级反向后钩子（`torch.autograd.Variable._execution_engine.queue_callback`）实现。

- **State Dictionary（状态字典）**

  当（任何由 Megatron-FSDP 管理的模块的）`module.state_dict()` 被调用时，Megatron-FSDP 会将所有参数引用切换为指向分片的 `DTensor` 主权重（main weights），以用于分布式优化与检查点。

  - 当 `MegatronFSDP.load_state_dict()` 被调用时，主权重与计算权重都会更新。使用量化模型计算时，主权重会被量化并分片。

#### 双缓冲（Double Buffering）

Megatron-FSDP 默认使用基于 `Tensor._typed_storage()._resize_(bytes)` 的分配器，用以即时分配与释放内存，而无需对未分片的参数与梯度依赖 `CUDACachingAllocator`。（缓存碎片化与垃圾回收会拖延大量的 `cudaMalloc` 与 `cudaFree` 操作，这些操作可能阻塞程序并造成内存峰值，尤其是在内存占用达到上限时。）然而，修改缓冲区底层存储与 NCCL 对称注册或 CUDA 图化（graphability）不兼容，后者要求运行时存在持久状态。

为支持这些优化，Megatron-FSDP 使用双缓冲（double-buffering），它以交替模式为 FSDP 单元分配 2 个持久分配的缓冲区，从而硬性限制参数与梯度缓冲分配的内存开销，并确保同时计算或通信的 FSDP 单元不超过 2 个。

![FSDP Double Buffering](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/fsdp_double_buffer.png)

*Megatron-FSDP 中双缓冲的可视化。偶数索引与奇数索引的 FSDP 单元共享同一对未分片的参数与梯度缓冲区，在运行时按需覆盖已有数据。Megatron-FSDP 确保运行时任意时刻未分片的 FSDP 单元不超过两个。*

借助双缓冲，Megatron-FSDP 在初始化之后无需再分配内存，这可以减少内存碎片并提升性能。然而，双缓冲要求深度方向的模型对称性，即偶数索引与奇数索引的 FSDP 单元在运行时具有相同大小。如果使用双缓冲，Megatron-FSDP 会计算 FSDP 单元大小的众数（mode）作为对称的双缓冲大小，任何与该计算大小不对称的 FSDP 单元将回退到基于 `_resize_(bytes)` 的分配器（或者，当底层参数 `fsdp_db_use_persist_buf_on_alloc_fail` 被设置时，对于那些会显著影响性能的超大非对称层，如 `torch.nn.Embedding`，则进行持久分配）。

并非所有模型架构都支持深度方向的模型对称性。例如，Nemotron 这类混合架构是 Transformer、Mamba 和 MoE 块的组合，它们在大小与数据类型上都是非对称的。要对这类模型架构进行双缓冲，我们需要一个可以支持任意 FSDP 单元的缓冲区池，该池可由所有 FSDP 单元的最大值计算得到；这种由最大尺寸、形状与 dtype 的（此时已对称的）缓冲区组成的“MaxPool”便可以被双缓冲。

![MaxPoolAllocator](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/maxpool_allocator.png)

*Megatron-FSDP 中 MaxPoolAllocator 初始化的可视化。遍历所有 FSDP 单元，数据桶（buckets）按数据类型分类、从小到大排序，并与当前 MaxPool 进行比较。如果池中没有足够的桶来支持该单元，则向池中添加桶（初始大小为 0）。如果池中最大的桶不足以支持该单元中的桶（按从小到大的顺序分配到池），则扩大池中的桶。经过此过程，我们得到一组最小的桶，可以对模型中每一个 FSDP 单元进行双缓冲。*

### 数据并行分片策略（Data-Parallel Sharding Strategies）

| 优化项 | 说明 | `Megatron-Core` 配置 | `fully_shard` 配置 |
| --- | --- | --- | --- |
| 数据并行分片策略 | FSDP 的主数据并行分片策略，支持 DDP、ZeRO-1（优化器）、ZeRO-2（优化器与梯度）和 ZeRO-3（优化器、梯度与参数）。通常使用节点内通信，即“inner”或“intra”DP。 | `--data-parallel-sharding-strategy {no_shard, optim, optim_grads, optim_grads_params}` | `zero_dp_strategy={no_shard, optim, optim_grads, optim_grads_params, 0, 1, 2, 3}` |
| DP-Outer 分片策略 | HSDP 的次级数据并行分片策略，支持混合分片数据并行（HSDP / `no_shard`）和混合 FSDP（HFSDP / `optim`）。通常使用节点间通信，即“outer”或“inter”DP。 | `--outer-dp-sharding-strategy {no_shard, optim}` | `outer_dp_sharding_strategy={no_shard, optim, 0, 1}` |
| 混合数据并行大小 | 指定 DP-Outer / Inter-DP 的并行大小。DP-Inner / Intra-DP 的大小将由其他并行的大小与 `torch.distributed.get_world_size()` 推导得出。 | `--num-distributed-optimizer-instances` | `dp_outer_dim=`（HFSDP 需要累积 DP 分组 `hybrid_fsdp_group` / `hybrid_fsdp_expt_group`。） |

Megatron-FSDP 支持在多种分布式拓扑上使用多种分片策略：

- **分布式数据并行（DDP）**

  - 模型状态在各 DP rank 间复制。

  - 梯度all-reduce与反向计算重叠，并在优化步骤前的最后一次反向传播期间发起。

- **ZeRO-1**

  - 优化器状态在各 DP rank 间分片。

  - 梯度reduce-scatte与反向计算重叠，并在优化步骤前的最后一次反向传播期间发起。（之所以用reduce-scatte代替all-reduce以提升性能，是因为优化只需要梯度的一个分片。）

- **ZeRO-2**

  - 优化器状态与梯度在各 DP rank 间分片。

  - 梯度reduce-scatte与反向计算重叠，并在每次反向传播期间累加。

- **全分片数据并行（FSDP / ZeRO-3）**

  - 优化器状态、梯度与参数在各 DP rank 间分片。

  - 梯度reduce-scatte与反向计算重叠，并在每次反向传播期间累加。

- **混合分片数据并行（HSDP）**

  - 优化器状态、梯度与参数在“inner”或“intra”DP rank 间分片。

  - 模型状态在“outer”/“inter”DP rank 间复制，outer 数据并行梯度在优化步骤前的最后一次反向传播期间被all-reduce。

- **混合 FSDP（HFSDP）**

  - 优化器状态、梯度与参数在“inner”或“intra”DP rank 间分片。

  - 优化器状态进一步在“outer”/“inter”DP rank 间分片。

    - outer 数据并行梯度在优化步骤前的最后一次反向传播期间被reduce-scatte。

    - outer 数据并行参数在优化步骤后的第一次前向传播期间被全收集。

  - HFSDP 的次级分片（`optim`）要求采用 FSDP 主分片（`optim_grads_params`）。

  - 需要向 Megatron-FSDP 传入累积的数据并行分组（`hybrid_fsdp_group` / `hybrid_fsdp_expt_group`），其中包含所有数据并行 rank。

    - 要使用 `DeviceMesh` 创建这些分组，请为累积 DP 组创建一个数据并行 `DeviceMesh`，并使用 `DeviceMesh._unflatten(dp_dim, mesh_sizes=(dp_outer_size, dp_inner_size), mesh_dim_names=("dp_outer_dim", "dp_shard_dim"))` 构造一个带有 DP-Inner 与 DP-Outer 网格维度的 `DeviceMesh`，用于混合 FSDP。

#### 理解混合 FSDP（HFSDP）

混合 FSDP（HFSDP）是 HSDP 的一种变体，其中优化器状态尤其会在 DP-Inner 与 DP-Outer（即所有数据并行 rank）之间分片，从而进一步降低内存占用。换言之，节点内分片与通信使用 ZeRO-3，而节点间分片与通信使用 ZeRO-1。参数与梯度仅在优化步骤期间与全分片的优化器状态之间相互转换，从而降低了节点间通信的频率。

![Hybrid-FSDP Topology](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/hfsdp.png)

*受 DHEN（Zhang, Luo, Liu, Meta, et al., 2022）论文中的巧思启发：https://arxiv.org/abs/2203.11014*

混合全分片数据并行（HFSDP）是对 HSDP 的一个细微修改，它将优化器状态在所有数据并行 rank 间全分片，并引入 outer 层级的全收集与reduce-scatte集合通信，将全分片的参数与梯度映射为部分分片的参数与梯度。

HFSDP 的内存分布是 FSDP（优化器状态）与 HSDP（梯度与模型权重）的“混合体”。另一种理解 HFSDP 功能的优雅方式是：ZeRO-1 与 ZeRO-3 的组合。

$$
\text{Hybrid-FSDP Memory Profile} = \frac{\text{Optimizer State}}{\text{DP-Inner} \times \text{DP-Outer}} + \frac{\text{Gradient} + \text{Weight}}{\text{DP-Inner}}
$$

这一改进后的算法具有以下特征：

- Megatron-FSDP 维护一份在所有数据并行 rank 间分片的模型参数视图。

  - 分布式检查点保存与加载全分片的模型参数。

  - 分布式优化器状态基于全分片的模型参数初始化。

- 在检查点或优化之后的第一次前向传播期间，全分片的模型权重被全收集为部分分片的模型权重。

- 在优化之前的最后一次反向传播期间，部分分片的模型梯度被reduce-scatte为全分片的模型梯度。

- 其余情况下，FSDP 在部分分片的模型权重与累加的梯度上执行。由于模型权重与梯度在每个优化周期内只更新与摄取一次，我们可以跳过或推迟所有昂贵的节点间 / DP-outer 集合通信，直到某次优化步骤。

除了改善内存占用外，HFSDP 的通信还在通信大小（通信字节数）、通信拓扑（DP-Inner 与 DP-Outer 分组）以及通信域（NVLink 与 InfiniBand）上，被拆分到两个分片阶段。

![Hybrid-FSDP Streams](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/fsdp_v_hfsdp_streams.png)

*节点间通信也可以使用独立的 CUDA 流与节点内通信并行化。*

#### 混合 FSDP 与模型并行（Mixing FSDP & Model Parallelism）

Megatron-FSDP 还兼容多种对模型状态进行分片的模型并行方式，例如张量并行（TP）与专家并行（EP）。当在设备拓扑的多个维度上对模型状态分片时，FSDP 分片总是最后执行，因为 FSDP 集合通信会在计算前后立即解除与重新分片参数和梯度。因此，FSDP 分片机制是在张量并行与专家并行的（跨步 strided）分片之上实现的。

![Mixed Model Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/mixed_sharding.png)

*每当 FSDP 与其他模型并行方式组合时，FSDP 分片总是最后执行，以无缝集成到既有的模型分片中。*

Megatron-FSDP 使用 `torch.distributed.DeviceMesh` 来描述并配置数据并行分组内各设备间的通信。由于具有混合层的异构模型——例如 [混合 Mamba-Transformer](https://arxiv.org/abs/2504.03624) 或 [专家混合（MoE）](https://arxiv.org/abs/1701.06538) 模型——需要不同的并行配置，因此对于需要不同分布式拓扑以实现最优内存效率与性能的特定层，可能需要多个 `DeviceMesh`。

目前，Megatron-FSDP 支持两个 `DeviceMesh`，一个用于稠密 / 非专家 `Module`，另一个用于 Megatron-Core MoE 稀疏 / 专家 `Module`。（Megatron-Core 中的专家模块与参数会被自动检测。）

- 稠密模块通常拥有一个带数据并行、张量并行与上下文并行维度的 `DeviceMesh`，其中数据并行维度用于 FSDP。通常，数据并行与上下文并行 rank 都用于 FSDP 中的分片。
- 专家混合模块通常拥有一个带数据并行、张量并行与专家并行维度的 `DeviceMesh`，其中数据并行维度用于 FSDP。

关于 Megatron-Core 中专家混合的更多信息，请参阅 [Megatron-Core 用户指南 - MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)。

#### 非均匀 / 非对等模型分片（Non-Uniform / Un-Even Model Sharding）

虽然 `torch.distributed.tensor.DTensor` 默认采用逐参数分片——即 Tensor 在数据并行域上沿 `dim=0` 均匀切分——但 Megatron-FSDP 对与某个 FSDP 单元关联的一组（已展平的）参数使用非均匀或非对等的 `DTensor` 分片。

![Non-Uniform Sharding](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/uneven_sharding.png)

*FSDP2 逐参数分片与 Megatron-FSDP 逐单元或逐模块分片的对比。FSDP2 需要 `COPY` 操作在通信缓冲区中搬入搬出参数与梯度，以减少 NCCL 集合通信调用的频率；而 Megatron-FSDP 则将连续通信缓冲区的切片视图分配给与某个 FSDP 单元关联的参数。*

尽管非对等分片的数据结构复杂且对用户不够直观，但它在无需引入昂贵的 `COPY` 操作来建立通信与计算缓冲区的前提下，带来了一些性能优势：

- 更少的 NCCL 调用，从而减少核启动与同步开销。只有当 FSDP 单元中的参数具有不同的通信相关属性（例如 `dtype` 或分布式拓扑）时，才会被合并到各自独立的 NCCL 调用中。
- 扁平的通信与计算缓冲区天然连续，支持那些要求缓冲区由连续内存支撑的优化 CUDA 核，例如 MoE 中使用的分组 GEMM（grouped GEMMs）。

实际上，这意味着同一个 `DTensor` 分片的模型参数在不同 rank 上可能具有完全不同的形状；如果整个参数被分配到其他 rank，则本地 `Tensor` 将为空。

> ℹ️ Megatron-FSDP 提供了一个便捷库（[`megatron_fsdp.uneven_dtensor`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py)），用于操作非对等分片的 `DTensors`，聚焦于逐参数操作，例如对在各 rank 上形状不同的参数进行解除分片或规约。虽然参数组对于 FSDP 集合通信是均匀分片的，但逐参数的集合通信（它们假设设备间通信的字节数是对称的）对于非对等分片的 `DTensors` 会一直等待那些永远不会到达的字节而挂起。

特别地，连续内存只是高性能 CUDA 核要求的一半。另一半要求是局部性（locality），而 FSDP 可能会破坏它，这在将 FSDP 与当前及未来的优化相结合时引入兼容性问题。例如，块级量化（为 MXFP8、NVFP4 等计算缩放因子 / `absmax`）在块被 FSDP 分片时需要 DP 通信与自定义的 max-reduce 核。

Megatron-FSDP 支持 `dim=0` 分片，它为 FSDP 单元中的所有参数 `p` 计算 `p.shape[1:]` 的最小公倍数（LCM），并将未分片缓冲区填充到最接近的 `DP x LCM(p.shape[1:])` 的倍数，形成一个带 `LCM` 长度分块的“DP-LCM”划分，以确保 DP 分片边界不会破坏 `dim=0` 坐标上的数据块。

![Flat Buffer Sharding Algorithm](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/lcm_dim0_shard.png)

*参数如何被非对等地分配到跨 DP rank 分片的扁平逐单元缓冲区的可视化。借助 LCM 算法，`dim=0` 的每一个切片都绝不会被 FSDP 一分为二。算法与计算核可以利用这种局部性与连续性。*

- 当某个参数能被 LCM 整除时，它可以被插入到缓冲区中任何空闲的、索引为 LCM 倍数的位置。该参数的 `p[i]` 分块按定义能整除 LCM，因而与 DP-LCM 分片网格对齐。
- 当某个参数大于 LCM 但不能被 LCM 整除时，余数 `r` 会占用另一个 LCM 分块的一部分，于是会安装一个同样超过 LCM、且其“共轭”余数 `r'` 小于或等于 `LCM - r` 的“共轭”参数，用以填满剩余空间并与 DP-LCM 分片网格对齐。
- 当某个参数小于 LCM 但不能被 LCM 整除时，会对扁平缓冲区中的剩余空间进行一次分配后扫描，所有大小为 LCM 倍数、且足以容纳整个参数的空隙都会被利用。一旦所有空隙填满，最后的参数会被分配到缓冲区尾部，并遵循 DP-LCM 分片网格。

> ℹ️ Megatron-FSDP 中对连续性与局部性的通用支持仍在开发中（work-in-progress），并将随 OSS 社区与 PyTorch 的贡献持续演进。关于核缓冲区要求如何影响 FSDP 数据结构设计的更多信息，请参阅全面分析这些要求的论文 [veScale: Consistent and Efficient Tensor Programming with Eager-Mode SPMD (Li, Youjie, ByteDance Seed, et al.)](https://arxiv.org/abs/2509.07003)。

### 混合精度与量化（Mixed-Precision & Quantization）

| 优化项 | 说明 | `Megatron-Core` 配置 | `fully_shard` 配置 |
| --- | --- | --- | --- |
| 量化参数 | Megatron-FSDP 会对 TransformerEngine 量化的参数进行分片与全收集以供计算。量化参数在每个优化步骤更新，对于 MXFP8 等不可转置的 1-D 量化配方，行主序（FWD）与列主序（BWD）数据都会被管理。否则，仅激活被量化。 | `--fp8-param-gather` | TransformerEngine `quantized_model_init()` |
| 主参数（优化 / 检查点）数据类型 | 用于优化与检查点参数的数据类型。若设为 `auto`，则改用模型计算权重。`--fp8-param-gather` 所必需。默认为 FP32。 | `--megatron-fsdp-main-params-dtype {fp32, bf16, fp16, auto}` | `MixedPrecisionPolicy(main_params_dtype=...)` |
| 主梯度（累加）数据类型 | 用于梯度累加的数据类型。若设为 `auto`，主梯度精度将由模型参数精度推导。默认为 `auto`。 | `--megatron-fsdp-main-grads-dtype {fp32, bf16, fp16, auto}` | `MixedPrecisionPolicy(main_grads_dtype=...)` |
| 梯度通信（规约）数据类型 | 用于梯度通信与规约的数据类型。若设为 `auto`，则使用主梯度精度进行通信。（使用 NCCL 对称注册时，低精度梯度在线路上以 FP32 规约。）默认为 `auto`。 | `--megatron-fsdp-grad-comm-dtype {fp32, bf16, fp16, auto}` | `MixedPrecisionPolicy(grad_comm_dtype=...)` |
| 权重梯度累加融合 | 使用 TransformerEngine 模块时，Megatron-FSDP 实现 `get_main_grad` 以分配供 TransformerEngine 调用的未分片梯度缓冲区，从而避免将梯度 `COPY` 到 Megatron-FSDP 通信缓冲区。默认启用，可通过 `--no-gradient-accumulation-fusion` 关闭。 | `--no-gradient-accumulation-fusion` | N/A（仅 Megatron-Core） |
| 精度感知优化器 | 使用 TransformerEngine 的 `FusedAdam` 优化器，Megatron-FSDP 会将梯度安装到临时属性 `Parameter.decoupled_grad` 中，供 `FusedAdam` 消费。Megatron-FSDP 管理主参数，但优化器状态精度可通过 `--exp-avg-dtype` 和 `--exp-avg-sq-dtype` 定制，两者都支持 `fp8` 优化状态。 | `--use-precision-aware-optimizer` | `use_decoupled_grad=True` |

#### 量化（Quantization）

量化对 Megatron-FSDP 而言是一项极其重要的特性，因为它减少了激活与参数的内存占用及通信大小，这直接影响 FSDP 的可行性与性能。

![Quantized Model Parameters & FSDP](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/quantized_param_gather.png)

*使用来自 TransformerEngine 的量化权重时 Megatron-FSDP 训练循环的可视化。每个优化步骤都会更新分片模型权重的量化表示，其通信大小得以减小。*

虽然 TransformerEngine 负责激活量化，但 Megatron-FSDP 对量化权重进行分片以供 AG。

- **量化模型初始化（Quantized Model Initialization）**——模型以量化权重（如 MXFP8 或 NVFP4）初始化。若使用 `meta` 设备初始化，Megatron-FSDP 会调用 `reset_parameters()` 逐层初始化量化权重。如果行主序与列主序数据不可转置，Megatron-FSDP 会对两者都进行分片与缓冲。此外，高精度主权重会被获取并分片，以用于分布式优化、检查点与量化。
- **前向 / 反向传播（Forward / Backward Pass）**——量化权重会在前向与反向传播中被解除分片。如果行主序与列主序数据不可转置，则行主序权重用于前向收集，列主序权重用于反向收集。
- **分布式优化步骤（Distributed Optimization Step）**——来自量化 GEMM 的非量化累加梯度分片被应用到高精度主权重分片上。
- **分片量化（Sharded Quantization）**——分片的主权重被量化，以更新量化的计算权重供后续训练步骤使用。

![Sharded Quantization](https://docs.nvidia.com/megatron-core/developer-guide/nightly/_images/sharded_quantization.png)

*分片量化涉及对最大值进行规约，以计算一组全局缩放因子用于本地 / 分片量化。*

特别地，分片量化通过传输缩放因子而非主权重，来最小化通信大小与内存占用。

- **本地绝对最大值（Local Abs-Max）**——对于 FSDP 单元中的一组参数，在全局未分片形状上计算本地的张量级或块级最大值，非本地数据以零填充。
- **全局绝对最大值（Global Abs-Max）**——全局all-reduce最大值，并由最大值推导缩放因子。
- **本地量化（Local Quantization）**——本地量化分片的主权重并安装到计算权重缓冲区。

#### 混合精度（Mixed-Precision）

Megatron-FSDP 的分片与通信缓冲区支持混合精度，使用户可以在模型计算所用的原生或量化 `dtype` 之外，定制用于主权重、梯度通信（规约）与梯度累加的 `dtype`。这些选项被封装在 `MixedPrecisionPolicy` 数据类中。

- **主权重精度（Main Weight Precision）**——控制负责分布式优化、分布式检查点与量化的参数的数据类型。若设为 `auto`（`None`），则使用原生模型计算参数的数据类型。使用 `--fp8-param-gather` 进行参数量化时必需。默认为 `torch.float32`。
- **主梯度精度（Main Gradient Precision）**——控制 `wgrad` 累加与分布式优化的数据类型。默认为 `auto`（`None`），即使用模型原生梯度数据类型。虽然在大规模下推荐使用 `torch.float32`（或更高）以保证精度，但由于 `main_grads_dtype` 控制梯度累加的数据类型，`auto` 更为灵活，在混合精度场景中会使用预定的参数梯度逻辑，例如对于经 TransformerEngine 量化为 `FP8`/`FP4` 的参数使用 `BF16`。
- **梯度通信精度（Gradient Communication Precision）**——控制规约梯度时梯度通信的数据类型。更低的精度可提升（通信）性能。默认为 `auto`（`None`），此时使用主梯度数据类型。如果使用 `no_shard`、`optim`、HSDP 或 HFSDP，分配 `dtype` 定制的梯度通信缓冲区可能会增加每单元内存开销，因此用户在使用此特性时应权衡性能与内存。

如果使用 NCCL 对称注册 `v2.27+`，梯度规约可能会依据网络域（NVLink 或 IB）以高精度执行，并可启用混合精度通信与累加——例如将 grad_comm_dtype 设为 `BF16` 时，即便通信输入与输出缓冲区都是 `BF16`，也可以支持 `FP32` 规约。否则，梯度将像往常一样以通信与累加精度进行规约和累加。

### NCCL

| 优化项 | 说明 | `Megatron-Core` 配置 | `fully_shard` 配置 |
| --- | --- | --- | --- |
| NCCL 用户缓冲区 | 分配 Megatron-FSDP 通信缓冲区并向 NCCL 注册，从而启用零 `COPY`、高精度规约、拷贝引擎集合通信以及对称核。使用双缓冲。 | `--use-nccl-ub` | `nccl_ub=True` |
| NCCL 手动注册 | 不在首次分配时注册 NCCL 用户缓冲区，而是在初始训练步骤结束时批量注册所有通信缓冲区。减少注册延迟。 | `--fsdp-manual-registration` | N/A（仅 Megatron-Core） |
| 禁用对称注册 | 禁用与 NCCL 的对称注册。可选，因为对称注册失败会默认回退到普通注册。 | `--disable-symmetric-registration` | `disable_symmetric_registration=True` |

[NVIDIA 集合通信库（NCCL）](https://developer.nvidia.com/nccl) 实现了面向 NVIDIA CUDA 设备与网络优化的多设备、多节点通信原语。Megatron-FSDP 的通信已向 NCCL 注册并与之深度集成，从而启用多种硬件级网络优化，例如拷贝引擎 AG、高精度 RS、SHARP 规约卸载以及对称核。

要利用 NCCL 的网络优化，需要进行 NCCL 用户缓冲区注册（User Buffer Registration，UBR），以告知 NCCL 哪些 PyTorch Tensor（“用户缓冲区”）直接充当 PyTorch `ProcessGroup` 的 NCCL 集合通信的输入与目标。由于已注册的通信缓冲区为 NCCL 所知，将集合通信输入送往 NCCL 缓冲区、将集合通信输出送往 PyTorch 缓冲区的 `COPY` 操作便不再需要，这使得 Megatron-FSDP 能够端到端零 `COPY`。

NCCL（`v2.27+`）支持为 NVLink 域上的通信器进行对称分配或注册，使得在各设备间共享相同虚拟地址的缓冲区能够受益于优化的集合通信：

- **对称核（Symmetric Kernels）**——在 NVLink 域上，作用于对称内存的对称核将单个通信核的 SM 占用降低到 1。
- **NVSwitch SHARP 卸载（NVSwitch SHARP Offloading）**——为进一步最小化 AG 与 RS 集合通信的 SM 占用，NCCL SHARP 将规约与聚合工作卸载到 NVLink 与 IB 交换机硬件，依据域（NVL、IB，或 NVL + IB）使用 1-6 个 SM。
- **拷贝引擎（CE）集合通信（Copy-Engine Collectives）**——对于 Megatron-FSDP 中 AG 等常见的非计算类集合通信，使用拷贝引擎来执行全收集集合通信，而非使用 SM（或 CTA），从而将 SM 资源专用于 FSDP 期间的计算与规约。需要 NCCL `v2.28+`。
- **高精度规约（High-Precision Reduction）**——训练大模型时，为保证精度与收敛性，期望进行高精度的梯度规约与累加，但通信 FP32 梯度代价高昂。借助对称注册，FP32 累加器使得梯度可以以 FP32 规约、但以 BF16 通信，从而在保持训练高精度的同时降低梯度 RS 通信延迟。Megatron-FSDP 支持 FP32 主梯度累加但以 BF16 进行梯度通信，可通过 `megatron_fsdp.MixedPrecisionPolicy` 定制。

这些优化显著降低了 FSDP 中重叠计算与通信核的 SM 资源争用。对称注册、分配与池化在 PyTorch 中也受支持：[`torch.distributed._symmetric_memory`](https://docs.pytorch.org/docs/stable/symmetric_memory.html)。
