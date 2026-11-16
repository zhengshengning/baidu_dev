# Megatron-FSDP 通信 Overlap / Prefetch / Buffer 管理实现分析

> 源码基线：`Megatron-LM` HEAD `d37b3a8fd`
> 主要路径：`megatron/core/distributed/fsdp/src/megatron_fsdp/`
> - `megatron_fsdp.py`：`MegatronFSDP` 模块封装、hook 注册、流创建、预取量计算
> - `param_and_grad_buffer.py`：bucket 划分、各类 buffer/allocator、`AllGatherPipeline`、`GradReducePipeline`
> - `distributed_data_parallel_config.py`：开关与默认值
> - `utils.py`：`FSDPDistributedIndex`（进程组）、`GlobalMemoryBuffer`
> - `experimental/`：MFSDP v2（`fully_shard` 风格，仍在演进）
>
> 下文行号均指上述文件。未特别说明时描述的是生产版 v1（`MegatronFSDP`）。

## 0. 结论速览

| 问题 | 答案 |
| --- | --- |
| 通信粒度 | bucket（= `ParameterGroup`），同一 FSDP unit 内同 `is_expert` 的所有 bucket 组成 bucket group，用 `_coalescing_manager` 合成一次 NCCL 调用 |
| 预取方向 | 静态 bucket 顺序：前向 `bucket_id+1`，反向 `bucket_id-1`（不是运行时录制的顺序） |
| 预取量 | `suggested_AG_prefetch_size = suggested_communication_unit_size // 2`。ZeRO-3 下默认 `unit_size = 平均每个 FSDP unit 参数量 × 2`，所以默认预取量约等于 1 个平均 FSDP unit，即只预取下一层 |
| RS 在途量 | `suggested_RS_queue_capacity = suggested_communication_unit_size`，约 2 个 unit 的梯度同时在 reduce-scatter 中 |
| 上限 | `fsdp_double_buffer=True` 时，AG 与 RS 同一时刻最多涉及 2 个 FSDP unit，在预取扩展和 RS 入队时都会强制检查 |
| 流 | 每个 `MegatronFSDP` 2 条 side stream：`side_stream_for_param_gather`（AG）和 `side_stream_for_buffer_copy_and_grad_accum`（RS）。HFSDP 再加 outer AG/RS 各 1 条 |
| 通信器 | AG 可以使用独立的 NCCL communicator（`--create-all-gather-group` → `dp_cp_ag`），避免 AG 与 RS 共用一个 comm 时出现 head-of-line 串行 |
| 内存释放 | 不用 `record_stream`；靠「下次 AG 前 `ag_stream.wait_stream(current_stream)`」和「`event.wait()` 后再 free」保证跨流安全 |
| 释放时机 | 前向 post-forward 立即释放；重计算时 lazy 释放（标记后推迟到下一次 AG 分配前再回收）；反向 post-backward 立即释放 |
| buffer | 持久：分片权重、main weight、main grad（每 dtype 一块大 buffer）。临时：未分片的权重 bucket、梯度通信 bucket，由 4 种 allocator 之一管理 |

## 1. 基本组织：strategy / unit / bucket / bucket group

### 1.1 分片策略

`_buffer_distribution_for_strategy`（`param_and_grad_buffer.py:270`）决定 3 类 buffer 是否分片：

| strategy | model_weight | main_weight(优化器 fp32) | main_grad |
| --- | --- | --- | --- |
| `no_shard` | 否 | 否 | 否 |
| `optim` (ZeRO-1) | 否 | 是 | 否 |
| `optim_grads` (ZeRO-2) | 否 | 是 | 是 |
| `optim_grads_params` (ZeRO-3) | 是 | 是 | 是 |

只有 `optim_grads_params` 会对模型权重分片，所以**只有它需要前向/反向 all-gather + 预取 + 释放**（`megatron_fsdp.py:489-499` 只对这类参数安装 safety check）。`optim_grads` 和 `optim_grads_params` 每个 micro-batch 都要做 RS；`no_shard` 和 `optim` 只在最后一个 micro-batch 做 all-reduce 或 RS。

MoE 时专家与非专家参数可以分别配置（`expert_data_parallel_sharding_strategy`）。HSDP/HFSDP 用 `outer_dp_sharding_strategy`（`no_shard` / `optim`）描述外层 DP 的分片方式。

### 1.2 FSDP unit

`fsdp_unit_modules`（一般是 `TransformerLayer`、embedding 等类）是参数生命周期的管理单位。unit 内的参数一起 AG、一起释放。非 unit 模块的参数即使在 ZeRO-3 下也**常驻**（状态记为 `PRESERVED`，见 `release_bucket` 的注释 `:4979`），因为它们可能被共享或同时修改。

### 1.3 bucket（ParameterGroup）划分

`_get_parameter_groups`（`:1790`）分 4 步：

1. 按 `(dtype, is_expert(".experts." in name), requires_grad, fsdp_unit_id)` 分组。同一 unit 的参数默认落进同一组。
2. 不属于任何 unit 的组，按 `BucketingPolicy.suggested_bucket_size`（默认 `40_000_000` 元素，`:256`）切分。
3. 按 `chunk_size_factor = shape[1:].numel()` 再切。注释说明了不取 LCM 的原因：混合形状取 LCM 会让 padding 爆炸，DeepSeek-V3 是例子。
4. 建 `bucket_to_bucket_group`，key 为 `(fsdp_unit_id, is_expert)`，同组 bucket 的 AG/RS 合并到同一个 `_coalescing_manager` 里发出。

bucket id 的顺序就是模块注册顺序（`named_parameters` 顺序），**预取所依赖的“下一个/上一个”就是 bucket id ±1**。

### 1.4 buffer 布局

`build_data_parallel_buffer_index`（`:287`）给每个 bucket 计算布局：

- bucket 尺寸 pad 到 `dp_world_size × chunk_size_factor` 的整数倍，保证每个 rank 的 shard 等长，可以直接用 `all_gather_into_tensor` / `reduce_scatter_tensor`。
- 小参数片段塞进 padding 空隙，减少浪费。
- 产出 `ShardBucketIndex`（本 rank shard 在 bucket 中的偏移和长度）。不分片的 buffer 也有一个“虚拟 shard”，这样 RS 输出可以直接写回全量 buffer 中本 rank 的那一段。

每个 bucket（`ParameterGroup`）挂载以下 `DataParallelBuffer`：

| buffer | 用途 |
| --- | --- |
| `model_weight_buffer` | 计算用权重（bf16/fp8），ZeRO-3 下持久部分只存 shard |
| `transpose_weight_buffer` | MXFP8 反向用的列向量化权重（只在 bwd 时 AG） |
| `main_weight_buffer` | 优化器高精度权重（`main_params_dtype`），分片 |
| `main_grad_buffer` | 高精度梯度，分片（ZeRO-2/3）或全量 |
| `hfsdp_helper_wbuf/wtbuf/gbuf` | HFSDP 时 DP-Shard 视角的辅助 buffer |
| `hsdp_comm_gbuf` | HSDP/HFSDP 外层通信用临时梯度 buffer（自定义通信 dtype 时） |

## 2. 流与通信器

### 2.1 CUDA Stream

`MegatronFSDP._init_fsdp_param_and_grad_buffer`（`megatron_fsdp.py:449-460`）：

```python
self.side_stream_for_buffer_copy_and_grad_accum = torch.cuda.Stream()   # RS 流
self.side_stream_for_param_gather = torch.cuda.Stream()                 # AG 流
self.grad_reduce_pipeline = GradReducePipeline(buf, rs_stream=<RS 流>)
self.all_gather_pipeline  = AllGatherPipeline(buf, ag_stream=<AG 流>)
```

使用 HFSDP 时两条 pipeline 各自再创建 1 条外层流：`outer_fsdp_group_param_gather_stream` 和 `outer_fsdp_group_grad_reduce_stream`。

| 流 | 上面跑的操作 |
| --- | --- |
| 计算流 (current) | fwd/bwd 计算、`_grad_acc` 拷梯度到 main_grad bucket、释放 bucket（CPU 侧 free） |
| AG 流 | 分配未分片 bucket（`fetch_bucket`）、lazy 回收、`all_gather_into_tensor` |
| RS 流 | `fetch_bucket(grad_comm_dtype)`、预缩放、`reduce_scatter_tensor`/`all_reduce`、`local_grad += reduced` |
| outer AG 流 | HFSDP 外层 DP 的 all-gather（优化器 shard → DP-Shard shard） |
| outer RS 流 | HSDP 外层 all-reduce / HFSDP 外层 reduce-scatter |

流之间的同步规则：

- **发 AG 前**：`all_gather_stream.wait_stream(torch.cuda.current_stream())`（`:4907`）。它保证此前计算流上所有 kernel，包括还在读旧 bucket 的计算，都排在本次 AG 写 buffer 之前。复用释放掉的内存时，安全性就靠这一步。
- **用权重前**：`wait_bucket_ready` → `event.wait()`（`_coalescing_manager` 返回的 work），让计算流等待 AG 完成。这是 GPU 侧的依赖，CPU 不阻塞。
- **发 RS 前**：`reduce_scatter_stream.wait_stream(current_stream)`（`:4254`），确保梯度已经写进 bucket。
- **RS 完成**：RS 流 `record_event()`，事件和 free 闭包一起入队；出队时先 `event.wait()`（计算流等 RS），再 free。
- **外层**：outer RS 流 `wait_stream(rs_stream)` 后再做外层 reduce；AG 流 `wait_event(outer_event)` 或 `wait_stream(outer_stream)` 后再做内层 AG。

这和 FSDP2 的思路一致（`docs/runtime_schedule.md`）：**不用 `record_stream`**，通过显式的流间同步保证“分配和复用的先后顺序”。这样显存占用是确定性的。

### 2.2 独立 AG 通信器

`runtime_schedule.md` 说明，AG 和 RS 如果共享同一个 NCCL communicator，即使分别放在不同流上，**NCCL 也会按 host 端 launch 顺序串行执行**（`docs/scripts/nccl_same_pg_two_streams_*.py` 可以复现）。于是反向时 `AG_{i-1}` 要等 `RS_i` 跑完，overlap 被破坏。

实现方式：

- `parallel_state.create_all_gather_groups()`（`parallel_state.py:1688`）用与 DP×CP 完全相同的 rank 集合，再 `create_group` 一个新的 communicator `DATA_PARALLEL_GROUP_WITH_CP_AG`。专家参数也有对应的 `EXPERT_DATA_PARALLEL_GROUP_AG`。
- 训练侧通过 `--create-all-gather-group` 开启（`training.py:2524`），结果放进 `pg_collection.dp_cp_ag` / `expt_dp_ag`。
- `mcore_fsdp_adapter.py:487-547` 把它们作为 `fsdp_group_ag` / `expt_fsdp_group_ag` 传给 `FSDPDistributedIndex`。
- `get_fsdp_group(independent_all_gather=True)`（`utils.py:734`）返回这个 AG 组。`model_weight_buffer` 的 `data_parallel_group` 用的就是它（未使用 HFSDP helper 时）。梯度 buffer 继续用普通 DP 组。

结果：AG 走 comm A + AG 流，RS 走 comm B + RS 流，两者可以真正并行。

> 注意：即使有独立通信器，**预取仍然必要**。否则 `AG_{i-1}` 要等 layer i 的 pre-backward hook 才会发起，根本没机会和 `Bwd_i` 重叠。

### 2.3 Coalescing

AG（`:4910`）和 RS（`:4259`）都包在 `torch.distributed._coalescing_manager(dp_group, async_ops=...)` 里。一个 bucket group（同一 unit、同一 expert 属性）的多个 bucket 合并成一次 NCCL group call，减少 launch 开销。AG 场景下，coalescing 返回的 work 会替换各 bucket 自己的 event（`:4918-4925`），后续 `wait_bucket_ready` 等的就是整组完成。

## 3. Hook 体系（`_register_fsdp_hooks`, `megatron_fsdp.py:565`）

| Hook | 注册方式 | 动作 |
| --- | --- | --- |
| pre-forward unshard | `register_forward_pre_hook(prepend=True)`，挂在每个 unit 上，非 unit 模块只处理直属参数；开启 fine-grained 时挂在所有子模块上 | `all_gather_and_wait_parameters_ready(prefetch=True, FORWARD_PASS_ORDER)`；如果模块状态为 `PRE_BACKWARD`（重计算中），关闭预取 |
| post-forward release | unit 上的 `register_forward_hook` | `release_module_parameters(bwd=False, lazy=重计算中)` |
| 注册 post-backward | unit 上的 forward_pre_hook | 用恒等 autograd 函数 `RegisterFSDPBackwardFunction` 包住模块输入，它的 backward 就是“该模块反向结束”的时刻 |
| pre-backward unshard | forward_hook 里对输出调用 `register_multi_grad_hook(mode="any")` | 输出梯度一到就执行 `all_gather_and_wait_parameters_ready(bwd=True, BACKWARD_PASS_ORDER)`，`prefetch` 默认 True |
| post-backward release | `RegisterFSDPBackwardFunction.backward` | `_post_backward_release_module`：立即释放 bwd 和 fwd 两套 bucket，状态回到 IDLE |
| grad 累加 + RS | `param.register_post_accumulate_grad_hook` | `_process_post_backward_gradients` → `_grad_acc` → `reduce_gradients` |
| root pre-backward | root 输出的梯度 hook | 把所有模块置为 `PRE_BACKWARD`，把所有 unit 的 fwd bucket 标记为 `can_be_released`（lazy），`queue_callback(_root_post_backward)` |
| root post-backward | autograd engine callback | 补处理剩余梯度（按名字排序），`reduce_gradients`，`grad_reduce_pipeline.reset()`，`microbatch_count += 1` |

## 4. 前向时序

以 L 个 `TransformerLayer` unit 为例，每层 1 个 bucket group，默认预取约 1 层：

```
CPU:   pre_fwd(0) ─────────── pre_fwd(1) ─────────── pre_fwd(2) ...
         AG0+AG1(prefetch)      AG2(prefetch)          AG3
         wait(0)                wait(1)                wait(2)
                    post_fwd(0): free B0     post_fwd(1): free B1
AG 流:  [AG0][AG1]             [AG2]                 [AG3]
计算流:     ...wait0[ Fwd0 ]wait1[ Fwd1 ]wait2[ Fwd2 ]...
```

细节：

1. `pre_fwd(i)` 调用 `all_gather_params([layer i 的参数], prefetch=True)`。
   - layer i 已经被上一层预取过时，状态为 `COMMUNICATING`/`READY_TO_USE`，过滤后不再发 AG（`:4874-4881`），只走预取扩展。
   - `_extend_by_prefetch_size` 从 i 往后扩展：i+1 的 bucket group 加入后，预取量达到阈值就停止。
2. `wait_bucket_ready(i)` 让计算流等待 AG_i。随后 Fwd_i 与 AG_{i+1} 在两条流上并行。
3. `post_fwd(i)` **立即**释放 layer i 的 bucket（`release_bucket(lazy=False)`）。CPU 侧 `free_bucket_storage()`，GPU 侧不需要等待，因为下一次复用前 AG 流会 `wait_stream(current)`，Fwd_i 必然先跑完。
4. 这正好满足 `runtime_schedule.md` 的要求：**`AG_{i+1}` 必须在 `DO_i`（释放 i）之前发起**。

峰值未分片权重 ≈ 当前层 + 预取层 = 2 个 unit，正好是 double buffer。

## 5. 反向时序

```
CPU:  pre_bwd(L-1): AG(L-1)+AG(L-2)   [ RS 由 grad hook 触发 ]   post_bwd(L-1): free
                    pre_bwd(L-2): AG(L-3) ...
AG 流:  [AG L-1][AG L-2]            [AG L-3]
计算流:   [ Bwd L-1 (含 grad_acc) ][ Bwd L-2 ] ...
RS 流:                     [RS L-1]             [RS L-2]
```

1. root pre-backward：所有模块置为 `PRE_BACKWARD`，前向遗留的 fwd bucket 全部标记为 lazy 可释放（主要是没被释放的最后几层，或被重计算保留下来的）。
2. `pre_bwd(i)`（layer i 输出梯度就绪时）：`all_gather_params(bwd=True, BACKWARD_PASS_ORDER, prefetch=True)`，从 i 向 i-1 扩展预取。
   - MXFP8 时 bwd 取 `transpose_weight_buffer`。key 为 `(bucket_id, bwd and has_transpose)`，所以 fwd/bwd 两套 bucket 状态互不干扰。
   - 非 FP8 时 bwd key 与 fwd 相同。最后一层前向刚算完时可能还处于 `READY_TO_USE`（root pre-backward 只做了 lazy 标记），这次 AG 会把 `can_be_released` 清为 False 并直接复用，省掉一次 AG。
3. Bwd_i 计算过程中，每个参数的 post-accumulate-grad hook 依次触发：
   - `_grad_acc`：`param.get_main_grad()` 分配或取得梯度通信 bucket，`copy_(param.grad)` 后 `del param.grad`。TE 开启 fused wgrad 时直接写入 `main_grad`，此时 `grad_added_to_main_grad=True`，跳过拷贝。
   - `reduce_gradients`：把参数加入 `bucket_grad_ready_params`。**整个 bucket group 的参数到齐**后，才 `wait_for_previous_grad_reduce(capacity)` + `_bucket_group_gradient_reduce(async_op=True)`。
4. `post_bwd(i)`（`RegisterFSDPBackwardFunction.backward`，即 layer i 输入梯度算完）：立即释放 bwd 和 fwd bucket。
5. 满足 `runtime_schedule.md` 的要求：`AG_{i-1}` 在 `RS_i` 和 `DO_i` 之前发起。独立 AG 通信器让 `AG_{i-1}` 与 `RS_i` 不再互相阻塞。
6. root post-backward：处理没触发 hook 的参数（共享参数等），排序后统一 reduce，然后 `grad_reduce_pipeline.reset()`。

## 6. 预取：预取多少、怎么扩展

### 6.1 默认值计算（`megatron_fsdp.py:462-487`）

```python
unit_size = ddp_config.suggested_communication_unit_size
if unit_size is None:
    if any strategy == "optim_grads_params":
        # 所有 FSDP unit 模块参数总量 / unit 个数 × 2
        unit_size = total_param_elements_in_units // num_units * 2
    elif bucket_size is not None:
        unit_size = bucket_size
    else:
        unit_size = 1_000_000_000
    unit_size = min(1_000_000_000, unit_size)          # 上限 1B 元素

suggested_RS_queue_capacity = unit_size              # RS 在途容量
suggested_AG_prefetch_size  = unit_size // 2         # AG 预取量
```

源码注释写的是“处理当前 FSDP 模块并尝试预取下一个”。按默认值，**AG 预取量 ≈ 1 个平均 unit，RS 队列容量 ≈ 2 个平均 unit**。单位是元素个数，统计的是 unit 模块的 `p.numel()`（TP 下是本 rank 的 TP 分片），不包括 embedding 等非 unit 参数。

### 6.2 扩展算法 `_extend_by_prefetch_size`（`param_and_grad_buffer.py:4676`）

```
result = 当前要 AG 的 bucket 列表（已排序）
base   = sum(result 中各 bucket 的 model_weight_buffer.bucket_index.size)   # 全量（padding 后）尺寸
loop:
    next = 前向: result 首部开始的连续段之后的第一个 id；反向: 尾部连续段之前的第一个 id
    越界 → 停止
    已预取量 = sum(result) - base
    已预取量 >= suggested_prefetch_size → 停止
    fsdp_double_buffer 且加入 next 的 unit 后涉及的 unit 数 > 2 → 停止
    result += bucket_to_bucket_group[next]   # 整组加入
```

要点：

- **先判断、后加入**：只要 `suggested_prefetch_size > 0`，**至少会预取 1 个 bucket group**。可能预取超出阈值（最后一组整组加入）。
- 要彻底关闭预取，需要 `suggested_communication_unit_size` 为 0 或 1（使 `//2 == 0`）。只把它“调小”到比一层还小，仍然会预取下一组（这一点与 `arguments.py` 帮助文本 “a smaller value disables prefetching” 的说法不完全一致）。
- 各层大小一致时，默认值恰好预取 1 层。层大小不一时（例如 MoE 层与 dense 层交替），下一层比平均小就可能连续预取 2 层以上。这时只有 `fsdp_double_buffer` 的 2-unit 上限能卡住。
- 直接调用 `all_gather_params` 且 `suggested_AG_prefetch_size=None` 时，内部兜底为 `500_000_000`（`:4705`）。`MegatronFSDP` 总会传入计算值。
- 预取方向完全由 bucket id 决定。模块 `forward()` 的调用顺序与注册顺序不一致，或同一模块被调用多次时，预取的就是“错的层”：浪费带宽，并多占一个 buffer，但结果仍然正确，因为真正使用前会 `wait_bucket_ready`，没到位就补发 AG。

### 6.3 不预取的场景

| 场景 | 处理 |
| --- | --- |
| 重计算（模块状态 `PRE_BACKWARD` 时进入 pre-forward） | `fsdp_forward_prefetch=False`（`megatron_fsdp.py:851`） |
| double buffer 且本次请求的参数都不属于任何 unit | `should_prefetch=False`（`:4842-4851`），避免非 unit 模块的预取抢走 unit 的双缓冲槽位 |
| `no_shard` | 整个 AG 流程直接 return（`:520`） |

### 6.4 HFSDP 外层预取

`hfsdp_param_gather_overlap=True` 且 `outer_dp_sharding_strategy=="optim"` 时，外层 DP 的 AG（优化器 shard → DP-Shard shard）在内层预取边界基础上**再多预取 1 个 FSDP unit**（`_extend_by_fsdp_units(..., 1 if should_prefetch else 0)`，`:4864-4870`）。外层 AG 在独立的 outer 流上执行，完成后记录 `outer_bucket_ready_events`。内层 AG 只 `wait_event` 自己所需 bucket 的外层事件，形成两级流水：outer(i+2) ‖ inner(i+1) ‖ compute(i)。

未开启该选项时，外层 AG 与内层 AG 串行：先 `_launch_outer_bucket_group`，再 `ag_stream.wait_stream(outer_stream)`。外层 AG 只在每个优化 step 的第一个 micro-batch（`microbatch_count==0`）或 `model_auto_sync` 时执行（`megatron_fsdp.py:532-540`），因为只有优化器刚更新完权重时才需要刷新。

## 7. AllGatherPipeline 状态机与释放

### 7.1 状态

`BucketStatus`（`:4014`）：`EMPTY`（无存储）→ `COMMUNICATING`（AG 已发出）→ `READY_TO_USE`（计算流已等待事件）。`PRESERVED` 表示非 unit bucket 存储常驻，但需要在下一个 step 刷新。

状态表的 key 是 `(bucket_id, bwd and has_transpose_buffer)`。辅助结构：

- `param_gather_event_map[key] = (work/event, mark_ready_closure)`
- `bucket_can_be_released[key]`：lazy 释放标记
- `outer_bucket_ready_events[key]`：HFSDP 外层事件

### 7.2 `async_bucket_gather`（`:5032`）

```
can_be_released[key] = False
若已 COMMUNICATING/READY → return（去重，这是预取命中的关键）
status = COMMUNICATING
recycle_unused_buckets()          # 先回收 lazy 标记的 bucket，再分配
bucket = wbuf.fetch_bucket(set_param_data=True)  # 分配全量 bucket，把 p.data 改为 bucket 内的 view
all_gather_into_tensor(bucket.data, wbuf.get_shard_from_local_buffer(), group=wbuf.dp_group, async_op=True)
event_map[key] = (work, mark_ready)
```

整个函数在 `torch.cuda.stream(ag_stream)` 和 `_coalescing_manager` 内执行，所以**分配发生在 AG 流上**。

### 7.3 释放

- 立即释放 `release_bucket(lazy=False)`（`:4953`）：先 `wait_bucket_ready(empty_ok=True)`，再 `free_bucket_storage()`（StorageResize 方式是 `storage.resize_(0)`，池方式是把槽位还给 idle），然后置为 `EMPTY`。
- lazy 释放：只设置 `can_be_released=True`，**真正的回收推迟到下一次 `async_bucket_gather` 分配前**（`recycle_unused_buckets`）。用于重计算的 post-forward，以及 root pre-backward 时批量标记前向残留的 bucket。好处是：如果紧接着的反向需要同一个 bucket，会直接命中，不必重新 AG。
- `reset()`（`:4574`）：等待所有未完成的 AG；对外层事件做 device 侧 `wait_event`（不做 host sync，兼容 CUDA Graph）；unit bucket 全部释放，非 unit bucket 标记为 `PRESERVED`。

### 7.4 跨流内存安全

AG bucket 在 AG 流上分配，在计算流上读，在 CPU 上释放。之所以不需要 `record_stream`，是因为释放后的复用只会发生在下一次 `async_bucket_gather` 中，而它之前一定执行过 `ag_stream.wait_stream(current_stream)`（`:4907`）。这样新的 AG 写入一定排在所有已入队计算之后。池 allocator 复用的是 `GlobalMemoryBuffer` 的固定张量，依赖的也是同一条规则。

## 8. GradReducePipeline：梯度归约与 overlap

### 8.1 梯度的落地：`get_main_grad`

`param_and_grad_buffer.py:3190-3219` 为每个参数注入 `main_grad_getter`：

```python
def main_grad_getter(p):
    grad_reduce_pipeline._enforce_double_buffer_limit([bucket_id])   # 先保证双缓冲容量
    bucket = gbuf.fetch_bucket(dtype=grad_comm_dtype if gbuf.is_data_distributed else None)
    return gbuf.get_item_from_bucket(bucket, item_id).view(p.shape)
```

- 梯度分片（ZeRO-2/3）时，`fetch_bucket` 会分配一个**未分片的临时梯度 bucket**，dtype 为 `grad_comm_dtype`（未设置时为 `grad_dtype`）。同一 bucket 的参数共用这块内存，第一次访问时分配，之后命中缓存。
- 梯度不分片（`no_shard`/`optim`）且 dtype 相同时，直接返回持久 `main_grad_buffer` 中对应的切片，不额外分配。
- `_grad_acc`（`megatron_fsdp.py`）：分片时执行 `main_grad.copy_(param.grad)` 后 `del param.grad`；不分片时执行 `main_grad.add_(param.grad)`，跨 micro-batch 累加。
- TE fused wgrad 直接写入 `main_grad`，并设置 `grad_added_to_main_grad=True`，此时跳过拷贝。前向 unshard 时会把它重置为 False，并根据是否是第一个 micro-batch 设置 `overwrite_main_grad`。

### 8.2 触发条件：整组到齐才发 RS

`reduce_gradients(params)`（`:4091`）：

1. 参数按 bucket id 排序后加入 `bucket_grad_ready_params[bucket_id]`。
2. 某 bucket 的参数全部到齐时，标记该 bucket ready。
3. 只有**整个 bucket group 的所有 bucket 都 ready**，才执行：
   - `wait_for_previous_grad_reduce(suggested_queue_capacity)`：在途量超过容量时先出队旧的 RS。
   - `_bucket_group_gradient_reduce(group, async_op=True)`。

所以 RS 的粒度和 AG 一样，都是 bucket group（≈ 一个 FSDP unit），时机是该 unit 最后一个参数梯度产生之后，一般在 `Bwd_i` 末尾、`post_bwd(i)` 之前。

### 8.3 `_bucket_group_gradient_reduce`（`:4226`）

```bash
if fsdp_double_buffer: _enforce_double_buffer_limit(bucket_group)
rs_stream.wait_stream(current_stream)                 # 梯度已写进 bucket
with torch.cuda.stream(rs_stream), _coalescing_manager(dp_group, async_ops=async_op):
    for bucket in group:
        gbuf = pg.main_grad_buffer
        unreduced = gbuf.fetch_bucket(grad_comm_dtype)   # 命中 8.1 中已分配的 bucket
        (不分片 + 自定义 comm dtype 时另 allocate_bucket_storage 并拷贝)
        gradient_reduce_preprocessing(unreduced, scaling, ReduceOp)   # 预缩放 / AVG
        no_shard: all_reduce(unreduced)
        其它:     reduce_scatter_tensor(grad_shard_tmp, unreduced)
        closure: local_grad(main_grad 分片) += reduced    # optim/no_shard 用 copy_
    status = COMMUNICATING; event = rs_stream.record_event()
HSDP/HFSDP 外层:
    outer_stream.wait_stream(rs_stream)
    outer=optim: reduce_scatter 到 main_grad 外层 shard (AVG/SUM)
    HSDP:        all_reduce
    event = outer_stream.record_event()
free 闭包: gbuf.free_bucket_storage(); hsdp_comm_gbuf 释放; reset_param_main_grad(); status = EMPTY
grad_reduce_queue.append((event, free_closure, bucket_group))
```

要点：

- RS 输出先写入一个临时 shard，再在 RS 流上 `+=` 到持久 `main_grad` 分片。这样既能跨 micro-batch 累加，也能做 dtype 转换（例如 bf16 通信、fp32 累加）。
- 预缩放规则在 `gradient_reduce_preprocessing`（`:5084`）中，于 RS 流上执行：
  - 无 scaling：`SUM`。
  - `average_in_collective`：`AVG`。
  - 组大小为 1：直接 `mul_`。
  - `gradient_reduce_div_fusion` 且非 bf16：使用 NCCL `premul_sum` 融合缩放。
  - 其余情况：先 `mul_` 再 `SUM`。
- free 闭包**不立刻执行**，跟着 event 进入队列，由后续的 `wait_for_previous_grad_reduce` 统一回收。

### 8.4 队列容量 `wait_for_previous_grad_reduce`（`:4135`）

```
while sum(queue 中各 bucket 的 main_grad_buffer.bucket_index.size) > capacity:
    event, free_fn, _ = queue.pop(0)
    event.wait()          # 当前流（调用方所在流）等待 RS 完成，GPU 侧依赖
    free_fn()             # 回收未分片梯度 bucket
```

- `capacity = suggested_RS_queue_capacity ≈ 2 个平均 unit`。最多约 2 个 unit 的未分片梯度 bucket 在途，第三个入队前会先回收最早的那个。
- `capacity=0` 表示全部等完，`reset()` 和 `finish_grad_sync` 用这种方式。
- 只有 `event.wait()`，没有 `synchronize`，CPU 不阻塞。被回收的内存下次分配在计算流上（`get_main_grad` 运行在计算流），而计算流已经 `wait` 过 RS 事件，所以复用是安全的。

### 8.5 双缓冲限制 `_enforce_double_buffer_limit`（`:4167`）

只在 `fsdp_double_buffer=True` 时生效，因为 FixedPool/MaxPool 每种 bucket 只有 2 个槽位：

1. 断言本次要加入的 bucket 至多属于 2 个 unit。
2. 从队尾往前数 unit 数，找到「保留后加上本次请求也不超过 2 个 unit」的保留条数 `keep_n`。
3. 在 RS 流上把更早的条目出队：等 event，然后 free。

它在两个地方调用：`get_main_grad` 首次分配梯度 bucket 时，以及发 RS 前。由此保证梯度 bucket 槽位不会被第 3 个 unit 抢占。

### 8.6 各策略下何时归约

| 策略 | 每个 micro-batch | 最后一个 micro-batch / `finish_grad_sync` |
| --- | --- | --- |
| `optim_grads` / `optim_grads_params` | 每层 RS（必须 `overlap_grad_reduce=True`，否则 `synchronize_gradient_reduce` 断言失败） | 等待全部 RS，`update_main_grads` |
| `optim` | 只在 `main_grad_buffer` 中累加，不通信 | `overlap_grad_reduce=True` 时在最后一个 micro-batch 的 hook 中逐组 RS；否则在 `start_grad_sync` 中统一 `reduce_scatter_gradients()` |
| `no_shard` | 同上 | all-reduce（同步或逐组异步） |

判断「最后一个 micro-batch」使用 `is_last_microbatch`（由 `no_sync()` 上下文维护）或 `model_auto_sync`。

## 9. Buffer 管理

### 9.1 持久 buffer 与临时 buffer

| 类别 | 内容 | 分配方式 |
| --- | --- | --- |
| 持久 | 分片后的 `model_weight`（ZeRO-3 只存 shard）、`main_weight`、`main_grad` 分片（或全量） | 初始化时**每种 dtype 只分配一块大 tensor**（`buffer_all_in_one=True`，`:3129`，fp32/fp16/bf16/float8(uint8) 各一块）。各 bucket 按 offset 切片，处于 `mem_alloc_context()` 内（NCCL UB 时直接从注册池分配） |
| 临时 | ZeRO-3 未分片的权重 bucket（fwd/bwd）、MXFP8 transpose 权重 bucket、未分片的梯度通信 bucket、HSDP 自定义 comm dtype 的梯度 bucket | 通过 `DataParallelBuffer.fetch_bucket` → `allocate_bucket_storage` → `temporary_bucket_allocator.allocate(bucket_id, size, dtype)` |

初始化阶段（`:3120` 附近），权重先以全量形式分配并完成初始化，再拷贝 shard 进持久 buffer，最后 `free_bucket_storage()`。因此整模型全量权重只在初始化时出现一次。

`fetch_bucket`（`:1328`）的逻辑：

```python
if 未分片 且 dtype == buffer.dtype:  返回持久 buffer 中该 bucket 的切片（零拷贝）
else:                                 allocate_bucket_storage(...)  # 走临时 allocator
if set_param_data:                    p.data = bucket 中对应 item 的 view（FP8 用 fp8_set_raw_data）
```

`set_param_data=True` 只在 AG 时使用。unshard 后，模块参数的 `.data` 直接指向 bucket 内存，计算代码不需要任何改动。

### 9.2 Allocator 选择（`:2592-2654`）

```python
if fsdp_double_buffer and 存在 fsdp_unit_modules:
    POOL = MaxPoolAllocator if megatron_fsdp_max_pool_double_buffer else FixedPoolAllocator
    weight_alloc           = POOL("fsdp_params", size=2)
    transpose_weight_alloc = POOL("fsdp_fp8_transpose_params", size=2)
    main_grad_alloc        = POOL("fsdp_grads", size=2, dtype_fn=grad_comm_dtype 或 grad_dtype)
    hsdp_grad_comm_alloc   = POOL("hsdp_grad_comm", size=2)        # 仅 HFSDP
else:
    weight_alloc = transpose_weight_alloc = StorageResizeBasedBucketAllocator()
    main_grad_alloc = None  → DataParallelBuffer 内部回退为 TemporaryBucketAllocator()
```

| Allocator | 分配 | 释放 | 特点 |
| --- | --- | --- | --- |
| `TemporaryBucketAllocator`（`:495`） | 每个 bucket_id 首次 `torch.empty(size)` | `resize_(0)` 并从 dict 删除 | 交给 PyTorch caching allocator。默认的梯度 bucket 用它 |
| `StorageResizeBasedBucketAllocator`（`:559`） | tensor 对象常驻，`_alloc_storage` 把 storage `resize_` 回 size | `_free_storage`：`storage.resize_(0)` | 默认的权重 bucket 用它。tensor 对象和 `p.data` 的 view 关系保持不变，只是底层 storage 被收回或重新申请 |
| `RotaryBucketAllocator`（`:594`） | 从 idle 列表取一个 `GlobalMemoryBuffer` 槽，没有就新建 | 槽位还回 idle | 环形复用，数量不设上限。当前主路径未使用 |
| `FixedPoolAllocator`（`:679`） | 2 组 × 每 unit 的 bucket 偏移，从 `GlobalMemoryBuffer` 取名为 `{name}_{buf}_{offset}` 的固定 tensor | 槽位还回 idle | 只服务「对称」的 unit：bucket 数、dtype、numel 都相同（如同构 TransformerLayer）。不对称的 unit 回退到 StorageResize，或在 `fsdp_db_use_persist_buf_on_alloc_fail=True` 时使用持久命名 buffer |
| `MaxPoolAllocator`（`:918`） | 对所有 unit 按 dtype、按 bucket 偏移取**最大尺寸**，建 2 组池；每个 bucket 预先绑定偏移 | 槽位还回 idle | 面向异构模型（MoE/dense 混合、Mamba 混合）。所有 unit 都能双缓冲，代价是每个槽按最大 unit 预留 |

### 9.3 双缓冲池的细节

- **槽位规划**：
  - FixedPool 先挑出「最大的对称 unit 集合」，再为每个 bucket 记录 `(buf=-1, offset)`。
  - MaxPool 把每个 unit 的 bucket 按尺寸降序排列，逐偏移取最大值，组成 `max_dtype_bucket_sizes[dtype]`，并记录每个 bucket 对应的 offset（`:1005-1055`）。
  - idle 池初始为 `{0,1} × 各 offset`（MaxPool 还要乘上各 dtype）。
- **strict_assignments**：bucket 第一次拿到某组槽位后会记住它，以后尽量拿同一块。这样同一 bucket 每个 step 的地址固定，满足 CUDA Graph 和 NCCL UB 的需求。拿不到时会 warn，并分配别的空闲槽。
- **槽位耗尽**：池中只有 2 份，第 3 个 unit 申请时没有空闲槽，分配会失败。这就是 AG 预取（`_extend_by_prefetch_size` 的 2-unit 上限、`all_gather_params` 中 >2 unit 抛 ValueError）和 RS（`_enforce_double_buffer_limit`）必须严格限制在 2 个 unit 的原因。
- **非 unit 参数**：不参与双缓冲，走 backup（StorageResize，此时 NCCL UBR 不生效）或持久 buffer。
- **底层内存**：`GlobalMemoryBuffer.get_tensor(shape, dtype, name, mem_alloc_context)`（`utils.py:815`）以 `(name, dtype)` 为 key，只有请求尺寸变大时才重新分配。FixedPool 在 `mem_alloc_context` 下需要扩容时，会先 `torch.cuda.synchronize()` 再分配，以免旧内存仍被异步 kernel 使用。

### 9.4 显存账（ZeRO-3，L 层同构，每层未分片权重 W，DP = N）

| 项 | 默认（StorageResize + caching allocator） | `fsdp_double_buffer` |
| --- | --- | --- |
| 持久权重 shard | L·W/N | L·W/N |
| 未分片权重 | 前向约 2W（当前层 + 预取层）；反向约 2W，root pre-backward 留下的 lazy bucket 回收前可能更多 | 固定 2W（池），fwd/bwd 共用；FP8 transpose 另有一套 2W |
| 未分片梯度 | ≤ RS 容量 ≈ 2 层的梯度（grad_comm_dtype） | 固定 2 层（池） |
| 碎片 | 依赖 caching allocator，长期可能碎片化 | 地址固定，无碎片，可注册 NCCL UB |

## 10. NCCL User Buffer / 对称内存注册

开启 `nccl_ub=True` 时（`param_and_grad_buffer.py:2117-2203`）：

1. 强制 `fsdp_double_buffer=True`，因为 UB 注册要求地址固定的持久内存。
2. `nccl_allocator.init()`，创建 `NCCL_MEMORY_POOL`（`create_nccl_mem_pool(symmetric=not disable_symmetric_registration)`）。
3. 要注册的通信组 `ubr_groups` 依次为：
   - DP-Shard 组
   - 专家 DP 组
   - 独立 AG 组（`dp_cp_ag`）
   - 专家独立 AG 组
   - HFSDP outer 组（IB 域，放在最后）

   每组先 `barrier` 一次，最后做一次全局 `barrier`，确保所有 rank 一起进入注册集合通信，避免死锁。
4. `mem_alloc_context` 的取值：
   - MCORE allocator：单组用 `nccl_mem(pool, group, symmetric)`，多组用 `MultiGroupMemPoolAllocator`。
   - `fsdp_manual_registration=True`：用 `MemPoolAllocatorWithoutRegistration`，只从池中分配、不立即注册。
   - APEX allocator：不支持对称注册和手动注册，会回退并 warn；多组用本文件的 `MultiGroupUBRAllocator`（`:175`）。
5. 持久 buffer 和双缓冲池都在这个 context 下分配，所以 AG/RS 的输入输出都位于注册内存中。NCCL 可以走零拷贝或 SM 更少的 kernel（对称内存下还可以用 NVLS 等算法）。
6. 手动注册：`training.py:5282-5297` 在 `iteration == start_iteration + 1`，即预热一个 iteration、池中 buffer 基本分配完毕之后，调用 `manual_buffer_registration()`（`:2298`）。流程是 `synchronize → barrier → synchronize`，然后对每个组执行 `register_mem_pool`。一次性注册整个池，避免每次分配都触发注册。

约束：使用 `nccl_ub` 且 torch < 2.11 时，不能开启 `expandable_segments`。在 arguments.py 中，`nccl_ub` 也会把 double buffer 和 manual registration 一起打开。

## 11. Step 边界：`start_param_sync` / `finish_grad_sync`

`megatron_fsdp.py:1344-1466`：

- `start_param_sync()`，在每个 step 前向开始前调用：
  - `overlap_param_gather=True` 且 `fsdp_all_gather_in_start_param_sync=True`：只对**第一个参数**发起 `all_gather_and_wait_parameters_ready(prefetch=True, wait_bucket_ready=False)`。效果是提前启动第 0 组的 AG 和下一组的预取，但不等待，让它们和 data loader、前一 step 的收尾工作重叠。
  - 不 overlap（或 `force_sync`）：先 `synchronize_param_gather()`，然后对**全部 bucket** 顺序执行 `async_bucket_gather`。分片权重每组都立即 `wait_bucket_ready`，最后再统一 wait 一遍。这是同步的全量 unshard。
- `finish_grad_sync()`，在 optimizer.step 前调用：
  1. `synchronize_gradient_reduce()`：在 overlap 模式下先 `wait_for_previous_grad_reduce(0)`，再 `grad_reduce_pipeline.reset()`。非 overlap 模式只允许 `no_shard`/`optim`，并同步执行 `start_grad_sync()`（all-reduce 或 `reduce_scatter_gradients`）。
  2. `update_main_grads()`：把 main_grad 分片挂到优化器的 DTensor 参数上。
  3. `synchronize_param_gather()`：`all_gather_pipeline.reset(preserve_non_fsdp_units=True)`，等待残留的 AG，释放 unit bucket，非 unit bucket 置为 `PRESERVED`。
  4. 把模块参数替换为分布式（DTensor）参数，供优化器使用。
  5. `microbatch_count = 0`。
- 优化器 step 之后调用 `install_optimized_model_weights()` → `copy_main_weights_to_model_weights()`，把高精度 main weight 转回模型权重 shard（必要时做 FP8 量化）。下一 step 的 HFSDP 外层 AG 只在 `microbatch_count == 0` 时执行，正是为了拿到这次更新后的权重。
- `zero_grad_buffer()`：清零持久梯度 buffer，并把 `grad_added_to_main_grad` 复位。

## 12. 激活重计算的处理

重计算会在反向阶段**再跑一次前向**，v1 用模块状态区分这两种前向：

1. root pre-backward 把所有模块置为 `PRE_BACKWARD`。
2. 重计算触发 `pre_fwd(i)` 时，看到状态是 `PRE_BACKWARD`，就关闭前向预取（`megatron_fsdp.py:851`）。这时如果按 +1 方向预取，会拿到反向不需要的层，白白占用双缓冲槽位。
3. 重计算的 `post_fwd(i)` 使用 lazy 释放（`release_module_parameters(lazy=True)`）。紧接着的 `pre_bwd(i)` 需要同一批权重，AG 时发现状态为 `READY_TO_USE`，直接清除 `can_be_released` 并复用，**不会重复 AG**。
4. 真正释放发生在 `post_bwd(i)`（立即释放），或者推迟到下一次 `async_bucket_gather` 分配前的 `recycle_unused_buckets()`。

v2 的做法相同：`pre_forward` 在重计算时不预取，`post_forward` 在重计算时直接跳过 reshard。

## 13. MFSDP v2（`experimental/`）的差异

v2 是 `fully_shard` 风格的重写，基于 DTensor 和 `DBuffer`，overlap 思路与 v1 一致，实现上有以下不同：

| 方面 | v1 `MegatronFSDP` | v2 `experimental` |
| --- | --- | --- |
| 流 | AG 流、RS 流（各模型实例各一对） | `FsdpContext` 全局共享一条 `allgather_stream`。`reduce_scatter_stream` 默认独立，`unify_communication_stream=True` 时与 AG 共用一条，以降低峰值显存（issue #6471） |
| 预取顺序 | bucket id ±1 | `ExecutionOrder`（`execution_order.py`），由 `named_modules` 静态构建：前向为先序遍历，反向为 children 逆序递归。`advance()` 推进游标，同一模块多次出现时校验游标；`next_item(offset)` 只看后继、不移动游标 |
| 预取量 | 全局 `suggested_communication_unit_size // 2` 元素 | 每个模块可单独配置 `SchedulePolicy(forward_prefetch_size, backward_prefetch_size)`（`schedule.py`）：`None` 表示恰好预取 1 个后继，`0` 表示关闭，正数为元素预算 |
| 去重 | `BucketStatus` 状态机 | 每个 parameter group 的 `_unshard_event`：已设置就跳过 |
| 释放 | 计算流侧 CPU free，依赖下一次 AG 前的 `wait_stream` | `_reshard_parameter_groups`：先 `ag.wait_stream(current)`，再**在 AG 流上** `release_unsharded_storage`。释放和下一次分配在同一条流上，天然有序，仍然不需要 `record_stream` |
| unshard | `all_gather_into_tensor` 写进 bucket，再设置 `p.data` view | `unshard_parameters`：在对称内存池下 `reallocate_storage`，用 DTensor `redistribute(Shard→Replicate)`，配合 `_unsafe_preserve_version_counter` 防止 autograd 版本号报错 |
| 梯度 | 未分片梯度 bucket + `reduce_scatter_tensor` + `+=` | `allocate_partial_grad_buffer`：在 RS 流、对称内存池下分配 `Partial("avg")` 的 `DBuffer`（dtype 为 `grad_comm_dtype`）。计算流先 `wait_stream(rs)`，再把梯度拷进 partial buffer，然后 `rs.wait_stream(current)`，最后在 RS 流上 `reduce_partial_gradients`：能直接 redistribute 进 main_grad 就直接写，否则先 `grad_scale`，再 add 或 copy |
| step 收尾 | `finish_grad_sync` 内统一等待 | `finish_grad_sync = current.wait_stream(rs_stream)`。`post_backward` 执行延迟回调，最后一个 micro-batch 自动 finish。`caller_managed_grad_sync` 可以把这一步交给调用方 |
| root 处理 | root pre/post backward hook | root `pre_forward` 先 `ag.wait_stream(current)`；root `pre_backward` 注册 post-backward 回调，并 `rs.wait_stream(current)`，以兼容 CUDA Graph capture |

v2 的预取由执行顺序驱动，按模块而非 bucket 粒度管理，同一份代码也能处理非层状模型。代价是缺少 v1 那套完整的双缓冲池和 HFSDP 外层流水，仍在演进中。

## 14. 关键配置与调优建议

| 参数 | 默认 | 作用 / 建议 |
| --- | --- | --- |
| `--data-parallel-sharding-strategy` | `optim_grads_params`（CLI）/ `no_shard`（config 类） | 只有 ZeRO-3 才有 AG 预取 |
| `--overlap-param-gather` / `--overlap-grad-reduce` | False | FSDP 场景都应开启。ZeRO-2/3 必须开启 `overlap_grad_reduce` |
| `--suggested-communication-unit-size` | None → 平均 unit×2，上限 1e9 | 预取量 = 值/2，RS 容量 = 值。想预取 2 层就设为平均 unit 的 4 倍；设成 0 或 1 才能关闭预取 |
| `--fsdp-double-buffer` | False | 固定 2 份未分片 buffer，显存确定、无碎片，是 NCCL UB 的前提。同时给预取和 RS 加上 2-unit 硬上限 |
| `--megatron-fsdp-max-pool-double-buffer` | False | 层大小不一（MoE/dense 混合、Mamba 混合）时用它代替 FixedPool。会自动开启 double buffer |
| `--fsdp-db-use-persist-buf-on-alloc-fail` | False | 不对称或非 unit 的 bucket 改用持久 buffer，换取 UB 可用 |
| `--create-all-gather-group` | False | 给 AG 单独建一个 NCCL communicator，解决 AG 和 RS 的 head-of-line 串行。反向 overlap 收益明显，建议开启 |
| `--use-nccl-ub` / `--fsdp-manual-registration` / `--disable-symmetric-registration` | False | 注册内存后通信 SM 占用更少、带宽更高。手动注册在第 2 个 iteration 一次性完成 |
| `--hfsdp-param-gather-overlap` | False | HFSDP 外层 AG 多预取 1 个 unit，并放到独立 outer 流 |
| `fsdp_all_gather_in_start_param_sync` | True | step 开始时提前发起第一组 AG |
| `gradient_reduce_div_fusion` / `average_in_collective` | – | 梯度缩放融进 NCCL（premul_sum / AVG），省掉一次 `mul_` |
| `grad_comm_dtype`（mp_policy） | None | 例如用 bf16 通信、fp32 累加，梯度通信 bucket 按该 dtype 分配 |

已知限制：

1. **预取顺序是静态的**：按 bucket id（即注册顺序）决定，不是运行时录制的。`forward()` 调用顺序与注册顺序不一致、模块被复用、或存在条件分支时，会预取错层。结果仍然正确，因为使用前都会 `wait`，但会浪费带宽和 buffer。
2. **层大小不均**：预取量按平均 unit 计算，遇到小层可能连续预取多层；没有 double buffer 时没有硬上限，显存峰值会上升。
3. **帮助文本与实现不符**：`arguments.py` 说「a smaller value disables prefetching」，实际上只要阈值大于 0 就至少预取 1 组。
4. **非 unit 参数常驻**：embedding、output layer 等如果不在 `fsdp_unit_modules` 中，ZeRO-3 下也会常驻未分片副本（`PRESERVED`）。
5. **共用 communicator**：没有独立 AG 组时，反向的 `AG_{i-1}` 会被 `RS_i` 阻塞，即使两者在不同 stream 上。

