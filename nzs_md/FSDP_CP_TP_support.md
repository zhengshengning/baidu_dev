# FSDP 支持 CP / TP 的梯度规约修改说明

涉及文件（Paddle 子仓库）：

- `python/paddle/distributed/fsdp/fully_shard_fusion.py`
- `test/legacy_test/test_fsdp_fully_shard_fusion.py`

| 部分 | 状态 | 来源 |
| --- | --- | --- |
| CP | 已提交 | Paddle commit `12c390870c` `[FSDP] Sum cp-partial grads over cp in the sharding reduce (#79948)` |
| TP（sequence parallel 梯度） | 未提交 | Paddle 工作区 `git diff HEAD`（基于 `8e379b722c`） |

## 背景：FSDP 梯度怎么规约

每个 `BufferGroup` 有一个 `grads_buffer`（`TensorFusionBuffer`），反向结束后在 FSDP 组（即 sharding 组）上做 reduce_scatter，再乘 `grad_scale = 1 / grad_div`：

- `grad_div * grad_accum_steps == fsdp_degree` 时直接用 NCCL `AVG`；
- 否则用 `SUM`，fp32 主梯度把缩放推迟到 `finish_grads_sync` 里对规约后的分片做（`grad_scale_pending`），其他 dtype 先缩放整块 buffer 再 `SUM`。

修改前 `grad_div` 固定为 `self._fsdp_group.nranks`，也就是对整个 sharding 组求平均。

---

## 一、CP 修改（已提交 `12c390870c`）

### 问题

`hybrid_parallel_topo_order: sharding_first` 下 sharding 组跨越 cp 维度（日志里 `cp_sharding_group` 和 `sharding_group` 是同一组 rank）。开 CP 后，同一条样本的序列切到 cp 个 rank 上，每个 rank 算出的梯度只是这条样本的一部分（cp-partial），正确做法是在 cp 维度上**求和**，只在数据维度（sharding / cp）上求平均。

原实现对整个 sharding 组求平均，相当于 cp-partial 梯度多除了一个 cp。

例外是 cp 维度上本来就相同（cp-replicated）的梯度，它们应该在整个 sharding 组上求平均：

- dense 参数：`context_parallel_disable_scale_grad = True`（embedding / MTP embedding，`ContextParallelScatterOp` 反向会 all-gather 梯度，每个 cp rank 拿到同一份完整梯度；由 `PaddleFleet/src/paddlefleet/context_parallel_utils.py` 设置）；
- expert 参数：`expert_cp_replicated = True`。

这与非 FSDP 路径 `ernie5/src/trainers/pretraining_trainer.py:8255` 附近的约定一致：sharding 规约本身跨 cp，大多数参数不需要 cp 系数，只有 cp-replicated 参数用 `S/cp`。

### 改动

1. `BufferGroup` 新增字段 `cp_replicated: bool = False`。
2. `FSDPBufferManager.__init__` 读取 `cp_degree = hcg.get_context_parallel_world_size()`（hcg 没有该方法时视为 1）。
3. 参数分组 key 增加一项“是否 cp-replicated”：
   ```python
   cp_degree > 1 and bool(
       getattr(param, "expert_cp_replicated", False)
       if is_expert
       else getattr(param, "context_parallel_disable_scale_grad", False)
   )
   ```
   cp-partial 和 cp-replicated 的参数因此进入不同的 buffer，各自有自己的 `grad_div`。`sort_key` 的解包和 `BufferGroup(...)` 构造同步加上这一项。
4. 构造 `grads_buffer` 时：
   ```python
   grad_div = self._fsdp_group.nranks // (1 if group.cp_replicated else cp_degree)
   ```
   - cp-partial：除以 `sharding_degree / cp`，等价于 cp 上求和、数据维上求平均；
   - cp-replicated：仍除以 `sharding_degree`。

### cp = 1 时保持原行为

key 里的标记写成 `cp_degree > 1 and ...`，cp = 1 时恒为 False，不会因为参数上带了 `context_parallel_disable_scale_grad` 等标记而多拆 buffer。这样 buffer 数量和 `fuse_params_<gid>` 编号与修改前完全一致，不影响已有 checkpoint 和对齐结果。

### 单测 `TestCpGradDiv`

用 `_FakeHcg`（sharding=4，cp 可配）和只记录 `grad_div` 的 `_RecordingFusionBuffer` 构造 `FSDPBufferManager`，4 个参数：`plain`、`dense_rep`（disable_scale_grad）、`expert`、`expert_rep`（expert_cp_replicated）。

- `test_cp_partial_grads_summed_over_cp`（cp=2）：`plain` / `expert` 的 `grad_div == 2`，`dense_rep` / `expert_rep` 的 `grad_div == 4`，共 4 个 buffer。
- `test_cp_one_keeps_grouping`（cp=1）：`plain` 与 `dense_rep` 同组、`expert` 与 `expert_rep` 同组，共 2 个 buffer，`grad_div` 都是 4。

---

## 二、TP 修改（未提交）

### 问题

TP 开启 sequence parallel 后，被标记 `param.sequence_parallel = True` 的参数（LayerNorm / RMSNorm 权重等在 tp 上复制、但输入按序列切到各 tp rank 的参数）在每个 tp rank 上只拿到部分序列的梯度，需要在 tp 组上**求和**。非 FSDP 路径由 `SPGradSyncCallback`（`use_sp_callback=True`，默认）或 `register_sequence_parallel_allreduce_hooks` 在 mp 组上做 SUM。FSDP 把梯度写进自己的 fused grads buffer，并在 sharding 组上 reduce_scatter，这一步不覆盖 tp 维度，所以由 FSDP 在分片上自己补一次 tp 求和。

### 改动

#### 1. `_sp_grad_slices(buffer_groups)`

遍历每个 `grads_buffer` 中的参数，对 `trainable` 且 `sequence_parallel` 的参数调用 `grads_buffer.param_shard(param)`，得到该参数落在本 rank 分片内的区间 `(lo, hi)`；参数不在本分片时返回 None，跳过。

在同一个 buffer 内，相邻的 SP 参数区间合并成一段，中间夹着非 SP 参数就断开。结果是 `[(grads_buffer, lo, hi), ...]`。

非分片 buffer（`is_sharded = False`）的 `shard_start = 0`、`shard_end = total_buffer_size`，区间覆盖整个参数，同样适用。

#### 2. `_sum_slices_over_group(slices, group)`

- 跳过 `_data_released` 的 buffer（keep_whole 组在非 owner rank 上 reduce 后会释放梯度 buffer）；
- 按 dtype 归类各区间的 `data_buffer._slice(lo, hi)` 视图；
- 每个 dtype 只发一次 `all_reduce(SUM, group=tp)`：只有一段时直接在视图上原地规约，多段时先 `concat` 再规约，最后 `assign` 回各视图。

#### 3. `FSDPBufferManager.__init__` 末尾

```python
self.tp_group = None
self.sp_grad_slices = []
if hcg.get_model_parallel_world_size() > 1:
    self.tp_group = self.hcg.get_model_parallel_group()
    self.sp_grad_slices = _sp_grad_slices(self.buffer_groups)
```

区间在初始化时算一次，之后每步复用。tp = 1 时列表为空，不发任何通信。

#### 4. `FSDPCommManager.finish_grads_sync`

在 sharding 组 reduce_scatter 完成、推迟的 fp32 缩放（`grad_scale_pending`）也做完之后，调用一次 `_sum_slices_over_group(sp_grad_slices, tp_group)`。

### 设计要点

- **只在规约后的分片上做**：sharding 规约之后每个 rank 只持有 `1/sharding_degree` 的梯度，在分片上做 tp all_reduce，通信量和非 FSDP 路径在完整梯度上做相比少 sharding_degree 倍。
- **先缩放后求和不影响结果**：缩放和求和都是线性的，顺序可交换。
- **tp 组内调用次数一致**：在 `sharding_first` 拓扑下，一个 rank 的 tp peer 在各自的 sharding 组里坐标相同；buffer 布局、分片区间、`_assign_whole_owners` 的 owner 选择也都由参数大小决定，tp peer 之间一致。所以 tp 组内各 rank 跳过/参与的 buffer 相同，concat 后长度相同，all_reduce 次数匹配，不会 hang。前提是 SP 参数在各 tp rank 上形状相同（SP 参数本身就是 tp 复制的，满足）。

### 单测 `TestSpGradSumOverTp`

用真实 `TensorFusionBuffer`（fsdp 组 nranks=1，不分片）放 4 个 `[3]` 参数：`sp_a`、`sp_b`、`plain`、`sp_c`。

- `test_slices_merge_adjacent_sp_params`：`sp_a`+`sp_b` 合并成一段，`sp_c` 单独一段，`plain` 不出现。
- `test_sum_touches_only_sp_slices`：mock `all_reduce` 为 ×2（模拟 tp=2），断言只调用一次且 group 为 tp，SP 参数位置变成 2，`plain` 保持 1。
- `test_no_slices_without_sp_params`：没有 SP 参数时返回空列表。

### 提交前待处理

**避免重复求和（需确认）**：`ernie5/src/trainers/pretraining_trainer.py:4276` 在 FSDP 模式下仍会在 `fully_shard` 之前调用 `enable_sequence_parallel(model)`，tp > 1 且开 SP 时会挂上 `SPGradSyncCallback` 或 SP allreduce hook。需要确认这两条路径在 FSDP 下不会再对 SP 梯度做一次 mp SUM（例如 sharded buffer 的 `main_grad` 在 `finish_grads_sync` 末尾已被清掉，但非分片 buffer 的 `main_grad` 仍在），否则 SP 梯度会被乘 tp。稳妥做法是 FSDP 模式下跳过 `enable_sequence_parallel`。

当前工作区的测试文件在 `TestReshardPolicy` 之后插入了一份新的 `_FakeGroup` / `_FakeHcg` / `_RecordingFusionBuffer` / `TestCpGradDiv`，与 commit `12c390870c` 已有的定义重复（`test_fsdp_fully_shard_fusion.py` 第 979 行和第 1128 行附近各一份）。后定义的会覆盖前面的，测试能跑，但应该删掉重复的那份，只保留新增的 `TestSpGradSumOverTp`。插入处 `class _FakeGroup` 前只有一个空行，flake8 会报 E302。

另外 `third_party/sleef` 子模块指针有变化（`7623d6c` → `6ee14bc`），与本修改无关，提交时不要带上。
