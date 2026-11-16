# FSDP 性能优化拆解：逐功能点修改与收益

本文分两部分：

- 第一部分（S0–S8）：PR #79881 按 hunk 拆分，加上当时的本地修改（side stream、`prefetch_units=2`、DeepGEMM SM 数）。
- 第二部分（T0–T6）：`fsdp_opt_pr.txt` 中其余 PR（#79887 / #79897 + Fleet #2371 / #79898 / #79899 / #79901 / #79903）在最终代码上按开关逐个累积打开，见文末[第二部分](#第二部分fsdp_opt_prtxt-各-pr-功能点拆解t0t6)。

# 第一部分：PR #79881 + 本地修改（S0–S8）

测试对象：ERNIELite T4（h1280 / L20，34.93B-A1.232B，MoE 768 专家 top12，EP8），2 机 16 卡 H 系列（132 SM），
`sharding: fsdp`，mb=2，ga=24，seq 8192，MuonH + AdamW + AdamH，FP8 MoE。
测试脚本：`run_fsdp_test.sh`（`max_steps=10`），指标取 rank0 `tokens_per_sec_per_card` 在 step 4–10 的均值（前 3 步为 warmup/编译）。

修改来源共三处：

| 来源 | 文件 | 说明 |
| --- | --- | --- |
| PR [#79881](https://github.com/PaddlePaddle/Paddle/pull/79881)（本地 commit `9030271e0c`） | `fully_shard_fusion.py` / `fully_shard.py` / `optimizer/muon.py` | FSDP 通信与调度优化、AdamW+Hyperball 分片 |
| 本地 Paddle 未提交修改 | `fully_shard_fusion.py` | 通信 side stream、`prefetch_units=2` |
| 本地 PaddleFleet 未提交修改 | `transformer/moe/fp8_utils.py` | dW overlap 后 DeepGEMM SM 数恢复值 |

Paddle 生效的是 `Paddle/build/python/paddle/...`（`run_fsdp_test.sh` 把 `Paddle/build/python` 放在 `PYTHONPATH` 最前），测试时逐阶段替换 build 下的三个文件。

## 结果汇总

各阶段为累积叠加（Sn = Sn-1 + 本功能点）。tokens/s/card 为 step 4–10 均值 ± 标准差；fwd-bwd / opt-step 取 trainer Profile 同区间均值；
loss 为 step 10 的值（各阶段一致到 1e-4 量级，未见精度回退）。

| 阶段 | 功能点 | 来源 | tokens/s/card | 相对上一阶段 | 相对 baseline | fwd-bwd (ms) | opt-step (ms) | max_mem_alloc (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | baseline（PR 前 Paddle + fp8_utils 原版 `set_num_sms(0)`） | — | 13874.9 ± 42 | — | — | 26911 | 1199 | 67.53 |
| S1 | gather buffer `empty` + reduce-scatter AVG | PR | 13886.1 ± 62 | +0.1% | +0.1% | 26906 | 1209 | 67.53 |
| S2 | 不再强制 Muon 逐矩阵 NS | PR | 14352.4 ± 72 | +3.4% | +3.4% | 26831 | 362 | 67.53 |
| S3 | AdamW+Hyperball 分片更新 | PR | 14358.0 ± 53 | +0.0% | +3.5% | 26814 | 359 | 67.53 |
| S4 | Muon owner 大块优先分配 | PR | 14329.9 ± 64 | −0.2% | +3.3% | 26860 | 362 | 70.09 |
| S5 | 预取 / 梯度队列调度重写 | PR | 15054.6 ± 58 | +5.1% | +8.5% | 25557 | 362 | 70.30 |
| S6 | 集合通信放 side stream | 本地 Paddle | 18104.8 ± 130 | **+20.3%** | +30.5% | 21142 | 361 | 70.31 |
| S7 | `prefetch_units` 1→2 | 本地 Paddle | 21177.5 ± 179 | **+17.0%** | +52.6% | 17990 | 365 | 70.74 |
| S8 | DeepGEMM dW overlap 后 `set_num_sms(120)` | 本地 PaddleFleet | 21160.6 ± 189 | −0.1% | +52.5% | 17973 | 365 | 70.74 |

结论：

- 总收益 **13875 → 21161 tokens/s/card，+52.5%**。PR #79881 整体贡献 +8.5%，本地 Paddle 两项修改贡献 +40.6%（相对 S5）。
- 收益最大的是通信与计算的 overlap：side stream（+20.3%）和 2-unit 预取（+17.0%），fwd-bwd 从 25.6s 降到 18.0s。
- S2 的收益全部来自 optimizer-step（1199 → 362 ms）：batched NS 替代逐矩阵 NS，kernel launch 大幅减少。
- S1 / S3 / S4 / S8 在本配置下都在噪声范围内（|Δ| ≤ 0.2%，标准差约 0.4–0.9%）：
  - S1：ga=24，AVG 分支不触发，只剩 `empty` 省的 memset。
  - S3：opt-step 已经只有 ~360 ms，AdamH 只作用于 lm_head 这类少数 2D 参数，省掉的 gather/broadcast 量很小。
  - S4：吞吐无变化，但 max_mem_allocated 上升约 2.6 GB（owner 分布变化导致某些 rank 持有的全量 Muon 矩阵更大），显存紧张时需留意。
  - S8：这套配置下 dW overlap 路径带来的 SM 竞争不是瓶颈；保留 12 个 SM 也没有带来负收益。
- 单次测试 10 步、每阶段 1 次运行；±0.5% 以内的差异不应视为确定结论。

## 功能点详解

### S1 · 通信微优化：gather buffer 用 `empty`、reduce-scatter 折叠 AVG（PR #79881）

`fully_shard_fusion.py` `TensorFusionBuffer.get_tmp_buffer` / `do_reduce_scatter`

- param all_gather 的临时 buffer 会被 all_gather 完整覆盖，`paddle.zeros` 改 `paddle.empty`，省一次全量 memset kernel。
- grad buffer 仍需 `zeros`（要累加）。
- reduce-scatter 的缩放系数恰好为 `1/fsdp_degree` 时用 `ReduceOp.AVG` 让 NCCL 顺带除，省一次 `scale_`；
  有梯度累积（本配置 ga=24）或 EP 组时条件不成立，仍走 `scale_ + SUM`，所以本配置下这一半不生效。

```python
if self.is_params:
    # Fully overwritten by all_gather.
    self.tmp_data_buffer = paddle.empty(shape=[self.total_buffer_size], dtype=self.dtype)
else:
    # Grads accumulate into it.
    self.tmp_data_buffer = paddle.zeros(shape=[self.total_buffer_size], dtype=self.main_grad_dtype)
...
if self.grad_div * self.grad_accum_steps == self.fsdp_degree:
    op = paddle.distributed.ReduceOp.AVG
else:
    tmp_buffer.scale_(self.grad_scale)
    op = paddle.distributed.ReduceOp.SUM
```

### S2 · 不再强制 Muon 逐矩阵 Newton-Schulz（PR #79881）

`fully_shard_fusion.py` 删除 `requires_muon_per_matrix_ns`；`muon.py` 删除 `_maybe_force_per_matrix_ns` 及其调用。

旧逻辑：只要 FSDP 按矩阵边界切 Muon 参数，就把 `ns_per_matrix` 强制改为 True，每个矩阵单独一次 NS（每矩阵一次 kernel launch）。
新逻辑：尊重用户配置（默认 False），同形状矩阵 batched NS（`paddle.stack + baddbmm`）。
代价：batched NS 的数值依赖 rank 持有的矩阵组合，不再与 stage1 逐位一致（对齐诊断时用 `MUON_FSDP_NS_SINGLE=1`，见本地 build 版 muon.py 的诊断 hack）。

```python
-            self._maybe_force_per_matrix_ns(fsdp_context)
...
-    def _maybe_force_per_matrix_ns(self, fsdp_context):
-        if self._ns_per_matrix:
-            return
-        fun = getattr(fsdp_context, "requires_muon_per_matrix_ns", None)
-        if fun is None or not fun():
-            return
-        self._ns_per_matrix = True
```

### S3 · AdamW+Hyperball（AdamH）参数改为分片更新（PR #79881）

`fully_shard.py` `_mark_params_for_optimizer`、`fully_shard_fusion.py` `FSDPBufferManager` / `bind_optimizer`、`muon.py` `_hyperball_apply`

旧逻辑：`use_hyperball` 参数一律标成 `use_muon`，走 Muon 的"owner rank 收全量 → 计算 → 广播"路径。
AdamH 只需要整矩阵 Frobenius 范数，不需要全量矩阵。
新逻辑：2D、非 Muon、无 `split_concat_func` 的 hyperball 参数标 `use_hyperball_sharded`，每个参数独占一个 buffer，
在本地分片上做 AdamW，范数用 `sum(x*x)` + 组内 `all_reduce` 求全局值，省掉 gather/broadcast。

```python
# fully_shard.py
param.use_hyperball_sharded = bool(
    hyperball and not info.use_muon
    and getattr(info, "split_concat_func", None) is None
    and len(shape) == 2
)
param.use_muon = bool(info is not None and (info.use_muon or hyperball)
                      and not param.use_hyperball_sharded)

# fully_shard_fusion.py：独立 buffer，并挂上 all-reduce 组
if group.hyperball_sharded:
    data_buffer = group.params_buffer.data_buffer
    data_buffer._hyperball_group = group.fsdp_group if group.params_buffer.is_sharded else None
# bind_optimizer：fused buffer 注册为 AdamW+Hyperball
info_map[group.params_buffer.data_buffer.name] = MuonParamInfo(use_muon=False, use_hyperball=True)

# muon.py _hyperball_apply
if reduce_group is not None or w.ndim == 1:
    def _fro(x):
        sq = paddle.sum(x * x).reshape([1])
        if reduce_group is not None:
            paddle.distributed.all_reduce(sq, group=reduce_group)
        return paddle.sqrt(sq)
```

### S4 · Muon owner 按大小降序分配（PR #79881）

`fully_shard_fusion.py` `_assign_muon_owners`

贪心分配 owner rank 时先放大 buffer，降低各 rank Muon 计算负载的最大值（LPT 调度）。

```python
-        for group in self.buffer_groups:
+        # Largest first to balance owner loads.
+        for group in sorted(self.buffer_groups, key=lambda g: -g.params_buffer.total_buffer_size):
```

### S5 · 预取 / 梯度队列调度重写（PR #79881）

`fully_shard_fusion.py` `FSDPCommManager`

- `double_buffer_limit` 改为 `prefetch_units`（按 unit 计数），一个 unit 包含多个 sharded group 时自动按 `groups_per_unit` 放大上限：
  `gather_buffer_limit = groups_per_unit * (prefetch_units + 1)`，`grad_queue_limit = groups_per_unit * max(prefetch_units, 1)`。
- 预取时机后移：先等待本 unit 的 all_gather 再发下一 unit 的预取，避免计算流被预取的 gather 阻塞。
- forward 中逐层回收 backward 残留的 reduce-scatter（每层退一个），不在 step 前集中等待。
- backward 末尾发出的预取拿到的是更新前的 shard，`zero_grad`/`reshard` 时等待并丢弃，保证下一次 forward 拿到新参数。
- step 末尾 `paddle.device.synchronize()`，让 backward 中在通信流释放的 buffer 回到显存池。

```python
unit_sizes = Counter(g.fsdp_unit_id for g in buffer_manager.buffer_groups
                     if g.params_buffer.is_sharded and g.fsdp_unit_id is not None)
groups_per_unit = max(2, max(unit_sizes.values(), default=0))
self.prefetch_units = max(int(prefetch_units), 0)
self.gather_buffer_limit = groups_per_unit * (self.prefetch_units + 1)
self.grad_queue_limit = groups_per_unit * max(self.prefetch_units, 1)
...
        elif self.grad_reduce_queue:
            # Retire leftover grad reduce-scatters, one per forward layer.
            self._wait_for_grad_comm(queue_limit=len(self.grad_reduce_queue) - 1)
...
        # Prefetch after waiting on this unit's gathers, or compute stalls on it.
        if self.enable_overlap:
            for next_gid in self._next_buffer_id(req_gids, is_backward):
                ...
                self._issue_async_gather(next_gid)
...
            if params_buffer.status == BufferState.SYNCING:
                params_buffer.comm_task.wait()
                params_buffer.comm_task = None
                params_buffer.status = BufferState.READY
```

### S6 · 集合通信放到独立 side stream（本地 Paddle 修改）

`fully_shard_fusion.py` `_async_collective` / `_StreamTask` / `do_all_gather` / `do_reduce_scatter`

`sync_op=False` 的 `paddle.distributed.all_gather/reduce_scatter` 走 ProcessGroup 内部通信流，并在调用时做 host 侧同步/事件管理。
改为每个通信组一条自建 side stream：先 `wait_stream(当前计算流)` 保证数据就绪，在 side stream 上以 `use_calc_stream=True` 发集合通信（host 不阻塞），
对涉及的 buffer `_record_stream()` 防止被分配器提前复用，返回的 task `.wait()` 只让计算流等 side stream 的 event。

```python
_comm_streams = {}  # group id -> side stream for the async collectives

class _StreamTask:
    def __init__(self, stream):
        self._event = stream.record_event()
    def wait(self):
        paddle.device.current_stream().wait_event(self._event)

def _async_collective(group, tensors, collective):
    stream = _comm_streams.get(group.id)
    if stream is None:
        stream = _comm_streams[group.id] = paddle.device.Stream()
    stream.wait_stream(paddle.device.current_stream())
    with paddle.device.stream_guard(stream):
        for t in tensors:
            t._record_stream()
        collective()
    return _StreamTask(stream)

def do_all_gather(self):
    tmp_buffer = self.get_tmp_buffer()
    return _async_collective(self.fsdp_group, (tmp_buffer, self.data_buffer),
        lambda: paddle.distributed.stream.all_gather(tmp_buffer, self.data_buffer,
            group=self.fsdp_group, sync_op=True, use_calc_stream=True))
```

### S7 · 默认预取 2 个 unit（本地 Paddle 修改）

`fully_shard_fusion.py` `FullyShardFusion.__init__`：`prefetch_units=1` → `2`。
配合 S5 的上限公式，gather buffer 上限变为 `groups_per_unit * 3`，可同时在途 2 个 unit 的 all_gather，用显存换更深的通信掩盖。

### S8 · DeepGEMM dW overlap 后 SM 数恢复为 120（本地 PaddleFleet 修改）

`PaddleFleet/src/paddlefleet/transformer/moe/fp8_utils.py` `ExpertsGroupGemmContiguousNode` `_compute_weight_grad`

dW 与通信 overlap 时先 `set_num_sms(118)` 跑 `k_grouped_bf16_gemm_tn_contiguous`，结束后原来恢复为 `0`（= 全部 132 SM）。
改为恢复 `120`，后续 DeepGEMM kernel 固定留出 12 个 SM 给并发的 NCCL / DeepEP 通信 kernel，减少通信与 GEMM 抢 SM。

```python
                 if overlap_gemm:
                     deep_gemm.set_num_sms(118)
                     deep_gemm.k_grouped_bf16_gemm_tn_contiguous(x, dy, weight_grad, ...)
-                    deep_gemm.set_num_sms(0)
+                    deep_gemm.set_num_sms(120)
```

### 不计入性能测试的本地修改

- `Paddle/build/python/paddle/optimizer/muon.py`：`MUON_FSDP_NS_SINGLE=1` 时逐矩阵 NS（stage1 对齐诊断，环境变量门控、默认关闭；各测试阶段的 muon.py 由 PR 前版本 + PR hunk 生成，不含此 hack，测试结束后原文件已恢复）。
- `PaddleFleet/src/paddlefleet/trainer/trainer.py`：`MUON_FSDP_STAGE1_CLIP_EPS` / `MUON_FSDP_EXPERT_NO_ACCUM_SCALE` 对齐诊断 hack（环境变量门控，测试中未开启）。
- `Paddle/cmake/cinn.cmake`：`cmake` → `${CMAKE_COMMAND}`，构建脚本修正，与运行时无关。

## 复现方式

- `perf_stages/make_stages.py`：把 PR / 本地 diff 按 hunk 拆分，生成各阶段累积版本到 `perf_stages/stages/S*/`
  （`python perf_stages/make_stages.py --list` 查看 hunk 编号与功能点映射）。
- `perf_stages/run_stages.sh [S...]`：逐阶段把文件装入 `Paddle/build`（S8 额外启用 fp8_utils 修改），
  跑 `run_fsdp_test.sh`，结果追加到 `perf_stages/results.log`，结束后自动恢复全部原始修改（备份在 `perf_stages/backup/`）。

## 附录：各功能点完整代码 diff

由 `perf_stages/stages/` 相邻阶段文件 `diff -U3` 生成（Sn-1 → Sn），即每个功能点在 Paddle 中实际新增/删除的代码。

### S1 · gather buffer empty + reduce-scatter AVG

```diff
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -270,10 +270,16 @@
             return self.data_buffer
         # Reuse tmp_buffer if exists, else create
         if self.tmp_data_buffer is None:
-            self.tmp_data_buffer = paddle.zeros(
-                shape=[self.total_buffer_size],
-                dtype=self.dtype if self.is_params else self.main_grad_dtype,
-            )
+            if self.is_params:
+                # Fully overwritten by all_gather.
+                self.tmp_data_buffer = paddle.empty(
+                    shape=[self.total_buffer_size], dtype=self.dtype
+                )
+            else:
+                # Grads accumulate into it.
+                self.tmp_data_buffer = paddle.zeros(
+                    shape=[self.total_buffer_size], dtype=self.main_grad_dtype
+                )
         return self.tmp_data_buffer
 
     def do_all_gather(self):
@@ -287,11 +293,17 @@
     def do_reduce_scatter(self):
         tmp_buffer = self.get_tmp_buffer()
         shard = tmp_buffer._slice(0, self.data_buffer.shape[0])
-        tmp_buffer.scale_(self.grad_scale)
+        # AVG folds the scale into NCCL when it is exactly 1/nranks of this
+        # group; otherwise (grad accumulation, EP groups) pre-scale and SUM.
+        if self.grad_div * self.grad_accum_steps == self.fsdp_degree:
+            op = paddle.distributed.ReduceOp.AVG
+        else:
+            tmp_buffer.scale_(self.grad_scale)
+            op = paddle.distributed.ReduceOp.SUM
         return paddle.distributed.reduce_scatter(
             shard,
             tmp_buffer,
-            op=paddle.distributed.ReduceOp.SUM,
+            op=op,
             group=self.fsdp_group,
             sync_op=False,
         )
```

### S2 · 不再强制 Muon 逐矩阵 NS

```diff
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -1182,12 +1182,6 @@
             self._shard_descs[params_buffer.data_buffer.name] = param_slice_info
         return result
 
-    def requires_muon_per_matrix_ns(self):
-        return any(
-            group.muon_shard_numel is not None
-            for group in self.buffer_manager.buffer_groups
-        )
-
     def bind_decay_param_fun(self, optimizer):
         """Answer ``apply_decay_param_fun`` for the fused buffers.
```

```diff
--- a/python/paddle/optimizer/muon.py
+++ b/python/paddle/optimizer/muon.py
@@ -321,8 +321,7 @@
             splitting the batch in half moves the result by 1.2e-2 relative).
             That makes a batched update depend on how many matrices a rank holds,
             which breaks sharding a group across ranks. Costs one kernel launch
-            per matrix. FSDP turns it on by itself for groups it shards on matrix
-            boundaries. Default: ``False``.
+            per matrix. Default: ``False``.
         multi_precision (bool): Maintain FP32 master weights when training in
             BF16/FP16. Default: ``False``.
         name (str | None): Optional name for the optimizer instance.
@@ -1089,7 +1088,6 @@
                 fsdp_context.bind_decay_param_fun(self)
             if hasattr(fsdp_context, "bind_optimizer"):
                 fsdp_context.bind_optimizer(self)
-            self._maybe_force_per_matrix_ns(fsdp_context)
             if self._grad_clip is not None:
                 self._grad_clip.should_comm_on_shard_dim = True
                 self._grad_clip.fsdp_group = (
@@ -1227,19 +1225,6 @@
 
     _FSDP_FUSED_PREFIX = "fuse_params_"
 
-    def _maybe_force_per_matrix_ns(self, fsdp_context):
-        if self._ns_per_matrix:
-            return
-        fun = getattr(fsdp_context, "requires_muon_per_matrix_ns", None)
-        if fun is None or not fun():
-            return
-        self._ns_per_matrix = True
-        _logger.warning(
-            "FSDP shards Muon params on matrix boundaries; forcing "
-            "ns_per_matrix=True so the update does not depend on how many "
-            "matrices a rank holds."
-        )
-
     def _fsdp_collect_params_grads(self, fsdp_context, params_grads):
         muon_entries = (
             fsdp_context.muon_params_grads()
```

### S3 · AdamW+Hyperball 分片更新

```diff
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -60,6 +60,7 @@
     is_tie: bool = False
     no_decay: bool = False
     use_muon: bool = False
+    hyperball_sharded: bool = False
     clip_attrs: tuple = ()
     muon_owner_rank: int = None
     muon_shard_numel: int = None
@@ -405,6 +406,12 @@
                 # A buffer must not mix params with and without weight decay.
                 bool(getattr(param, "no_weight_decay", False)),
                 bool(getattr(param, "use_muon", False)),
+                # Sharded-Hyperball params get their own buffer.
+                (
+                    param.name
+                    if getattr(param, "use_hyperball_sharded", False)
+                    else None
+                ),
                 # A buffer must not mix params that grad clip treats differently.
                 tuple(
                     (attr, getattr(param, attr, default))
@@ -429,6 +436,7 @@
                 _,
                 use_muon,
                 _,
+                _,
             ) = item[0]
             return (
                 0 if is_tie else (1 if not trainable else 2),
@@ -448,6 +456,7 @@
                 is_tie=is_tie,
                 no_decay=no_decay,
                 use_muon=use_muon,
+                hyperball_sharded=hyperball_name is not None,
                 clip_attrs=clip_attrs,
             )
             for (
@@ -459,6 +468,7 @@
                 is_tie,
                 no_decay,
                 use_muon,
+                hyperball_name,
                 clip_attrs,
             ), params in sorted(keyed_params.items(), key=sort_key)
         ]
@@ -509,6 +519,11 @@
                     shard_align_numel=muon_align,
                 )
             group.muon_shard_numel = group.params_buffer.shard_align_numel
+            if group.hyperball_sharded:
+                data_buffer = group.params_buffer.data_buffer
+                data_buffer._hyperball_group = (
+                    group.fsdp_group if group.params_buffer.is_sharded else None
+                )
             if group.muon_shard_numel is not None:
                 for param in params:
                     full = tuple(
@@ -1227,6 +1242,15 @@
     def bind_optimizer(self, optimizer):
         if getattr(optimizer, "_fsdp_state_dict_ctx", None) is self:
             return
+        info_map = getattr(optimizer, "_muon_param_info_map", None)
+        if info_map is not None:
+            from paddle.optimizer.muon import MuonParamInfo
+
+            for group in self.buffer_manager.buffer_groups:
+                if group.hyperball_sharded:
+                    info_map[group.params_buffer.data_buffer.name] = (
+                        MuonParamInfo(use_muon=False, use_hyperball=True)
+                    )
         optimizer.sharded_state_dict = lambda model_sharded_state_dict=None: (
             self.optimizer_sharded_state_dict(optimizer)
         )
```

```diff
--- a/python/paddle/distributed/fsdp/fully_shard.py
+++ b/python/paddle/distributed/fsdp/fully_shard.py
@@ -62,9 +62,19 @@
         info = muon_info_map.get(param.name)
         if info is None and not param.stop_gradient:
             missing.append(name)
+        hyperball = info is not None and getattr(info, "use_hyperball", False)
+        # AdamW+Hyperball runs on shards with all-reduced norms.
+        shape = getattr(param, "original_shape", None) or param.shape
+        param.use_hyperball_sharded = bool(
+            hyperball
+            and not info.use_muon
+            and getattr(info, "split_concat_func", None) is None
+            and len(shape) == 2
+        )
         param.use_muon = bool(
             info is not None
-            and (info.use_muon or getattr(info, "use_hyperball", False))
+            and (info.use_muon or hyperball)
+            and not param.use_hyperball_sharded
         )
     if missing:
         raise ValueError(
```

```diff
--- a/python/paddle/optimizer/muon.py
+++ b/python/paddle/optimizer/muon.py
@@ -704,7 +704,9 @@
         return orthogonal_update * scale * extra_scale_factor
 
     @staticmethod
-    def _hyperball_apply(w, update, lr, eps=1e-12, split_concat_func=None):
+    def _hyperball_apply(
+        w, update, lr, eps=1e-12, split_concat_func=None, reduce_group=None
+    ):
         r"""Hyperball step, shared by Muon+Hyperball and AdamW+Hyperball.
 
         Given the current weight ``w`` and a base-optimizer update direction
@@ -742,13 +744,23 @@
                 lambda u: Muon._hyperball_apply(next(w_iter), u, lr, eps),
             )
 
-        def _fro(x):
-            # Frobenius norm over the last two dims, kept for broadcasting.
-            # 2D [H, I] -> [1, 1] (whole matrix); 3D [E, H, I] -> [E, 1, 1].
-            # paddle.norm(p='fro') is numerically identical to
-            # sqrt(sum(x*x)) here but fused (no full-size x*x temp), so it is
-            # faster on large 2D / 3D-expert weights.
-            return paddle.norm(x, p='fro', axis=[-2, -1], keepdim=True)
+        if reduce_group is not None or w.ndim == 1:
+            # Flat shards of one matrix: all-reduce the squared sums.
+            def _fro(x):
+                sq = paddle.sum(x * x).reshape([1])
+                if reduce_group is not None:
+                    paddle.distributed.all_reduce(sq, group=reduce_group)
+                return paddle.sqrt(sq)
+
+        else:
+
+            def _fro(x):
+                # Frobenius norm over the last two dims, kept for broadcasting.
+                # 2D [H, I] -> [1, 1] (whole matrix); 3D [E, H, I] -> [E, 1, 1].
+                # paddle.norm(p='fro') is numerically identical to
+                # sqrt(sum(x*x)) here but fused (no full-size x*x temp), so it
+                # is faster on large 2D / 3D-expert weights.
+                return paddle.norm(x, p='fro', axis=[-2, -1], keepdim=True)
 
         w32 = w.astype(paddle.float32)
         u32 = update.astype(paddle.float32)
@@ -862,7 +874,11 @@
                 param_info.split_concat_func if param_info else None
             )
             new_w = Muon._hyperball_apply(
-                w, update, lr * lr_ratio, split_concat_func=split_concat_func
+                w,
+                update,
+                lr * lr_ratio,
+                split_concat_func=split_concat_func,
+                reduce_group=getattr(param, "_hyperball_group", None),
             )
             if find_master:
                 paddle.assign(new_w, w)
```

### S4 · Muon owner 大块优先

```diff
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -556,7 +556,11 @@
 
     def _assign_muon_owners(self):
         loads = {}
-        for group in self.buffer_groups:
+        # Largest first to balance owner loads.
+        for group in sorted(
+            self.buffer_groups,
+            key=lambda g: -g.params_buffer.total_buffer_size,
+        ):
             if not group.use_muon or group.grads_buffer is None:
                 continue
             if group.muon_shard_numel is not None:
```

### S5 · 预取 / 梯度队列调度重写

```diff
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -15,7 +15,7 @@
 import contextlib
 import math
 import re
-from collections import OrderedDict
+from collections import Counter, OrderedDict
 from dataclasses import dataclass, field
 from enum import Enum
 
@@ -586,30 +586,34 @@
         self,
         buffer_manager,
         enable_overlap=True,
-        double_buffer_limit=None,
+        prefetch_units=1,
     ):
         self.buffer_manager = buffer_manager
         self.enable_overlap = enable_overlap
         self.grad_reduce_queue = []
 
-        # for double buffer mechanism config
-        groups_per_unit = 2
-        self.double_buffer_limit = double_buffer_limit or (groups_per_unit * 2)
-        self.prefetch_units = max(
-            self.double_buffer_limit // groups_per_unit - 1, 0
+        # Size limits in whole units: a unit may own several sharded groups.
+        unit_sizes = Counter(
+            g.fsdp_unit_id
+            for g in buffer_manager.buffer_groups
+            if g.params_buffer.is_sharded and g.fsdp_unit_id is not None
         )
+        groups_per_unit = max(2, max(unit_sizes.values(), default=0))
+        self.prefetch_units = max(int(prefetch_units), 0)
+        self.gather_buffer_limit = groups_per_unit * (self.prefetch_units + 1)
+        self.grad_queue_limit = groups_per_unit * max(self.prefetch_units, 1)
         self.buffer_cnt_in_using = 0
         self.need_zero_grads = True
         self._last_backward_unit_id = None
 
     # ------------------------------------------------------------------
     # Params all_gather double buffer. Owns buffer_cnt_in_using /
-    # double_buffer_limit / prefetch_units and the BufferState machine.
+    # gather_buffer_limit / prefetch_units and the BufferState machine.
     # ------------------------------------------------------------------
 
     def _release_one_buffer_if_needed(self, keep=()):
         # Release a buffer with the READY status if needed
-        while self.buffer_cnt_in_using >= self.double_buffer_limit:
+        while self.buffer_cnt_in_using >= self.gather_buffer_limit:
             found = False
             for gid_idx, group in enumerate(self.buffer_manager.buffer_groups):
                 if gid_idx in keep:
@@ -676,6 +680,11 @@
                 self._flush_expert_grads_after_unit(unit_id)
                 self._release_params_after_unit(unit_id)
             self._bind_sharded_expert_main_grads(params)
+        elif self.grad_reduce_queue:
+            # Retire leftover grad reduce-scatters, one per forward layer.
+            self._wait_for_grad_comm(
+                queue_limit=len(self.grad_reduce_queue) - 1
+            )
 
         if self.enable_overlap:
             keep = set(req_gids)
@@ -687,19 +696,6 @@
                     self._release_one_buffer_if_needed(keep)
                     self._issue_async_gather(gid)
 
-            for next_gid in self._next_buffer_id(req_gids, is_backward):
-                next_params_buffer = self.buffer_manager.buffer_groups[
-                    next_gid
-                ].params_buffer
-                if not next_params_buffer.is_sharded:
-                    continue
-                if next_params_buffer.status != BufferState.FREED:
-                    continue
-                self._release_one_buffer_if_needed(keep)
-                if self.buffer_cnt_in_using >= self.double_buffer_limit:
-                    break
-                self._issue_async_gather(next_gid)
-
         for param in params:
             gid = self.buffer_manager.param_to_buffer_id[param.name]
             group = self.buffer_manager.buffer_groups[gid]
@@ -736,6 +732,22 @@
             tmp_param.get_tensor()._set_dims(param.shape)
             param.get_tensor()._share_data_with(tmp_param.get_tensor())
 
+        # Prefetch after waiting on this unit's gathers, or compute stalls on it.
+        if self.enable_overlap:
+            keep = set(req_gids)
+            for next_gid in self._next_buffer_id(req_gids, is_backward):
+                next_params_buffer = self.buffer_manager.buffer_groups[
+                    next_gid
+                ].params_buffer
+                if not next_params_buffer.is_sharded:
+                    continue
+                if next_params_buffer.status != BufferState.FREED:
+                    continue
+                self._release_one_buffer_if_needed(keep)
+                if self.buffer_cnt_in_using >= self.gather_buffer_limit:
+                    break
+                self._issue_async_gather(next_gid)
+
     def shard_params(self, params, is_backward=False):
         affected_gids = set()
         for param in params:
@@ -758,6 +770,12 @@
         self._last_backward_unit_id = None
         for group in self.buffer_manager.buffer_groups:
             params_buffer = group.params_buffer
+            # A prefetch issued at the end of backward gathers the pre-step
+            # shard; drop it so the next forward gathers the updated one.
+            if params_buffer.status == BufferState.SYNCING:
+                params_buffer.comm_task.wait()
+                params_buffer.comm_task = None
+                params_buffer.status = BufferState.READY
             if params_buffer.status in (BufferState.READY, BufferState.USING):
                 for param in group.params:
                     stop_gradient = param.stop_gradient
@@ -902,7 +920,9 @@
                 param.main_grad = None
                 param._main_grad_addr = None
 
-    def _wait_for_grad_comm(self, queue_limit=2):
+    def _wait_for_grad_comm(self, queue_limit=None):
+        if queue_limit is None:
+            queue_limit = self.grad_queue_limit
         while len(self.grad_reduce_queue) > queue_limit:
             grads_buffer = self.grad_reduce_queue.pop(0)
             if grads_buffer.comm_task is None:
@@ -1052,8 +1072,8 @@
         model,
         fsdp_unit_layers=None,
         enable_tensor_fusion_and_overlap=True,
-        double_buffer_limit=None,
         mp_policy=None,
+        prefetch_units=1,
     ):
         self.model = model
         self.mp_policy = mp_policy
@@ -1065,7 +1085,7 @@
         self.comm_manager = FSDPCommManager(
             self.buffer_manager,
             enable_overlap=enable_tensor_fusion_and_overlap,
-            double_buffer_limit=double_buffer_limit,
+            prefetch_units=prefetch_units,
         )
         self.register_tensor_fusion_hooks(self.model)
         register_fsdp_context(self)
@@ -1530,6 +1550,8 @@
         for param in self.model.parameters():
             if param.trainable:
                 param.main_grad = None
+        # Sync so comm-stream buffers freed in backward return to the pool before
+        paddle.device.synchronize()
 
     @paddle.autograd.no_grad()
     def _p2p_gather_full(self, shard, fsdp_group, owner_local):
```

### S6 · 集合通信 side stream

```diff
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -85,6 +85,30 @@
     return granularity
 
 
+_comm_streams = {}  # group id -> side stream for the async collectives
+
+
+class _StreamTask:
+    def __init__(self, stream):
+        self._event = stream.record_event()
+
+    def wait(self):
+        paddle.device.current_stream().wait_event(self._event)
+
+
+def _async_collective(group, tensors, collective):
+    stream = _comm_streams.get(group.id)
+    if stream is None:
+        stream = _comm_streams[group.id] = paddle.device.Stream()
+    stream.wait_stream(paddle.device.current_stream())
+    with paddle.device.stream_guard(stream):
+        # Keep the buffers from being reused before the side stream is done.
+        for t in tensors:
+            t._record_stream()
+        collective()
+    return _StreamTask(stream)
+
+
 class TensorFusionBuffer:
     class _CompletedTask:
         # Placeholder for the broadcast (sync) path where callers still .wait().
@@ -284,11 +308,17 @@
         return self.tmp_data_buffer
 
     def do_all_gather(self):
-        return paddle.distributed.all_gather(
-            self.get_tmp_buffer(),
-            self.data_buffer,
-            group=self.fsdp_group,
-            sync_op=False,
+        tmp_buffer = self.get_tmp_buffer()
+        return _async_collective(
+            self.fsdp_group,
+            (tmp_buffer, self.data_buffer),
+            lambda: paddle.distributed.stream.all_gather(
+                tmp_buffer,
+                self.data_buffer,
+                group=self.fsdp_group,
+                sync_op=True,  # on the side stream; the host does not block
+                use_calc_stream=True,
+            ),
         )
 
     def do_reduce_scatter(self):
@@ -301,12 +331,17 @@
         else:
             tmp_buffer.scale_(self.grad_scale)
             op = paddle.distributed.ReduceOp.SUM
-        return paddle.distributed.reduce_scatter(
-            shard,
-            tmp_buffer,
-            op=op,
-            group=self.fsdp_group,
-            sync_op=False,
+        return _async_collective(
+            self.fsdp_group,
+            (tmp_buffer,),
+            lambda: paddle.distributed.stream.reduce_scatter(
+                shard,
+                tmp_buffer,
+                op=op,
+                group=self.fsdp_group,
+                sync_op=True,  # on the side stream; the host does not block
+                use_calc_stream=True,
+            ),
         )
 
     def accumulate_reduced_grad(self):
```

### S7 · prefetch_units 1→2

```diff
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -1108,7 +1108,7 @@
         fsdp_unit_layers=None,
         enable_tensor_fusion_and_overlap=True,
         mp_policy=None,
-        prefetch_units=1,
+        prefetch_units=2,
     ):
         self.model = model
         self.mp_policy = mp_policy
```

### S8 · DeepGEMM set_num_sms(120)（PaddleFleet）

```diff
diff --git a/src/paddlefleet/transformer/moe/fp8_utils.py b/src/paddlefleet/transformer/moe/fp8_utils.py
index 86f454e7..670ee821 100644
--- a/src/paddlefleet/transformer/moe/fp8_utils.py
+++ b/src/paddlefleet/transformer/moe/fp8_utils.py
@@ -2895,7 +2895,7 @@ class ExpertsGroupGemmContiguousNode:
                         tokens_per_expert_tensor,
                         weight_grad,
                     )
-                    deep_gemm.set_num_sms(0)
+                    deep_gemm.set_num_sms(120)
                 else:
                     deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                         x,
```

# 第二部分：fsdp_opt_pr.txt 各 PR 功能点拆解（T0–T6）

## 测试对象与方法

| PR | 功能点 | 阶段 |
| --- | --- | --- |
| [#79881](https://github.com/PaddlePaddle/Paddle/pull/79881) Optimize FSDP Performance | 已在第一部分 S1–S5 拆解，本部分作为公共底座，不再单独测 | — |
| [#79898](https://github.com/PaddlePaddle/Paddle/pull/79898) Optimize FSDP Comm Stream | 集合通信放到每个通信组一条 side stream | T1 |
| [#79901](https://github.com/PaddlePaddle/Paddle/pull/79901) Support prefetch by size | `comm_unit_size`：按元素数控制预取深度和梯度队列 | T2 |
| [#79897](https://github.com/PaddlePaddle/Paddle/pull/79897) + [Fleet #2371](https://github.com/PaddlePaddle/PaddleFleet/pull/2371) fp8 param gather | 专家权重在 shard 上量化，all_gather fp8 数据和 scale | T3 |
| [#79899](https://github.com/PaddlePaddle/Paddle/pull/79899) reduce_scatter dtype | `MixedPrecisionPolicy.grad_comm_dtype`：梯度 reduce-scatter 用 bf16 | T4 |
| [#79887](https://github.com/PaddlePaddle/Paddle/pull/79887) reshard_policy / reshard_filter | 参数 gather 后保留到 backward 或整个 step 结束 | T5a–T5d |
| [#79903](https://github.com/PaddlePaddle/Paddle/pull/79903) sync_grads_each_microbatch | 梯度只在最后一个 micro-batch 做 reduce-scatter | T6 |

测试对象与第一部分相同：ERNIELite T4（h1280 / L20，34.93B-A1.232B，MoE 768 专家 top12，EP8），2 机 16 卡，`sharding: fsdp`，seq 8192，MuonH + AdamW + AdamH，FP8 MoE。

方法与第一部分的区别：

- **基于最终代码按开关累积。** 这几个 PR 前后依赖（#79897 的 fp8 gather 建在 #79898 的 `run_on_stream` 上，#79887 的 `keep_gathered` 改的是 #79901 / #79897 加的 gather 计数逻辑，#79903 的 diff 基于 #79897 之后的 `fully_shard_fusion.py`），无法像第一部分那样逐个回退到 PR 前。
  因此用最终代码（Paddle 本地 HEAD + #79887 + #79903，PaddleFleet 含 #2371），加一组只用于测试的开关（`perf_stages2/knobs.json`），T0 全部关掉，之后每阶段多打开一项。
  关掉某项时走的是 PR 前的等价路径（见下表），其余代码完全相同。
- **脚本：** `run_fsdp_test_v2.sh` 的副本 `perf_stages2/run_fsdp_test_v2_stage.sh`，只多拼了一行 `--kwargs`（`max_steps=20`，`internal_medicine_monitor_interval=0`）。
- **统计区间：** 指标取 rank0 `workerlog.0` 中 step 4–20（n=17）的 `tokens_per_sec_per_card` 均值 ± 标准差；fwd-bwd / opt-step 取 trainer Profile 同区间均值；max_mem 为区间内最大值。
- **mb2 / ga24：** yaml 当前是 mb3 / ga16，但在 mb3 下各阶段都会在 step 6–12 随机挂掉：
  - VMM allocator 报 `(AlreadyExists) Duplicate underlying allocation base ... in VMM V2 registry`，大量 `Recovered ... after memory defragmentation`。
  - T1 出现过 `CUDA error 700 illegal memory access`。
  - max_mem_alloc 约 73.3 GB，贴着上限；吞吐逐步掉（15.8k → 9.4k）。

  所以所有阶段统一用 `EXTRA_KWARGS="per_device_train_batch_size=2 gradient_accumulation_steps=24"` 覆盖（global batch 不变），max_mem_alloc 约 60 GB，全部稳定。yaml 本身没改。
- **yaml 与第一部分不同：** `using_cache_weight_quant_transpose: false`、`fused_linear_ce_loss_chunk: 24`、`recompute` 新增 `mlp: [0..5]`，所以 T 系列的绝对值比 S 系列低 3–4%，两部分之间不要直接比。

| 开关（knobs.json） | 对应 PR | 关闭（T0） | 打开 |
| --- | --- | --- | --- |
| `legacy_comm` | #79898 | `true`：`paddle.distributed.all_gather/reduce_scatter(sync_op=False)`，走 ProcessGroup 内部通信流 | `false`：`run_on_stream` + `paddle.distributed.stream.*(use_calc_stream=True)` |
| `comm_unit_size` | #79901 | 不设，按层预取（`prefetch_units=1`） | `703647744`（= 3 个 unit 的元素数） |
| `no_fp8_gather` | #79897 / #2371 | `true`：清掉 `_fsdp_fp8_quantizer`，专家权重 bf16 gather，Fleet `FP8QuantWeightCallback` 每步逐层 gather 后预量化 | `false`：FSDP 在 shard 上量化、gather fp8 |
| `grad_comm_dtype` | #79899 | 不设，reduce-scatter 用 fp32 main grad | `"bfloat16"` |
| `reshard_policy` / `reshard_filter` | #79887 | `after_forward` / `all` | `after_backward` 或 `after_step`，filter `all` / `dense` / `expert` |
| `sync_grads_each_microbatch` | #79903 | `true`：每个 micro-batch 都 reduce-scatter | `false`：只在最后一个 micro-batch 做 |

## 第二部分结果汇总

T0–T4 为累积叠加（Tn = Tn-1 + 本功能点），T3 起都建在 T2（`comm_unit_size`）上。
T5a–T5d 是 T4 上分别换一种 reshard 策略的并列方案，不再互相叠加。
统计口径为 mb2 / ga24、step 4–20（n=17）。各阶段 step 1、2 的 loss 完全相同（12.24116135 / 12.24004745），step 20 的 loss 一致到 1e-5 量级，未见精度回退。

| 阶段 | 功能点 | 来源 | tokens/s/card | 相对上一阶段 | 相对 baseline | fwd-bwd (ms) | opt-step (ms) | max_mem_alloc (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | baseline（全部开关关闭） | — | 14525.4 ± 33（剔除 2 步异常，见下） | — | — | 26503 | 364 | 61.46 |
| T1 | 集合通信放 side stream | #79898 | 17618.4 ± 51 | **+21.3%** | +21.3% | 21733 | 361 | 61.46 |
| T2 | `comm_unit_size=703647744` | #79901 | 20262.2 ± 57 | **+15.0%** | +39.5% | 18825 | 362 | 62.20 |
| （T2ref） | 对照：`prefetch_units=2` | — | 20322.3 ± 68 | +15.3%（相对 T1） | +39.9% | 18775 | 360 | 61.90 |
| T3 | 专家权重 fp8 param gather | #79897 + Fleet #2371 | 20986.2 ± 66 | +3.6% | +44.5% | 18300 | 351 | 59.38 |
| T4 | 梯度 reduce-scatter 用 bf16 | #79899 | 21372.4 ± 87 | +1.8% | +47.1% | 17958 | 348 | 59.38 |
| T5a | T4 + `after_backward` / `all` | #79887 | 21471.9 ± 66 | +0.5%（相对 T4） | +47.8% | 17892 | 348 | 60.29 |
| T5b | T4 + `after_step` / `dense` | #79887 | 21585.2 ± 66 | +1.0%（相对 T4） | +48.6% | 17794 | 350 | 60.29 |
| T5c | T4 + `after_step` / `expert` | #79887 | 21610.7 ± 61 | +1.1%（相对 T4） | +48.8% | 17762 | 347 | 59.81 |
| T5d | T4 + `after_step` / `all` | #79887 | 21735.6 ± 79 | **+1.7%（相对 T4）** | +49.6% | 17656 | 350 | 60.29 |
| T6 | T4 + `sync_grads_each_microbatch=False` | #79903 | mb2 / ga24 下 step 2 OOM | — | — | — | — | 66.5（step 1） |

T0 的 17 步里有两步异常掉速（step 10：12150.4，step 13：8336.2），全部 17 步的均值是 14021.6 ± 1574，中位数是 14509.7。
其余阶段都没有这种波动（标准差 ≤ 87）。表中 T0 用剔除这两步后的均值（n=15）作为 baseline，它和中位数只差 0.1%。
T0 的 fwd-bwd 26503 ms 含这两步，偏高。

T6 在 mb2 / ga24 下 step 1 的 max_mem_alloc 已到 66.5 GB，step 2 报 `Out of memory error on GPU 0. Cannot allocate 222.187500MB`（cuMemCreate 失败）。
原因是每个 unit 的完整 fp32 grad buffer 在整个 step 内常驻。所以 T6 改在 mb1 / ga48（global batch 不变）下和 T4 成对重测：

| 阶段 | 配置 | tokens/s/card | 相对 T4ref | fwd-bwd (ms) | opt-step (ms) | max_mem_alloc (GB) | loss@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T4ref_mb1ga48 | T4，mb1 / ga48 | 15148.1 ± 38 | — | 25516 | 347 | 47.48 | 12.050840 |
| T6_mb1ga48 | T4 + 只在最后一个 micro-batch 同步梯度 | 17114.2 ± 79 | **+13.0%** | 22525 | 353 | 63.73 | 12.050831 |

结论：

- 累积链 T0 → T4 总收益 **14525 → 21372 tokens/s/card，+47.1%**；再加 T5d（`after_step` / `all`）到 21736，+49.6%。
- 收益最大的两项和第一部分一致，都是通信与计算的 overlap：
  - T1 side stream：+21.3%，fwd-bwd 26.5s → 21.7s。
  - T2 按元素数预取：+15.0%。`comm_unit_size=703647744` 是 3 个 unit 的元素数，减去当前 unit 后的预取窗口约 351.8M 元素，实际预取深度是 2 个 unit，和 `prefetch_units=2`（T2ref）等价，两者差 0.3%，在噪声内。
    `comm_unit_size` 的价值在于按大小而不是层数控制，层大小不均匀时更稳。
- T3 fp8 param gather：+3.6%，同时 max_mem_alloc 降 2.8 GB（62.20 → 59.38）。专家权重 gather 量减半，Fleet `FP8QuantWeightCallback` 也不再每步逐层 gather bf16 权重再量化；opt-step 同时从 362 降到 351 ms。
- T4 bf16 reduce-scatter：+1.8%。本配置 `amp_master_grad: true`，main grad 是 fp32，改成 bf16 后梯度通信量减半。显存不变。
- T5 reshard 策略的收益都不大，`after_step` / `all` 最好，相对 T4 +1.7%，多占 0.9 GB。
  收益小的原因是 fp8 buffer 不计入 gather 预算，在 `after_forward` 下 fp8 专家 buffer 已经从 forward 保留到 backward，T5 只能再省掉 dense 参数和 backward 之后的那部分 gather。
  T5c（只保留 expert）的显存最低（59.81 GB），收益和 T5b 相同。
- T6 在显存放得下时收益明显：mb1 / ga48 下 +13.0%（fwd-bwd 25.5s → 22.5s），代价是 max_mem_alloc +16.3 GB（47.48 → 63.73）。
  但 mb1 / ga48 + T6 的 17114 仍低于 mb2 / ga24 + T4 的 21372。在本配置下，T6 省下的显存不如拿去加大 micro-batch 划算，不建议开启。
  micro-batch 已经受显存限制、而 ga 很大时，T6 才值得考虑。
- 每阶段 1 次运行，17 步；除 T0 外标准差约 0.3–0.4%，±0.5% 以内的差异（如 T5a）不应视为确定结论。

## 第二部分功能点详解

### T1 · 集合通信放到 side stream（PR #79898）

`fully_shard_fusion.py` `StreamTask` / `run_on_stream` / `FSDPBufferManager._comm_stream` / `do_all_gather` / `do_reduce_scatter`

与第一部分 S6 的本地修改思路相同，PR 版本的实现差异：

- side stream 由 `FSDPBufferManager` 按通信组 id 创建，构造 `TensorFusionBuffer` 时传入 `comm_stream`；`nranks <= 1` 的组不建流。
- `_record_stream()` 放在集合通信之后调用。
- `all_gather_params` 里的同步 gather（buffer 处于 FREED 时）也改走 `do_all_gather().wait()`，不再直接调 `process_group.all_gather`。

```python
class StreamTask:
    def __init__(self, stream):
        self._event = stream.record_event()

    def wait(self):
        paddle.device.current_stream().wait_event(self._event)


def run_on_stream(stream, tensors, collective):
    stream.wait_stream(paddle.device.current_stream())
    with paddle.device.stream_guard(stream):
        collective()
        for t in tensors:
            t._record_stream()
    return StreamTask(stream)

    def do_all_gather(self):
        full, shard = self.get_tmp_buffer(), self.data_buffer

        def gather():
            paddle.distributed.stream.all_gather(
                full, shard, group=self.fsdp_group, sync_op=True, use_calc_stream=True)

        return run_on_stream(self.comm_stream, (full, shard), gather)
```

### T2 · 按元素数预取（PR #79901）

`fully_shard.py` `fully_shard(prefetch_units=..., comm_unit_size=...)`；`fully_shard_fusion.py` `FSDPCommManager._init_prefetch_by_size` / `_extend_by_prefetch_size` / `_prefetch_plan` / `_release_ready_buffers` / `_wait_for_grad_comm`

原来的 `prefetch_units` 按 unit 个数限制预取和梯度队列。设置 `comm_unit_size` 后改为按元素数：

- **参数 gather：** 从当前请求的 buffer 往执行方向（backward 时反向）走，以"同一 unit 的同类（dense / expert）sharded buffer"为整体加入预取，直到累计元素数达到 `comm_unit_size // 2`。backward 跳过冻结参数。
- **显存上限：** 已 gather 的元素数（`gathered_numel`）超过 `max(comm_unit_size, 本次请求 + 预取目标)` 时，按顺序释放 READY 状态、且不在本次 keep 集合里的 buffer。
- **梯度队列：** 在途 reduce-scatter 的元素数（含即将发出的一个）不超过 `comm_unit_size`，超出时从队头开始 wait。
- 预取计划按 `(req_gids, is_backward)` 缓存，hook 在 host 关键路径上，不每次重算。

本模型每个 unit 有一个 dense buffer（22,212,608 元素）和一个 expert buffer（212,336,640 元素），共 234,549,248。
测试值 `comm_unit_size = 703,647,744` 正好是 3 个 unit，预取窗口 351,823,872：加入下一个 unit 后累计 234.5M 仍不足窗口，会继续加入第二个，
所以实际预取深度是 2 个 unit，与 `prefetch_units=2` 相同。T2ref（`prefetch_units=2`）就是用来对照这一点的。

```python
        unit_size = max(int(unit_size), 0)
        self.comm_unit_size = unit_size
        self.grad_queue_capacity = unit_size
        self.gather_prefetch_size = unit_size // 2
...
            if prefetched >= prefetch_size:
                break
            for gid in self._gid_to_bucket_group[candidate]:
                ...
                selected.add(gid)
                targets.append(gid)
                prefetched += params_buffer.total_buffer_size
...
    def _wait_for_grad_comm(self, queue_limit=None, next_numel=0):
        if queue_limit is None and self.prefetch_by_size:
            while (self.grad_reduce_queue
                   and self._grad_queue_numel + next_numel > self.grad_queue_capacity):
                self._retire_grad_comm()
            return
```

### T3 · 专家权重 fp8 param gather（PR #79897 + PaddleFleet #2371）

Paddle：新增 `fsdp/_fp8_utils.py`；`fully_shard_fusion.py` `FSDPBufferManager` / `TensorFusionBuffer.do_all_gather` / `FSDPCommManager.all_gather_params`。
PaddleFleet：`fp8_utils.mark_fsdp_fp8_params`、`FP8QuantWeightCallback.on_init_end` / `on_step_begin`。

PR 前的流程：专家权重以 bf16 分片存放，Fleet 的 `FP8QuantWeightCallback` 在每步开始时逐层 gather bf16 权重、量化成 fp8 缓存（`_fsdp_fp8_quant_weight`），forward 里 FSDP 再 gather 一次 bf16。

PR 后的流程：

1. Fleet 在 `on_init_end`（`fully_shard` 之前）给 `GroupedMLPExpert.weight1/weight2` 挂上 `_fsdp_fp8_quantizer`（128×128 block 量化，`offline_quant_expert_weight` 开启时生效）。
2. FSDP 按 quantizer 分组建 buffer，shard 对齐到 `lcm(128², rows*cols)`，每个 rank 的 shard 都是整数个专家矩阵，可以在本地独立量化。
   这类 buffer 的 main weight 直接用 fp32（optimizer 开了 multi_precision 时），不再另存 bf16 shard。
3. 每步第一次 gather 前 `quantize_fp8_shard` 量化本地 shard，之后 all_gather 的是 fp8 数据（1 B/元素）和 fp32 scale（每 128×128 block 一个），bf16 是 2 B/元素。
4. gather 完 `bind_fp8_param` 把 `fp8_weight_stacked` / `fp8_scale_stacked` 直接指向 gather 出的 buffer，param 本身是零字节 storage。
5. 检测到 quantizer 后，Fleet callback 跳过每步的 `_fsdp_fp8_quant_weight`。

本 yaml `using_cache_weight_quant_transpose: false`，只 gather 非转置一份；为 true 时 data 和 transpose 两份都要 gather，通信量与 bf16 相同。

```python
# PaddleFleet fp8_utils.py
def mark_fsdp_fp8_params(args, model):
    ...
    def quantizer(weights, transpose):
        return fused_stack_quant_without_cache(
            [weights[i] for i in range(weights.shape[0])], transpose=transpose)

    quantizer.block = 128
    keep_transpose = getattr(args, "using_cache_weight_quant_transpose", True)
    quantizer.attrs = (
        ("fp8_weight_stacked", "fp8_scale_stacked", False),
        ("fp8_weight_stacked_transpose", "fp8_scale_stacked_transpose", True),
    )[: 2 if keep_transpose else 1]
    for layer in model.sublayers():
        if isinstance(layer, GroupedMLPExpert) and not isinstance(layer, SonicMoEExpert):
            layer.weight1._fsdp_fp8_quantizer = quantizer
            layer.weight2._fsdp_fp8_quantizer = quantizer

# Paddle fully_shard_fusion.py
    def do_all_gather(self):
        if self.fp8 is not None:
            pairs = get_fp8_tmp_buffer(self)   # 先 quantize_fp8_shard，再分配 fp8 / scale 的 full buffer
        else:
            pairs = [(self.get_tmp_buffer(), self.data_buffer)]

        def gather():
            for full, shard in pairs:
                paddle.distributed.stream.all_gather(
                    full, shard, group=self.fsdp_group, sync_op=True, use_calc_stream=True)
        ...
```

fp8 buffer 不计入 gather 预算（`_count_gathered` / `_free_params_buffer` / `_release_ready_buffers` 都跳过 `fp8 is not None`）。
forward 结束后 `shard_params` 只把 buffer 置为 READY，真正释放要靠预算或 backward 的 `_release_params_after_unit`，
所以在 `after_forward` 下，fp8 专家 buffer 实际是从 forward gather 一直留到本 unit 的 backward 结束，backward 不再 gather。
效果上专家参数已经是 `after_backward`，代价是 forward 期间整模型的 fp8 专家权重都在显存里（每 rank 20 × 212M × 1 B ≈ 4.2 GB）。
这也是 T5 各变体收益偏小的原因，见下文。

#### T3 修改前后对比（教学版）

一句话：修改前 FSDP 通信 bf16 专家权重、另外再做一遍 fp8 量化；修改后每张卡在自己的 shard 上先量化，FSDP 直接通信 fp8。通信量和多余的 gather 都省掉了，数值结果不变。

**背景。** 本模型的 MoE 专家 GEMM 用 fp8（`offline_quant_expert_weight: true`），计算前专家权重要按 128×128 block 量化成 fp8 数据加每 block 一个 fp32 scale，挂在参数的 `fp8_weight_stacked` / `fp8_scale_stacked` 属性上，GEMM 读的就是它们。
FSDP 下每张卡只存参数的一个 shard，用之前要 all_gather 成完整权重，所以要回答两个问题：gather 什么（bf16 还是 fp8），以及谁来量化。
每个 unit 有 dense buffer 22,212,608 元素、expert buffer 212,336,640 元素，专家权重约占 90%，它的处理方式基本决定了 FSDP 的通信量。

##### 修改前

1. **step 开始时 Fleet callback 预量化。** `FP8QuantWeightCallback.on_step_begin` 调 `_fsdp_fp8_quant_weight`（`PaddleFleet/src/paddlefleet/trainer/trainer_callback.py`）：

   ```python
   for layer in model.sublayers(include_self=True):
       if not isinstance(layer, TransformerLayer):
           continue
       with fsdp_context.unshard_params(_fp8_quant_target_params(layer)):   # all_gather bf16
           layer.fp8_quant_weight(batch_mode=batch_mode, quant_transpose=quant_transpose)
   ```

   逐层把 bf16 专家权重 all_gather 成完整权重、量化成 fp8 缓存，再释放 bf16。fp8 缓存是参数对象上的属性，reshard 后仍在，forward / recompute / dgrad 都复用它。
   代价是每步多一轮完整的 bf16 专家权重 gather，加 20 层的量化计算。
2. **每个 micro-batch 的 forward / backward，FSDP 照常 gather bf16。** FSDP 只知道"这个 unit 的 buffer 要 unshard"，不知道专家 GEMM 实际读的是 fp8 缓存，所以每次进入 unit 都 all_gather 整个 bf16 buffer（2 B/元素）。
   `after_forward` 下 ga=24 时每层每步 gather 48 次（forward、backward 各一次），而其中专家部分 gather 回来的 bf16 GEMM 根本不读。
3. **存储。** 每张卡存专家权重的 bf16 shard 加 fp32 master weight，另外还有完整的 fp8 缓存。

问题：通信量大（专家部分按 2 B/元素传，且没人用）；callback 每步多一轮 gather，只为了量化；bf16 shard 和 fp32 master 是两份冗余。

##### 修改后

思路是让 FSDP 知道"这组参数用的时候是 fp8"，把量化挪到 gather 之前、在 shard 上做，通信直接传 fp8。

1. **Fleet 给专家权重挂 quantizer。** `mark_fsdp_fp8_params` 在 `on_init_end`（`fully_shard` 建 buffer 之前）给 `GroupedMLPExpert.weight1/weight2` 挂 `_fsdp_fp8_quantizer`。
   quantizer 是一个函数，带 `block = 128`（量化 block 大小）和 `attrs`（量化结果写到参数的哪些属性；`using_cache_weight_quant_transpose: false` 时只有非转置一组），调用 `quantizer(weights, transpose)` 返回 `(fp8 data, scale)`。
   Paddle 不依赖 Fleet 的量化算子，Fleet 只通过"参数上挂一个函数"这个约定告诉 FSDP 怎么量化。
2. **FSDP 按 quantizer 分组，并对齐 shard 边界。** `FSDPBufferManager` 分组时把 `_fsdp_fp8_quantizer` 放进分组 key，带 quantizer 的专家权重单独成一个 buffer。shard 边界按 `fp8_shard_numel`（`_fp8_utils.py`）对齐：

   ```python
   numel = block**2
   for p in params:
       numel = math.lcm(numel, shape[1] * shape[2])   # 128² 与单个专家矩阵大小的最小公倍数
   ```

   这样每个 rank 的 shard 恰好是整数个专家矩阵，且矩阵边界与 128×128 block 对齐，每张卡只看本地 shard 就能算出正确的量化结果，不需要跨卡数据。
   不对齐的话，一个 block 可能跨两张卡，本地量化出的 scale 就是错的。
   这类 buffer 的 main weight 直接是 fp32（optimizer 开了 `multi_precision` 时），不再另存 bf16 shard。
3. **每步第一次 gather 前量化本地 shard。** `do_all_gather` 遇到 fp8 buffer 时调 `get_fp8_tmp_buffer`，先执行 `quantize_fp8_shard`：

   ```python
   if fp8.quantized:          # 同一步内只量化一次
       return
   for param in fp8.params:
       lo, hi, _ = params_buffer.param_shard(param)          # 本卡 shard 里属于这个 param 的区间
       weights = view_slice(data_buffer, lo, hi)              # fp32 main shard
       weights = weights.astype(param.dtype)                  # 先转 bf16，与修改前的量化输入一致
       data, scale = quantizer(weights, transpose)
       paddle.assign(data.flatten(), fp8.shards[i]._slice(lo, hi))
       paddle.assign(scale.flatten(), fp8.shards[k + i]._slice(lo // bn, hi // bn))
   fp8.quantized = True
   ```

   量化输入先 `astype` 成 bf16，与修改前"gather bf16 再量化"看到的数值相同；加上第 2 步保证 block 不跨卡，量化结果与修改前逐位一致。
   `quantized` 标志保证一步内只量化一次，之后每个 micro-batch 的 gather 都复用这份 fp8 shard。
4. **all_gather 传 fp8 和 scale。**

   ```python
   pairs = get_fp8_tmp_buffer(self)       # [(fp8 full, fp8 shard), (scale full, scale shard)]
   for full, shard in pairs:
       paddle.distributed.stream.all_gather(full, shard, group=self.fsdp_group,
                                            sync_op=True, use_calc_stream=True)
   ```

   每层专家部分：bf16 是 212.3M × 2 B ≈ 425 MB，fp8 是 212.3M × 1 B ≈ 212 MB，scale 是 212.3M / 16384 block × 4 B ≈ 52 KB，可忽略。专家部分通信量减半，dense 部分（44 MB）不变。
5. **参数直接指向 gather 出的 buffer。** `bind_fp8_param` 把 fp8 属性设成 buffer 切片的 view（零拷贝），再把参数本身变成零字节 storage，只保留 shape 供下游做形状检查：

   ```python
   setattr(param, "fp8_weight_stacked", view(data, [num * rows, cols]))
   setattr(param, "fp8_scale_stacked",  view(scale, [num * rows // block, cols // block]))
   param._clear_to_zero_allocation()
   param.get_tensor()._set_dims(dims)
   ```

6. **Fleet callback 不再预量化。** callback 检测到参数上有 quantizer 就跳过 `_fsdp_fp8_quant_weight`（`ernie5/src/callbacks/fp8_quant_weight_callback.py` `on_step_begin`），修改前每步那一轮 bf16 gather 和逐层量化没有了。

##### fp8 buffer 的生命周期

fp8 buffer 不计入 gather 预算，forward 结束时 `shard_params` 只把它标为 READY，真正释放在本 unit 的 backward 之后（`_release_params_after_unit`）。
所以即使 `reshard_policy` 是默认的 `after_forward`，专家 fp8 权重实际也是从 forward 留到 backward，backward 不再 gather，效果上已经是 `after_backward`。
代价是 forward 结束时所有层的 fp8 专家权重都在显存里（每 rank 约 20 × 212M × 1 B ≈ 4.2 GB）。这也是 T5 收益小的原因：专家部分最大的收益已经被拿走，`after_backward` / `after_step` 只能再省 dense 参数和跨 micro-batch 的 gather。

##### 前后对比

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 专家权重 shard 的存储 | bf16 shard + fp32 master | 只有 fp32 master |
| 谁量化、在哪量化 | Fleet callback，每步逐层 gather 完整 bf16 后量化 | FSDP 在本地 shard 上量化，每步一次 |
| forward / backward 里 gather 的内容 | bf16（2 B/元素），GEMM 不读 | fp8 + scale（约 1 B/元素），GEMM 直接读 |
| 每步额外的 gather | callback 一轮完整 bf16 gather | 无 |
| gather 后的 param | 完整 bf16 tensor | 零字节 storage，fp8 属性指向 gather buffer |
| 数值 | — | 逐位一致（量化输入相同，block 不跨卡） |

实测（`fsdp_perf_opt_reorder_t2_last.md`，在 T1 上只打开这一项）：tokens/s/card 17597.6 → 20970.0（+19.2%），fwd-bwd 21767 → 18327 ms，opt-step 359 → 351 ms，max_mem_alloc 61.46 → 59.03 GB，step 1 loss 都是 12.24116135，step 20 loss 差 < 1e-5。
本文累积链里 T3 只记 +3.6%，是因为它排在 T2（`comm_unit_size`）之后，二者覆盖的是同一段暴露通信，收益先记到了 T2 上。

##### 要点

1. 按数据实际怎么被用来决定通信什么：专家 GEMM 读 fp8，就不该传 bf16。
2. 分布式下本地量化的前提是 block 不跨 shard，`lcm(128², rows*cols)` 对齐是正确性的关键。
3. 跨库协作用约定而不是硬依赖：Fleet 在参数上挂 quantizer，Paddle FSDP 只认这个约定，量化算子留在 Fleet。

### T4 · 梯度 reduce-scatter 用 bf16（PR #79899）

`fully_shard.py` `MixedPrecisionPolicy.grad_comm_dtype`；`fully_shard_fusion.py` `TensorFusionBuffer.do_reduce_scatter` / `accumulate_reduced_grad`

本配置 `amp_master_grad: true`，trainer 不传 `mp_policy`，FSDP main grad 是 fp32，reduce-scatter 也是 fp32。
`grad_comm_dtype=bfloat16` 时，先把 fp32 的完整梯度 cast 成 bf16 放到 `_rs_comm_buffer`，立即释放 fp32 临时 buffer，用 bf16 做 reduce-scatter（通信量减半），
收到的 shard 再 `astype` 回 fp32 累加到 main grad。梯度累加仍是 fp32，只有跨 rank 求和这一步是 bf16。
`grad_comm_dtype` 与 main grad dtype 相同时视为 None，不做 cast。另外 reduce-scatter 发出后立即 `_release_reduced_grad_views`。

```python
        self.grad_comm_dtype = (
            grad_comm_dtype if grad_comm_dtype != self.main_grad_dtype else None)
...
        if self.grad_comm_dtype is None:
            self._rs_comm_buffer = tmp_buffer
        else:
            self._rs_comm_buffer = paddle.cast(tmp_buffer, self.grad_comm_dtype)
            self.tmp_data_buffer = None
        full = self._rs_comm_buffer
        shard = full._slice(0, self.data_buffer.shape[0])
...
    def accumulate_reduced_grad(self):
        shard = self._rs_comm_buffer._slice(0, self.data_buffer.shape[0])
        self._rs_comm_buffer = None
        if shard.dtype != self.data_buffer.dtype:
            shard = shard.astype(self.data_buffer.dtype)
        self.data_buffer.add_(shard)
        self.clear_tmp_buffer()
```

#### T4 修改前后对比（教学版）

一句话：梯度累加和 main grad 仍是 fp32，只把跨 rank 求和那一次通信改成 bf16。每个 micro-batch 的梯度通信量减半，在途梯度占的显存也减半，代价是跨 rank 求和的精度从 fp32 降到 bf16。

**背景。** 本配置 `amp_master_grad: true`，FSDP 的 main grad 是 fp32（`TensorFusionBuffer.main_grad_dtype`）。
每个 unit 的 grads buffer 有两块：分片的 `data_buffer`（每 rank 只存 1/N，跨 micro-batch 累加的 main grad），和完整的 `tmp_data_buffer`（本 micro-batch backward 写入的完整梯度，`get_tmp_buffer` 里用 `paddle.zeros(..., dtype=main_grad_dtype)` 分配）。
每个 unit 共 234,549,248 个元素（dense 22.2M + expert 212.3M），fp32 下是 938 MB。
默认 `sync_grads_each_microbatch=True`，ga=24 时每个 unit 每步要做 24 次 reduce-scatter，梯度通信在 backward 里占的比重很大。

##### 修改前

1. **backward 写完整 fp32 梯度。** 本 unit 的参数梯度累加到 `tmp_data_buffer`（fp32，938 MB）。
2. **unit 的 backward 结束后发起 reduce-scatter。** `_reduce_group_grads` → `do_reduce_scatter`：

   ```python
   tmp_buffer = self.get_tmp_buffer()                     # fp32 完整梯度
   if self.grad_div * self.grad_accum_steps == self.fsdp_degree:
       op = paddle.distributed.ReduceOp.AVG               # 缩放正好是 1/nranks，交给 NCCL 做
   else:
       tmp_buffer.scale_(self.grad_scale)                 # 否则先缩放，再 SUM
       op = paddle.distributed.ReduceOp.SUM
   shard = tmp_buffer._slice(0, self.data_buffer.shape[0])
   reduce_scatter(shard, tmp_buffer, op=op, ...)          # 原地：结果写回 tmp_buffer 的前 1/N
   ```

   通信数据是 fp32，每个 unit 每个 micro-batch 938 MB。
3. **等通信完成后累加。** `_retire_grad_comm` → `accumulate_reduced_grad`，把 `tmp_buffer` 前 1/N 段加到 `data_buffer`，然后释放 `tmp_buffer`。

问题：
- 通信量按 4 B/元素算。
- reduce-scatter 在途期间（`grad_reduce_queue` 里排队的 buffer），整块 938 MB 的 fp32 `tmp_buffer` 一直不能释放。在途 buffer 越多，显存越高。

##### 修改后

思路是只在跨 rank 求和这一步降精度。backward 写梯度、跨 micro-batch 累加、optimizer 读 main grad 都保持 fp32。

1. **配置入口。** `MixedPrecisionPolicy` 新增字段 `grad_comm_dtype`（`fully_shard.py`）。`fully_shard` 只接受 `None` / `float32` / `bfloat16`，其他值抛 `ValueError`；走 auto parallel 路径时传了会抛 `NotImplementedError`。
   这个值经 `FullyShardFusion` → `FSDPBufferManager` → `TensorFusionBuffer` 一路传下去。
2. **和 main grad dtype 相同时视为未设置。**

   ```python
   self.grad_comm_dtype = (
       grad_comm_dtype if grad_comm_dtype != self.main_grad_dtype else None)
   ```

   main grad 本身就是 bf16 时（例如不开 `amp_master_grad`、`reduce_dtype=bf16`），不会多做一次无意义的 cast。
3. **先缩放，再 cast，再通信。** `do_reduce_scatter` 里先在 fp32 上做 `scale_`（如果需要的话），然后 cast：

   ```python
   if self.grad_comm_dtype is None:
       self._rs_comm_buffer = tmp_buffer
   else:
       self._rs_comm_buffer = paddle.cast(tmp_buffer, self.grad_comm_dtype)   # fp32 -> bf16 新 buffer
       self.tmp_data_buffer = None                                            # 立即释放 fp32 完整梯度
   full = self._rs_comm_buffer
   shard = full._slice(0, self.data_buffer.shape[0])
   reduce_scatter(shard, full, op=op, ...)
   ```

   - 顺序很关键：缩放在 fp32 上做，cast 之后 bf16 的数值范围已经是除以 N 之后的结果，不容易溢出。用 AVG 时，除法由 NCCL 在 bf16 上完成。
   - 通信 buffer 换成 bf16，每个 unit 每个 micro-batch 从 938 MB 降到 469 MB。
   - `self.tmp_data_buffer = None` 去掉了 fp32 完整梯度的最后一个引用，它在 cast 完成后就能被回收。在途期间只占一块 bf16 buffer。cast 的那一刻会短暂同时存在 fp32 和 bf16 两份。
4. **收到 shard 后 cast 回 fp32 再累加。** `accumulate_reduced_grad`：

   ```python
   shard = self._rs_comm_buffer._slice(0, self.data_buffer.shape[0])
   self._rs_comm_buffer = None
   if shard.dtype != self.data_buffer.dtype:
       shard = shard.astype(self.data_buffer.dtype)      # bf16 -> fp32
   self.data_buffer.add_(shard)                           # 跨 micro-batch 累加仍是 fp32
   self.clear_tmp_buffer()
   ```

5. **reduce-scatter 发出后马上释放梯度 view。** `_reduce_group_grads` 发起通信后立即调 `_release_reduced_grad_views`，清掉参数上指向完整梯度的 `main_grad` view，buffer 的引用不会再被参数拖住。

##### 精度影响在哪里

- **不变的部分**：参数梯度的计算、跨 micro-batch 的累加、optimizer 读到的 main grad，都是 fp32。
- **变的部分**：每个 micro-batch 内 N 个 rank 的梯度求和在 bf16 上做，每个 micro-batch 的贡献会带一次 bf16 舍入，精度约 3 位有效十进制数字。ga=24 时，每个 shard 的最终 main grad 是 24 个 bf16 求和结果在 fp32 上的累加，舍入误差不会在 fp32 累加里进一步放大。
- **实测**：20 步内各阶段 step 20 的 loss 差 < 1e-5。这只能说明短期没有明显回退，不能证明长程训练等价，正式训练前建议做长程 loss 对比。

##### 前后对比

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| backward 写入的完整梯度 | fp32 `tmp_data_buffer` | fp32 `tmp_data_buffer`（不变） |
| reduce-scatter 的数据 | fp32，938 MB / unit / micro-batch | bf16，469 MB / unit / micro-batch |
| 缩放（1/N、1/ga） | fp32 上 `scale_`，或 NCCL AVG | 先在 fp32 上 `scale_` 再 cast，或 NCCL 在 bf16 上 AVG |
| 在途梯度的显存 | 一块 fp32 完整 buffer | 一块 bf16 完整 buffer，fp32 在 cast 后释放 |
| 跨 micro-batch 累加 | fp32 `data_buffer` | fp32 `data_buffer`（bf16 shard 先 `astype` 回 fp32） |
| 每个 unit 每步的 reduce-scatter 次数 | 24（ga=24） | 24（不变，减次数要靠 T6） |

##### 怎么打开

当前 trainer（`ernie5/src/trainers/pretraining_trainer.py` 的 `_wrap_model` → `fully_shard`）没有传 `mp_policy`，也没有对应的训练参数，所以在 yaml 里打不开这一项。
`fsdp_perf_opt_reorder_t2_last.md` 的测试是靠 `perf_stages3/knobs.json` 的 `grad_comm_dtype: "bfloat16"` 直接覆盖 `FSDPBufferManager` 打开的。
要在正式训练里用，需要给 trainer 加一个参数，并传 `mp_policy=MixedPrecisionPolicy(grad_comm_dtype=paddle.bfloat16)`。

##### 实测

- 本文累积链（T3 → T4）：20986.2 → 21372.4 tokens/s/card（+1.8%），fwd-bwd 18300 → 17958 ms，显存不变（59.38 GB）。
- 重排复测（`fsdp_perf_opt_reorder_t2_last.md`，T3 → T4）：20970.0 → 21430.3（+2.2%），fwd-bwd 18327 → 17914 ms，显存不变（59.03 GB）。

显存没有下降，是因为峰值不在梯度在途阶段，而在 forward 结束时，那时 fp8 专家权重和激活都在显存里。

##### 要点

1. 混合精度要分清"存储/累加精度"和"通信精度"，两者可以分开设。累加保持 fp32，只降通信这一步，收益是通信减半，精度风险集中在一次求和上。
2. 降精度前先在高精度上缩放，可以避免 bf16 溢出，也减少舍入误差的相对影响。
3. 降精度的 cast 会新建 buffer，要及时释放原来的高精度 buffer，否则显存反而比修改前多一份。

### T5 · reshard_policy / reshard_filter（PR #79887）

`fully_shard.py` `fully_shard(reshard_policy=..., reshard_filter=...)`；`fully_shard_fusion.py` `FSDPCommManager`

决定 gather 出来的完整参数什么时候释放：

| `reshard_policy` | 释放时机 | 每层每个 step 的 gather 次数（ga=24） |
| --- | --- | --- |
| `after_forward`（默认） | forward 后可被回收，backward 重新 gather | 48（每个 micro-batch forward + backward 各一次） |
| `after_backward` | 保留到本 unit 的 backward 结束 | 24 |
| `after_step` | 跨 micro-batch 保留，optimizer step 前统一释放 | 1 |

`reshard_filter` 决定策略作用于哪些参数：`all`、`expert`（`is_moe_param`）、`dense`，其余参数仍按 `after_forward`。callable 预留，当前会抛 `NotImplementedError`。

实现上给 buffer 打 `keep_gathered` 标记：

- 这类 buffer 不计入 gather 预算，按层 / 按元素数回收时都跳过。
- `after_step` 的 buffer 在 backward 的 `_release_params_after_unit` 里也跳过，由 step 结束时的 `reset_params_buffer_status` 释放。

```python
        for group in buffer_manager.buffer_groups:
            group.params_buffer.keep_gathered = self._keep_after_forward(group)
...
    def _reshard_policy_of(self, group):
        if self._in_reshard_filter(group):
            return self.reshard_policy
        return "after_forward"

    def _keep_after_forward(self, group):
        return self._reshard_policy_of(group) != "after_forward"

    def _keep_after_backward(self, group):
        return self._reshard_policy_of(group) == "after_step"
...
    def _release_params_after_unit(self, unit_id):
        ...
            if self._keep_after_backward(group):
                # Freed by reset_params_buffer_status at the end of the step.
                continue
```

如 T3 所述，fp8 专家 buffer 在 `after_forward` 下已经留到 backward，所以：

- `after_backward` + `all`（T5a）实际只改变 dense 参数，显存多出约 20 × 22.2M × 2 B ≈ 0.9 GB，与实测 +0.91 GB 一致。
- `after_step` 对专家参数的增量是把 fp8 权重从"每个 micro-batch gather 一次"变成"每步一次"；专家权重原本就在 forward 期间常驻，所以 T5c 显存只多 0.4 GB。

#### T5 修改前后对比（教学版）

一句话：FSDP 默认 forward 用完就把完整参数释放，backward 再 gather 一次。reshard_policy 让用户用显存换通信，把 gather 出来的完整参数多留一段时间，backward 或后续 micro-batch 直接复用。reshard_filter 决定这笔交换只作用在 dense 参数、专家参数，还是全部参数。

**背景：FSDP 参数 buffer 的状态机。** 每个 unit 的参数 buffer 有四个状态（`BufferState`）：

```
FREED ──all_gather──> USING ──本 unit 计算完──> READY ──释放──> FREED
                        ^                         │
                        └──────── 复用 ───────────┘
```

- `READY` 表示"完整参数还在显存里，但当前没人用"。再次进入这个 unit 时，READY 直接转回 USING，不用 gather（`fully_shard_fusion.py` `all_gather_params`，`READY -> USING, no need to all_gather again`）。
- 所以"gather 几次"完全取决于 buffer 什么时候从 READY 被释放回 FREED。

释放发生在三个地方：
1. **gather 预算回收**：gather 新 unit 前，`_release_one_buffer_if_needed`（按层）或 `_release_ready_buffers`（按元素数）把超出预算的 READY buffer 释放掉。按层预取时，预算是 `gather_buffer_limit = groups_per_unit × (prefetch_units + 1)`。
2. **backward 里本 unit 算完后**：`_release_params_after_unit` 释放已经完成 backward 的 unit。
3. **step 结束**：`reset_params_buffer_status` 释放所有还在显存里的 buffer。

##### 修改前（等价于现在的 `after_forward`）

只有一种行为：
- **forward**：进入 unit 时 gather，离开时 `shard_params` 把 buffer 标成 READY。gather 下一个 unit 时预算不够，这个 READY buffer 就被回收。默认 `prefetch_units=1` 时，dense 参数同时只留约 2 个 unit。
- **backward**：buffer 已经是 FREED，要重新 gather。
- **下一个 micro-batch**：再从头 gather。

ga=24 时，一个 bf16 unit 每步 gather 48 次（每个 micro-batch 的 forward、backward 各一次）。
fp8 专家 buffer 是个例外：它不计入预算，forward 后一直留到本 unit 的 backward（见 T3），所以每步 gather 24 次。

问题：显存够用时也没法用它换通信。dense 参数每个 micro-batch 都要 gather 两次；专家 fp8 参数每个 micro-batch 也要 gather 一次。

##### 修改后

思路是给 buffer 打一个 `keep_gathered` 标记，让"预算回收"和"backward 后释放"这两个释放点按策略跳过它。状态机本身不变。

1. **配置入口与校验。** `fully_shard(reshard_policy=..., reshard_filter=...)` 一路传到 `FSDPCommManager.__init__`：
   - `reshard_policy` 只能是 `after_forward` / `after_backward` / `after_step`，`reshard_filter` 只能是 `all` / `expert` / `dense`，否则抛 `ValueError`。
   - `reshard_filter` 传 callable（按层选择）会抛 `NotImplementedError`，接口已预留。
   - 训练侧已有对应参数 `fsdp_reshard_policy` / `fsdp_reshard_filter`（PaddleFleet `training_args.py`），trainer 会传给 `fully_shard`，可以直接写在 yaml 里。
2. **每个 buffer 算出自己实际的策略。** filter 选中的 buffer 用配置的策略，没选中的回退到 `after_forward`：

   ```python
   def _in_reshard_filter(self, group):
       if self.reshard_filter == "expert":
           return group.is_expert_param
       if self.reshard_filter == "dense":
           return not group.is_expert_param
       return True

   def _reshard_policy_of(self, group):
       if self._in_reshard_filter(group):
           return self.reshard_policy
       return "after_forward"
   ```

3. **两个谓词对应两个释放点。**

   ```python
   def _keep_after_forward(self, group):    # forward 后不让预算回收
       return self._reshard_policy_of(group) != "after_forward"

   def _keep_after_backward(self, group):   # backward 后也不释放
       return self._reshard_policy_of(group) == "after_step"
   ```

   初始化时 `group.params_buffer.keep_gathered = self._keep_after_forward(group)`。
4. **预算回收跳过 keep 的 buffer。**
   - 这类 buffer 不计入预算：`_count_gathered` 和 `_free_params_buffer` 都会跳过 `keep_gathered`，不增减 `buffer_cnt_in_using` / `gathered_numel`。
   - 回收时也跳过它们：`_release_one_buffer_if_needed` 检查 `_keep_after_forward(group)`，`_release_ready_buffers` 检查 `keep_gathered`。
   - 结果是 forward 后 buffer 停在 READY，backward 进入同一个 unit 时直接复用。
5. **backward 后按策略决定是否释放。** `_release_params_after_unit` 里：

   ```python
   if self._keep_after_backward(group):
       # Freed by reset_params_buffer_status at the end of the step.
       continue
   ```

   `after_backward` 在这里释放；`after_step` 跳过，下一个 micro-batch 的 forward 再进入这个 unit 时，buffer 仍是 READY，直接复用。
6. **step 结束统一释放。** `reset_params_buffer_status` 把所有 READY / USING 的 buffer 释放回 FREED。
   - 这一步必须在 optimizer 更新参数之前做完。一个 step 内的各个 micro-batch 参数不变，跨 micro-batch 复用完整参数是安全的；optimizer 一更新，旧的完整参数就作废了。
   - backward 末尾发出的预取 gather 拿到的是更新前的 shard，这里会先 `wait` 再丢掉，下一步 forward 重新 gather 更新后的参数（代码注释 `A prefetch issued at the end of backward gathers the pre-step shard`）。

##### 每个 unit 每步的 gather 次数（ga=24）

| 策略 | dense（bf16，44 MB / 次） | expert（fp8，212 MB / 次） |
| --- | --- | --- |
| `after_forward` | 48 | 24（fp8 buffer 自带 forward → backward 保留） |
| `after_backward` | 24 | 24（与 `after_forward` 相同） |
| `after_step` | 1 | 1 |

上面原有表格里 `after_forward` 写的是 48，那是 bf16 参数的次数；fp8 专家参数在 `after_forward` 下已经是 24。

##### 显存代价

keep 的 buffer 在保留期间一直占着完整参数的显存。显存峰值出现在 forward 结束时，这时所有 unit 的完整参数都还没释放：

- dense 全部保留：20 × 22.2M × 2 B ≈ 0.9 GB。T5a（`after_backward` / `all`）、T5b（`after_step` / `dense`）、T5d（`after_step` / `all`）实测都是 +0.91 GB。
- expert 全部保留：fp8 专家权重在 `after_forward` 下本来就会在 forward 结束时全部留在显存里（约 4.2 GB），所以 T5c（`after_step` / `expert`）只多 0.43 GB。
- T5d 的显存和 T5b 相同，没有再叠加 T5c 那 0.43 GB。推测是峰值位置变了，本次未验证。

##### 前后对比

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 可选策略 | 只有 forward 后可回收 | `after_forward` / `after_backward` / `after_step` |
| 作用范围 | 全部参数 | `all` / `expert` / `dense` |
| 预算回收 | 所有 READY buffer 都可能被回收 | 跳过 `keep_gathered` 的 buffer |
| backward 后释放 | 一律释放 | `after_step` 的 buffer 跳过，留到 step 结束 |
| 显存 | 同时只留约 `prefetch_units + 1` 个 unit | 保留范围内的完整参数全部常驻 |
| yaml 配置 | 无 | `fsdp_reshard_policy` / `fsdp_reshard_filter` |

##### 实测

- 本文 T5（都相对 T4 21372.4，均带 `comm_unit_size=703647744`）：T5a +0.5%，T5b +1.0%，T5c +1.1%，T5d +1.7%。
- 重排复测（`fsdp_perf_opt_reorder_t2_last.md`）：T5d 21899.8，相对 T4 21430.3 为 +2.2%，fwd-bwd 17914 → 17533 ms，max_mem_alloc 59.03 → 60.29 GB。T5d 再叠 `comm_unit_size` 反而 -0.7%。

收益偏小有两个原因：
- dense 每次 gather 只有 44 MB，省下来的通信量本来就少。
- expert 省得多：每个 unit 每步少 23 次 212 MB 的 fp8 gather。但这些 gather 在 side stream 和预取下大部分已经被计算盖住了，省掉之后只缩短了暴露在外的那部分。

这一点是根据 fwd-bwd 的变化推断的，没有采 nsys 验证。

##### 要点

1. FSDP 的"释放时机"是显存和通信之间的一个旋钮。默认选省显存；显存有余量时，把完整参数留久一点，就能少 gather。
2. 跨 micro-batch 复用完整参数之所以安全，是因为一个 step 内参数不变。释放必须在 optimizer 更新之前做完，预取的旧 shard 也要丢掉。
3. 不同参数的保留收益不同，所以 filter 有意义：dense 参数小、gather 便宜；expert 参数大，但 fp8 方案已经替它做了一半的保留。先搞清楚每类参数现在 gather 几次，再决定保留谁。

### T6 · 只在最后一个 micro-batch 同步梯度（PR #79903）

`fully_shard_fusion.py` `FSDPCommManager`（`sync_grads_each_microbatch` / `_backward_cnt`）/ `FullyShardFusion.set_grad_accum_steps`

PR 中标为 Experimental，`fully_shard` 不暴露参数，要直接设 `comm_manager.sync_grads_each_microbatch = False`。
backward 进入编号更大的 unit 时 `_backward_cnt += 1`，以此计数 micro-batch。
`_backward_cnt < grad_accum_steps` 时梯度不做 reduce-scatter，留在未分片的完整 grad buffer 里继续累加，只在最后一个 micro-batch 做一次 reduce-scatter，`zero_grad` 时清零计数。

代价是每个 unit 的完整 fp32 grad buffer 在整个 step 内常驻，而不是 reduce-scatter 后就释放。

```python
        # Experimental: False syncs grads only on the last micro-batch.
        self.sync_grads_each_microbatch = True
        self.grad_accum_steps = 1
        self._backward_cnt = 0
...
            if unit_id != self._last_backward_unit_id:
                prev_unit_id = self._last_backward_unit_id
                if unit_id is not None and (prev_unit_id is None or unit_id > prev_unit_id):
                    self._backward_cnt += 1
...
        if (not self.sync_grads_each_microbatch
                and self._backward_cnt < self.grad_accum_steps):
            self._release_reduced_grad_views(grads_buffer)
            return
```

## 第二部分复现方式

文件都在 `perf_stages2/`：

| 文件 | 作用 |
| --- | --- |
| `final/{fully_shard.py,fully_shard_fusion.py,_fp8_utils.py}` | 最终代码 + 测试开关，开关从 `perf_stages2/knobs.json` 读取（mpirun 不透传环境变量，所以走共享盘文件），启动时打印 `[perf_stages2] effective: ...` |
| `backup/` | `Paddle/build/python/paddle/distributed/fsdp/` 下原始的三个文件 |
| `specs/all.txt` | T0–T6 的阶段定义，每行 `<name> <knobs-json>` |
| `specs/t6.txt` | T6 在 mb1 / ga48 下的成对对照（T4ref_mb1ga48、T6_sync_last_mb_mb1ga48） |
| `run_stages.sh` | 把 `final/*` 装进 `Paddle/build`，逐行写 `knobs.json`，跑 `run_fsdp_test_v2_stage.sh`，解析 step 4–20 结果追加到 `results.log`，退出时恢复 `backup/` |
| `run_fsdp_test_v2_stage.sh` | `run_fsdp_test_v2.sh` 的副本，多拼 `extra_kwargs.txt` 里的 `--kwargs` |
| `cleanup.sh` | 每阶段结束后经 `script/mpi.sh` 在两台机器上清理残留训练进程（mpirun 异常退出时另一台机器的 worker 会继续占 81 GB 显存，导致下一阶段 OOM） |
| `parse.py` | `python parse.py <workerlog> 4 20`，输出吞吐均值 / 中位数、fwd-bwd、opt-step、显存、loss |
| `patches/*.diff` | 各 PR 不含测试开关的 diff，即下方附录 |

```bash
# 需在多机环境（IS_STANDALONE=0），任一节点执行
EXTRA_KWARGS="per_device_train_batch_size=2 gradient_accumulation_steps=24" \
  bash perf_stages2/run_stages.sh perf_stages2/specs/all.txt

# T6 对照（mb2 / ga24 下 T6 OOM，改用 mb1 / ga48）
EXTRA_KWARGS="per_device_train_batch_size=1 gradient_accumulation_steps=48" \
  bash perf_stages2/run_stages.sh perf_stages2/specs/t6.txt
```

测试前已用 `script/evict_be_pod.py` 驱逐两台机器上的混部 BE 任务。

## 第二部分附录：各 PR 代码 diff

每个 PR 相对其前一个 PR 的 diff（不含测试开关），即该功能点在 Paddle / PaddleFleet 中实际新增和删除的代码。
#79881 的 diff 见第一部分附录。

### T1 · PR #79898 Comm Stream

```diff
diff --git a/python/paddle/distributed/fsdp/fully_shard_fusion.py b/python/paddle/distributed/fsdp/fully_shard_fusion.py
index f9cafb8a0d..484952b629 100644
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -35,6 +35,23 @@ from paddle.distributed.fsdp._fsdp_context import (
 )


+class StreamTask:
+    def __init__(self, stream):
+        self._event = stream.record_event()
+
+    def wait(self):
+        paddle.device.current_stream().wait_event(self._event)
+
+
+def run_on_stream(stream, tensors, collective):
+    stream.wait_stream(paddle.device.current_stream())
+    with paddle.device.stream_guard(stream):
+        collective()
+        for t in tensors:
+            t._record_stream()
+    return StreamTask(stream)
+
+
 class BufferState(Enum):
     # Buffer status for lazy double buffer mechanism
     #
@@ -102,10 +119,12 @@ class TensorFusionBuffer:
         grad_div=None,
         shard_buffer=True,
         shard_align_numel=None,
+        comm_stream=None,
     ):
         # Calculate total buffer size needed (with padding)
         self.unique_key = unique_key
         self.fsdp_group = fsdp_group
+        self.comm_stream = comm_stream
         self.fsdp_degree = fsdp_group.nranks
         self.is_sharded = shard_buffer and fsdp_group.nranks > 1
         self.is_params = is_params
@@ -284,16 +303,21 @@ class TensorFusionBuffer:
         return self.tmp_data_buffer

     def do_all_gather(self):
-        return paddle.distributed.all_gather(
-            self.get_tmp_buffer(),
-            self.data_buffer,
-            group=self.fsdp_group,
-            sync_op=False,
-        )
+        full, shard = self.get_tmp_buffer(), self.data_buffer
+
+        def gather():
+            paddle.distributed.stream.all_gather(
+                full,
+                shard,
+                group=self.fsdp_group,
+                sync_op=True,
+                use_calc_stream=True,
+            )
+
+        return run_on_stream(self.comm_stream, (full, shard), gather)

     def do_reduce_scatter(self):
         tmp_buffer = self.get_tmp_buffer()
-        shard = tmp_buffer._slice(0, self.data_buffer.shape[0])
         # AVG folds the scale into NCCL when it is exactly 1/nranks of this
         # group; otherwise (grad accumulation, EP groups) pre-scale and SUM.
         if self.grad_div * self.grad_accum_steps == self.fsdp_degree:
@@ -301,13 +325,18 @@ class TensorFusionBuffer:
         else:
             tmp_buffer.scale_(self.grad_scale)
             op = paddle.distributed.ReduceOp.SUM
-        return paddle.distributed.reduce_scatter(
-            shard,
-            tmp_buffer,
-            op=op,
-            group=self.fsdp_group,
-            sync_op=False,
-        )
+
+        def reduce():
+            paddle.distributed.stream.reduce_scatter(
+                tmp_buffer._slice(0, self.data_buffer.shape[0]),
+                tmp_buffer,
+                op=op,
+                group=self.fsdp_group,
+                sync_op=True,
+                use_calc_stream=True,
+            )
+
+        return run_on_stream(self.comm_stream, (tmp_buffer,), reduce)

     def accumulate_reduced_grad(self):
         shard = self.get_tmp_buffer()._slice(0, self.data_buffer.shape[0])
@@ -347,6 +376,7 @@ class FSDPBufferManager:
         main_grad_dtype=None,
     ):
         self.model = model
+        self._comm_streams = {}
         self.hcg = fleet.get_hybrid_communicate_group()
         self._fsdp_group = self.hcg.get_sharding_parallel_group()
         use_ep = (
@@ -499,6 +529,7 @@ class FSDPBufferManager:
                 is_params=True,
                 shard_buffer=shard_buffer,
                 shard_align_numel=muon_align,
+                comm_stream=self._comm_stream(group.fsdp_group),
             )
             if not params[0].stop_gradient:
                 grads_shard_buffer = (
@@ -517,6 +548,7 @@ class FSDPBufferManager:
                     grad_div=self._fsdp_group.nranks,
                     shard_buffer=grads_shard_buffer,
                     shard_align_numel=muon_align,
+                    comm_stream=self._comm_stream(group.fsdp_group),
                 )
             group.muon_shard_numel = group.params_buffer.shard_align_numel
             if group.hyperball_sharded:
@@ -539,6 +571,13 @@ class FSDPBufferManager:
         self._assign_muon_owners()
         self._set_clip_attrs()

+    def _comm_stream(self, group):
+        if group.nranks <= 1:
+            return None
+        if group.id not in self._comm_streams:
+            self._comm_streams[group.id] = paddle.device.Stream()
+        return self._comm_streams[group.id]
+
     def _set_clip_attrs(self):
         for group in self.buffer_groups:
             data_buffer = group.params_buffer.data_buffer
@@ -712,20 +751,17 @@ class FSDPCommManager:
                 params_buffer.comm_task.wait()
                 params_buffer.comm_task = None

-            tmp_buffer = params_buffer.get_tmp_buffer()
             # Do all_gather in sync: FREED -> USING
             if params_buffer.status == BufferState.FREED:
                 if params_buffer.is_sharded:
-                    params_buffer.fsdp_group.process_group.all_gather(
-                        params_buffer.data_buffer, tmp_buffer
-                    ).wait()
+                    params_buffer.do_all_gather().wait()
                     self.buffer_cnt_in_using += 1
                 params_buffer.status = BufferState.USING

             # Bind the unsharded param to the real param
             offset = params_buffer.param_offsets[param.name]
             tmp_param = paddle._C_ops.view_slice(
-                tmp_buffer,
+                params_buffer.get_tmp_buffer(),
                 offset,
                 offset + param._numel(),
             )
```

### T2 · PR #79901 prefetch by size

```diff
diff --git a/python/paddle/distributed/fsdp/fully_shard.py b/python/paddle/distributed/fsdp/fully_shard.py
index b57ed3abd4..ececa0cded 100644
--- a/python/paddle/distributed/fsdp/fully_shard.py
+++ b/python/paddle/distributed/fsdp/fully_shard.py
@@ -93,6 +93,8 @@ def _fully_shard_manual_parallel(
     enable_tensor_fusion_and_overlap,
     optimizer=None,
     grad_accum_steps=1,
+    prefetch_units=1,
+    comm_unit_size=None,
 ):
     layers = module._layers if isinstance(module, MetaParallelBase) else module
     if optimizer is not None:
@@ -101,6 +103,8 @@ def _fully_shard_manual_parallel(
         layers,
         enable_tensor_fusion_and_overlap=enable_tensor_fusion_and_overlap,
         mp_policy=mp_policy,
+        prefetch_units=prefetch_units,
+        comm_unit_size=comm_unit_size,
     )
     layers._fsdp_context.set_grad_accum_steps(grad_accum_steps)
     if optimizer is not None:
@@ -120,6 +124,8 @@ def _fully_shard_auto_parallel(
     enable_tensor_fusion_and_overlap,
     optimizer=None,
     grad_accum_steps=1,
+    prefetch_units=1,
+    comm_unit_size=None,
 ):
     module._fsdp_context = FullyShardAuto(
         module,
@@ -142,6 +148,8 @@ def fully_shard(
     enable_tensor_fusion_and_overlap: bool = True,
     optimizer: paddle.optimizer.Optimizer | None = None,
     grad_accum_steps: int = 1,
+    prefetch_units: int = 1,
+    comm_unit_size: int | None = None,
 ) -> paddle.nn.Layer:
     """
     Apply fully sharded data parallel (FSDP) to the given module.
@@ -178,6 +186,14 @@ def fully_shard(
             if needed.
         grad_accum_steps (int, optional): Manual parallel mode only. Gradient
             accumulation steps. Defaults to 1.
+        prefetch_units (int, optional): Number of FSDP units all_gathered
+            ahead (prefetch by layer). 0 disables prefetch. Ignored when
+            ``comm_unit_size`` is set. Defaults to 1.
+        comm_unit_size (int | None, optional): If set, bound communication
+            by size: params are all_gathered ``comm_unit_size // 2``
+            elements ahead and at most ``comm_unit_size`` grad elements
+            stay in reduce_scatter. 0 disables prefetch. Defaults to None
+            (prefetch by layer).

     Returns:
         module: A wrapper module that applies FSDP to the input module.
@@ -222,6 +238,8 @@ def fully_shard(
         enable_tensor_fusion_and_overlap,
         optimizer,
         grad_accum_steps,
+        prefetch_units,
+        comm_unit_size,
     )

     if in_auto_parallel_mode():
diff --git a/python/paddle/distributed/fsdp/fully_shard_fusion.py b/python/paddle/distributed/fsdp/fully_shard_fusion.py
index 484952b629..f6f7dfc149 100644
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -626,10 +626,12 @@ class FSDPCommManager:
         buffer_manager,
         enable_overlap=True,
         prefetch_units=1,
+        comm_unit_size=None,
     ):
         self.buffer_manager = buffer_manager
         self.enable_overlap = enable_overlap
         self.grad_reduce_queue = []
+        self.prefetch_by_size = comm_unit_size is not None

         # Size limits in whole units: a unit may own several sharded groups.
         unit_sizes = Counter(
@@ -645,11 +647,64 @@ class FSDPCommManager:
         self.need_zero_grads = True
         self._last_backward_unit_id = None

+        # By size: gathered param elements and queued grad elements.
+        self.gathered_numel = 0
+        self._grad_queue_numel = 0
+        if self.prefetch_by_size:
+            self._init_prefetch_by_size(comm_unit_size)
+
+    def _init_prefetch_by_size(self, unit_size):
+        groups = self.buffer_manager.buffer_groups
+
+        # Sharded buffers of one (unit, is_expert) are prefetched together.
+        self._gid_to_bucket_group = []
+        bucket_groups = {}
+        for gid, g in enumerate(groups):
+            if g.fsdp_unit_id is None or not g.params_buffer.is_sharded:
+                self._gid_to_bucket_group.append([gid])
+                continue
+            members = bucket_groups.setdefault(
+                (g.fsdp_unit_id, g.is_expert_param), []
+            )
+            members.append(gid)
+            self._gid_to_bucket_group.append(members)
+
+        unit_size = max(int(unit_size), 0)
+        self.comm_unit_size = unit_size
+        self.grad_queue_capacity = unit_size
+        self.gather_prefetch_size = unit_size // 2
+
+        # Cache the fixed layout; the hooks are on the host critical path.
+        self._params_buffers = [g.params_buffer for g in groups]
+        self._sharded_gids = [
+            gid for gid, pb in enumerate(self._params_buffers) if pb.is_sharded
+        ]
+        # (req_gids, is_backward) -> (targets, keep, limit)
+        self._prefetch_plans = {}
+
     # ------------------------------------------------------------------
-    # Params all_gather double buffer. Owns buffer_cnt_in_using /
-    # gather_buffer_limit / prefetch_units and the BufferState machine.
+    # Params all_gather pipeline. Owns the BufferState machine,
+    # buffer_cnt_in_using (by layer) and gathered_numel (by size).
     # ------------------------------------------------------------------

+    def _count_gathered(self, params_buffer):
+        # A sharded buffer just got its full storage allocated.
+        if self.prefetch_by_size:
+            self.gathered_numel += params_buffer.total_buffer_size
+        else:
+            self.buffer_cnt_in_using += 1
+
+    def _free_params_buffer(self, params_buffer):
+        # USING / READY / SYNCING-waited -> FREED, keeping the accounting.
+        params_buffer.clear_tmp_buffer()
+        params_buffer.status = BufferState.FREED
+        if not params_buffer.is_sharded:
+            return
+        if self.prefetch_by_size:
+            self.gathered_numel -= params_buffer.total_buffer_size
+        elif self.buffer_cnt_in_using > 0:
+            self.buffer_cnt_in_using -= 1
+
     def _release_one_buffer_if_needed(self, keep=()):
         # Release a buffer with the READY status if needed
         while self.buffer_cnt_in_using >= self.gather_buffer_limit:
@@ -694,11 +749,81 @@ class FSDPCommManager:
             next_gid += step
         return gids

+    def _release_ready_buffers(self, need_numel, limit, keep=()):
+        # Free READY buffers outside `keep`, oldest first, until it fits.
+        if self.gathered_numel + need_numel <= limit:
+            return
+        for gid in self._sharded_gids:
+            if gid in keep:
+                continue
+            params_buffer = self._params_buffers[gid]
+            if params_buffer.status == BufferState.READY:
+                self._free_params_buffer(params_buffer)
+                if self.gathered_numel + need_numel <= limit:
+                    return
+
+    def _extend_by_prefetch_size(self, req_gids, is_backward):
+        # Walk buffers after the requested ones in execution order, adding a
+        # whole bucket group while under gather_prefetch_size.
+        groups = self.buffer_manager.buffer_groups
+        prefetch_size = self.gather_prefetch_size
+        if prefetch_size <= 0:
+            return []
+        step = -1 if is_backward else 1
+        selected = set(req_gids)
+        targets = []
+        prefetched = 0
+        candidate = (min(req_gids) if is_backward else max(req_gids)) + step
+        while 0 <= candidate < len(groups):
+            if candidate in selected:
+                candidate += step
+                continue
+            # Backward never touches frozen params.
+            if is_backward and not groups[candidate].params_buffer.trainable:
+                selected.add(candidate)
+                candidate += step
+                continue
+            if prefetched >= prefetch_size:
+                break
+            for gid in self._gid_to_bucket_group[candidate]:
+                if gid in selected:
+                    continue
+                params_buffer = groups[gid].params_buffer
+                # Unsharded buckets are never gathered.
+                if not params_buffer.is_sharded or (
+                    is_backward and not params_buffer.trainable
+                ):
+                    continue
+                selected.add(gid)
+                targets.append(gid)
+                prefetched += params_buffer.total_buffer_size
+            candidate += step
+        return targets
+
+    def _prefetch_plan(self, req_gids, is_backward):
+        # Cached per request: prefetch targets, buffers to keep, and the
+        # bound on gathered elements (comm_unit_size or the window if larger).
+        key = (tuple(req_gids), is_backward)
+        plan = self._prefetch_plans.get(key)
+        if plan is None:
+            targets = self._extend_by_prefetch_size(req_gids, is_backward)
+            keep = frozenset(req_gids) | frozenset(targets)
+            limit = max(
+                self.comm_unit_size,
+                sum(
+                    self._params_buffers[gid].total_buffer_size
+                    for gid in keep
+                    if self._params_buffers[gid].is_sharded
+                ),
+            )
+            plan = self._prefetch_plans[key] = (tuple(targets), keep, limit)
+        return plan
+
     def _issue_async_gather(self, gid):
         params_buffer = self.buffer_manager.buffer_groups[gid].params_buffer
         params_buffer.status = BufferState.SYNCING
         params_buffer.comm_task = params_buffer.do_all_gather()
-        self.buffer_cnt_in_using += 1
+        self._count_gathered(params_buffer)

     def all_gather_params(self, params, is_backward=False):
         if len(params) == 0:
@@ -726,14 +851,28 @@ class FSDPCommManager:
             )

         if self.enable_overlap:
-            keep = set(req_gids)
-            for gid in req_gids:
-                group = self.buffer_manager.buffer_groups[gid]
-                if not group.params_buffer.is_sharded:
-                    continue
-                if group.params_buffer.status == BufferState.FREED:
-                    self._release_one_buffer_if_needed(keep)
-                    self._issue_async_gather(gid)
+            if self.prefetch_by_size:
+                targets, keep, limit = self._prefetch_plan(
+                    req_gids, is_backward
+                )
+                for gid in req_gids:
+                    params_buffer = self._params_buffers[gid]
+                    if not params_buffer.is_sharded:
+                        continue
+                    if params_buffer.status == BufferState.FREED:
+                        self._release_ready_buffers(
+                            params_buffer.total_buffer_size, limit, keep
+                        )
+                        self._issue_async_gather(gid)
+            else:
+                keep = set(req_gids)
+                for gid in req_gids:
+                    group = self.buffer_manager.buffer_groups[gid]
+                    if not group.params_buffer.is_sharded:
+                        continue
+                    if group.params_buffer.status == BufferState.FREED:
+                        self._release_one_buffer_if_needed(keep)
+                        self._issue_async_gather(gid)

         for param in params:
             gid = self.buffer_manager.param_to_buffer_id[param.name]
@@ -755,7 +894,7 @@ class FSDPCommManager:
             if params_buffer.status == BufferState.FREED:
                 if params_buffer.is_sharded:
                     params_buffer.do_all_gather().wait()
-                    self.buffer_cnt_in_using += 1
+                    self._count_gathered(params_buffer)
                 params_buffer.status = BufferState.USING

             # Bind the unsharded param to the real param
@@ -769,7 +908,18 @@ class FSDPCommManager:
             param.get_tensor()._share_data_with(tmp_param.get_tensor())

         # Prefetch after waiting on this unit's gathers, or compute stalls on it.
-        if self.enable_overlap:
+        if self.enable_overlap and self.prefetch_by_size:
+            for next_gid in targets:
+                next_params_buffer = self._params_buffers[next_gid]
+                if not next_params_buffer.is_sharded:
+                    continue
+                if next_params_buffer.status != BufferState.FREED:
+                    continue
+                self._release_ready_buffers(
+                    next_params_buffer.total_buffer_size, limit, keep
+                )
+                self._issue_async_gather(next_gid)
+        elif self.enable_overlap:
             keep = set(req_gids)
             for next_gid in self._next_buffer_id(req_gids, is_backward):
                 next_params_buffer = self.buffer_manager.buffer_groups[
@@ -819,12 +969,7 @@ class FSDPCommManager:
                     param._clear_data()
                     param.stop_gradient = stop_gradient
                     param.get_tensor()._set_dims(_shape)
-                params_buffer.clear_tmp_buffer()
-                params_buffer.status = BufferState.FREED
-                if not params_buffer.is_sharded:
-                    continue
-                if self.buffer_cnt_in_using > 0:
-                    self.buffer_cnt_in_using -= 1
+                self._free_params_buffer(params_buffer)

     # ------------------------------------------------------------------
     # Grads reduce_scatter queue. Owns grad_reduce_queue / need_zero_grads;
@@ -902,9 +1047,10 @@ class FSDPCommManager:
         ):
             return
         # Grad queue mechanism: wait and release completed reduce_scatter async tasks
-        self._wait_for_grad_comm()
+        self._wait_for_grad_comm(next_numel=grads_buffer.total_buffer_size)
         grads_buffer.comm_task = grads_buffer.do_reduce_scatter()
         self.grad_reduce_queue.append(grads_buffer)
+        self._grad_queue_numel += grads_buffer.total_buffer_size
         if not self.enable_overlap:
             self._wait_for_grad_comm(queue_limit=0)

@@ -932,16 +1078,13 @@ class FSDPCommManager:
             if not params_buffer.is_sharded:
                 continue
             if params_buffer.status in (BufferState.READY, BufferState.USING):
-                params_buffer.status = BufferState.FREED
                 for param in group.params:
                     stop_gradient = param.stop_gradient
                     _shape = param.shape
                     param._clear_data()
                     param.stop_gradient = stop_gradient
                     param.get_tensor()._set_dims(_shape)
-                params_buffer.clear_tmp_buffer()
-                if self.buffer_cnt_in_using > 0:
-                    self.buffer_cnt_in_using -= 1
+                self._free_params_buffer(params_buffer)

     def _release_reduced_grad_views(self, grads_buffer):
         if not grads_buffer.is_sharded:
@@ -956,19 +1099,33 @@ class FSDPCommManager:
                 param.main_grad = None
                 param._main_grad_addr = None

-    def _wait_for_grad_comm(self, queue_limit=None):
+    def _retire_grad_comm(self):
+        grads_buffer = self.grad_reduce_queue.pop(0)
+        self._grad_queue_numel -= grads_buffer.total_buffer_size
+        if grads_buffer.comm_task is None:
+            grads_buffer.clear_tmp_buffer()
+            self._release_reduced_grad_views(grads_buffer)
+            return
+        grads_buffer.comm_task.wait()
+        grads_buffer.comm_task = None
+        grads_buffer.accumulate_reduced_grad()
+        self._release_reduced_grad_views(grads_buffer)
+
+    def _wait_for_grad_comm(self, queue_limit=None, next_numel=0):
+        # Default bound: grad_queue_limit buffers by layer or
+        # grad_queue_capacity elements (including the next one) by size.
+        if queue_limit is None and self.prefetch_by_size:
+            while (
+                self.grad_reduce_queue
+                and self._grad_queue_numel + next_numel
+                > self.grad_queue_capacity
+            ):
+                self._retire_grad_comm()
+            return
         if queue_limit is None:
             queue_limit = self.grad_queue_limit
         while len(self.grad_reduce_queue) > queue_limit:
-            grads_buffer = self.grad_reduce_queue.pop(0)
-            if grads_buffer.comm_task is None:
-                grads_buffer.clear_tmp_buffer()
-                self._release_reduced_grad_views(grads_buffer)
-                continue
-            grads_buffer.comm_task.wait()
-            grads_buffer.comm_task = None
-            grads_buffer.accumulate_reduced_grad()
-            self._release_reduced_grad_views(grads_buffer)
+            self._retire_grad_comm()

     def finish_grads_sync(self):
         # Wait for all async reduce_scatter tasks, call before optimizer.step()
@@ -1110,6 +1267,7 @@ class FullyShardFusion:
         enable_tensor_fusion_and_overlap=True,
         mp_policy=None,
         prefetch_units=1,
+        comm_unit_size=None,
     ):
         self.model = model
         self.mp_policy = mp_policy
@@ -1122,6 +1280,7 @@ class FullyShardFusion:
             self.buffer_manager,
             enable_overlap=enable_tensor_fusion_and_overlap,
             prefetch_units=prefetch_units,
+            comm_unit_size=comm_unit_size,
         )
         self.register_tensor_fusion_hooks(self.model)
         register_fsdp_context(self)
@@ -1449,10 +1608,8 @@ class FullyShardFusion:
             cm.shard_params(params)
             for gid in was_freed:
                 params_buffer = bm.buffer_groups[gid].params_buffer
-                params_buffer.clear_tmp_buffer()
-                params_buffer.status = BufferState.FREED
-                if params_buffer.is_sharded and cm.buffer_cnt_in_using > 0:
-                    cm.buffer_cnt_in_using -= 1
+                if params_buffer.status != BufferState.FREED:
+                    cm._free_params_buffer(params_buffer)

     def optimizer_sharded_state_dict(self, optimizer):
         """Split the state keyed by ``fuse_params_<gid>`` into per-param flattened shards."""
```

### T3 · PR #79897 fp8 param gather（Paddle）

```diff
diff --git a/python/paddle/distributed/fsdp/_fp8_utils.py b/python/paddle/distributed/fsdp/_fp8_utils.py
new file mode 100644
index 0000000000..ea25f63112
--- /dev/null
+++ b/python/paddle/distributed/fsdp/_fp8_utils.py
@@ -0,0 +1,151 @@
+# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
+# (Apache License 2.0 header omitted)
+
+import math
+import types
+
+import paddle
+
+
+def fp8_shard_numel(params):
+    # Buffer groups are keyed by quantizer, so all params share one.
+    quantizer = getattr(params[0], "_fsdp_fp8_quantizer", None)
+    if quantizer is None:
+        return None
+    block = quantizer.block
+    numel = block**2
+    for p in params:
+        shape = tuple(getattr(p, "original_shape", None) or p.shape)
+        if len(shape) != 3 or shape[1] % block or shape[2] % block:
+            raise ValueError(f"_fsdp_fp8_quantizer: {p.name}: {shape}")
+        numel = math.lcm(numel, shape[1] * shape[2])
+    return numel
+
+
+def init_fp8_buffer(params_buffer, params):
+    if not params_buffer.is_sharded:
+        raise ValueError(
+            f"_fsdp_fp8_quantizer: group {params_buffer.unique_key} is not "
+            "sharded"
+        )
+    quantizer = params[0]._fsdp_fp8_quantizer
+    k, full = len(quantizer.attrs), params_buffer.total_buffer_size
+    numels = [full] * k + [full // quantizer.block**2] * k
+    dtypes = [paddle.float8_e4m3fn] * k + [paddle.float32] * k
+    params_buffer.fp8 = types.SimpleNamespace(
+        params=params,
+        quantizer=quantizer,
+        quantized=False,
+        shards=[
+            paddle.empty([n // params_buffer.fsdp_degree], d)
+            for n, d in zip(numels, dtypes)
+        ],
+        fulls=None,
+        weight_copy=None,
+    )
+
+
+def build_ckpt_weight(params_buffer, dtype):
+    # A param-dtype main weight is saved as is; an fp32 one as a copy.
+    main = params_buffer.data_buffer
+    weight = main if main.dtype == dtype else main.astype(dtype)
+    params_buffer.fp8.weight_copy = weight
+    return weight
+
+
+@paddle.autograd.no_grad()
+def sync_ckpt_weight_to_main(params_buffer):
+    fp8 = params_buffer.fp8
+    weight, fp8.weight_copy = fp8.weight_copy, None
+    if weight is None:
+        return
+    main = params_buffer.data_buffer
+    if weight is not main and not bool(
+        paddle.equal_all(main.astype(weight.dtype), weight)
+    ):
+        paddle.assign(weight.astype(main.dtype), main)
+    fp8.quantized = False
+
+
+@paddle.autograd.no_grad()
+def quantize_fp8_shard(params_buffer):
+    fp8 = params_buffer.fp8
+    if fp8.quantized:
+        return
+    quantizer = fp8.quantizer
+    k, bn = len(quantizer.attrs), quantizer.block**2
+    for param in fp8.params:
+        shard = params_buffer.param_shard(param)
+        if shard is None:
+            continue
+        lo, hi, _ = shard
+        _, rows, cols = getattr(param, "original_shape", None) or param.shape
+        weights = paddle._C_ops.view_slice(params_buffer.data_buffer, lo, hi)
+        weights.get_tensor()._set_dims([(hi - lo) // (rows * cols), rows, cols])
+        weights = weights.astype(param.dtype)
+        for i, (_, _, transpose) in enumerate(quantizer.attrs):
+            data, scale = quantizer(weights, transpose)
+            paddle.assign(data.flatten(), fp8.shards[i]._slice(lo, hi))
+            paddle.assign(
+                scale.flatten(), fp8.shards[k + i]._slice(lo // bn, hi // bn)
+            )
+    fp8.quantized = True
+
+
+def get_fp8_tmp_buffer(params_buffer):
+    quantize_fp8_shard(params_buffer)
+    fp8 = params_buffer.fp8
+    fp8.fulls = [
+        paddle.empty([s.shape[0] * params_buffer.fsdp_degree], s.dtype)
+        for s in fp8.shards
+    ]
+    return list(zip(fp8.fulls, fp8.shards))
+
+
+def bind_fp8_param(params_buffer, param):
+    fp8 = params_buffer.fp8
+    attrs, block = fp8.quantizer.attrs, fp8.quantizer.block
+    k, bn = len(attrs), block**2
+    shape = list(getattr(param, "original_shape", None) or param.shape)
+    begin = params_buffer.param_offsets[param.name]
+    end = begin + math.prod(shape)
+    num, kn, nn = shape
+    view = paddle._C_ops.view_shape
+    for i, (d, s, transpose) in enumerate(attrs):
+        rows, cols = (nn, kn) if transpose else (kn, nn)
+        data = fp8.fulls[i]._slice(begin, end)
+        scale = fp8.fulls[k + i]._slice(begin // bn, end // bn)
+        setattr(param, d, view(data, [num * rows, cols]))
+        setattr(param, s, view(scale, [num * rows // block, cols // block]))
+    # Zero-byte storage: read only through the fp8 caches.
+    dims = param.shape
+    share_scratch(param, param.dtype)
+    param._clear_to_zero_allocation()
+    param.get_tensor()._set_dims(dims)
+
+
+def share_scratch(param, dtype):
+    stop_gradient = param.stop_gradient
+    scratch = paddle.empty([1], dtype=dtype)
+    param.get_tensor()._share_data_with(scratch.get_tensor())
+    param.stop_gradient = stop_gradient
+
+
+def clear_fp8_tmp_buffer(params_buffer):
+    for param in params_buffer.fp8.params:
+        for d, s, _ in params_buffer.fp8.quantizer.attrs:
+            for attr in (d, s):
+                if hasattr(param, attr):
+                    delattr(param, attr)
+    params_buffer.fp8.fulls = None
diff --git a/python/paddle/distributed/fsdp/fully_shard.py b/python/paddle/distributed/fsdp/fully_shard.py
index ececa0cded..146be2f748 100644
--- a/python/paddle/distributed/fsdp/fully_shard.py
+++ b/python/paddle/distributed/fsdp/fully_shard.py
@@ -99,10 +99,16 @@ def _fully_shard_manual_parallel(
     layers = module._layers if isinstance(module, MetaParallelBase) else module
     if optimizer is not None:
         _mark_params_for_optimizer(layers, optimizer)
+    # fp8 groups: fp32 main weight iff an fp32 master (default: yes).
+    inner = optimizer
+    while hasattr(inner, "_inner_opt"):
+        inner = inner._inner_opt
+    multi_precision = getattr(inner, "_multi_precision", True)
     layers._fsdp_context = FullyShardFusion(
         layers,
         enable_tensor_fusion_and_overlap=enable_tensor_fusion_and_overlap,
         mp_policy=mp_policy,
+        fp8_main_dtype=paddle.float32 if multi_precision else None,
         prefetch_units=prefetch_units,
         comm_unit_size=comm_unit_size,
     )
diff --git a/python/paddle/distributed/fsdp/fully_shard_fusion.py b/python/paddle/distributed/fsdp/fully_shard_fusion.py
index f6f7dfc149..e1b502e6b1 100644
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -30,6 +30,17 @@ from paddle.distributed.fleet.utils.tensor_fusion_helper import (
     get_current_device_type,
 )
 from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
+from paddle.distributed.fsdp._fp8_utils import (
+    bind_fp8_param,
+    build_ckpt_weight,
+    clear_fp8_tmp_buffer,
+    fp8_shard_numel,
+    get_fp8_tmp_buffer,
+    init_fp8_buffer,
+    quantize_fp8_shard,
+    share_scratch,
+    sync_ckpt_weight_to_main,
+)
 from paddle.distributed.fsdp._fsdp_context import (
     register_fsdp_context,
 )
@@ -140,6 +151,7 @@ class TensorFusionBuffer:
         self.param_offsets = {}
         self.tmp_data_buffer = None
         self.comm_task = None
+        self.fp8 = None  # set by init_fp8_buffer
         self.trainable = params[0].trainable

         for param in params:
@@ -177,7 +189,7 @@ class TensorFusionBuffer:
                 param.stop_gradient = True
                 param.flatten_()
                 paddle.assign(
-                    param,
+                    param if param.dtype == dtype else param.astype(dtype),
                     self.data_buffer._slice(
                         offset,
                         offset + param._numel(),
@@ -303,18 +315,23 @@ class TensorFusionBuffer:
         return self.tmp_data_buffer

     def do_all_gather(self):
-        full, shard = self.get_tmp_buffer(), self.data_buffer
+        if self.fp8 is not None:
+            pairs = get_fp8_tmp_buffer(self)
+        else:
+            pairs = [(self.get_tmp_buffer(), self.data_buffer)]

         def gather():
-            paddle.distributed.stream.all_gather(
-                full,
-                shard,
-                group=self.fsdp_group,
-                sync_op=True,
-                use_calc_stream=True,
-            )
+            for full, shard in pairs:
+                paddle.distributed.stream.all_gather(
+                    full,
+                    shard,
+                    group=self.fsdp_group,
+                    sync_op=True,
+                    use_calc_stream=True,
+                )

-        return run_on_stream(self.comm_stream, (full, shard), gather)
+        tensors = [t for pair in pairs for t in pair]
+        return run_on_stream(self.comm_stream, tensors, gather)

     def do_reduce_scatter(self):
         tmp_buffer = self.get_tmp_buffer()
@@ -344,6 +361,8 @@ class TensorFusionBuffer:
         self.clear_tmp_buffer()

     def clear_tmp_buffer(self):
+        if self.fp8 is not None:
+            clear_fp8_tmp_buffer(self)
         if self.tmp_data_buffer is not None:
             self.tmp_data_buffer._clear_data()
             self.tmp_data_buffer = None
@@ -374,6 +393,7 @@ class FSDPBufferManager:
         model,
         fsdp_unit_layers=None,
         main_grad_dtype=None,
+        fp8_main_dtype=paddle.float32,
     ):
         self.model = model
         self._comm_streams = {}
@@ -452,6 +472,7 @@ class FSDPBufferManager:
                         ("is_firstly_shared", True),
                     )
                 ),
+                getattr(param, "_fsdp_fp8_quantizer", None),
             )
             keyed_params.setdefault(key, []).append(param)

@@ -467,6 +488,7 @@ class FSDPBufferManager:
                 use_muon,
                 _,
                 _,
+                _,
             ) = item[0]
             return (
                 0 if is_tie else (1 if not trainable else 2),
@@ -500,6 +522,7 @@ class FSDPBufferManager:
                 use_muon,
                 hyperball_name,
                 clip_attrs,
+                _,
             ), params in sorted(keyed_params.items(), key=sort_key)
         ]

@@ -521,16 +544,26 @@ class FSDPBufferManager:
                 or muon_align is not None
                 or dense_muon_shard
             )
+            fp8_numel = fp8_shard_numel(params)
+            shard_align = muon_align
+            if fp8_numel:
+                shard_align = math.lcm(muon_align or 1, fp8_numel)
+            # fp8 groups: fp8_main_dtype main weight (None: param dtype).
+            main_dtype = (
+                (fp8_main_dtype or group.dtype) if fp8_numel else group.dtype
+            )
             group.params_buffer = TensorFusionBuffer(
                 gid,
                 params,
                 group.fsdp_group,
-                group.dtype,
+                main_dtype,
                 is_params=True,
                 shard_buffer=shard_buffer,
-                shard_align_numel=muon_align,
+                shard_align_numel=shard_align,
                 comm_stream=self._comm_stream(group.fsdp_group),
             )
+            if fp8_numel:
+                init_fp8_buffer(group.params_buffer, params)
             if not params[0].stop_gradient:
                 grads_shard_buffer = (
                     (not group.use_muon)
@@ -547,10 +580,14 @@ class FSDPBufferManager:
                     else self.main_grad_dtype,
                     grad_div=self._fsdp_group.nranks,
                     shard_buffer=grads_shard_buffer,
-                    shard_align_numel=muon_align,
+                    shard_align_numel=shard_align,
                     comm_stream=self._comm_stream(group.fsdp_group),
                 )
-            group.muon_shard_numel = group.params_buffer.shard_align_numel
+            group.muon_shard_numel = (
+                group.params_buffer.shard_align_numel
+                if group.use_muon
+                else None
+            )
             if group.hyperball_sharded:
                 data_buffer = group.params_buffer.data_buffer
                 data_buffer._hyperball_group = (
@@ -689,6 +726,8 @@ class FSDPCommManager:

     def _count_gathered(self, params_buffer):
         # A sharded buffer just got its full storage allocated.
+        if params_buffer.fp8 is not None:
+            return
         if self.prefetch_by_size:
             self.gathered_numel += params_buffer.total_buffer_size
         else:
@@ -698,7 +737,7 @@ class FSDPCommManager:
         # USING / READY / SYNCING-waited -> FREED, keeping the accounting.
         params_buffer.clear_tmp_buffer()
         params_buffer.status = BufferState.FREED
-        if not params_buffer.is_sharded:
+        if not params_buffer.is_sharded or params_buffer.fp8 is not None:
             return
         if self.prefetch_by_size:
             self.gathered_numel -= params_buffer.total_buffer_size
@@ -712,7 +751,10 @@ class FSDPCommManager:
             for gid_idx, group in enumerate(self.buffer_manager.buffer_groups):
                 if gid_idx in keep:
                     continue
-                if not group.params_buffer.is_sharded:
+                if (
+                    not group.params_buffer.is_sharded
+                    or group.params_buffer.fp8 is not None
+                ):
                     continue
                 if group.params_buffer.status == BufferState.READY:
                     group.params_buffer.status = BufferState.FREED
@@ -757,6 +799,8 @@ class FSDPCommManager:
             if gid in keep:
                 continue
             params_buffer = self._params_buffers[gid]
+            if params_buffer.fp8 is not None:
+                continue
             if params_buffer.status == BufferState.READY:
                 self._free_params_buffer(params_buffer)
                 if self.gathered_numel + need_numel <= limit:
@@ -897,6 +941,9 @@ class FSDPCommManager:
                     self._count_gathered(params_buffer)
                 params_buffer.status = BufferState.USING

+            if params_buffer.fp8 is not None:
+                bind_fp8_param(params_buffer, param)
+                continue
             # Bind the unsharded param to the real param
             offset = params_buffer.param_offsets[param.name]
             tmp_param = paddle._C_ops.view_slice(
@@ -956,6 +1003,10 @@ class FSDPCommManager:
         self._last_backward_unit_id = None
         for group in self.buffer_manager.buffer_groups:
             params_buffer = group.params_buffer
+            if params_buffer.fp8 is not None:
+                params_buffer.fp8.quantized = False
+                params_buffer.fp8.weight_copy = None
+                clear_fp8_tmp_buffer(params_buffer)
             # A prefetch issued at the end of backward gathers the pre-step
             # shard; drop it so the next forward gathers the updated one.
             if params_buffer.status == BufferState.SYNCING:
@@ -1266,6 +1317,7 @@ class FullyShardFusion:
         fsdp_unit_layers=None,
         enable_tensor_fusion_and_overlap=True,
         mp_policy=None,
+        fp8_main_dtype=paddle.float32,
         prefetch_units=1,
         comm_unit_size=None,
     ):
@@ -1275,6 +1327,7 @@ class FullyShardFusion:
             self.model,
             fsdp_unit_layers,
             None if mp_policy is None else mp_policy.reduce_dtype,
+            fp8_main_dtype=fp8_main_dtype,
         )
         self.comm_manager = FSDPCommManager(
             self.buffer_manager,
@@ -1357,6 +1410,9 @@ class FullyShardFusion:
         hcg = self.buffer_manager.hcg
         for group in self.buffer_manager.buffer_groups:
             params_buffer = group.params_buffer
+            weight = params_buffer.data_buffer
+            if params_buffer.fp8 is not None:
+                weight = build_ckpt_weight(params_buffer, group.dtype)
             param_slice_info = {}
             # (nranks, rank) of the expert parallel group, (1, 0) without EP.
             ep_nranks, ep_rank = 1, 0
@@ -1411,7 +1467,7 @@ class FullyShardFusion:
                 param_slice_info[struct_names[0]] = slice_info
                 for struct_name in struct_names:
                     result[struct_name] = self._create_sharded_weight(
-                        struct_name, params_buffer.data_buffer, slice_info
+                        struct_name, weight, slice_info
                     )
             self._shard_descs[params_buffer.data_buffer.name] = param_slice_info
         return result
@@ -1513,8 +1569,7 @@ class FullyShardFusion:
                     if self._bind_muon_shard_view(param, params_buffer) is None:
                         continue
                     optimizer._create_accumulators(block, [param])
-                    param.get_tensor()._set_dims(param._muon_full_shape)
-                    param.original_shape = list(param._muon_full_shape)
+                    self._unbind_muon_shard_view(param, group)
                 continue
             owner = group.muon_owner_rank
             if owner is not None and group.fsdp_group.ranks[owner] != my_rank:
@@ -1590,6 +1645,8 @@ class FullyShardFusion:
         bm = self.buffer_manager
         params = [p for p in params if p.name in bm.param_to_buffer_id]
         gids = {bm.param_to_buffer_id[p.name] for p in params}
+        if any(bm.buffer_groups[g].params_buffer.fp8 is not None for g in gids):
+            raise RuntimeError("unshard_params: fp8 groups have no bf16 weight")
         was_freed = {
             gid
             for gid in gids
@@ -1706,6 +1763,18 @@ class FullyShardFusion:
                         key, tensor, slice_info
                     )

+        for group in self.buffer_manager.buffer_groups:
+            if group.params_buffer.fp8 is None:
+                continue
+            main = group.params_buffer.data_buffer
+            if main.dtype == group.dtype:  # no fp32 main, no master to save
+                continue
+            for struct_name, slice_info in self._shard_descs[main.name].items():
+                key = f"{struct_name}.w_0"
+                sharded_state[key] = self._create_sharded_weight(
+                    key, main, slice_info
+                )
+
         if master_weights:
             for base_name, tensor in master_weights.items():
                 desc = self._muon_state_descs.get(base_name)
@@ -1876,6 +1945,16 @@ class FullyShardFusion:
                 out.append((param, grad))
         return out

+    @staticmethod
+    def _unbind_muon_shard_view(param, group):
+        # Back to a storage-less param-dtype shell of the full shape.
+        full = param._muon_full_shape
+        if group.params_buffer.fp8 is not None:
+            share_scratch(param, group.dtype)
+            param._clear_data()
+        param.get_tensor()._set_dims(full)
+        param.original_shape = list(full)
+
     def _bind_muon_shard_view(self, param, params_buffer):
         offset = params_buffer.param_offsets[param.name]
         begin = max(offset, params_buffer.shard_start)
@@ -1924,11 +2003,8 @@ class FullyShardFusion:
         for group in self.buffer_manager.buffer_groups:
             if group.muon_shard_numel is not None:
                 for param in group.params:
-                    full = getattr(param, "_muon_full_shape", None)
-                    if full is None:
-                        continue
-                    param.get_tensor()._set_dims(full)
-                    param.original_shape = list(full)
+                    if getattr(param, "_muon_full_shape", None) is not None:
+                        self._unbind_muon_shard_view(param, group)
                 continue
             if not group.use_muon or group.muon_owner_rank is None:
                 continue
@@ -2021,6 +2097,10 @@ class FullyShardFusion:

             @paddle.autograd.no_grad()
             def all_gather_comm(*_):
+                for group in self.buffer_manager.buffer_groups:
+                    if group.params_buffer.fp8 is not None:
+                        sync_ckpt_weight_to_main(group.params_buffer)
+                        quantize_fp8_shard(group.params_buffer)
                 params = sublayers.parameters(include_sublayers=recursive)
                 comm_manager.all_gather_params(params)
                 self._bind_expert_main_grads(params)
```

### T3 · PaddleFleet #2371 fp8 param gather（Fleet 侧）

```diff
diff --git a/src/paddlefleet/trainer/trainer_callback.py b/src/paddlefleet/trainer/trainer_callback.py
index b4f13a49..bac3547b 100644
--- a/src/paddlefleet/trainer/trainer_callback.py
+++ b/src/paddlefleet/trainer/trainer_callback.py
@@ -1048,6 +1048,16 @@ class FP8QuantWeightCallback(TrainerCallback):
     Callback for FP8 weight quantization during training
     """

+    def on_init_end(self, args, state, control, **kwargs):
+        """
+        fsdp: mark expert weights for fp8 param gather; runs before fully_shard.
+        """
+        if args.using_sonic_moe or not _fsdp_owns_param_storage(args):
+            return
+        from paddlefleet.transformer.moe.fp8_utils import mark_fsdp_fp8_params
+
+        mark_fsdp_fp8_params(args, kwargs["model"])
+
     def on_step_begin(self, args, state, control, **kwargs):
         """
         Quantize expert weights to FP8 before each training step
@@ -1073,7 +1083,11 @@ class FP8QuantWeightCallback(TrainerCallback):
                 # fsdp: weights are storage-less shells here and the plain
                 # optimizer has no clear_param_storage; gather layer by layer
                 # to quantize, and leave the bf16 shards to fsdp.
-                _fsdp_fp8_quant_weight(model, True, quant_transpose=False)
+                if not any(
+                    getattr(p, "_fsdp_fp8_quantizer", None) is not None
+                    for p in model.parameters()
+                ):
+                    _fsdp_fp8_quant_weight(model, True, quant_transpose=False)
                 moe_weights_name = _fsdp_expert_master_weight_names()
             else:
                 model.fp8_quant_weight(True, quant_transpose=False)
diff --git a/src/paddlefleet/transformer/moe/fp8_utils.py b/src/paddlefleet/transformer/moe/fp8_utils.py
index ec3ac6ae..beb1f8d3 100644
--- a/src/paddlefleet/transformer/moe/fp8_utils.py
+++ b/src/paddlefleet/transformer/moe/fp8_utils.py
@@ -236,6 +236,42 @@ def fused_stack_quant_without_cache(
     return w, scale


+def mark_fsdp_fp8_params(args, model):
+    """Let fsdp gather grouped fp8 weights."""
+    if not getattr(args, "offline_quant_expert_weight", False):
+        return
+    try:
+        import paddle.distributed.fsdp._fp8_utils  # noqa: F401
+    except ImportError:
+        return  # paddle without fsdp fp8 gather: keep per-step pre-quant
+    if any(getattr(m, "use_ue8m0", False) for m in model.sublayers()):
+        raise NotImplementedError("fsdp fp8 param gather: no ue8m0 support.")
+    from paddlefleet.transformer.moe.moe_expert import (
+        GroupedMLPExpert,
+        SonicMoEExpert,
+    )
+
+    # fsdp quantizes each shard to fp8 and gathers it into these caches.
+    def quantizer(weights, transpose):
+        return fused_stack_quant_without_cache(
+            [weights[i] for i in range(weights.shape[0])], transpose=transpose
+        )
+
+    quantizer.block = 128
+    # (data attr, scale attr, transpose): forward reads the transposed copy.
+    keep_transpose = getattr(args, "using_cache_weight_quant_transpose", True)
+    quantizer.attrs = (
+        ("fp8_weight_stacked", "fp8_scale_stacked", False),
+        ("fp8_weight_stacked_transpose", "fp8_scale_stacked_transpose", True),
+    )[: 2 if keep_transpose else 1]
+    for layer in model.sublayers():
+        if isinstance(layer, GroupedMLPExpert) and not isinstance(
+            layer, SonicMoEExpert
+        ):
+            layer.weight1._fsdp_fp8_quantizer = quantizer
+            layer.weight2._fsdp_fp8_quantizer = quantizer
+
+
 def fused_stack_quant(
     expert_weight_list, transpose=False, use_ue8m0=False, num_expert=None
 ):
```

### T4 · PR #79899 reduce_scatter dtype

```diff
diff --git a/python/paddle/distributed/fsdp/fully_shard.py b/python/paddle/distributed/fsdp/fully_shard.py
index 146be2f748..c557698b6e 100644
--- a/python/paddle/distributed/fsdp/fully_shard.py
+++ b/python/paddle/distributed/fsdp/fully_shard.py
@@ -42,6 +42,7 @@ class MixedPrecisionPolicy:
     reduce_dtype: paddle.dtype | None = None
     output_dtype: paddle.dtype | None = None
     cast_forward_inputs: bool = True
+    grad_comm_dtype: paddle.dtype | None = None
 
 
 # @dataclass
@@ -133,6 +134,8 @@ def _fully_shard_auto_parallel(
     prefetch_units=1,
     comm_unit_size=None,
 ):
+    if mp_policy.grad_comm_dtype is not None:
+        raise NotImplementedError("grad_comm_dtype is manual parallel only.")
     module._fsdp_context = FullyShardAuto(
         module,
         mesh,
@@ -177,6 +180,8 @@ def fully_shard(
             Defaults to None.
         mp_policy (MixedPrecisionPolicy | None, optional): Mixed precision policy configuration.
             If None, creates a default MixedPrecisionPolicy. Defaults to None.
+            ``reduce_dtype`` is the grad accumulation dtype. ``grad_comm_dtype``
+            is the grad reduce-scatter dtype (float32 or bfloat16), None means ``reduce_dtype``.
         offload_policy (OffloadPolicy | None, optional): Offload policy configuration for CPU offloading.
             If None, creates a default OffloadPolicy. Defaults to None.
         ignored_params (Iterable[paddle.Tensor] | None, optional): Parameters that should not be sharded.
@@ -227,6 +232,10 @@ def fully_shard(
     """
     if mp_policy is None:
         mp_policy = MixedPrecisionPolicy()
+    if mp_policy.grad_comm_dtype not in (None, paddle.float32, paddle.bfloat16):
+        raise ValueError(
+            f"grad_comm_dtype must be float32 or bfloat16, got {mp_policy.grad_comm_dtype}."
+        )
     if offload_policy is None:
         offload_policy = OffloadPolicy()
     ignored_params_set: set[paddle.Tensor] = (
diff --git a/python/paddle/distributed/fsdp/fully_shard_fusion.py b/python/paddle/distributed/fsdp/fully_shard_fusion.py
index e1b502e6b1..14a5cb1dbd 100644
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -130,6 +130,7 @@ class TensorFusionBuffer:
         grad_div=None,
         shard_buffer=True,
         shard_align_numel=None,
+        grad_comm_dtype=None,
         comm_stream=None,
     ):
         # Calculate total buffer size needed (with padding)
@@ -143,6 +144,10 @@ class TensorFusionBuffer:
         self.main_grad_dtype = (
             main_grad_dtype if main_grad_dtype is not None else dtype
         )
+        self.grad_comm_dtype = (
+            grad_comm_dtype if grad_comm_dtype != self.main_grad_dtype else None
+        )
+        self._rs_comm_buffer = None
         self.grad_div = grad_div if grad_div is not None else fsdp_group.nranks
         self.grad_accum_steps = 1
         self.grad_scale = 1.0 / self.grad_div
@@ -342,21 +347,31 @@ class TensorFusionBuffer:
         else:
             tmp_buffer.scale_(self.grad_scale)
             op = paddle.distributed.ReduceOp.SUM
+        if self.grad_comm_dtype is None:
+            self._rs_comm_buffer = tmp_buffer
+        else:
+            self._rs_comm_buffer = paddle.cast(tmp_buffer, self.grad_comm_dtype)
+            self.tmp_data_buffer = None
+        full = self._rs_comm_buffer
+        shard = full._slice(0, self.data_buffer.shape[0])
 
         def reduce():
             paddle.distributed.stream.reduce_scatter(
-                tmp_buffer._slice(0, self.data_buffer.shape[0]),
-                tmp_buffer,
+                shard,
+                full,
                 op=op,
                 group=self.fsdp_group,
                 sync_op=True,
                 use_calc_stream=True,
             )
 
-        return run_on_stream(self.comm_stream, (tmp_buffer,), reduce)
+        return run_on_stream(self.comm_stream, (full,), reduce)
 
     def accumulate_reduced_grad(self):
-        shard = self.get_tmp_buffer()._slice(0, self.data_buffer.shape[0])
+        shard = self._rs_comm_buffer._slice(0, self.data_buffer.shape[0])
+        self._rs_comm_buffer = None
+        if shard.dtype != self.data_buffer.dtype:
+            shard = shard.astype(self.data_buffer.dtype)
         self.data_buffer.add_(shard)
         self.clear_tmp_buffer()
 
@@ -393,9 +408,11 @@ class FSDPBufferManager:
         model,
         fsdp_unit_layers=None,
         main_grad_dtype=None,
+        grad_comm_dtype=None,
         fp8_main_dtype=paddle.float32,
     ):
         self.model = model
+        self.grad_comm_dtype = grad_comm_dtype
         self._comm_streams = {}
         self.hcg = fleet.get_hybrid_communicate_group()
         self._fsdp_group = self.hcg.get_sharding_parallel_group()
@@ -581,6 +598,7 @@ class FSDPBufferManager:
                     grad_div=self._fsdp_group.nranks,
                     shard_buffer=grads_shard_buffer,
                     shard_align_numel=shard_align,
+                    grad_comm_dtype=self.grad_comm_dtype,
                     comm_stream=self._comm_stream(group.fsdp_group),
                 )
             group.muon_shard_numel = (
@@ -1102,6 +1120,7 @@ class FSDPCommManager:
         grads_buffer.comm_task = grads_buffer.do_reduce_scatter()
         self.grad_reduce_queue.append(grads_buffer)
         self._grad_queue_numel += grads_buffer.total_buffer_size
+        self._release_reduced_grad_views(grads_buffer)
         if not self.enable_overlap:
             self._wait_for_grad_comm(queue_limit=0)
 
@@ -1327,6 +1346,9 @@ class FullyShardFusion:
             self.model,
             fsdp_unit_layers,
             None if mp_policy is None else mp_policy.reduce_dtype,
+            grad_comm_dtype=None
+            if mp_policy is None
+            else mp_policy.grad_comm_dtype,
             fp8_main_dtype=fp8_main_dtype,
         )
         self.comm_manager = FSDPCommManager(
```

### T5 · PR #79887 reshard_policy / reshard_filter

```diff
diff --git a/python/paddle/distributed/fsdp/fully_shard.py b/python/paddle/distributed/fsdp/fully_shard.py
index c557698b6e..5e7ae6fa90 100644
--- a/python/paddle/distributed/fsdp/fully_shard.py
+++ b/python/paddle/distributed/fsdp/fully_shard.py
@@ -96,6 +96,8 @@ def _fully_shard_manual_parallel(
     grad_accum_steps=1,
     prefetch_units=1,
     comm_unit_size=None,
+    reshard_policy="after_forward",
+    reshard_filter="all",
 ):
     layers = module._layers if isinstance(module, MetaParallelBase) else module
     if optimizer is not None:
@@ -112,6 +114,8 @@ def _fully_shard_manual_parallel(
         fp8_main_dtype=paddle.float32 if multi_precision else None,
         prefetch_units=prefetch_units,
         comm_unit_size=comm_unit_size,
+        reshard_policy=reshard_policy,
+        reshard_filter=reshard_filter,
     )
     layers._fsdp_context.set_grad_accum_steps(grad_accum_steps)
     if optimizer is not None:
@@ -159,6 +163,8 @@ def fully_shard(
     grad_accum_steps: int = 1,
     prefetch_units: int = 1,
     comm_unit_size: int | None = None,
+    reshard_policy: str = "after_forward",
+    reshard_filter: str | Callable[..., bool] = "all",
 ) -> paddle.nn.Layer:
     """
     Apply fully sharded data parallel (FSDP) to the given module.
@@ -205,6 +211,27 @@ def fully_shard(
             elements ahead and at most ``comm_unit_size`` grad elements
             stay in reduce_scatter. 0 disables prefetch. Defaults to None
             (prefetch by layer).
+        reshard_policy (str, optional): Manual parallel mode only. When the
+            all-gathered (unsharded) parameters selected by ``reshard_filter``
+            are freed again.
+
+            - ``"after_forward"``: after forward; backward all-gathers them
+              again. Lowest memory.
+            - ``"after_backward"``: kept from forward until their unit's
+              backward is done, so backward needs no all-gather.
+            - ``"after_step"``: also kept across the micro-batches of a
+              gradient accumulation step and freed before the optimizer step,
+              so each layer is all-gathered once per step.
+
+            The last two cost one full copy of each kept layer's local
+            parameters. Defaults to ``"after_forward"``.
+        reshard_filter (str | Callable, optional): Manual parallel mode only.
+            Which parameters ``reshard_policy`` applies to; the others use
+            ``"after_forward"``. ``"all"``: all sharded parameters;
+            ``"expert"``: expert parameters (``is_moe_param``) only;
+            ``"dense"``: non-expert parameters only. A callable for per-layer
+            selection is reserved and not supported yet. Defaults to
+            ``"all"``.
 
     Returns:
         module: A wrapper module that applies FSDP to the input module.
@@ -260,4 +287,8 @@ def fully_shard(
     if in_auto_parallel_mode():
         return _fully_shard_auto_parallel(*args)
     else:
-        return _fully_shard_manual_parallel(*args)
+        return _fully_shard_manual_parallel(
+            *args,
+            reshard_policy=reshard_policy,
+            reshard_filter=reshard_filter,
+        )
diff --git a/python/paddle/distributed/fsdp/fully_shard_fusion.py b/python/paddle/distributed/fsdp/fully_shard_fusion.py
index 14a5cb1dbd..0e2372f2b9 100644
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -676,15 +676,39 @@ class FSDPBufferManager:
 
 
 class FSDPCommManager:
+    RESHARD_POLICIES = ("after_forward", "after_backward", "after_step")
+    RESHARD_FILTERS = ("all", "expert", "dense")
+
     def __init__(
         self,
         buffer_manager,
         enable_overlap=True,
         prefetch_units=1,
         comm_unit_size=None,
+        reshard_policy="after_forward",
+        reshard_filter="all",
     ):
+        if reshard_policy not in self.RESHARD_POLICIES:
+            raise ValueError(
+                f"reshard_policy must be one of {self.RESHARD_POLICIES}, "
+                f"got {reshard_policy!r}"
+            )
+        if callable(reshard_filter):
+            raise NotImplementedError(
+                "a callable reshard_filter (per-layer selection) is not "
+                f"supported yet; use one of {self.RESHARD_FILTERS}"
+            )
+        if reshard_filter not in self.RESHARD_FILTERS:
+            raise ValueError(
+                f"reshard_filter must be one of {self.RESHARD_FILTERS}, "
+                f"got {reshard_filter!r}"
+            )
         self.buffer_manager = buffer_manager
         self.enable_overlap = enable_overlap
+        self.reshard_policy = reshard_policy
+        self.reshard_filter = reshard_filter
+        for group in buffer_manager.buffer_groups:
+            group.params_buffer.keep_gathered = self._keep_after_forward(group)
         self.grad_reduce_queue = []
         self.prefetch_by_size = comm_unit_size is not None
 
@@ -742,9 +766,30 @@ class FSDPCommManager:
     # buffer_cnt_in_using (by layer) and gathered_numel (by size).
     # ------------------------------------------------------------------
 
+    def _in_reshard_filter(self, group):
+        if self.reshard_filter == "expert":
+            return group.is_expert_param
+        if self.reshard_filter == "dense":
+            return not group.is_expert_param
+        return True
+
+    def _reshard_policy_of(self, group):
+        if self._in_reshard_filter(group):
+            return self.reshard_policy
+        return "after_forward"
+
+    def _keep_after_forward(self, group):
+        return self._reshard_policy_of(group) != "after_forward"
+
+    def _keep_after_backward(self, group):
+        return self._reshard_policy_of(group) == "after_step"
+
     def _count_gathered(self, params_buffer):
-        # A sharded buffer just got its full storage allocated.
-        if params_buffer.fp8 is not None:
+        # A sharded buffer just got its full storage allocated. fp8 and kept
+        # buffers stay outside the gather budget (by layer or by size).
+        if params_buffer.fp8 is not None or getattr(
+            params_buffer, "keep_gathered", False
+        ):
             return
         if self.prefetch_by_size:
             self.gathered_numel += params_buffer.total_buffer_size
@@ -755,7 +800,11 @@ class FSDPCommManager:
         # USING / READY / SYNCING-waited -> FREED, keeping the accounting.
         params_buffer.clear_tmp_buffer()
         params_buffer.status = BufferState.FREED
-        if not params_buffer.is_sharded or params_buffer.fp8 is not None:
+        if (
+            not params_buffer.is_sharded
+            or params_buffer.fp8 is not None
+            or getattr(params_buffer, "keep_gathered", False)
+        ):
             return
         if self.prefetch_by_size:
             self.gathered_numel -= params_buffer.total_buffer_size
@@ -767,7 +816,7 @@ class FSDPCommManager:
         while self.buffer_cnt_in_using >= self.gather_buffer_limit:
             found = False
             for gid_idx, group in enumerate(self.buffer_manager.buffer_groups):
-                if gid_idx in keep:
+                if gid_idx in keep or self._keep_after_forward(group):
                     continue
                 if (
                     not group.params_buffer.is_sharded
@@ -817,7 +866,9 @@ class FSDPCommManager:
             if gid in keep:
                 continue
             params_buffer = self._params_buffers[gid]
-            if params_buffer.fp8 is not None:
+            if params_buffer.fp8 is not None or getattr(
+                params_buffer, "keep_gathered", False
+            ):
                 continue
             if params_buffer.status == BufferState.READY:
                 self._free_params_buffer(params_buffer)
@@ -1144,6 +1195,9 @@ class FSDPCommManager:
                 continue
             if group.fsdp_unit_id <= unit_id:
                 continue
+            if self._keep_after_backward(group):
+                # Freed by reset_params_buffer_status at the end of the step.
+                continue
             params_buffer = group.params_buffer
             if not params_buffer.is_sharded:
                 continue
@@ -1339,6 +1393,8 @@ class FullyShardFusion:
         fp8_main_dtype=paddle.float32,
         prefetch_units=1,
         comm_unit_size=None,
+        reshard_policy="after_forward",
+        reshard_filter="all",
     ):
         self.model = model
         self.mp_policy = mp_policy
@@ -1356,6 +1412,8 @@ class FullyShardFusion:
             enable_overlap=enable_tensor_fusion_and_overlap,
             prefetch_units=prefetch_units,
             comm_unit_size=comm_unit_size,
+            reshard_policy=reshard_policy,
+            reshard_filter=reshard_filter,
         )
         self.register_tensor_fusion_hooks(self.model)
         register_fsdp_context(self)
```

### T6 · PR #79903 sync_grads_each_microbatch

```diff
diff --git a/python/paddle/distributed/fsdp/fully_shard_fusion.py b/python/paddle/distributed/fsdp/fully_shard_fusion.py
index e1b502e6b1..043242dfb3 100644
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -683,6 +683,10 @@ class FSDPCommManager:
         self.buffer_cnt_in_using = 0
         self.need_zero_grads = True
         self._last_backward_unit_id = None
+        # Experimental: False syncs grads only on the last micro-batch.
+        self.sync_grads_each_microbatch = True
+        self.grad_accum_steps = 1
+        self._backward_cnt = 0
 
         # By size: gathered param elements and queued grad elements.
         self.gathered_numel = 0
@@ -884,6 +888,11 @@ class FSDPCommManager:
                 req_gids[0]
             ].fsdp_unit_id
             if unit_id != self._last_backward_unit_id:
+                prev_unit_id = self._last_backward_unit_id
+                if unit_id is not None and (
+                    prev_unit_id is None or unit_id > prev_unit_id
+                ):
+                    self._backward_cnt += 1
                 self._last_backward_unit_id = unit_id
                 self._flush_expert_grads_after_unit(unit_id)
                 self._release_params_after_unit(unit_id)
@@ -1097,6 +1106,12 @@ class FSDPCommManager:
             or grads_buffer in self.grad_reduce_queue
         ):
             return
+        if (
+            not self.sync_grads_each_microbatch
+            and self._backward_cnt < self.grad_accum_steps
+        ):
+            self._release_reduced_grad_views(grads_buffer)
+            return
         # Grad queue mechanism: wait and release completed reduce_scatter async tasks
         self._wait_for_grad_comm(next_numel=grads_buffer.total_buffer_size)
         grads_buffer.comm_task = grads_buffer.do_reduce_scatter()
@@ -1232,6 +1247,7 @@ class FSDPCommManager:
                 mg._clear_data()
                 param.main_grad = None
                 param._main_grad_addr = None
+        self._backward_cnt = 0
 
 
 def _find_exposed_param_names(outputs, own_param_names, _depth=0):
@@ -1797,6 +1813,7 @@ class FullyShardFusion:
 
     def set_grad_accum_steps(self, steps):
         steps = max(int(steps), 1)
+        self.comm_manager.grad_accum_steps = steps
         for group in self.buffer_manager.buffer_groups:
             grads_buffer = group.grads_buffer
             if grads_buffer is None:
```
