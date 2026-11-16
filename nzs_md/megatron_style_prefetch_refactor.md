# Paddle FSDP 预取重构：从按 unit 计数改为 Megatron 式按大小（bucket）预取

代码：`Paddle/build/python/paddle/distributed/fsdp/fully_shard_fusion.py`（`FSDPCommManager`）、`fully_shard.py`（参数透传）。
快照与 diff：`perf_stages/megatron_prefetch/`（`fusion.diff`、`fully_shard.diff`，基线为 `perf_stages/backup/`，即 S7/S8 版本）。
Megatron 更完整的实现分析见 `megatron_fsdp_overlap_prefetch.md`，本文只展开与预取量控制相关的部分。

## 1. Megatron-FSDP 预取实现分析

### 1.1 粒度：bucket 与 bucket group

- bucket = 一个 `ParameterGroup`，即一段连续的 fused buffer（同 dtype / 同 FSDP unit / 同 expert 属性；非 unit 参数按 `bucket_size`≈40M 元素再切）。
- bucket group = 同一 `(fsdp_unit_id, is_expert)` 的 bucket 集合（`bucket_to_bucket_group`），一次 coalesced NCCL 调用发出；预取扩展时整组加入。
- 预取顺序是静态的 bucket id 顺序：前向 +1、反向 −1（`PrefetchOrder`），不做运行时 trace。

### 1.2 三个参数及默认值（`megatron_fsdp.py:462-487`）

```python
if suggested_communication_unit_size is None:
    # optim_grads_params（ZeRO-3）：每个 FSDP unit 平均参数量 × 2，上限 1e9
    suggested_communication_unit_size = total_param_elements // total_fsdp_module * 2
    suggested_communication_unit_size = min(1_000_000_000, suggested_communication_unit_size)
self.suggested_RS_queue_capacity = suggested_communication_unit_size
self.suggested_AG_prefetch_size = suggested_communication_unit_size // 2
```

| 参数 | 含义 | 默认 | 对应的 unit 数（同构层） |
| --- | --- | --- | --- |
| `suggested_communication_unit_size` | "当前 unit + 下一个 unit" 的元素量，派生另外两个 | 2 × avg_unit | — |
| `suggested_AG_prefetch_size` | 在请求的 bucket 之外再 all_gather 的元素量 | unit_size // 2 | ≈1 个 unit |
| `suggested_RS_queue_capacity` | 飞行中 reduce_scatter 的梯度元素量上限 | unit_size | ≈2 个 unit |

### 1.3 AG 预取扩展 `_extend_by_prefetch_size`（`param_and_grad_buffer.py:4676`）

```python
bucket_id = next_bucket_id()             # 紧邻已选集合的下一个 id（前向 +1 / 反向 -1）
while bucket_id is not None:
    prefetched_size = sum(size(b) for b in result) - base_all_gather_size
    if prefetched_size >= suggested_prefetch_size:   # 先判断，再加
        break
    if fsdp_double_buffer: ...                        # 仅双缓冲模式下限 2 个 unit
    result.extend(bucket_to_bucket_group[bucket_id])  # 整个 bucket group 加入
    bucket_id = next_bucket_id()
```

要点：
1. 大小用 bucket 的完整（未分片、含 padding）元素数 `bucket_index.size`。
2. "先判断再加整组"：只要阈值 > 0，至少预取一个 bucket group；最后一组可能让预取量超过阈值。
3. 已经 gather 过的 bucket 也计入预取量，扩展后再按状态过滤，只对 `EMPTY/PRESERVED` 的发 AG（`all_gather_params:4874`），重复命中自然去重。
4. 预取深度由"元素量"决定，与 unit 数解耦：层大小不均（dense 层 / MoE 层、embedding）时自动调节。

### 1.4 RS 队列 `wait_for_previous_grad_reduce`（`:4135`）

发新 RS 之前，按队列里梯度 bucket 的元素总量与 `suggested_RS_queue_capacity` 比较，超出则从队头依次 wait + 释放：

```python
queue_space = sum(size(b) for b in grad_reduce_queue)
while queue_space > suggested_queue_capacity:
    event, free_up_grad_bucket, b = grad_reduce_queue.pop(0)
    event.wait(); free_up_grad_bucket(); queue_space -= size(b)
```

RS 触发：bucket group 内所有参数梯度到齐（`get_ready_bucket_group_for_reduction`）。

### 1.5 释放

前向用完立即释放（`release_bucket`），重计算 / root pre-backward 场景走 lazy 释放；双缓冲模式另有 2 unit 上限。可选独立 AG 流 + 独立 AG 通信器（`--create-all-gather-group`）。

## 2. Paddle FSDP 原实现（S7/S8）与差异

| 维度 | Paddle 原实现 | Megatron |
| --- | --- | --- |
| 预取单位 | `prefetch_units`（unit 个数，默认 2） | `suggested_AG_prefetch_size`（元素数） |
| 已 gather 显存上限 | `gather_buffer_limit = groups_per_unit × (prefetch_units+1)`（buffer 个数） | 无全局上限（双缓冲模式除外） |
| RS 队列 | `grad_queue_limit = groups_per_unit × prefetch_units`（个数） | `suggested_RS_queue_capacity`（元素数） |
| 扩展方式 | `_next_buffer_id`：按 gid ±1 走，数到 N 个不同 unit_id 停 | 按大小，先判后加整组 |
| 计数 | `buffer_cnt_in_using`（个数，embedding 与一层 expert 同权） | bucket 元素数 |

按个数计数的问题：一个 unit 的 buffer 数与大小无关（embedding/LM head 一个 buffer 可能比整层还大；dense 层与 MoE 层大小差一个量级），`gather_buffer_limit` 无法对应真实显存，`prefetch_units` 也无法按通信量调优。

## 3. 重构方案

保留 Paddle 的结构（fused BufferGroup、BufferState 状态机与 READY lazy 复用、侧流 `_async_collective`、按 unit 结构触发 RS），只把"预取多少 / 保留多少 / RS 排队多少"改成 Megatron 语义：

1. **参数**：与 Megatron 一样只暴露 `suggested_communication_unit_size`（`FullyShardFusion` / `FSDPCommManager` / `fully_shard(...)`，取代 `prefetch_units`），内部派生 `suggested_AG_prefetch_size = unit // 2`、`suggested_RS_queue_capacity = unit`；未传入时用 Megatron 默认值。PaddleFleet 侧通过 yaml / `--kwargs` 配置（见 4.1）。rank0 打印一行 `FSDP prefetch: ...`。
2. **大小口径**：sharded params buffer 的 `total_buffer_size`（完整、含 padding）；非分片 buffer 记 0（不需要 AG）。`avg_unit_numel` = 各 FSDP unit 的 sharded buffer 大小之和的平均。
3. **bucket group**：同 `(fsdp_unit_id, is_expert)` 的 sharded buffer 组成一组；非 unit / 非分片 buffer 自成一组。
4. **`_extend_by_prefetch_size`**：照搬 Megatron——从请求集合的下一个 gid 开始按执行顺序走，先判断已预取量是否 ≥ 阈值，再整组加入；反向跳过不可训练 buffer；已 gather 的也计入大小，发 AG 时只对 FREED 的发。阈值为 0 关闭预取。
5. **显存账**：`buffer_cnt_in_using` → `gathered_numel`（SYNCING/USING/READY 的 sharded buffer 元素和）。窗口上限 `limit = max(unit_size, 本次请求+预取目标的大小)`；发 AG 前若 `gathered + new > limit`，按 gid 顺序释放窗口外的 READY buffer（Megatron 没有 READY 复用，这一步是为保留 Paddle 的 lazy 复用而加的上限）。`limit` 只控制 READY 保留量，不阻止预取（第一版曾用它阻止预取，导致回退，见 6.1）。所有释放路径统一走 `_free_params_buffer`，保证计数精确。
6. **RS 队列**：`_wait_for_grad_comm()` 默认按队列内梯度 buffer 元素总量与 `suggested_RS_queue_capacity` 比较，在发新 RS 之前出队等待（= `wait_for_previous_grad_reduce`）；显式 `queue_limit=k` 的按个数模式保留（前向逐层回收、`finish_grads_sync` 全清、`_ensure_grads_writable`）。
7. **不纳入本次重构（仅记录）**：独立 AG 通信器 / 独立 AG、RS 流、coalescing、非 unit 参数按 40M 切 bucket、整组到齐才触发 RS（Paddle 保持按 unit 边界的结构触发）。

## 4. 实现

### 4.1 接口

```python
fully_shard(model, ..., suggested_communication_unit_size=None)
FullyShardFusion(model, ..., suggested_communication_unit_size=None)
# AG = unit // 2，RS = unit，与 Megatron 一致，不单独暴露
```

PaddleFleet `TrainingArguments.suggested_communication_unit_size`（默认 None）透传给 `fully_shard`，yaml 或 `--kwargs suggested_communication_unit_size=9.4e8` 配置（OmegaConf 解析成 float，`FSDPCommManager` 内取 int）。早期版本的 `PADDLE_FSDP_*` 环境变量和独立的 AG / RS 参数已删除。

`prefetch_units` 参数删除（仓内无外部调用方：PaddleFleet `trainer.py` 只调 `fully_shard`，不传预取参数）。

### 4.2 关键代码（`FSDPCommManager`）

默认值与 bucket group：

```python
self._gid_to_bucket_group = []          # gid -> 同 (unit, is_expert) 的 sharded gid 列表
...
unit_size = suggested_communication_unit_size
if unit_size is None:
    unit_size = self.avg_unit_numel * 2 or 1_000_000_000
    unit_size = min(1_000_000_000, unit_size)
self.suggested_RS_queue_capacity = unit_size
self.suggested_AG_prefetch_size = unit_size // 2
self.gathered_numel = 0
```

预取扩展（Megatron `_extend_by_prefetch_size` 的移植）：

```python
while 0 <= candidate < len(groups):
    if candidate in selected: candidate += step; continue
    if is_backward and not groups[candidate].params_buffer.trainable: ...; continue
    if prefetched >= prefetch_size:      # 先判断
        break
    for gid in self._gid_to_bucket_group[candidate]:   # 再整组加入
        ...; targets.append(gid); prefetched += self._bucket_numel(gid)
    candidate += step
```

`all_gather_params` 中的窗口与发 AG：

```python
targets = self._extend_by_prefetch_size(req_gids, is_backward)
keep = set(req_gids) | set(targets)
limit = max(self.suggested_communication_unit_size,
            sum(self._bucket_numel(gid) for gid in keep))
# 请求的 FREED buffer：_release_ready_buffers(need, limit, keep) 后同步路径发 AG
# 等待/绑定当前 unit 参数后，对 targets 中 FREED 的：
self._release_ready_buffers(next_params_buffer.total_buffer_size, limit, keep)
self._issue_async_gather(next_gid)   # limit 只决定释放多少 READY，不阻止预取（见 6.1）
```

RS 队列（按大小）：

```python
def _wait_for_grad_comm(self, queue_limit=None):
    if queue_limit is not None:                 # 按个数（前向回收 / 全清）
        while len(self.grad_reduce_queue) > queue_limit:
            self._retire_grad_comm()
        return
    queue_numel = sum(g.total_buffer_size for g in self.grad_reduce_queue)
    while queue_numel > self.suggested_RS_queue_capacity:
        queue_numel -= self.grad_reduce_queue[0].total_buffer_size
        self._retire_grad_comm()
```

释放统一入口 `_free_params_buffer`，`reset_params_buffer_status`、`_release_params_after_unit`、`unshard_params`、`_release_ready_buffers` 都改走它，不再有 `if cnt > 0: cnt -= 1` 这种防御式计数。

完整 diff：`perf_stages/megatron_prefetch/fusion.diff`、`fully_shard.diff`。`Paddle/python/...` 源码目录已同步为同一版本。

## 5. 正确性验证

1. 算法单测（mock buffer group，CPU）：默认值推导（avg=100 → unit 200 / AG 100 / RS 200）、bucket group 划分、前向/反向扩展结果、"先判后加"（AG=1 仍预取 1 组，AG=101 → 3 组，AG=200 → 两个完整 unit）、AG=0 关闭、RS 按大小出队、`queue_limit=0` 全清、环境变量 `2e2` 解析，全部通过。
2. Paddle 分布式单测 `test/collective/fleet/dygraph_group_fsdp.py`（2 卡；MoE EP2、fused linear CE、Muon shard/replicated、AdamH、Muon checkpoint，内部与非 FSDP 参考逐项比对）在 4 组设置下全部通过（rc=0）：
   - 默认值
   - `AG=0, RS=0`（不预取、RS 即刻等待）
   - `unit=AG=RS=1`（每次只预取一个 bucket group，窗口最小）
   - `unit=AG=RS=1e9`（全部预取）

## 6. 性能验证

环境：2 机 16 卡，T4 ERNIELite（h1280、L20、768 专家），EP8，ga=24，10 步，取第 4–10 步 tokens/s/card 均值。
T4 的 `avg_unit_numel = 234,549,248`，默认 unit 4.69e8 / AG 2.35e8 / RS 4.69e8。

### 6.1 第一版的回退及原因

第一版（4.2 节的代码）默认值只有 14,566 tokens/s/card，同时段重跑的 old_S7 是 21,003，低 31%。loss 一致（12.15490 vs 12.15489），问题只在调度。

在调试副本里加计数后定位到原因（rank0，每步）：

- buffer 布局：每层是 3 个很小的 dense buffer（共 2.2e7）+ 1 个 2.12e8 的 expert buffer；另有 gid 0（embedding，2.58e8）和 gid 82（LM head，2.58e8），各自单独成组。
- LM head（`fused_linear_ce_loss_chunk > 0` 时 `GPTLMHead` 把 weight 作为输出交给 loss）从第 1 个 micro-batch 的反向起一直处于 USING，直到 step 末 `reset_params_buffer_status`：
  1. 前向：post-forward hook 发现 weight 被暴露，跳过 shard，前向结束仍为 USING。
  2. 反向：CE backward 同时产出 weight 与 hidden_states 的梯度；Paddle 把叶子的 `GradNodeAccumulation` 放到队头（`backward.cc:640-643`），weight 的 `comm_hook` 先执行，`shard_params` 将其置为 READY；随后 hidden_states 上的 `FusionBackwardHook.backward` 调 `all_gather_params`，又置回 USING。
  3. 此后没有路径把它改回 READY：`GPTLMHead.forward` 的输入是 `dict_args`，PyLayer 只追踪 Tensor 及其 list/tuple，`FusionForwardHook` 没有反向节点；LM head 不属于任何 unit，`_release_params_after_unit` 跳过它；`_release_ready_buffers` 只释放 READY。
- 当前层 2.34e8 + LM head 2.58e8 = 4.92e8，已经超过 `limit = max(4.69e8, 当前层+预取目标)`，于是除第 1 个 micro-batch 的前向外，前向和反向每一层的预取都在 `gathered + need > limit` 处 break：反向 24×20 = 480 次，前向 23×20 = 460 次，预取几乎全部失效。
- T4 未 tie embedding（`tie_word_embeddings: false`），embedding 在反向末尾停在 READY、可被释放，前向被挡同样是 LM head 造成的。
- old_S7 按 buffer 个数计数（上限 = 4×3 = 12 个），LM head 只占 1 个名额，所以不受影响。

结论：Megatron 只用 `suggested_AG_prefetch_size` 决定预取多少，没有"已 gather 总量上限"这一步；第一版为了限制 READY lazy 复用占的显存，额外加了一个会阻塞预取的上限，这是偏离 Megatron 的部分。

### 6.2 修正

预取循环里去掉 `gathered + need > limit` 的 break。`limit` 只用于决定释放多少 READY buffer（`_release_ready_buffers`），不再阻止发 AG。这样在途的预取量只由 `suggested_AG_prefetch_size` 决定，与 Megatron 一致；READY 复用占用的显存仍受 `limit` 约束。

修正后 4 组单测设置（默认、unit=AG=RS=1、全 1e9、AG=0/RS=0）重新跑 `dygraph_group_fsdp.py`，全部 rc=0。

### 6.3 结果

单参数版本（当前代码），通过 `--kwargs suggested_communication_unit_size=...` 配置，各配置的 `FSDP prefetch` 日志行已确认取值生效。2026-10-23 在同一两台机器上测了两轮（`perf_stages/prefetch_results.log` 的 sweep2 / sweep3，avg = 2.35e8）：

| 配置 | unit / AG / RS（×avg） | tokens/s/card | vs old_S7 | max_mem (GB) | step10 loss |
| --- | --- | --- | --- | --- | --- |
| old_S7（`prefetch_units=2`） | — | 21183.1 ± 98 | — | 70.74 | 12.15489388 |
| 默认（null） | 2 / 1 / 2 | 20907.8 ± 122 | −1.3% | 70.46 | 12.15485001 |
| 7.04e8 | 3 / 1.5 / 3 | 21093.2 ± 119 | −0.4% | 70.85 | 12.1548996 |
| 9.38e8（sweep2） | 4 / 2 / 4 | **21222.4 ± 169** | +0.2% | 70.90 | 12.15490723 |
| 1.41e9（sweep2） | 6 / 3 / 6 | OOM（step 3） | — | 71.59 | — |

未标注的行来自 sweep3（17:48–18:13 连续跑）。重复测过的配置只保留性能较好的一次。另有 3 次运行（sweep2 的 old_S7 / 7.04e8，sweep3 的 5×avg 1.17e9）在 step 1–4 出现 DeepEP dispatch 超时，没有数据。这些超时都发生在 BE pod 抢占、驱逐前后，重跑后正常（5×avg 未重跑），判断为环境问题。

loss 在所有配置下差异 ≤ 5e-5（第 10 步），属于通信顺序带来的浮点非确定性。

分析：

- 默认值（AG ≈ 1 个 unit）比 old_S7 慢 1.3%，重复测量结果一致。本配置单层计算时间不足以掩盖下一层 2.3e8 元素的 AG 加跨机延迟，需要约 2 层的预取深度，与 S7 中 `prefetch_units` 1→2 带来 +17% 的结论一致（那次是修复前的旧调度，差距被放大）。
- unit 越大越快：3×avg −0.4%，4×avg +0.2%，4×avg 与 old_S7 持平。4×avg 比默认快 1.5%，显存多 0.44 GB。
- 6×avg 时 GPU 0 OOM（`Cannot allocate 938MB ... available memory is only 808MB`）。T4 在 ga=24 下显存余量约 1 GB，unit 上限在 4×–5×avg 之间。

早期三参数版本（AG / RS 可独立设置）的扫描结论与此一致：AG = 2×avg 时与 old_S7 持平；RS 容量降到 1×avg 会慢约 1%，放大到 4×avg 收益不到 1%。所以只保留一个参数（RS = unit，AG = unit/2）对 T4 没有损失。

### 6.4 推荐

- 代码默认值保持 Megatron 公式（unit = 2×avg，AG = unit/2，RS = unit），语义与 Megatron 一致，模型结构不同时自适应。
- T4 这类"单层计算短、单层参数大"的 MoE 配置，建议把 unit 设为 4×avg。实测与 old_S7 持平（+0.2%），比默认快 1.5%，显存多 0.44 GB：

```yaml
suggested_communication_unit_size: 9.38e8   # 4×avg，AG = 4.69e8，RS = 9.38e8
```

  6×avg 会 OOM，5×avg 没测出结果，所以 4×avg 已接近显存上限。显存更紧的配置可以退到 3×avg（7.04e8，−0.4%），或者保持默认（null，−1.3%）。
- 如果希望不配置就达到 S7 水平，可以把 AG 默认改成 `unit_size`（即 2×avg）。这是偏离 Megatron 的选择，本次未改。

### 6.5 其他

- 早期三参数版本通过 `PADDLE_FSDP_*` 环境变量配置。`script/train_gpu.sh` 会 unset 所有含 `PADDLE` 的环境变量，当时临时把 `PADDLE_FSDP_*` 排除在外。现在改为 yaml 参数，这个排除已撤销，6.3 的表格也换成了单参数版本的数据。
- 扫描脚本：`perf_stages/run_prefetch_sweep.sh "NAME:CODE:UNIT" ...`。它每轮改写 `run_fsdp_test.sh` 里的 `suggested_communication_unit_size`，结束后还原。训练崩溃后用 `pkill -9` 清理残留进程，否则会占住 GPU，阻塞后续轮次。
- 未移植的 Megatron 特性见第 3 节第 7 点。

## 7. 待完成优化：双缓冲与 NCCL UB（按 Megatron 实现）

状态：未实现，本节是设计。Megatron 侧细节见 `megatron_fsdp_overlap_prefetch.md` 第 8.5、9、10 节。

### 7.1 Megatron 的实现

| 项 | Megatron 做法 |
| --- | --- |
| 开关 | `fsdp_double_buffer`；`megatron_fsdp_max_pool_double_buffer`（自动开双缓冲）；`fsdp_db_use_persist_buf_on_alloc_fail`；`nccl_ub`（强制开双缓冲和手动注册）；`fsdp_manual_registration`；`disable_symmetric_registration` |
| 池 | 每类临时 bucket 一个池，`size=2`：未分片权重、FP8 transpose 权重、未分片梯度、HFSDP 梯度通信。FixedPool 只服务同构 unit；MaxPool 按 dtype、按 unit 内 bucket 位置取所有 unit 的最大尺寸 |
| 槽位分配 | `strict_assignments`：bucket 第一次拿到哪个槽，以后尽量拿同一个，地址每步固定（NCCL UB、CUDA Graph 的前提） |
| 不进池的 bucket | 非 unit 参数和不同构 unit：回退 StorageResize（UB 不生效），或在 `use_persist_buf_on_alloc_fail` 时用持久 buffer |
| AG 约束 | `_extend_by_prefetch_size` 加入下一组后涉及 unit 数 > 2 就停；`all_gather_params` 请求超过 2 个 unit 抛 ValueError；请求全是非 unit 参数时不预取；重计算的前向不预取 |
| RS 约束 | `_enforce_double_buffer_limit`：首次 `get_main_grad` 分配梯度 bucket 前、发 RS 前，从队头出队等待，直到"队列里的 unit + 本次" ≤ 2 |
| 流安全 | 不用 `record_stream`：槽位复用只发生在下一次 AG，而 AG 前一定执行 `ag_stream.wait_stream(current_stream)` |
| NCCL UB | `nccl_allocator.init()` 建 NCCL 内存池（可对称）；持久 buffer 和双缓冲池都在 `mem_alloc_context` 下分配；依次对 DP-Shard、专家 DP、独立 AG 组、HFSDP outer 组注册，每组前 barrier；手动注册在第 2 个 iteration 一次性 `register_mem_pool` |
| 限制 | torch < 2.11 时与 `expandable_segments` 不兼容 |

收益：未分片权重和梯度的显存固定为 2 个 unit，没有碎片；注册内存后 AG/RS 可以走零拷贝、占 SM 更少的 kernel（对称内存下可用 NVLS），减少与计算 kernel 的 SM 竞争。代价：预取最多提前 1 个 unit。

### 7.2 Paddle 现状与差距

| 项 | Paddle 现状（`fully_shard_fusion.py`） | 差距 |
| --- | --- | --- |
| 临时 buffer | `TensorFusionBuffer.get_tmp_buffer()` 每次 `paddle.empty`（权重）/ `paddle.zeros`（梯度）分配 `total_buffer_size`，`clear_tmp_buffer()` 用 `_clear_data()` 归还 caching allocator | 地址每步变化，无法做窗口注册，也有碎片 |
| 数量上限 | 只有按大小的预取 / RS 队列（第 3 节），READY lazy 复用由 `limit` 约束 | 没有"槽位数"这一硬约束 |
| 流安全 | `_async_collective` 在侧流上先 `wait_stream(current)`，再对输入输出 `_record_stream()` | 槽位固定后 `record_stream` 不再需要（见 7.3.4） |
| 持久 buffer | 分片权重 `data_buffer`、分片梯度 `data_buffer` 由普通 allocator 分配 | AG 的输入、RS 的输出不在注册内存里 |
| 非 unit buffer | embedding（gid 0，2.58e8）、LM head（gid 82，2.58e8）各自成组，且常驻 USING（见 6.1） | 进不了按 unit 构建的池 |

Paddle 已有的注册能力（本仓 `Paddle/` 源码，已确认存在，未在 FSDP 中用过）：

- `core.nccl_mem_alloc(shape, dtype)`（`paddle/fluid/pybind/distributed_py.cc:1324`）：用 `ncclMemAlloc` 分配，大小按 4096 B 对齐；Python 封装 `paddle.distributed.collective._nccl_symmetric_empty`。
- `ProcessGroupNCCL.register_comm_buffer(tensor, win_flags=NCCL_WIN_COLL_SYMMETRIC)` / `deregister_comm_buffer`（`distributed_py.cc:1280-1322`）：调用 `ncclCommWindowRegister`，要求 tensor 从 allocation 起点开始；封装为 `collective._register_comm_buffer(tensor, group)`。`NCCLCommContext::IsRegistered` 按地址区间判断，注册块的切片也算已注册。
- 通信组 `nccl_config` 的 `cta_policy=2`（`NCCL_CTA_POLICY_ZERO`）：已注册窗口上的集合通信走 Copy Engine（机内）/ RMA proxy（机间），不占 SM。需要 NCCL ≥ 2.30.7；当前环境 `nvidia-nccl-cu13 2.30.7`。
- `paddle.distributed.communication.zero_sm`：带环形注册池 `_ZeroSMPool` 的 `all_gather` / `alltoall_single`。它的环按分配次数轮转，FSDP 的槽位需要按 unit 固定，不适合直接复用，但注册方式和"注册必须各 rank 对称"的约束可以照搬。
- `ncclCommRegister`（非窗口的普通 UB 注册）在 dynload 里有符号，但没有 Python 绑定。

### 7.3 设计

#### 7.3.1 Bucket allocator 抽象

把 `get_tmp_buffer` / `clear_tmp_buffer` 改为通过 `FSDPCommManager` 持有的 allocator 拿/还临时 buffer：

```python
class BucketAllocator:                      # 默认实现 = 现状
    def allocate(self, gid, numel, dtype): return paddle.empty([numel], dtype)
    def free(self, gid, tensor): tensor._clear_data()

class SlotPoolAllocator(BucketAllocator):  # 对应 Megatron FixedPool / MaxPool
    ...
```

- 权重和梯度各一个 allocator（梯度用 `main_grad_dtype`；FP8 transpose 权重若以后需要，再加一个）。
- 梯度 buffer 由池分配后不再是零初始化，拿到槽位后要显式 `zero_()`。
- 默认 allocator 必须和现在行为逐字节一致，保证不开开关时性能、loss 不变。

#### 7.3.2 槽位池

- key = `(dtype, unit 内 bucket 序号)`；每个 key 一组 `num_slots` 个槽。
  - FixedPool：只收与第一个 unit 的 bucket 布局（个数、dtype、numel）完全一致的 unit。
  - MaxPool：每个 key 取所有 unit 的最大 numel，异构 unit（dense / MoE 层、MTP 层）也能进。
- 只给 sharded 且属于 FSDP unit 的 buffer 分配。embedding / LM head / gid 81 这类非 unit buffer 回退到默认 allocator（UB 不生效）；可选 `use_persist_buf_on_alloc_fail`：给它们各分一块常驻全量 buffer。T4 上两个各 2.58e8 bf16，共约 1 GB。
- 槽位分配照搬 `strict_assignments`：unit 第一次拿到槽 k，以后优先拿 k；k 被占用时才拿别的槽。目的是让每步的地址序列固定，所有 rank 对称。
- 池在第一次使用时整体分配，之后不再增长。Megatron 的 `GlobalMemoryBuffer` 在增长时要 `synchronize`；Paddle 在构建 buffer group 时就知道所有尺寸，可以直接按最终尺寸一次分配。
- 槽位数默认 2，与 Megatron 一致；做成参数 `num_slots`。6.3 中 T4 预取 2 个 unit 比 1 个快 1.4%，2 槽只能提前 1 个 unit，所以需要在 T4 上对比 2 / 3 槽。

#### 7.3.3 槽位约束（AG / RS）

开启池后，在第 3 节的按大小逻辑上再加 Megatron 的 unit 数约束（`N = num_slots`）：

- `_extend_by_prefetch_size`：加入下一组后，涉及的不同 `fsdp_unit_id` 数（非 unit buffer 不计）> N 时停止。实际预取量 = min(大小阈值, N−1 个 unit)。
- `all_gather_params`：
  - 请求涉及的 unit 数 > N 时抛错。
  - 请求全是非 unit buffer 时不预取。
  - 重计算的前向（Paddle 侧需要识别 recompute 重放的 pre-forward hook）不预取，避免占槽。
- READY lazy 复用：一个 unit 拿槽前，如果所有槽都被别的 unit 占着，先按 gid 顺序释放窗口外 READY unit 的槽；被 USING / SYNCING 占满时抛错（说明调用顺序超出了 N 个 unit）。之后 `limit` 只对非池 buffer 生效。
- RS（对应 `_enforce_double_buffer_limit`）：
  - 在梯度 `get_tmp_buffer()`（首次 `get_main_grad`）之前、`do_reduce_scatter` 之前各调用一次。
  - 从 `grad_reduce_queue` 队尾往前数不同 unit，保留 `N − 1` 个（加上本次共 N），更早的执行 `_retire_grad_comm()`（wait + `accumulate_reduced_grad` + 归还槽）。
  - 与 `suggested_RS_queue_capacity` 同时生效，取更严格的那个。

#### 7.3.4 流安全

照 Megatron 的做法，开启池时去掉 `_record_stream()`，靠流顺序保证安全：

- 权重槽：只在下一次 AG 时被覆盖，而 `_async_collective` 发 AG 之前已经 `comm_stream.wait_stream(current)`。槽在 host 侧归还时，上一个 unit 读它的计算 kernel 都已入队，所以 AG 一定排在这些 kernel 之后。
- 梯度槽：`_retire_grad_comm()` 先 `comm_task.wait()`（当前流等 RS 事件），再在当前流上 `accumulate_reduced_grad`，然后才归还槽。下一个使用者在当前流上 `zero_()` / 累加梯度，同流有序。
- 槽归还后，旧 tensor 视图（`param` 绑定的 `view_slice`、`main_grad`）必须解绑，否则会被下一个 unit 静默覆盖。参考 `_ZeroSMPool` 对已发出 tensor 的弱引用检查，加一个 debug 开关：归还时检查是否还有外部引用。
- 默认 allocator 保留 `_record_stream()`，行为不变。

#### 7.3.5 NCCL UB（窗口注册）

1. 通信组：给 FSDP 用到的组（`_fsdp_group`、`_ep_fsdp_group`、按 color 建的组）设置 `nccl_config` 的 `cta_policy=2`。sharding 组由 fleet topology 创建，需要从 topology 的 `nccl_config` 透传。
2. 分配：NCCL 要求一次调用的输入输出要么都注册、要么都不注册。
   - AG：输入是分片权重 `data_buffer`，输出是权重槽，两者都要用 `_nccl_symmetric_empty` 分配。
   - RS：在梯度槽内原地进行（输出是 `tmp_buffer._slice`），只需注册梯度槽。
   - 每个池可以分配成一整块，再切成槽，以减少窗口数。`IsRegistered` 按地址区间判断，切片也算已注册。
3. 注册时机：在 `FSDPCommManager` 构建完 buffer group、池按最终尺寸分配后，按固定顺序逐组调用 `_register_comm_buffer`，每组前 barrier。
   - 顺序：dense 组 → EP 组 → color 组。
   - 尺寸在初始化时已全部确定，所以不需要 Megatron 的"第 2 个 iteration 手动注册"（那是为了等内存池在第一个 iteration 里长到最终大小）。
4. 回退：`register_comm_buffer` 返回 0（NCCL 版本过低）时 warning，并关闭 `cta_policy` 路径，只保留双缓冲。
5. 待确认：
   - AG 在窗口上能否走 zero-SM。Paddle 只对 AllToAll 显式判断了 `IsCommBufferRegistered`，AG/RS 依赖 NCCL 自身识别窗口。
   - RS 需要做规约，Copy Engine 做不了，预计仍会占 SM，具体要看 NCCL 2.30 的行为。
   - 机间 RMA proxy 在 IB 上的带宽是否不低于 SM kernel。
   - `ncclMemAlloc` 与 `FLAGS_use_virtual_memory_auto_growth=1`（`train_gpu.sh` 已开启）能否共存。
   - `train_gpu.sh` 中 `NCCL_NVLS_ENABLE=0`，NVLS 收益不可用。
   - `ncclMemAlloc` 的内存不经过 Paddle allocator，`max_mem` 统计里看不到，需要单独计入。

#### 7.3.6 配置

```python
fully_shard(model, ..., fsdp_double_buffer=False, fsdp_buffer_pool="max",   # "fixed" | "max"
            fsdp_num_buffer_slots=2, fsdp_use_persist_buf_on_alloc_fail=False,
            fsdp_use_nccl_ub=False)                                          # True 时强制开双缓冲
# 环境变量：PADDLE_FSDP_DOUBLE_BUFFER / PADDLE_FSDP_BUFFER_POOL / PADDLE_FSDP_NUM_SLOTS / PADDLE_FSDP_NCCL_UB
```

rank0 在 `FSDP prefetch: ...` 后追加一行，打印池类型、槽数、每槽大小、没进池的 gid 列表。

### 7.4 显存估算（T4）

每个 unit 未分片 234,549,248 元素，最大 unit 与平均相同（20 层同构，FixedPool 和 MaxPool 结果一样）。权重按 bf16 算，梯度按 `main_grad_dtype` 默认的 fp32 算：

| 槽数 | 权重池 | 梯度池 | 合计 |
| --- | --- | --- | --- |
| 2 | 0.94 GB | 1.88 GB | 2.81 GB |
| 3 | 1.41 GB | 2.81 GB | 4.22 GB |

- embedding / LM head 不进池；开 `use_persist_buf_on_alloc_fail` 时再常驻约 1 GB（bf16）。
- 现状下这部分显存是动态的，峰值取决于预取量和 READY 保留量。双缓冲的收益是峰值固定、没有碎片，不一定更省。实测时要和 fix_ag2（70.89 GB）比较峰值和 `reserved`，并单独统计 `ncclMemAlloc` 分配的部分。

### 7.5 实施步骤与验证

1. allocator 抽象 + 默认实现：`dygraph_group_fsdp.py` 4 组设置全过，T4 性能、loss 与 fix_ag2 一致。
2. SlotPool + 槽位约束 + 去掉 `record_stream`：
   - 单测：mock buffer group 验证槽位分配固定、unit 数约束、READY 让槽、RS 按 unit 出队。
   - 分布式单测在 `fixed` / `max` 两种池、`num_slots` = 2 / 3 下全过。
   - 每步打印每个 gid 拿到的槽地址，确认第 2 步起不变、所有 rank 相同。
3. T4 性能：`num_slots` = 2 / 3 与 fix_ag2 对比 tokens/s/card、max_mem、loss。
4. NCCL UB：
   - 先用 2 卡小例子确认 AG/RS 在窗口 + `cta_policy=2` 下走 zero-SM 路径（`NCCL_DEBUG=INFO` 日志、nsys 里通信 kernel 消失或 SM 占用下降）。
   - 再上 T4 测端到端。
5. 每步都确认不开开关时行为与当前版本一致。

### 7.6 风险

- 非 unit buffer：embedding（`is_tie` 分组，tie 时与 LM head 共享）、fused linear CE 暴露的 LM head（整个 step 保持 USING，见 6.1）。它们不进池，仍走 `limit` 和默认 allocator。
- 异构 unit：MTP 层、首尾 dense 层。FixedPool 会把它们排除在外，所以优先用 MaxPool。
- 显式 `queue_limit` 的调用方（前向逐层回收、`finish_grads_sync`、`_ensure_grads_writable`）要保证在槽位约束下不死锁。
- 槽内存被下一个 unit 复用后，外部还持有旧视图会静默写坏：`unshard_params`、Muon / AdamH 的 sharded 路径、checkpoint 保存、offline FP8 quant。这些路径要逐个确认在归还槽之前已用完视图。
- 注册必须各 rank 对称：池尺寸要由全组一致的信息决定（buffer 布局各 rank 相同，目前满足）；Muon owner 负载均衡等按 rank 不同的逻辑不能影响池尺寸。
- 弹性重启 / 通信组重建后窗口失效，需要和 `_ZeroSMPool.discard` 一样重新注册。

## 15. Paddle 实现：Megatron 式预取的 host 开销优化

> 改动文件：`Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`（已同步到 `Paddle/build/python/...`）
> 副本与 diff：`perf_stages/prefetch_opt/fully_shard_fusion.py`、`perf_stages/prefetch_opt/opt.diff`（基线为 `perf_stages/megatron_prefetch/`）
> 重构本身（按大小预取、AG = unit/2、RS 容量 = unit）见 `megatron_style_prefetch_refactor.md`。

### 15.1 问题

T4 ERNIELite（h1280，L20，768 expert，EP8，ga=24，2 机 16 卡，10 step，取 step 4–10 均值）上，重构版设 `suggested_communication_unit_size=7.04e8`（3×avg_unit_numel，avg = 234,549,248）时：

| 版本 | tokens/s/card | max_mem_allocated_gb |
| --- | --- | --- |
| old_S7（按个数预取，`prefetch_units=2`） | 21109 ~ 21183 | 70.74 ~ 70.79 |
| 重构版 u3 | 20925 ~ 21093 | 70.85 |

吞吐低约 0.5%~1%，显存高约 0.06~0.11 GB。

### 15.2 定位方法

用 `perf_stages/prefetch_debug/make.py` 给 fully_shard_fusion.py 追加插桩（`instrument_block.py`，`--deep` 再加 `instrument_block2.py`），不改源码本体。每个 step 在 rank 0/8 打一行 `[FSDPDBG]`：

- `stall_ms`：在 `_StreamTask.wait` 前后各记录一个计时 event，得到计算流因等待 AG/RS 而空转的 GPU 时间，按前向/反向分开统计。
- `req`：unit 请求某个 bucket 时该 bucket 所处的状态（READY 表示已被预取完成，SYNCING 表示在途）。
- `ag/rs`：本 step 发起的集合通信次数和元素总量。
- `cpu_ms(agp/rgg)`：用 `perf_counter` 统计 `all_gather_params`、`_reduce_group_grads` 的 host 耗时。`--deep` 进一步拆到各 helper。
- `peak_GB(p/g)`：未分片参数 buffer、临时梯度 buffer 的峰值字节。

注意：`--deep` 会重置设备峰值计数，这类 run 的 `max_mem_allocated_gb`（49.88）无意义。

### 15.3 测量结果（rank 0，单 step）

| 指标 | old_S7 | 重构版 u3 | 说明 |
| --- | --- | --- | --- |
| AG stall 前向 / 反向（ms） | 22–26 / 14–24 | 9–13 / 6–7 | 重构版预取更充分，等待更少 |
| RS stall 反向（ms） | 21–23 | 21–25 | 持平 |
| bucket 请求状态 | fwd SYNCING 1853 / READY 92；bwd SYNCING 1752 / READY 240 | 完全相同 | 预取命中情况一致 |
| AG / RS 次数与元素量 | fwd 1854 / 107.70e9，bwd 1752 / 107.51e9，RS 1831 / 107.70e9 | 完全相同 | 没有多发或少发通信 |
| `all_gather_params` host（ms） | 999–1020 | 1030–1076 | 重构版多 30–75 ms |
| `_reduce_group_grads` host（ms） | 320–325 | 328–329 | 重构版略多 |
| 峰值 p / g（GB） | 1.84 / 5.41 | 1.83 / 6.29 | 梯度临时 buffer 峰值高 0.88 GB |

结论：重构版在 GPU 侧的调度更好，AG stall 每 step 少约 25 ms，通信量相同。吞吐反而更低，原因在 host 侧。

### 15.4 根因：hook 的 host 开销落在关键路径上

- 每个 step 有 24 个 micro-batch，`all_gather_params` 调用数千次（前后向各约 1800 次 AG）。
- DeepEP 的 dispatch/combine 每层都会让 host 与 device 同步一次，host 没法跑在 GPU 前面积累队列。于是 hook 里每多花 1 ms 的 host 时间，基本就多 1 ms 的 step 时间。
- 重构版按大小决定预取，每次调用都要做以下事情：
  1. 现场遍历 bucket、计算 `total_buffer_size`，以此确定预取目标和需要保留的集合（`keep`）；
  2. `_release_ready_buffers` 遍历全部 bucket（包括未分片的）找可释放的 READY buffer；
  3. `_wait_for_grad_comm` 每次对 RS 队列求和得到在途元素量。
- 这些操作都是 O(bucket 数) 的 Python 循环，单次耗时很小，但乘以调用次数后每 step 多出约 100 ms host 时间。这个量超过了 AG stall 节省的约 25 ms。step 耗时 18.57 s（old）对 18.64–18.8 s（u3）也印证了这一点。

### 15.5 显存差异分析

- u3 的 RS 容量 = 7.04e8 元素，约 3 个 unit 的梯度同时在途；old_S7 的 `grad_queue_limit` 相当于 2 个。临时梯度 buffer 峰值因此从 5.41 GB 升到 6.29 GB。
- 验证实验 `dbg2_u3_rs2`：把 RS 容量强制设为 2×avg，梯度峰值回到 5.41 GB，但 `max_mem_allocated_gb` 仍为 70.86，吞吐 20958，没有改善。
- 说明 `max_mem_allocated` 的峰值不在梯度 buffer 最大的时刻，0.06–0.11 GB 的差异来自通信/释放时序和 allocator 碎片，与单 step 约 0.2 GB 的波动同量级。因此保持 Megatron 语义（RS 容量 = unit），不为这部分显存修改 RS 容量。

### 15.6 优化改动

原则：bucket 布局在 `FSDPCommManager.__init__` 结束后不再变化，凡是只依赖布局的量都在初始化时算好，hook 里只做查表和状态判断。预取策略、释放条件、RS 容量语义都不变。

| # | 位置 | 原实现 | 优化后 | 理由 |
| --- | --- | --- | --- | --- |
| 1 | `__init__` | 无 | 预先算好 `_params_buffers`（gid → params_buffer）、`_bucket_numels`（未分片 bucket 记 0）、`_sharded_gids` | 消除 hook 中反复的 `buffer_groups[gid].params_buffer` 属性链和 `is_sharded` 判断 |
| 2 | `_bucket_numel` | 每次读 params_buffer 并判断 `is_sharded` | `return self._bucket_numels[gid]` | 在预取扩展和释放判断里被频繁调用 |
| 3 | `_prefetch_plan`（新增） | `all_gather_params` 每次都调用 `_extend_by_prefetch_size`，再构造 `keep` 集合、求和得到 `limit` | 以 `(tuple(req_gids), is_backward)` 为 key 缓存 `(targets, frozenset keep, limit)` | 预取目标只取决于请求的 bucket 和方向，与运行时状态无关。每个 step 请求的组合是固定的几十种，首个 micro-batch 之后全部命中缓存 |
| 4 | `_release_ready_buffers` | 遍历全部 bucket，每轮检查一次 limit | 先检查 limit，未超出直接返回；只遍历 `_sharded_gids`；每释放一个后再检查 | 大多数调用并不需要释放，提前返回最省；未分片 bucket 不可能是 READY 状态，跳过 |
| 5 | RS 队列在途量 | `_wait_for_grad_comm` 每次 `sum(g.total_buffer_size for g in grad_reduce_queue)` | 维护计数 `_grad_queue_numel`：`_reduce_group_grads` 入队时加，`_retire_grad_comm` 出队时减；等待循环直接比较 `_grad_queue_numel > suggested_RS_queue_capacity` | O(队列长度) 降为 O(1) |


改动共 +50 / −30 行，完整 diff 见 `perf_stages/prefetch_opt/opt.diff`。

`_flush_expert_grads_after_unit` 保留原实现，没有改。曾试过一个水位（floor）版本：只处理本轮反向中新完成的 unit，把每轮反向的扫描从 O(L²) 降到 O(L)。后来撤掉了，原因有两点：

- 收益很小。原实现每次扫描遇到已经 reduce 过的 group，`_reduce_group_grads` 在入口判断 `tmp_data_buffer is None` 后直接返回。按 T4 的规模（20 个 unit，每个约 4 个 bucket，24 个 micro-batch）做 CPU 微基准，扫描本身每个 step 约 2.0 ms，水位版约 0.9 ms，只省约 1 ms，不到总节省量（约 100 ms）的 1%。
- 水位版要额外维护跨调用状态，还要靠「`unit_id >= floor` 即新一轮反向」的推断，不如原逻辑直观。

插桩里 `_flush_expert_grads_after_unit` 从 338 降到 300 ms，这部分下降主要来自它内部调用的 `_reduce_group_grads`（RS 队列计数 O(1)、`_wait_for_grad_comm` 变快），不是来自扫描方式的改变。

### 15.7 正确性验证

- 单测 `dygraph_group_fsdp.py`（2 卡，`/tmp/fsdp_ut/run_ut.sh`）：unit size 取默认、`0`（关闭预取）、`2`（最小预取）、`1e9`（大窗口）四种设置均 rc=0。撤掉 `_flush_expert_grads_after_unit` 的水位改动后，四种设置全部重跑，均 rc=0。
- T4 端到端：step 10 loss 为 12.15488 / 12.15487（两次 opt run），与 old_S7、重构版的 12.15485–12.15491 一致，差异在该模型的 run-to-run 波动范围内。
- 插桩计数：bucket 请求状态分布、AG/RS 次数与元素量、峰值 p/g 与优化前完全相同，说明调度行为没有改变。

### 15.8 性能结果（同一批机器，step 4–10 均值）

| 版本 | tokens/s/card | max_mem_allocated_gb | step10 loss |
| --- | --- | --- | --- |
| old_S7 | 21109 ± 83 | 70.79 | – |
| 重构版 u3（优化前） | 20925 ± 240 | 70.85 | 12.15490 |
| 重构版 u3（优化后） | 21138 ± 133 | 70.86 | 12.15488 |
| 重构版 u3（优化后，复跑） | 21099 ± 147 | 70.87 | 12.15487 |

插桩对比（优化后，rank 0）：`all_gather_params` host 时间 1030–1076 → 925–979 ms；`_reduce_group_grads` 328 → 284–303 ms；`--deep` 拆分中 `_release_ready_buffers` 65 → 37 ms，`_flush_expert_grads_after_unit`（含其中的 RS 发起）338 → 300 ms。AG stall 前向 / 反向 11–20 / 6–10 ms，仍低于 old_S7。

注：上表的端到端数据是在含水位改动的版本上测得的。撤掉水位改动后，估计每 step 只多约 1 ms host 时间（约 0.005%），远小于 run 间标准差，因此没有重跑端到端测试。

优化后吞吐与 old_S7 持平（差值小于单次 run 的标准差），比优化前高约 1%。显存仍高约 0.07 GB，原因见 15.5。

### 15.9 调参建议修正

- 优化后，`suggested_communication_unit_size = 3 × avg_unit_numel`（T4 上为 7.04e8）即可达到 old_S7 的吞吐。
- `megatron_style_prefetch_refactor.md` 6.4 节推荐的 4×avg（9.38e8）在多次复跑中为 20970–21222，与 3×avg 的差距在噪声内，且显存略高，不再优先推荐。
- 默认值 2×avg（4.69e8）稳定低约 1%（20897–20936），原因是只预取 1 个 unit，AG 来不及完全隐藏。
- 结论推广：只要 DeepEP 这类每层同步 host 的算子存在，FSDP hook 的 host 时间就直接计入 step 时间。后续在 hook 中增加逻辑时，应避免每次调用都做 O(bucket 数) 的遍历，能预计算的都放到初始化。