# PaddleFleet FSDP（Paddle 手动模式 Tensor Fusion FSDP）overlap / prefetch 实现分析与优化建议

> 源码基线：
> - Paddle `9030271e0c` + 本地 staged 修改：`Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`（2067 行，下文行号均指该文件）、`fully_shard.py`、`_fsdp_context.py`
> - Paddle C++：`Paddle/paddle/fluid/distributed/collective/process_group_nccl.cc`
> - PaddleFleet trainer：`PaddleFleet/.../trainer.py`（`_setup_fsdp` 约 L5620）
> - 示例配置：`conf/experiment/ERNIELite/ScalingLadder/*-hzz1_fsdp.yaml`
> - 对照文档：`megatron_fsdp_overlap_prefetch.md`、`pytorch_fsdp_overlap_prefetch.md`
>
> 说明：本文全部结论来自静态阅读源码，未采集 profile。涉及"实际是否 overlap""通信耗时占比"的判断均为推断，文中标注【推断】。

---

## 0. 结论速览

| 维度 | Paddle FSDP（manual, `enable_tensor_fusion_and_overlap=True`） | Megatron-FSDP |
|---|---|---|
| 分片粒度 | BufferGroup：按 `(dtype, trainable, unit_id, is_expert, fsdp_group, tie, no_wd, muon, hyperball, clip_attrs)` 分组的扁平 buffer；一个 unit 可有多个 group | bucket（按 module/size 切）+ bucket group 合并 |
| unit 定义 | `fsdp_unit_layers` 类名匹配（默认 `TransformerLayer` 等），`fully_shard()` 未暴露 | `fsdp_unit_modules`，由用户/框架指定 |
| prefetch 数量 | `prefetch_units=1`（硬编码默认，未暴露），按 group 下标 ±1 静态顺序，预取下一个 unit 的全部 group | 静态 bucket ±1，`suggested_AG_prefetch_size ≈ unit_size//2`（约 1 个 unit），按元素数计 |
| AG buffer 上限 | `gather_buffer_limit = groups_per_unit × (prefetch_units+1)`，按 **buffer 个数** 计 | 按元素数 / FixedPool 双 buffer |
| 驱逐策略 | 从下标 0 线性扫描第一个 READY buffer 释放（非 LRU、非距离） | 显式 release + `can_be_released` 懒释放 |
| 通信流 | AG 和 RS 走同一 PG → **同一 NCCL comm + 同一 comm stream**，互相串行 | AG / RS 独立 side stream；`--create-all-gather-group` 独立 communicator |
| 跨流内存安全 | `record_stream`（StreamSafe allocator，默认开）+ `task.wait()` 让 calc stream 等 comm event | 不用 `record_stream`，纯 `wait_stream` / event |
| RS 触发 | **结构触发**：反向进入 unit i-1 的 pre-backward 时，才对 unit_id > i-1 的所有 group 发 RS | bucket group 内全部 grad ready 即发 RS |
| RS 未覆盖部分 | 第一个 unit（unit 0）+ 所有 `unit_id=None` 的 group（embedding / final norm / lm_head 等）→ 在 `finish_grads_sync` 中**同步** RS，无 overlap | 无此问题 |
| RS 并发 | `grad_queue_limit = groups_per_unit × max(prefetch_units,1)` 个在飞 RS | RS 队列容量约 2 个 unit（按元素数） |
| 梯度累加 | **无 `no_sync`**：unit 1..N-1 每个 micro-batch 都做 RS；unit 0/None group 反而全步只做一次（副作用，见 §8.5） | `optim`/`no_shard` 只在最后一个 micro-batch reduce；ZeRO-2/3 每个 micro-batch reduce |
| 梯度缩放 | `grad_div*accum == degree` 时用 AVG，否则对整块 tmp 做 `scale_` + SUM | AVG / premul_sum / `mul_`+SUM |
| RS 输出 | 原地写到输入 buffer 头部 `tmp[0:shard]`，再在 **calc stream** 上 `data_buffer.add_` | 写临时 shard，在 RS stream 上 `+=` main_grad |
| recompute | **无感知**：重算 forward 的 pre-hook 按 +1 方向预取（方向错误） | root pre-backward 标记 `can_be_released`，重算时禁用 forward prefetch |
| step 边界 | `finish_grads_sync` → `reset_params_buffer_status` → `paddle.device.synchronize()`（每步全设备同步） | 可配 `fsdp_all_gather_in_start_param_sync` 提前发首个 AG |
| allocator | 普通 caching allocator，tmp buffer 每次 `paddle.empty/zeros` 分配、`_clear_data` 释放 | 4 种 allocator（含 FixedPool / MaxPool 双 buffer）、NCCL UB / symmetric memory |

一句话：Paddle 实现的"骨架"（扁平 buffer、±1 静态预取、异步 AG/RS、有限在飞队列）与 Megatron 一致，但在**通信并发（共用 comm stream）**、**RS 触发时机（滞后一个 unit + 首尾同步兜底）**、**梯度累加（无 no_sync）**、**recompute 感知**、**buffer 预算与驱逐**五个方面明显弱于 Megatron，这也是 §13 优化点的主线。

---

## 1. 入口与调用链

```
PaddleFleet Trainer (--sharding fsdp, ShardingOption.FSDP)
  └─ _setup_fsdp(module, mp_policy)                       # trainer.py ~L5620
       ├─ fully_shard(module, enable_tensor_fusion_and_overlap=True, mp_policy=...)
       │    └─ FSDP context (fully_shard_fusion.py)
       │         ├─ BufferManager: 分组 / 排序 / 构建 TensorFusionBuffer      (§2, §3)
       │         ├─ FSDPCommManager(prefetch_units=1)                          (§6, §8)
       │         └─ register_tensor_fusion_hooks(model)                       (§5)
       ├─ ctx.set_grad_accum_steps(gradient_accumulation_steps)               # L1612
       └─ ctx.bind_optimizer(optimizer)
Optimizer._apply_optimize (optimizer.py ~L1702)
  └─ fsdp_context.comm_sync_and_reset_status()                                # L1621
       └─ 把 params_grads 换成 fused shard；grad clip 走 should_comm_on_shard_dim
保存：fsdp_context.all_gather_params()                                         # trainer.py ~L6388
FP8 量化：unshard_params(params) 上下文（同步 gather，不预取）                  # L1458
```

- `mp_policy`：`amp_master_grad=False` 且 amp 为 bf16/fp16 时 `reduce_dtype=amp_dtype`，否则 reduce 为 fp32。示例配置 `amp_master_grad: true` → grad buffer / RS 都是 **fp32**。
- PP 场景下对 `model._layers` 做 `fully_shard`。
- auto-parallel 路径（`fully_shard(model, mesh=...)`，`auto_parallel/fully_shard_fusion.py`）是另一套实现，仍用 `grads_use_cnt == grads_use_sum` 计数触发 RS，本文不展开。
- `fully_shard()` 的 `reshard_after_forward / shard_placement_fn / offload_policy / ignored_params / mesh` 在 manual 模式下**未接线**；`fsdp_unit_layers`、`prefetch_units` 也未经 `fully_shard()` 暴露（L1151 内部默认 `prefetch_units=1`）。

---

## 2. BufferGroup：分组与排序（决定预取顺序）

### 2.1 分组 key（L445–471）

每个参数按下列 tuple 归入一个 BufferGroup：

```
(dtype, trainable, fsdp_unit_id, is_expert, fsdp_group, is_tie,
 no_weight_decay, use_muon, hyperball_name_or_None,
 clip_attrs = (need_clip, no_sync, is_distributed, is_firstly_shared))
```

- `fsdp_group`：`param.color["group"]` > expert 参数用 EP 对应的 fsdp group > `hcg.get_sharding_parallel_group()`。
- `fsdp_unit_id`：对 `enumerate(model.modules())` 中类名（沿 `__mro__` 匹配）属于 `fsdp_unit_layers` 的模块取其下标；默认 `['TransformerLayer','Qwen3VLTextDecoderLayer','Qwen3MoeDecoderLayer']`。不在任何 unit 内的参数（embedding、final norm、lm_head、MTP 头等）`unit_id=None`。

结果：**一个 unit 常常对应多个 group**（dense / expert / muon / no_decay / 不同 clip 属性），每个 group 一块独立 buffer、一次独立 AG / RS。

### 2.2 排序（L474–492）

```
tie group 最前 → 非 trainable → trainable
同类内：unit_id 升序（None 视为 +inf）→ is_expert → use_muon
```

排序后的 group 下标 `gid` 就是**前向预取顺序**（反向取反）。后果：

- tie group（embedding/lm_head 共享权重）排在 unit 0 之前；
- `unit_id=None` 的非 tie 参数（final norm、lm_head 非 tie 时）排在最后；
- 预取完全依赖这个静态顺序，与真实执行顺序（如 MTP、条件分支）不一致时预取会落空或预取错对象。

---

## 3. TensorFusionBuffer：布局、分片与临时 buffer（L104–385）

每个 BufferGroup 有一对 buffer：`params_buffer`（`is_params=True`）和 `grads_buffer`（trainable 才有）。

### 3.1 布局与 padding

- 参数按 group 内顺序拼接，`param_offsets[name]` 记录起点。
- `get_padded_size`（L262）：每个参数按 `alignment[device] // align[dtype] × fsdp_degree` 向上取整，所以 `total_buffer_size` 必然是 `fsdp_degree` 的整数倍；分片是对**整块扁平 buffer 均分**，单个参数可以跨 rank。
- Muon 3D 参数传 `shard_align_numel`（`_muon_3d_shard_numel`，即 `s[-2]*s[-1]` 的 lcm）；若无法对齐，则该 buffer **退化为不分片**（`is_sharded=False`，L147–155），通信改为 step 末的 all_reduce / reduce(owner)。

### 3.2 params_buffer

| 成员 | 形状 / dtype | 生命周期 |
|---|---|---|
| `data_buffer` | `[total/degree]`，参数 dtype | 常驻，即优化器看到的 fused 参数 shard |
| `tmp_data_buffer` | `[total]`，`paddle.empty` | AG 目标；按需分配，`clear_tmp_buffer()` 立即释放 |
| `status` | BufferState | 见 §6 |
| `comm_task` | NCCL task | 异步 AG 句柄 |

初始化（L160–200）：先在 GPU 上 `paddle.zeros([total])` 建完整 buffer → 逐参数 `assign` → `param._clear_data()` → `empty_cache()` → 切出本 rank 的 shard 并 `.clone()`。代码自带 TODO：完整 buffer 应在 CPU 上构建，否则初始化显存峰值 ≈ 未分片模型。

参数绑定：`all_gather_params` 中用 `view_slice(tmp, off, off+numel)` + `_share_data_with` 让原 `param` 指向 tmp 的切片（L771–779），所以 param 本身零拷贝；`shard_params` 时 `param._clear_data()` 只解除引用。

### 3.3 grads_buffer

| 成员 | 形状 / dtype | 生命周期 |
|---|---|---|
| `data_buffer` | `[total/degree]`，`main_grad_dtype` | 常驻，跨 micro-batch 累加的 grad shard（optimizer 输入） |
| `tmp_data_buffer` | `[total]`，`paddle.zeros` | 本次 backward 的**完整** grad，`param.main_grad` 是它的 `view_slice`；RS 完成后释放 |

- `main_grad_dtype`：expert / fp32 参数为 fp32；否则取 `mp_policy.reduce_dtype`（默认 fp32）。示例配置下 dense 与 expert 都是 fp32。
- `grad_div = fsdp_group.nranks`（可被 trainer 的 expert hack 改写），`grad_scale = 1/(grad_div × grad_accum_steps)`。
- `param.get_main_grad` 是绑定到参数上的闭包（L231–246），首次访问时 lazy 分配 tmp。

显存含义：每个"正在累加梯度"的 group 都持有一块**完整的 fp32** tmp grad（参数 bf16 时是参数完整大小的 2 倍）。在飞 RS 队列的上限决定了同时存在多少块（§11）。

---

## 4. 通信与流：C++ ProcessGroupNCCL 行为

### 4.1 Python 侧

```python
# AG（L302）
paddle.distributed.all_gather(tmp, data_buffer, group=fsdp_group, sync_op=False)
# RS（L350），输出原地写到输入 buffer 头部
paddle.distributed.reduce_scatter(tmp[0:shard], tmp, op=AVG|SUM, group=fsdp_group, sync_op=False)
# 同步 AG 兜底（L765）
fsdp_group.process_group.all_gather(data_buffer, tmp).wait()
```

### 4.2 C++ `ProcessGroupNCCL::Collective()`（process_group_nccl.cc ~L1132）

```
if (!use_calc_stream) SyncCalcStream(place)      # comm stream 等 calc stream 的 event
                                                  # （outer_wait_ 为真时跳过，是否设置未核实）
comm_ctx = GetCommContext(place)                  # 每 (PG, place) 一个 NCCL comm + 一个 comm stream
run nccl collective on comm_ctx->stream()
if FLAGS_use_stream_safe_cuda_allocator (默认 true, allocator_facade.cc:114):
    memory::RecordStream(holder, nccl_stream)    # 输入/输出 allocation 记录 comm stream
task->UpdateWaitChain(comm_event)
allocation_stream_pairs_.push_back(...)
```

`NCCLTask::Wait()`（~L108）：让 **calc stream 等 comm_event**（GPU 侧等待，CPU 不阻塞，除非 `FLAGS_nccl_blocking_wait`），随后 `RemoveHolderStreamInGroup` → `EraseTensorHolders`（~L1401）清掉该 PG **所有**已记录的 allocation-stream 对。

### 4.3 推论

1. **AG 与 RS 同 PG → 同 comm stream、同 communicator，严格 FIFO 串行。** 反向时 RS(unit i) 与 AG 预取(unit i-2) 被排在同一条流上，RS（fp32、2× 字节）会挡住后面的 AG（head-of-line blocking）。Megatron 用独立 AG/RS stream + 可选独立 AG communicator 规避这一点。
2. **内存安全靠 `record_stream`**：`_clear_data()` 释放一块仍在 comm stream 上使用的 tmp 是安全的，但 allocator 只有在 comm stream 的 event 完成后才能复用该块，复用被推迟；在 tmp 频繁分配/释放的模式下容易造成碎片和峰值上升。这是 Megatron 与 FSDP2 都刻意避免的做法。
3. `SyncCalcStream` 让每次发起集合通信时 comm stream 都等 calc stream 当前所有已发出的 kernel，因此**预取 AG 的最早启动时刻 = 发起时刻之前的所有计算完成**。预取发起点越晚（§6.3），overlap 窗口越小。
4. step 末 `paddle.device.synchronize()` 的注释是"让 backward 中 comm stream 上释放的 buffer 回到池子"——正是 `record_stream` 延迟回收的副作用。

---

## 5. Hook 体系（`register_tensor_fusion_hooks`，L1921–2068）

注册规则：递归遍历模型；遇到 FSDP unit 层注册 `recursive=True` 的 hook 并停止下钻；非 unit 层只要自身直接持有参数，就注册 `recursive=False` 的 hook（embedding、final norm、lm_head 都属于这一类）。

| 时机 | 挂载方式 | 动作 | 对应 Megatron |
|---|---|---|---|
| 顶层 model pre-forward | `register_forward_pre_hook` | `_bind_expert_main_grads(all params)`：非分片 expert 参数重绑 main_grad | — |
| layer pre-forward | `register_forward_pre_hook` | `all_gather_params(params)` → 等待本层 AG → 预取下一 unit；`_bind_expert_main_grads` | `_pre_forward_param_unshard` |
| layer post-forward | `register_forward_post_hook` | `shard_params(params)`：`_clear_data` + USING→READY（**懒释放**，tmp 仍保留）；被输出"暴露"的参数（fused linear+CE 的 lm_head 权重）跳过 | `_post_forward` release |
| layer pre-backward | post-forward 把输出包进 `FusionBackwardHook`（PyLayer，前向恒等），其 `backward` 调 `all_gather_params(trainable, is_backward=True)` | 结构触发 RS（§8.2）→ 释放后续 unit 参数 → 等本层 AG → 预取前一 unit | `_pre_backward` + `_register_pre_backward_param_unshard_hook` |
| 参数 grad 就绪 | `param._register_grad_hook(comm_hook)`（L1867） | 累加到 tmp grad → `shard_params([param])` → `reduce_scatter_grads(param)`（只计数，不发通信） | `_process_post_backward_gradients` |
| tie 参数所在层 | 额外 `FusionForwardHook`（pre-forward 包输入），其 backward 中 `shard_params` | 共享权重第二次使用的反向结束后 reshard | — |
| step 末 | `Optimizer._apply_optimize` 调 `comm_sync_and_reset_status` | §9 | `finish_grad_sync` + `_root_post_backward` |

关键差异：

- **没有 module 级 post-backward hook。** 参数 reshard 是逐参数在 grad hook 里做的；RS 不由"本 group 全部 grad 就绪"驱动，而是由"反向进入了前一个 unit"驱动（§8）。
- **没有 root pre-backward / backward-final hook**，因此无法区分"真实 forward"与"recompute forward"（§10.1），也无法在反向结束时异步 flush 剩余 RS。
- `comm_hook` 中 `reduce_scatter_grads` 只做 `grads_use_cnt += 1` 和 `main_grad=None`，`grads_use_cnt` 不再被任何逻辑读取（死代码）。

---

## 6. BufferState 状态机与 prefetch

### 6.1 状态机（params_buffer，L54）

```
            _issue_async_gather                      comm_task.wait()
   FREED ───────────────────────────> SYNCING ─────────────────────────> USING
     ^  \                                                                 |  ^
     |   \ 同步 all_gather().wait()（预取未命中）                          |  |
     |    └───────────────────────────────────────────────────────────────┘  |
     |                                                       shard_params    |
     |   _release_one_buffer_if_needed /                          v          |
     |   _release_params_after_unit /                           READY ───────┘
     └── reset_params_buffer_status  <────────────────────────── (再次 all_gather_params 时直接复用)
```

- READY = 参数已 reshard（param 不再引用 tmp），但 tmp 中的完整参数仍然有效，可零通信复用。前向最后几个 unit 在反向开始时通常仍是 READY → 反向第一个 unit 免 AG（类似 FSDP2 对最后一层 `reshard_after_forward=False`）。
- `buffer_cnt_in_using` 统计 SYNCING + USING + READY 的分片 buffer 数，是唯一的显存预算变量。

### 6.2 预算参数（L641–651）

```python
unit_sizes      = Counter(g.fsdp_unit_id for g in groups if g.params_buffer.is_sharded and g.fsdp_unit_id is not None)
groups_per_unit = max(2, max(unit_sizes.values(), default=0))   # 取"最多 group 的 unit"
prefetch_units  = 1                                              # 默认，未暴露
gather_buffer_limit = groups_per_unit * (prefetch_units + 1)     # 当前 unit + 预取 unit
grad_queue_limit    = groups_per_unit * max(prefetch_units, 1)   # 在飞 RS 数
```

例：每个 MoE 层有 dense + expert 两个分片 group → `groups_per_unit=2`，`gather_buffer_limit=4`，`grad_queue_limit=2`。

注意这三个上限都是**个数**而非字节：dense group 与 expert group、embedding 大 group 与 norm 小 group 一视同仁。若某个 unit 的 group 更多（如加了 no_decay / muon 变体），所有 unit 都按最大值放宽预算。

### 6.3 `all_gather_params(params, is_backward)` 流程（L710–795）

```
req_gids = params 所属 group 的去重列表
[仅反向] 若 unit_id 变化：
    _flush_expert_grads_after_unit(unit_id)   # 对 unit_id > 当前 的所有 group 发 RS（§8.2）
    _release_params_after_unit(unit_id)       # 把 unit_id > 当前 的 READY/USING 参数 buffer 直接 FREED
    _bind_sharded_expert_main_grads(params)
[仅前向] 若 grad_reduce_queue 非空：退役 1 个在飞 RS（上个 micro-batch 遗留）
[overlap] 对 req_gids 中仍为 FREED 的分片 group：先 _release_one_buffer_if_needed(keep=req) 再异步 AG
逐参数：
    READY   -> USING                      （复用，无通信）
    SYNCING -> USING  comm_task.wait()    （calc stream 等 comm event）
    FREED   -> USING  同步 AG().wait()     （仅 overlap=False 或特殊情况）
    view_slice + _share_data_with 绑定参数
[overlap] 预取：for gid in _next_buffer_id(req_gids, is_backward):
    跳过非分片 / 非 FREED；_release_one_buffer_if_needed(keep=req)
    若仍 >= gather_buffer_limit 则停止；否则 _issue_async_gather(gid)
```

要点：

- **预取在 wait 之后发起**（L781 注释："否则 compute 会卡在它上面"）。按 comm stream FIFO 的语义，AG(i) 已先入队，先发预取并不会推迟 AG(i)；源码注释所说的卡顿机制未从 C++ 侧得到印证，较可能的实际作用是保证当前 unit 的 buffer 先拿到预算【推断】。
- 由于 `SyncCalcStream`，AG(i+1) 在 comm stream 上要等 unit i pre-forward 之前已发出的全部计算完成才启动，随后与 unit i 的计算并行。这与 Megatron/FSDP2 一致：**提前量正好 1 个 unit**。

### 6.4 `_next_buffer_id`（L678–702）：预取多少

- 方向：前向 `+1`、反向 `-1`，从 `max(req_gids)+1`（反向 `min-1`）开始，跳过本次请求的 gid。
- 反向跳过非 trainable group（它们不需要反向 AG）。
- 连续收集，直到遇到第 `prefetch_units+1` 个不同的 `fsdp_unit_id` 为止 → **默认预取紧邻的下一个 unit 的所有 group**。
- `unit_id=None` 被当作一个"unit"：例如前向在最后一个 unit 时，会把 final norm / lm_head 等所有 None group 一并作为下一个 unit 预取；反向在 unit 0 时会预取 tie group（排在最前）。
- 受 `gather_buffer_limit` 截断：如果 READY buffer 都在 `keep` 中或不存在，预取直接放弃（静默，不报告）。

### 6.5 驱逐：`_release_one_buffer_if_needed`（L660–676）

```python
while buffer_cnt_in_using >= gather_buffer_limit:
    for gid, group in enumerate(groups):          # 从 0 开始顺序扫描
        if gid in keep or not sharded: continue
        if status == READY: FREED + clear_tmp_buffer; cnt -= 1; break
```

问题：

1. **不是 LRU，也不按距离**：前向时，下标小的 READY（刚用完的 unit）被优先释放，符合预期；但**反向时**，下标小的 READY 恰恰是**即将用到**的 unit（例如前向末尾留下的 READY、或 tie group），会被先驱逐，造成"刚预取/保留的 buffer 被扔掉、马上又同步 AG"。
2. tie group 永远是 gid 0，只要处于 READY 就第一个被释放，lm_head 与 embedding 共享权重时容易反复 AG【推断】。
3. 反向时的主要释放手段其实是 `_release_params_after_unit`（进入新 unit 时把所有后续 unit 的参数直接 FREED），它在预取之前执行，所以反向 READY 驱逐问题通常被掩盖；但 recompute 和 None group 会打破这个前提（§10.1）。

---

## 7. 时间线

记号：E = embedding（tie group，gid 0，unit None），L0..L{N-1} = FSDP unit，H = final norm + lm_head 等非 tie 的 None group；每个 unit 2 个分片 group，`gather_buffer_limit=4`，`grad_queue_limit=2`。单一 comm stream 记为 `comm`。

### 7.1 前向（稳态，unit i）

```
CPU   pre-fwd(Li): wait AG(Li) ── issue AG(Li+1) ── [launch fwd(Li) kernels] ── post-fwd(Li): Li USING->READY
calc  ...fwd(Li-1)...|wait ev(AG Li)| fwd(Li) ..................................|
comm                  | AG(Li+1)（先等 calc 上已发出的 fwd(Li-1)，再与 fwd(Li) 并行）|
buf   Li-1 READY 被 _release_one_buffer_if_needed 释放 → 腾出预算给 Li+1
```

- micro-batch 内第一个前向层（E）以及每个 step 的第一个 micro-batch：E 的 AG 没有预取者，属于**暴露的通信**；L0 的 AG 在 E 的 pre-forward 才发起，只能与 embedding 计算重叠（embedding 计算很轻，基本也暴露）。
- 非首个 micro-batch：上一轮反向末尾 L0、E 仍为 READY（`_release_params_after_unit(None)` 直接 return），前向可零通信复用。
- 前向每进入一个层，若 `grad_reduce_queue` 非空，退役一个上个 micro-batch 遗留的 RS（`task.wait()` + calc stream 上 `add_` 累加）。

### 7.2 反向（稳态，unit i）

```
CPU   pre-bwd(Li):  flush: issue RS(Li+1 的全部 group) ── release Li+1 params ── wait AG(Li) ── issue AG(Li-1)
calc  ..bwd(Li+1)..|wait ev(AG Li)| bwd(Li)（grad hook: main_grad.add_ 到 tmp(Li)；参数逐个 reshard）|
comm               | RS(Li+1) fp32 ──────────────── | AG(Li-1) bf16 ── |
                     ^ 与 AG(Li-1) 同一条流，AG 必须排在 RS 之后
```

- RS(Li+1) 与 AG(Li-1) 同时在 pre-bwd(Li) 发起，但串行执行：AG(Li-1) 的完成时刻 = RS(Li+1) + AG(Li-1) 的总通信时间，必须落在 bwd(Li) 之内才不暴露。fp32 RS 的字节数是 bf16 AG 的 2 倍，所以反向每个 unit 的通信量约为前向的 3 倍，而计算约为 2 倍；**反向比前向更容易通信受限**【推断，需 profile 验证】。
- `_reduce_group_grads` 发起 RS 前调 `_wait_for_grad_comm()`，在飞 RS 超过 `grad_queue_limit` 时，calc stream 等最老的 RS 完成并在 calc stream 上做 `data_buffer.add_(shard)`。

### 7.3 反向尾部与 step 边界

```
pre-bwd(L0):  issue RS(L1) ; wait AG(L0) ; prefetch E(tie)
bwd(L0)
pre-bwd(E):   unit_id None ⇒ _flush_expert_grads_after_unit(None) 直接返回 ⇒ RS(L0) 不发！
bwd(E)
-- micro-batch 结束：tmp(L0)、tmp(E)、tmp(H) 保持未 reduce，继续跨 micro-batch 累加 --
...（共 accum 个 micro-batch）
optimizer step → comm_sync_and_reset_status():
    finish_grads_sync:  等完 RS 队列 → 对 L0 / E / H 等 tmp 非空的分片 group 依次 **同步** RS + add_
    reset_params_buffer_status: 等完 SYNCING，全部 buffer FREED
    need_zero_grads = True
    paddle.device.synchronize()
```

---

## 8. 梯度路径

### 8.1 grad hook（`comm_hook`，L1867–1917）

```
_maybe_zero_grads()                 # 每 step 第一次：所有 grads data_buffer.zero_()
_ensure_grads_writable(param)       # 若本 group 的 tmp 仍在 RS 队列里（recompute / tie / 迟到 grad），
                                    # 循环退役队列直到它出队 —— 之后 get_main_grad 会新建一块 zeros tmp
if param._fused_kernel_fp32_grad:   # fused kernel 预先算好的 fp32 grad（MTP 时是 list，按序逐个 add）
    main_grad.add_(fp32_grad); grad._clear_data()
else:
    main_grad = view_slice(tmp, ...); main_grad.add_(grad); grad._clear_data()
shard_params([param], is_backward=True)   # 参数逐个解除对 tmp 的引用；group USING->READY
reduce_scatter_grads(param)               # 仅 grads_use_cnt += 1; main_grad = None
```

梯度先落在 autograd 产生的 bf16/fp32 `grad` 上，再 `add_` 进 fp32 tmp —— 每个参数一次额外的 elementwise kernel 和一次 grad 张量的分配/释放。

### 8.2 RS 触发：结构触发（L903–931）

```python
# 在 all_gather_params(is_backward=True) 中，unit_id 变化时：
def _flush_expert_grads_after_unit(unit_id):   # 名字带 expert，但对 dense group 同样生效
    if unit_id is None: return
    for group in reversed(groups):
        if group.fsdp_unit_id is not None and group.fsdp_unit_id > unit_id:
            _reduce_group_grads(group)

def _reduce_group_grads(group):
    if 未分片 or tmp is None or 已在队列: return      # 幂等守卫
    _wait_for_grad_comm()                             # 队列 > grad_queue_limit 时退役最老的
    grads_buffer.comm_task = do_reduce_scatter()      # 异步
    grad_reduce_queue.append(grads_buffer)
    if not enable_overlap: _wait_for_grad_comm(0)
```

含义：

- unit i 的 RS 在 **unit i-1 的 pre-backward** 才发起，比 Megatron"bucket group grad 齐了就发"晚。晚多少取决于 unit i 最后一个 grad hook 与 unit i-1 输出梯度到达之间的间隔：对标准 Transformer 层两者紧挨着（输入梯度与权重梯度在同一个 grad node 里），损失不大；但对于权重梯度节点不在"到输入的路径"上的情况，unit i 的部分 grad 可能在 RS 发起之后才到，此时 `_ensure_grads_writable` 会让 calc stream 等待该 RS、再新建 tmp，残余部分留到 `finish_grads_sync` 同步 RS —— 正确，但多一次通信且无 overlap【推断】。
- 使用"unit_id 变化"作为信号，因此**所有 `unit_id=None` 的 group 和 unit 0 都没有触发点**（§7.3）。

### 8.3 RS 本体（L310–356）

- 输出 `shard = tmp[0:shard_numel]`，与输入 `tmp` **部分重叠**。NCCL 定义的 in-place RS 要求 `recvbuff == sendbuff + rank*recvcount`；这里对 rank 0 恰好是 in-place，对 rank≠0 则是"输出覆盖了别人那段输入"的非标准别名，NCCL 文档对此不作保证（ring 算法下最终写发生在最后一步，实践中大概率无问题，但依赖实现细节）。`grads_buffer.shard_start = rank*shard_size` 已有，改成标准 in-place 只需换切片起点（见 §13 P2-5）。
- 缩放：`grad_div × grad_accum_steps == fsdp_degree` → `ReduceOp.AVG`；否则先对**整块完整 tmp** 做 `scale_(1/(div×accum))` 再 SUM。示例配置 accum=32 永远走 `scale_` 分支：每个 micro-batch 每个 group 多一次完整 fp32 buffer 的读写。
- `MUON_FSDP_BCAST_GRAD=1`：对齐调试用的同步 broadcast 路径，不在性能讨论范围。

### 8.4 退役与累加（L969–981, L358）

```
_wait_for_grad_comm(limit):
    while len(queue) > limit:
        buf = queue.pop(0)
        buf.comm_task.wait()                 # calc stream 等 RS event
        data_buffer.add_(tmp[0:shard])       # 在 calc stream 上累加
        tmp._clear_data()                    # 释放（record_stream 保证 comm 用完才复用）
        释放各 param 的 main_grad view
```

退役点：发起新 RS 前（limit=`grad_queue_limit`）、下一 micro-batch 前向每层退役 1 个、`_ensure_grads_writable`、`finish_grads_sync`（limit=0）。累加 `add_` 放在 calc stream 上，与 Megatron 在 RS stream 上 `+=` 不同：这一步占用计算流，并且迫使 calc stream 在此处等待 RS 完成。

### 8.5 梯度累加语义（无 `no_sync`）

| group 类别 | 每个 micro-batch 的行为 | 一个 step（accum=A）的 RS 次数 | 常驻显存 |
|---|---|---|---|
| unit 1..N-1 的分片 group | 反向中结构触发异步 RS，结果 `add_` 进 shard | **A 次** | shard + 在飞的 tmp |
| unit 0 的分片 group | 无触发点，tmp 跨 micro-batch 累加 | 1 次（step 末同步） | **完整 fp32 tmp 常驻整个 step** |
| `unit_id=None` 的分片 group（E / H） | 同上 | 1 次（step 末同步） | 完整 fp32 tmp 常驻整个 step |
| 未分片 group（Muon 不对齐、nranks=1 的 EP group） | 直接累加在 `data_buffer` | step 末 1 次 all_reduce/reduce | 完整 buffer |

即：中间各层是 ZeRO-3 式"每 micro-batch RS"，首层和非 unit 部分则意外地是"no_sync"式。示例配置 `gradient_accumulation_steps=32`，中间层的 RS 通信量是理论最小值的 32 倍，而 embedding/lm_head（词表大，往往是最大的 group）的 RS 又全部堆在 step 末同步执行。

### 8.6 `finish_grads_sync`（L983–1065）

1. `_wait_for_grad_comm(0)`：退役全部在飞 RS。
2. 未分片 group：Muon 对齐的 `all_reduce(SUM)`，或按 owner `reduce`（非 owner 随后释放 grad buffer）；然后对 `data_buffer` 做一次 `scale_(grad_scale)`。全部 `sync_op=True`。
3. 分片且 tmp 非空的 group（unit 0、E、H、迟到 grad 残余）：`do_reduce_scatter().wait()` + `accumulate_reduced_grad()`，**逐 group 串行、同步**。
4. 清理所有 `main_grad` view。

步骤 2、3 全部在关键路径上，与任何计算都不重叠。

---

## 9. step 边界：`comm_sync_and_reset_status`（L1621–1631）

```
finish_grads_sync()               # §8.6
reset_params_buffer_status()      # SYNCING 的预取（反向末尾发出、拿的是旧权重）先 wait 再丢弃；
                                  # READY/USING 全部 _clear_data + clear_tmp_buffer -> FREED
need_zero_grads = True            # 下一 step 第一个 hook 时 zero_ 所有 grad shard
main_grad = None
paddle.device.synchronize()       # 全设备同步，让 record_stream 延迟回收的块回到池子
```

与 Megatron 对比：

- Megatron 的 optimizer step 之后可以用 `fsdp_all_gather_in_start_param_sync` 立刻对第一个 bucket 发 AG，与 optimizer 尾部/数据加载重叠；Paddle 下一 step 第一层的 AG 完全暴露。
- `paddle.device.synchronize()` 阻塞 CPU，CPU 侧 launch 无法提前于 GPU，下一 step 开头会有一段 GPU 空泡（大小 ≈ CPU 从 sync 返回到下一个 kernel 发出的时间）。
- 反向中发出、但在 step 结束前未被消费的预取 AG（静态 gid 顺序与实际执行顺序不一致、或 recompute 触发的前向方向预取，见 §10.1）会在这里被 wait 后丢弃，属于浪费的通信。

---

## 10. 特殊场景

### 10.1 recompute

`fully_shard_fusion.py` 中没有任何 recompute 相关逻辑（grep 无结果）。分两种情况：

- **unit 内部的 selective recompute**（示例配置 `recompute_granularity: selective`、`recompute_modules: ["moe_gate_up"]`）：hook 只挂在 unit 层上、不下钻子模块，重算子模块时不会触发 FSDP hook，参数在 unit 反向期间本来就是 USING，**没有问题**。
- **整层 recompute**（unit 层本身被 recompute 包裹）【推断，未实测】：反向中重跑 unit i 的 forward 会触发 pre-forward hook → `all_gather_params(is_backward=False)`：
  - 按前向方向预取 unit i+1 —— 它刚在 `_release_params_after_unit` 中被释放，且反向不再需要，是**白做的 AG**，还占用 comm stream 和 buffer 预算，进一步推迟真正需要的 AG(i-1)；
  - 走"前向退役一个 RS"分支，把本应与计算重叠的 RS 提前 wait；
  - post-forward 再包一层 `FusionBackwardHook`，unit i 的 pre-backward 逻辑执行两次（第二次因幂等守卫基本无副作用）。

Megatron 的做法：root pre-backward 把所有 bucket 标记为 `can_be_released`（懒释放），并在 backward 阶段禁用 forward prefetch。

### 10.2 tie 权重与"暴露参数"

- tie group 排在 gid 0；共享权重的层额外挂 `FusionForwardHook`，在该层输入梯度回传时 reshard。
- fused linear + CE 的 lm_head 会把 `weight` 作为输出交给 loss 模块分块计算：post-forward 跳过对它的 reshard（`_find_exposed_param_names`），由其 grad hook 在梯度到达后 reshard。
- 两者都依赖 `_ensure_grads_writable`：tie 权重第二次产生梯度时，若第一份已进了 RS 队列，会先等它完成再新建 tmp。

### 10.3 Muon / 未分片 group

Muon 3D 参数要求分片边界按矩阵对齐，不满足时整个 group 不分片：参数常驻完整副本、不走 AG；梯度直接累加进完整 `data_buffer`，step 末 all_reduce 或 reduce 到 owner。这些 group 不计入 `buffer_cnt_in_using`，通信全部同步、在 step 末。

### 10.4 `unshard_params` / 保存

`unshard_params(params)`（L1458）是同步 gather、不预取的上下文管理器，供 FP8 离线量化使用；保存前 `all_gather_params()`。两者都不在训练热路径上。

---

## 11. 显存账（单卡，峰值估算）

记 `P_u` = 一个 unit 的参数字节（bf16），`G_u` = 一个 unit 的 fp32 grad 字节（≈ 2 P_u），`d` = fsdp degree，`P_E/P_H` 为 E/H 的参数字节。

| 项 | 大小 | 说明 |
|---|---|---|
| 参数 shard | `ΣP / d` | 常驻 |
| grad shard（fp32） | `2ΣP / d` | 常驻 |
| AG tmp（参数完整） | `≤ gather_buffer_limit` 块，约 `(prefetch_units+1) × P_u` | 当前 unit + 预取 unit；若 E/H 大于 `P_u`，以它们为准 |
| 在飞 RS 的 tmp（fp32 完整） | `≤ (grad_queue_limit+1)` 块，约 `G_u ~ 1.5 G_u` | 队列上限 + 正在累加的那一块 |
| 当前正在累加的 tmp(Li) | `G_u` | |
| unit 0 / E / H 的 tmp | `G_0 + 2P_E + 2P_H` | **跨 micro-batch 常驻整个 step**；大词表时 E/H 可能是最大的一块 |
| `record_stream` 延迟回收 | 不定 | tmp 释放后要等 comm event 才能复用，峰值可能比上式高 |
| 初始化峰值 | `≈ P_group` 的完整 GPU buffer + 原参数 | 每个 group 依次构建，见 §3.2 TODO |

---

## 12. 与 Megatron-FSDP（及 PyTorch FSDP2）的逐项对比

| 方面 | Paddle FSDP | Megatron-FSDP | FSDP2（参考） | 差距影响 |
|---|---|---|---|---|
| 预取顺序 | 静态 gid 顺序（tie / 非 trainable 在前，unit 升序） | 静态 bucket 顺序 ±1 | 前向记录的真实 post-forward 顺序反转 | 非线性结构下 Paddle 易预取错 |
| 预取量 | 1 个 unit（`prefetch_units=1`，不可配） | 约 1 个 unit（`suggested_AG_prefetch_size`，可配） | 1 个 module，可手动 `set_modules_to_forward/backward_prefetch` | 无法针对慢网络加深预取 |
| AG 预算 | buffer 个数 `groups_per_unit×2` | 元素数 / 固定池 | 由 `reshard_after_forward` + 预取数决定 | 大小 group 混合时不准 |
| 驱逐 | 从 gid 0 扫第一个 READY | 显式 release，recompute 下懒释放 | post-forward / post-backward 显式 reshard | 反向可能逐出即将用到的 buffer |
| AG / RS 流 | 同一 comm stream | 独立 AG stream、RS stream | 独立 AG stream、RS stream | **反向 AG 被 RS 阻塞** |
| communicator | AG/RS 同一 NCCL comm | 可 `--create-all-gather-group` 独立 | 同一 PG，但流独立 | 同上 |
| 内存安全 | `record_stream` + event | event / `wait_stream`，无 `record_stream` | 同 Megatron（显式持有到 event 完成） | 复用延迟、碎片，需要每步 device sync |
| AG 合并 | 每个 group 单独一次 AG | bucket group 合并成一次 | 一个 module 的参数一次 AG（foreach copy-in） | 一个 unit 多次小 AG，launch/latency 开销 |
| RS 触发 | 进入前一 unit 时结构触发 | bucket group grad 齐即触发 | post-backward hook 触发 | 首/尾无 overlap |
| 首层 / 非 unit RS | step 末同步 | 正常 overlap | root post-backward 中 flush | **step 末串行暴露** |
| RS 并发 | `groups_per_unit` 个 | 约 2 个 unit | 1 个（下一个 RS 前 wait 上一个） | — |
| RS 输出与累加 | 写回输入头部；calc stream `add_` | 临时 shard；RS stream `+=` | RS stream 上 `+=` 或直接写 | 占用计算流 |
| 缩放 | accum>1 时整块 `scale_` + SUM | AVG / `premul_sum` / `mul_`+SUM | `premul_sum` / divide factor | 额外一次 fp32 全量读写 |
| reduce dtype | fp32（`amp_master_grad`） | 可 bf16 通信 + fp32 累加 | `reduce_dtype` 可配 | RS 字节翻倍 |
| 梯度累加 | 中间层每 micro-batch RS；无 `no_sync` | `optim`/`no_shard` 可只在最后一个 micro-batch reduce | `set_requires_gradient_sync(False)` | **RS 通信 × accum** |
| recompute | 无感知 | 有专门处理 | 通过 hook 状态区分 | 整层重算时浪费 AG |
| step 起点 | 首个 AG 暴露 + `device.synchronize()` | 可在 param sync 起点提前发 AG | 首 AG 暴露 | 每步固定空泡 |
| buffer 分配 | caching allocator，每次 `empty/zeros` | FixedPool / MaxPool 双 buffer、NCCL UB | caching allocator + 显式生命周期 | 碎片 / memset 开销，无 NVLS/UB |

---

## 13. 优化点（按优先级）

记号：A = gradient_accumulation_steps，P = 本卡对应的完整参数字节（bf16）。一个 step 内中间层的通信字节量（按 ring 的 `(d-1)/d` 系数省略）：

```
前向 AG：A·P      反向 AG：A·P（最后一层 READY 复用可略减）      RS（fp32）：A·2P
合计 ≈ 4·A·P，其中 RS 占一半
```

### P0-1 梯度累加期间跳过 RS（no_sync / 延迟 reduce）

- 现状：中间层每个 micro-batch 都 RS（§8.5），示例配置 A=32。
- 方案：
  1. `FSDPCommManager` 增加 `grad_sync_enabled` 状态，`fully_shard` 上下文暴露 `no_sync()` / `set_requires_gradient_sync(bool)`；PaddleFleet 在非最后一个 micro-batch 关闭。
  2. 关闭时 `_reduce_group_grads` 不发 RS，tmp grad 跨 micro-batch 保留（目前 unit 0 / E / H 已经是这种行为，可复用）；最后一个 micro-batch 恢复正常的结构触发 RS。
  3. 由于完整 fp32 grad 常驻需要 `2P` 显存，增加按字节预算的**混合策略**：预算内的 group（优先选反向最后才算到的 unit，因为它们的 RS 本来就最难 overlap）走 no_sync，其余仍每 micro-batch RS。
- 收益：全部 no_sync 时 RS 从 `A·2P` 降到 `2P`，step 总通信字节约减半（A=32 时 ≈ 4AP → 2AP+2P）。
- 风险：显存增加最多 `2P`（MoE 大模型基本只能做部分 group）；数值上与现在等价（都是 fp32 求和后统一缩放），但求和顺序变化导致 bitwise 不同。

### P0-2 AG 与 RS 拆分 communicator / comm stream

- 现状：AG、RS 同一 PG → 同一 comm stream，反向 AG(i-1) 排在 RS(i+1) 后面（§4.3、§7.2）。
- 方案：在 `BufferManager` 构建时为每个分片 `fsdp_group` 额外 `paddle.distributed.new_group(fsdp_group.ranks)` 一个 AG 专用 group，`params_buffer.do_all_gather` 与同步 AG 兜底都改用它；RS 和 step 末 all_reduce 仍用原 group。所有 rank 必须以相同顺序创建。对应 Megatron 的 `--create-all-gather-group`。
- 收益：反向 AG 与 RS 并行，AG 不再被 2× 字节的 fp32 RS 阻塞；预期主要改善反向通信受限的 unit【需 profile 验证】。
- 风险：多一份 NCCL comm 的显存（每个 comm 数百 MB 级，与 `NCCL_BUFFSIZE`、channel 数有关）；两个 NCCL kernel 并发会抢 SM 和带宽，可能需要配合 `NCCL_MAX_NCHANNELS` 调优；需确认 BCCL 在同一组 rank 上多 communicator 的行为。

### P0-3 修正 RS 触发点：消灭 step 末同步 RS

- 现状：unit 0 与所有 `unit_id=None` 的分片 group 只能在 `finish_grads_sync` 串行同步 RS（§7.3、§8.6）；`grads_use_cnt` 已在 grad hook 里递增但无人读取。
- 方案 A（最小改动）：
  - `_flush_expert_grads_after_unit` 在 `unit_id is None` 且上一个 unit 不是 None 时（即反向从 unit 0 进入 embedding），flush 所有 `unit_id is not None` 的 group → unit 0 的 RS 与 embedding 反向重叠；
  - 反向首次进入某个 unit 时，flush 非 tie 的 None group（final norm / 非 tie lm_head 已经算完）。
- 方案 B（通用）：在 BufferGroup 上记录"本次 backward 预期的 grad hook 次数"（trainable 参数数；tie / exposed / MTP 多次使用的参数第一次 backward 时统计得到），`grads_use_cnt` 达到预期即发 RS；结构触发保留作为兜底，`finish_grads_sync` 只处理残余。
- 方案 C：用 `core.eager._add_backward_final_hook`（`group_sharded_fully_shard.py:469` 已有用例）在反向结束时**异步**发起剩余 RS，退役推迟到下一 micro-batch 前向，或最后一个 micro-batch 的 `finish_grads_sync`。
- 收益：step 末不再有串行的同步 RS；E/H（大词表时最大的 group）的 RS 可以和反向尾部或下个 micro-batch 前向重叠。
- 风险：方案 A/B 会让 unit 0 / E / H 也变成每 micro-batch RS，需要与 P0-1 一并设计，否则会**增加**这几个 group 的通信次数；tie 参数必须等两处梯度都到齐再发。

### P1-1 recompute 感知

- 现状：整层 recompute 时，反向中重跑的 forward 会做 +1 方向预取、提前退役 RS（§10.1）。
- 方案：`FSDPCommManager` 增加 `in_backward` 标志：第一次 `FusionBackwardHook.backward` 时置位，在 backward final hook 或 `comm_sync_and_reset_status` 中清除。`all_gather_params(is_backward=False)` 在 `in_backward` 时跳过前向预取和"退役一个 RS"，post-forward 的 `shard_params` 保持 READY（等于 Megatron 的懒释放 `can_be_released`）。
- 收益：去掉整层 recompute 下每个 unit 一次白做的 AG。示例配置是 unit 内 selective recompute，**当前不受影响**；整层 recompute 的配置受益。
- 风险：低；需要覆盖 PP 下多个 micro-batch 交错 forward/backward 的情况（1F1B 中 `in_backward` 不能用全局布尔，应按 micro-batch 或按 hook 调用栈判断）。

### P1-2 显存预算按字节计、驱逐按距离、暴露旋钮

- 现状：`gather_buffer_limit` 按 buffer 个数；驱逐从 gid 0 扫第一个 READY；`prefetch_units`、`fsdp_unit_layers` 不能从 `fully_shard` / PaddleFleet 配置。
- 方案：
  1. 预算改为字节：`max_gather_bytes = (prefetch_units+1) × max_unit_bytes`（或用户指定），`buffer_cnt_in_using` 换成在用字节数。
  2. 驱逐选择"沿当前执行方向距离最远 / 已经走过"的 READY buffer：前向优先释放 `gid < min(req)`，反向优先释放 `gid > max(req)`；tie group 特殊对待。
  3. `fully_shard(..., prefetch_units=, fsdp_unit_layers=)` 透传，PaddleFleet 增加对应 `trainer_args`。
- 收益：慢网络 / 大 EP 场景可以加深预取；避免反向逐出即将使用的 buffer。
- 风险：低；加深预取会增加 AG tmp 显存。

### P1-3 一个 unit 内多个 group 的 AG 合并

- 现状：dense / expert / muon / no_decay 等变体各自一次 AG；`groups_per_unit` 大时 NCCL launch 与延迟开销叠加。
- 方案（二选一）：
  1. 放宽分组 key：`no_weight_decay`、`clip_attrs`、`hyperball` 等只影响优化器的属性改为 buffer 内的分段元数据，而不是 buffer 的切分维度；同一 `(dtype, unit, fsdp_group)` 只保留一块 buffer。
  2. 保留分组，但同一 `fsdp_group` 上的多个 AG 用 `ncclGroupStart/End` 合并发出（需确认 Paddle Python 侧是否已有 coalescing 接口，未核实）。
- 收益：每个 unit 的 AG/RS 次数从 `groups_per_unit` 降到"不同 fsdp_group 数"（通常 dense 1 + expert 1）。
- 风险：方案 1 改动优化器侧（weight decay、grad clip 需要按段处理），工作量较大。

### P1-4 RS 缩放与累加

- 现状：A>1 时每次 RS 前对整块 fp32 tmp 做 `scale_`，然后 SUM；累加 `add_` 放在 calc stream。
- 方案：
  1. RS 一律用 `AVG`（除以 nranks），`1/A`（以及 EP hack 的额外系数）推迟到 `finish_grads_sync` 中对 **shard**（`1/d` 大小）做一次 `scale_`。每 step 从 `A` 次完整 buffer 读写降为 1 次 shard 读写。
  2. `accumulate_reduced_grad` 的 `add_` 放到 comm stream 上（`paddle.device.stream_guard`），calc stream 只等累加完成的 event；或改为 RS 直接输出到独立 shard 后在 comm stream 上累加。
  3. 输出切片改为标准 in-place 位置 `tmp[shard_start:shard_end]`（§8.3）。
- 收益：去掉 `A × groups` 次大 elementwise kernel；消除对 NCCL 非标准别名的依赖。
- 风险：方案 1 改变浮点求和顺序（先和后除 vs 先除后和），fp32 下差异很小，但对齐实验需重新出基线；方案 2 要处理 comm stream 句柄的获取与跨流生命周期。

### P2-1 step 边界：提前发首个 AG，去掉 `device.synchronize()`

- 方案：
  1. 优化器更新完参数 shard 后，立即对前向最先用到的 group（tie group、unit 0）发异步 AG，等价于 Megatron 的 `fsdp_all_gather_in_start_param_sync`，与 optimizer 尾部、日志、dataloader 取数重叠。需要一个"optimizer step 之后"的挂点（可放在 `_apply_optimize` 返回后）。
  2. `paddle.device.synchronize()` 是为了让 `record_stream` 延迟回收的 tmp 回池。若 tmp 改为 P2-2 的固定池，或改为显式持有到 event 完成再释放（不调用 `record_stream`），即可删除该同步。
- 收益：每个 step 去掉一次首层 AG 暴露 + 一段 CPU 同步造成的 GPU 空泡。A 很大时占比小，A 小（如 RL / SFT）时明显。
- 风险：提前 AG 必须在参数更新的 kernel 之后（同一 calc stream 上发起，靠 `SyncCalcStream` 保证）；去掉 synchronize 后需要检查显存峰值是否回升。

### P2-2 AG / RS tmp 用固定池（双 buffer）替代每次分配

- 现状：每次 AG 都 `paddle.empty([total])`、每次累加都 `paddle.zeros([total])`，释放走 `_clear_data` + `record_stream`。
- 方案：按 `max_unit_bytes` 预分配 `prefetch_units+1` 个 AG 槽和 `grad_queue_limit+1` 个 RS 槽，buffer 状态机只在槽之间轮转（Megatron FixedPool / MaxPool 思路）。grad 槽在 RS 完成后用一次 `zero_` 复用；或者记录每个参数是否首次写入，第一次用 `copy_` 代替 `add_`，省去 memset。
- 收益：消除 allocator 碎片和 `record_stream` 依赖；固定地址也是后续 NCCL user buffer 注册 / symmetric memory（NVLS）以及 CUDA graph 的前提。
- 风险：固定池按最大 unit 预留，E/H 大 group 需要单独槽，否则浪费显存。

### P2-3 bf16 RS（可选）

- 现状：`amp_master_grad=True` 时 RS 为 fp32。
- 方案：增加"通信 dtype 与累加 dtype 分离"的选项：tmp 仍 fp32 累加单个 micro-batch，RS 前转 bf16，RS 结果在 fp32 shard 中累加。
- 收益：RS 字节减半；与 P0-1 叠加时边际收益变小。
- 风险：数值精度，需要 loss 对齐验证，默认关闭。

### P2-4 其它

- 初始化：按代码 TODO 在 CPU 构建完整 buffer，只把 shard 拷到 GPU，消除初始化峰值。
- grad hook 中 `main_grad.add_(grad)`：若 fused wgrad kernel 能直接累加进 `main_grad`（类似 Megatron 的 gradient accumulation fusion，项目中已有 `_fused_kernel_fp32_grad` 的雏形），可省掉 grad 张量的分配和一次 add。
- 删除死代码 `reduce_scatter_grads` 中的 `grads_use_cnt`，或按 P0-3 方案 B 赋予其实际作用。
- `_flush_expert_grads_after_unit` 实际对所有 group 生效，名字有误导性，建议改名。

### P2-5 正确性 / 鲁棒性

- RS 输出改为标准 in-place 偏移（§8.3、P1-4）。
- `reset_params_buffer_status` 中丢弃过期预取的逻辑依赖 SYNCING 状态；若引入 P2-1 的提前 AG，需要确保它发生在 reset 之后，否则会被当成过期预取丢掉。

### 优先级汇总

| 编号 | 内容 | 预期收益 | 改动量 | 风险 |
|---|---|---|---|---|
| P0-1 | 累加期间跳过 RS / 混合 no_sync | 通信字节最多约 -50%（A=32） | 中 | 显存 |
| P0-2 | AG/RS 独立 communicator | 反向 AG 不再被 RS 阻塞 | 小 | NCCL 资源 |
| P0-3 | 修 RS 触发点，消灭 step 末同步 RS | 去掉 step 末串行通信 | 小（A）/ 中（B） | 需与 P0-1 协同 |
| P1-1 | recompute 感知 | 整层重算时省一次 AG/unit | 小 | PP 交错 |
| P1-2 | 字节预算 / 距离驱逐 / 暴露旋钮 | 可调、减少误驱逐 | 小 | 低 |
| P1-3 | 一个 unit 的 AG 合并 | 减少 launch/延迟 | 中~大 | 优化器侧 |
| P1-4 | AVG + 末尾缩放、comm stream 累加 | 省 A×groups 次大 kernel | 小 | 对齐基线 |
| P2-1 | 提前首个 AG、去掉 device sync | 每步固定空泡 | 小 | 显存峰值 |
| P2-2 | 固定池 | 碎片、为 UB/CUDA graph 铺路 | 中 | 显存预留 |
| P2-3 | bf16 RS | RS 字节减半 | 小 | 精度 |

建议实施顺序：先采一次 nsys（确认 comm stream 上 AG 被 RS 阻塞的时长、`finish_grads_sync` 耗时、step 首层 AG 暴露时长），再按 P0-3（A）→ P0-2 → P1-4 → P0-1 推进；前三项改动小、彼此独立，P0-1 涉及 trainer 接口与显存策略，放在最后。

---

## 14. 现有旋钮与注意事项

| 旋钮 | 位置 | 默认 | 说明 |
|---|---|---|---|
| `enable_tensor_fusion_and_overlap` | `fully_shard()` | PaddleFleet 传 True | False 时走同步路径，无预取 |
| `prefetch_units` | `FSDPCommManager` / context 构造（L635, L1151） | 1 | 未经 `fully_shard()` 暴露 |
| `fsdp_unit_layers` | BufferManager | `TransformerLayer` 等 3 个类名 | 未经 `fully_shard()` 暴露；新模型的层类名不在列表中时**整个模型都没有 unit**，全部退化为 None group（无预取、step 末同步 RS） |
| `mp_policy.reduce_dtype` | PaddleFleet `_setup_fsdp` | `amp_master_grad` 时 fp32 | 决定 grad buffer 与 RS dtype |
| `set_grad_accum_steps` | context | trainer 传 `gradient_accumulation_steps` | 决定 AVG/SUM 分支与 `grad_scale` |
| `FLAGS_use_stream_safe_cuda_allocator` | Paddle flag | true | 跨流安全的前提，不要关 |
| `FLAGS_nccl_blocking_wait` | Paddle flag | false | 打开后 `task.wait()` 变成 CPU 阻塞，破坏 overlap |
| `MUON_FSDP_BCAST_GRAD` | 环境变量 | 未设 | 对齐调试，全部通信改同步 broadcast，仅用于数值对齐 |

注意事项：

1. 新接入模型首先确认其 decoder layer 类名（或其 `__mro__` 中的父类）在 `fsdp_unit_layers` 中，否则 overlap 全部失效。
2. `FSDP_TUTORIAL_zh.md` 中 `double_buffer_limit=4`、"queue 默认 2"等描述与行号已过时，以本文与源码为准；其 §6.5 时间线、§9 显存公式、§13.1 限制列表仍可参考。
3. 本文对"是否通信受限""收益大小"的判断都基于字节量推算，未做 profile；EP 场景下 expert 的 fsdp group 大小（是否为 1、是否分片）取决于 `hybrid_parallel_topo_order` 与 EP/sharding 度，未在线上作业中核实。
4. `outer_wait_` 是否被设置（决定 `SyncCalcStream` 是否生效）未核实；若被设置，预取 AG 的启动时机会比 §6.3 描述的更早，但 comm 与 calc 的数据依赖需要由调用方保证。

