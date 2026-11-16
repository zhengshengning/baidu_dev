# FSDP 支持 `use_mixed_optimizer`：本地 Paddle 修改说明与正确性检查

代码位置：`erniebot/Paddle`（分支 `fsdp_prefetch_refactor_2`，基于 `bfe92db268`），改动在暂存区（`git diff --cached`），共 3 个文件 +82/-1。`run_fsdp_test.sh` 通过 `PYTHONPATH=.../Paddle/build/python` 使用这份代码，`build/python` 下的 3 个文件与源码一致。

| 文件 | 改动 |
|---|---|
| `python/paddle/distributed/fsdp/fully_shard.py` | 新增 `_mark_params_for_mixed_optimizer`，按子优化器给参数打标 |
| `python/paddle/distributed/fsdp/fully_shard_fusion.py` | 融合 buffer 的分组 key 加入子优化器编号；新增 `_bind_mixed_routes`，把 `fuse_params_*` 路由到子优化器 |
| `python/paddle/optimizer/mixed_inner_optimizer.py` | `_apply_optimize` 在 FSDP 下改用 FSDP 产出的 `params_grads`，step 后广播整矩阵参数 |

## 1. 背景

`MixedInnerOptimizer` 持有 N 个子优化器（`_sub_opts`），按 `param.name` 查 `_route` 把每个 `(param, grad)` 派发给所属子优化器。每个子优化器声明 `keep_whole`：

- `keep_whole=True`（Muon、AdamH、MuonH、RMSpropH 等）：更新需要整矩阵（Newton-Schulz、Frobenius 投影），不能在 element-wise 分片上算。
- `keep_whole=False/None`（AdamW、RMSprop）：逐元素更新，可以直接在分片上算。

FSDP（`FullyShardFusion`）把参数拼成融合 buffer，每个 rank 只持有一段分片，optimizer 实际更新的是名为 `fuse_params_<gid>` 的 buffer，不是模型参数本身。原有 FSDP 只认两种优化器：

- 普通 `Optimizer`：所有 buffer 都按分片更新（`optimizer.py` 的 FSDP 分支）。
- `paddle.optimizer.Muon`：靠 `_muon_param_info_map` 区分 Muon 参数和 AdamW 参数。`use_muon` 的 buffer 走 `muon_params_grads()`，先 P2P 汇聚到 owner rank 得到整矩阵，step 后再 `broadcast_muon_params()` 分发回去；其余 buffer 按分片更新。

`MixedInnerOptimizer` 不属于这两种：它没有 `_muon_param_info_map`，`_route` 里也没有 `fuse_params_*` 名字，原来的 `_apply_optimize` 也不认识 FSDP。本次修改让它复用 Muon 的那套机制：`keep_whole` 的子优化器走整矩阵路径，其余子优化器走融合分片路径。

## 2. 改动逐段说明

### 2.1 `fully_shard.py`：按子优化器给参数打标

```python
def _mark_params_for_mixed_optimizer(module, optimizer, decay_fun):
    # Buffers are split by owning sub; keep_whole subs take the Muon path.
    route = optimizer._route
    sub_ids = {id(sub): i for i, sub in enumerate(optimizer._sub_opts)}
    missing = []
    for name, param in module.named_parameters():
        param.no_weight_decay = decay_fun is not None and not decay_fun(param.name)
        param.use_hyperball_sharded = False
        sub = route.get(param.name)
        if sub is None:
            if not param.stop_gradient:
                missing.append(name)
            param.fsdp_optimizer_group = None
            param.use_muon = False
            continue
        param.fsdp_optimizer_group = sub_ids[id(sub)]
        param.use_muon = bool(sub.keep_whole_of(param.name))
    if missing:
        raise ValueError(...)

def _mark_params_for_optimizer(module, optimizer):
    decay_fun = getattr(optimizer, "_apply_decay_param_fun", None)
    if getattr(optimizer, "_sub_opts", None) is not None:
        _mark_params_for_mixed_optimizer(module, optimizer, decay_fun)
        return
    ...
```

`fully_shard(model, optimizer=...)` 在建融合 buffer 之前调用 `_mark_params_for_optimizer`，参数上的标记决定 buffer 怎么分组。检测到 `_sub_opts` 就走 mixed 分支。经过 `MixPrecisionOptimizer` 包一层也能识别，因为它的 `__getattr__` 会转发到内层。每个参数打 4 个标：

- `no_weight_decay`：与原逻辑相同，保证一个 buffer 内 weight decay 一致。
- `use_hyperball_sharded = False`：原 Muon 路径会把 2D 的 AdamW+Hyperball 放到分片上算（norm 做 all-reduce）。mixed 下 AdamH 等子优化器的 `keep_whole=True`，统一走整矩阵路径，不使用分片 Hyperball。
- `fsdp_optimizer_group`：所属子优化器在 `_sub_opts` 里的下标，用于拆分 buffer（见 2.2）。
- `use_muon = sub.keep_whole_of(name)`：`keep_whole` 的参数复用 Muon 的整矩阵路径。这里的 `use_muon` 含义是“需要整矩阵”，不要求子优化器真的是 Muon。

没有路由的可训练参数直接报错，没有路由的冻结参数标成 `None/False`，进入冻结 buffer。

### 2.2 `fully_shard_fusion.py`：buffer 分组 key 加入子优化器编号

```python
key = (
    ...,
    bool(getattr(param, "use_muon", False)),
    # A buffer is stepped by one sub-optimizer of a mixed optimizer.
    getattr(param, "fsdp_optimizer_group", None),
    ...,
)
```

`sort_key` 和构造 `BufferGroup` 的解包同步多加一个 `_` 占位。

一个融合 buffer 作为一个整体交给一个子优化器更新，所以 buffer 内的参数必须属于同一个子优化器。例如两个分片型子优化器（AdamW 组 A、AdamW 组 B，超参不同）的参数原来可能落进同一个 buffer，加了这一维以后会拆开。非 mixed 场景下这个属性不存在，取值恒为 `None`，分组结果不变。`sort_key` 不含这一维，`sorted` 是稳定排序，同一 unit 内拆出来的 buffer 保持参数出现顺序。

### 2.3 `fully_shard_fusion.py`：把 `fuse_params_*` 路由到子优化器

```python
def _bind_mixed_routes(self, optimizer):
    # Route each fused buffer to the sub owning its params.
    inner = optimizer
    while hasattr(inner, "_inner_opt"):
        inner = inner._inner_opt
    route = getattr(inner, "_route", None)
    if getattr(inner, "_sub_opts", None) is None or route is None:
        return False
    for group in self.buffer_manager.buffer_groups:
        name = group.params_buffer.data_buffer.name
        if name not in route and group.params[0].name in route:
            route[name] = route[group.params[0].name]
    return True

def bind_optimizer(self, optimizer):
    self._bind_mixed_routes(optimizer)
    ...

def init_optimizer_state(self, optimizer):
    ...
    # Mixed subs that keep whole matrices are stepped like Muon's.
    is_muon = isinstance(inner, Muon) or self._bind_mixed_routes(inner)
```

`MixedInnerOptimizer` 的 `_apply_optimize`、`_create_accumulators`、`_set_master_weight`、`set_state_dict` 都按 `param.name` 查 `_route`。FSDP 下实际更新的对象是 `fuse_params_<gid>`，所以要把每个 buffer 名映射到其首个参数所属的子优化器。2.2 已经保证同一 buffer 的参数属于同一个子优化器，所以取首个参数就够了。冻结 buffer 的首参不在 `_route` 中，会被跳过。

调用时机：

- `fully_shard()` 末尾的 `bind_optimizer(optimizer)`：buffer 刚建好就注册路由。
- 每步 `_fsdp_params_grads` 里的 `bind_optimizer(self)`：已有的名字会跳过，开销只是遍历一遍 buffer。
- `init_optimizer_state`（加载 checkpoint 前创建 accumulator 目标）：确保路由已注册。返回 `True` 时，按 Muon 的方式处理：`use_muon` 的 buffer 不创建融合 accumulator，改由 `_init_muon_optimizer_state` 按原参数逐个创建。

### 2.4 `mixed_inner_optimizer.py`：FSDP 下的 step

```python
def _apply_optimize(self, loss, startup_program, params_grads):
    ...
    fsdp_context = get_fsdp_context()
    if fsdp_context is not None:
        params_grads = self._fsdp_params_grads(fsdp_context)
    if self._grad_clip is not None:
        params_grads = self._grad_clip(params_grads)
    ...  # 按 _route 分桶、逐个子优化器 _apply_optimize（原逻辑）
    self._maybe_refuse()
    if fsdp_context is not None and hasattr(fsdp_context, "broadcast_muon_params"):
        fsdp_context.broadcast_muon_params()

def _fsdp_params_grads(self, fsdp_context):
    fsdp_context.comm_sync_and_reset_status()
    params_grads = list(fsdp_context.muon_params_grads())
    for group in fsdp_context.buffer_manager.buffer_groups:
        if group.use_muon or group.grads_buffer is None:
            continue
        data_buffer = group.params_buffer.data_buffer
        if data_buffer.stop_gradient:
            continue
        params_grads.append((data_buffer, group.grads_buffer.data_buffer))
    fsdp_context.bind_optimizer(self)
    for sub in self._sub_opts:
        fsdp_context.bind_decay_param_fun(sub)
    fsdp_context.bind_decay_param_fun(self)
    if self._grad_clip is not None:
        self._grad_clip.should_comm_on_shard_dim = True
        self._grad_clip.fsdp_group = fsdp_context.buffer_manager._fsdp_group
    return params_grads
```

这段与 `Muon._apply_optimize` 的 FSDP 分支、`Muon._fsdp_collect_params_grads` 逐项对应：

1. `comm_sync_and_reset_status()`：等待 backward 中所有异步 reduce-scatter 完成，重置 buffer 状态。
2. `muon_params_grads()`：处理 `use_muon`（即 keep_whole）的 buffer，把参数和梯度 P2P 汇聚到 owner rank，让原参数的 storage 指向整矩阵视图，返回 `(原参数, 整梯度)`，只在 owner 上返回。3D expert 参数按 expert 维对齐切分，返回 `[n_local, m, n]` 视图。
3. 其余可训练 buffer 以 `(fuse_params_x, 梯度分片)` 加入列表。顺序与 Muon 路径一致（先整矩阵、后分片），clip 中 fp32 `add_n` 的累加顺序也就一致。
4. `bind_decay_param_fun(sub)`：子优化器按 `param.name` 调用 `_apply_decay_param_fun`。`fuse_params_*` 不是模型参数名，不包一层会被判成“不 decay”，weight decay 就会悄悄丢掉。这里给每个子优化器都包一层（各自有 `_fsdp_decay_fun_bound` 标志，只包一次）。
5. grad clip 设置 `should_comm_on_shard_dim` 和 `fsdp_group`，让 global norm 在 FSDP 组内 all-reduce。分片梯度每个 rank 贡献自己那段，整矩阵梯度只在 owner 上出现，all-reduce 后得到全局范数。
6. 后面的分桶派发沿用原逻辑。`fuse_params_*` 经 2.3 的路由找到子优化器，原参数直接命中 `_route`。非 owner rank 上某个子优化器可能分不到参数，桶为空时跳过。
7. 所有子优化器更新完以后，`broadcast_muon_params()` 把 owner 上更新好的整矩阵 scatter 回各 rank 的分片，并释放原参数的整矩阵 storage。

整体时序：

```
fully_shard(model, optimizer=mixed)
 ├─ _mark_params_for_mixed_optimizer   # 打 use_muon / fsdp_optimizer_group
 ├─ FullyShardFusion(...)              # 按 key 建融合 buffer
 └─ bind_optimizer → _bind_mixed_routes # 注册 fuse_params_* → sub

optimizer.step()  (MixPrecisionOptimizer → mixed._apply_optimize)
 ├─ _fsdp_params_grads
 │   ├─ comm_sync_and_reset_status
 │   ├─ muon_params_grads              # keep_whole：P2P 汇聚整矩阵到 owner
 │   ├─ + 融合分片 (fuse_params_x, grad_shard)
 │   └─ bind decay / clip
 ├─ grad_clip(全局一次)
 ├─ 按 _route 分桶 → sub._apply_optimize
 └─ broadcast_muon_params              # 整矩阵 scatter 回分片
```

## 3. 正确性检查

### 3.1 训练 step：未发现问题

逐项对照 `Muon` 的 FSDP 路径检查，step 路径的逻辑是对的：

- buffer 不会跨子优化器，也不会混放整矩阵和分片两类参数（分组 key 同时包含 `use_muon` 和 `fsdp_optimizer_group`）。
- `fuse_params_*` 都能路由到子优化器（`_bind_mixed_routes`），不会触发 `_apply_optimize` 中的 "has no sub-optimizer route"。
- 融合 buffer 的 weight decay 由每个子优化器各自绑定，各组的 `weight_decay` 数值仍由子优化器自己决定。
- grad clip 在全局只做一次，范数在 FSDP 组内 all-reduce，与 Muon 路径相同。
- 整矩阵参数先汇聚、后更新、再 scatter，与 Muon 路径同序。owner 选择沿用 `_assign_muon_owners` 的负载均衡。
- `set_state_dict` 的最长前缀匹配会把 `fuse_params_*` 也纳入 `names_by_len`，按长度倒序匹配，`fuse_params_1` 不会误吞 `fuse_params_10_*`。
- 非 mixed 场景：`fsdp_optimizer_group` 恒为 `None`，`_bind_mixed_routes` 返回 `False`，行为与修改前一致。

### 3.2 问题 1（会报错，已修复）：checkpoint 保存时识别不出 accumulator 名

`FullyShardFusion.optimizer_sharded_state_dict` 从 `type(inner).__mro__` 里收集 `_*_acc*_str` 属性，用来拆分 `fuse_params_0_moment1_0` 这类名字。mixed 下 `inner` 是 `MixedInnerOptimizer`，它自己没有任何 accumulator 字符串属性（这些属性在子优化器类上），所以 `tags` 为空：

- 融合 buffer 的状态会触发 `NotImplementedError: FSDP cannot save optimizer state 'fuse_params_0_moment1_0' of MixedInnerOptimizer ...`，到 `save_steps` 时保存直接失败。
- 整矩阵参数的状态（`<param>_moment1_0`）得到的 `base_name` 为 `None`，会被静默跳过。

复现（单卡，`Paddle/build/python` + `PaddleFleet/src`）：

```
state keys: ['fuse_params_0_moment1_0', 'fuse_params_0_moment2_0', ...]
NotImplementedError FSDP cannot save optimizer state 'fuse_params_0_moment1_0' of MixedInnerOptimizer on fused buffer 'fuse_params_0': ...
```

修复（已改入 `fully_shard_fusion.py` 的 `optimizer_sharded_state_dict`，并同步到 `build/python`）：tags 从每个子优化器的类上收集，非 mixed 优化器仍只看 `inner` 自己。

```python
tags = sorted(
    {
        getattr(cls, attr)
        for opt in (getattr(inner, "_sub_opts", None) or [inner])
        for cls in type(opt).__mro__
        for attr in vars(cls)
        if attr.endswith("_str")
        and "_acc" in attr
        and isinstance(getattr(cls, attr), str)
    },
    key=len,
    reverse=True,
)
```

### 3.3 问题 2（静默丢数据，已修复）：融合 buffer 的 fp32 master weight 不进 `state_dict`

`MixedInnerOptimizer.state_dict()` 只保留属于 `sub._parameter_list` 的 master weight：

```python
owned_names = {p.name for p in sub._parameter_list}
for param_name, master_weight in sub_master_weights.items():
    if param_name not in owned_names:
        continue
```

FSDP 下子优化器为 `fuse_params_*` 创建了 master weight，但 `_parameter_list` 里只有原模型参数，所以这些 master weight 全被过滤掉。分片型子优化器（AdamW 等）的 fp32 master weight 因此不会写进 checkpoint。即使问题 1 修好，保存也不报错，但恢复训练时这部分 master weight 只能从 bf16 参数重建，精度会掉。整矩阵参数用的是原参数名，不受影响。

复现：

```
sub master keys: ['fuse_params_0']
mixed.state_dict master: []
```

修复（已改入 `mixed_inner_optimizer.py` 的 `state_dict`，并同步到 `build/python`）：在原有 `_parameter_list` 的基础上，再把 `_route` 里归属该子优化器的名字并进来。`_route` 已经包含 `_bind_mixed_routes` 注册的 buffer 名：

```python
# _route also holds the FSDP fused buffers (fuse_params_*).
owned_names = {p.name for p in sub._parameter_list}
owned_names.update(
    n for n, s in self._route.items() if s is sub
)
```

保留 `_parameter_list` 是为了不改变已有语义：`test_mixed_inner_optimizer.py::test_state_dict_master_weight_in_two_subs` 会直接往 `sub._parameter_list` 里追加参数，构造“同一参数归属两个子优化器”的场景，要求报 "more than one"。只用 `_route` 时这个用例会失败。

验证：`repro_issue2_master_weights.py` 输出 `mixed.state_dict() master: ['fuse_params_0']`、`RESULT: OK`；`test/legacy_test/test_mixed_inner_optimizer.py` 47 个用例全部通过。多机端到端验证未执行，原因是第二台节点（10.95.224.35）的 8 张卡被其他任务占满（约 79 GB、利用率 100%）。

### 3.4 修复验证

用 monkeypatch 套上 3.2 和 3.3 两处修复，构造 AdamW 组（融合 buffer）加 Muon 组（整矩阵参数）的 mixed 优化器，跑一步后调用 `optimizer_sharded_state_dict`：

```
master in state_dict: ['fuse_params_0', 'linear_2.w_0']
sharded keys: ['model.fused.beta1_pow_acc_0', 'model.fused.beta2_pow_acc_0',
               'model.fused.moment1_0', 'model.fused.moment2_0', 'model.fused.w_0',
               'model.muon.moment1_0', 'model.muon.w_0']
```

两类参数的 moment 和 master weight 都能正确映射到结构化 key。这个验证只覆盖保存时的命名映射，没有跑多卡训练，也没有做 save→load 往返。

### 3.5 复现方式

**单卡最小复现**（脚本在 `repro_mixed_fsdp/`，约 30 秒，只需 1 张卡）：

```bash
bash repro_mixed_fsdp/run.sh
```

- `repro_issue1_save_tags.py`：用名为 `fuse_params_0` 的参数模拟融合 buffer，跑一步 AdamW 子优化器后直接调用 `FullyShardFusion.optimizer_sharded_state_dict`，也就是真实保存路径会调用的同一个函数。
- `repro_issue2_master_weights.py`：用 bf16 参数加 `multi_precision=True` 让子优化器为 `fuse_params_0` 创建 master weight，再对比 `sub._master_weights` 和 `mixed.state_dict()["master_weights"]`。

修复前的输出：

```
===== repro_issue1_save_tags.py
optimizer state keys: ['fuse_params_0_moment1_0', 'fuse_params_0_moment2_0', 'fuse_params_0_beta1_pow_acc_0', 'fuse_params_0_beta2_pow_acc_0']
RESULT: REPRODUCED -> FSDP cannot save optimizer state 'fuse_params_0_moment1_0' of MixedInnerOptimizer on fused buffer 'fuse_params_0': ...
===== repro_issue2_master_weights.py
sub._master_weights      : ['fuse_params_0']
mixed.state_dict() master: []
RESULT: REPRODUCED -> dropped ['fuse_params_0']
```

修复后两个脚本都应输出 `RESULT: OK`。问题 1 修复后的当前输出：issue 1 为 `RESULT: OK (issue 1 not reproduced)`，issue 2 仍为 `RESULT: REPRODUCED -> dropped ['fuse_params_0']`。

**端到端复现**（未执行，会通过 `script/mpi.sh` 按 hostfile 多机启动）：把 `run_fsdp_test.sh` 里的 `save_steps=200` 改成 `save_steps=5`（`max_steps=10` 不变）后运行 `bash run_fsdp_test.sh`。第 5 步保存时，`_save_flex_optimizer_state` 调用 `self.optimizer.sharded_state_dict()`，这个方法已被 `bind_optimizer` 替换为 `optimizer_sharded_state_dict`，应抛出与问题 1 相同的 `NotImplementedError`。问题 2 在问题 1 修复后才会暴露：保存不报错，但 checkpoint 里缺少分片型子优化器的 `*.w_0`（master weight），可以通过加载后检查 `optimizer.state_dict()["master_weights"]` 是否包含 `fuse_params_*` 来确认。

### 3.6 问题 1 修复后的端到端验证（2026-10-28，2 机 16 卡）

配置：`run_fsdp_test.sh`，`max_steps=10`、`save_steps=5`、`prefetch_units=2`，`output_dir` 通过 `--kwargs` 指到新目录，避免续训到修复前残留的半个 checkpoint。

| 运行 | 结果 |
|---|---|
| 修复前（`outputs/20261028_1609_*`） | 第 5 步 `Saving optimizer files.` 后抛 `NotImplementedError ... 'fuse_params_0_moment1_0' of MixedInnerOptimizer` |
| 修复后保存（`outputs/20261028_1617_*`，checkpoint 在 `output/mixed_fsdp_e2e_fix1.run1/checkpoint-5`） | 保存成功，耗时 64 s。16 个 `saved_signal_*` 齐全。`model_state` 66G、`optimizer_state` 133G、`master_weight` 130G |
| 从该 checkpoint 续训（`outputs/20261028_1659_*`） | model / optimizer / master_weight 三份都加载成功，step 6 到 10 正常训练，退出码 0 |

checkpoint 内容：

- `optimizer_state` 共 792 个 key：`moment1_0` 363 个、`moment2_0` / `beta1_pow_acc_0` / `beta2_pow_acc_0` 各 143 个。融合 buffer 和整矩阵参数的 moment 都映射到了结构化 key。
- 没有 optimizer state 的 60 个模型 key 都是冻结参数（`e_score_correction_bias`、`qb_bin_max`、`qb_bin_min` 各 20 个），符合预期。
- `master_weight` 只有 221 个，模型有 423 个 key。缺的 202 个是 embedding、各层 norm、gate 等分片型子优化器的参数，这就是问题 2：恢复时它们的 fp32 master weight 只能从 bf16 参数重建。

续训 loss 与不保存的连续训练（`outputs/20261028_1432_*`，同配置）对比：

| step | 连续训练 | 续训 |
|---|---|---|
| 6 | 12.21790 | 12.21781 |
| 7 | 12.20771 | 12.23128 |
| 8 | 12.19455 | 12.21653 |
| 9 | 12.18240 | 12.20130 |
| 10 | 12.16868 | 12.18526 |

lr 逐步一致，step 6 基本吻合，step 7 起续训 loss 高约 0.02。可能原因包括问题 2 丢失的 master weight（bf16 重建引入舍入误差），以及续训时数据跳过和随机数状态与连续训练不完全一致（`_load_rng_state` 在 ernie5 trainer 里被置空）。这里没有拆分这两个因素，修完问题 2 后再对比一次可以判断。

## 4. 其他说明

- 性能：mixed 下所有 `keep_whole` 子优化器（包括 AdamH）都走 P2P 汇聚整矩阵的路径，不使用原 Muon 路径里 2D AdamW+Hyperball 的分片实现（`use_hyperball_sharded`），通信量会比单 Muon 路径大一些。分片型子优化器超过一个时，每个 unit 的 buffer 数也会变多。
