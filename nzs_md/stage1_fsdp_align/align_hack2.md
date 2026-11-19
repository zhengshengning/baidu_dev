# ERNIELite stage1 × FSDP 逐位对齐 —— 诊断 hack 修改清单（erniebot_nzs 版）

> **用途**：在 erniebot `ernie5` trainer 上让 `sharding: stage1`（Fleet MuonShardingOptimizer）与
> `sharding: fsdp`（Paddle 原生 FullyShardFusion）同配置下 loss 逐位一致。
> **性质**：纯诊断改动，全部由环境变量控制、默认关闭。**开启后训练语义是错的，不可用于生产。**
> **上游参考**：`/root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/FSDP/stage1_fsdp_align_hack.md`
> （下称"参考文档"，改动 A–G）。本文只记录**本仓库实际落地的版本**，与参考文档不同之处标 ⚠️。
>
> 范围：只记录"环境变量控制"的部分。`pretraining_trainer.py` 里 FSDP 功能支持本身
> （FullyShardFusion 包装、unit layers、`use_muon` 标记、`set_grad_accum_steps` 等，无开关、常开）
> 不在本文范围，复现时随整个文件一起带上即可。

## 0. 代码基线

| 组件 | 位置 | 基线 | 状态 |
|---|---|---|---|
| Paddle | `erniebot_nzs/Paddle` | fork `zhengshengning/Paddle` @ `51af813a03`（"several fused calls (main + MTP heads)"） | 4 个文件改动，**已 `git add`、未 commit** |
| trainer | `erniebot_nzs/ernie5/src/trainers/pretraining_trainer.py` | erniebot `dev` @ `6e0ee06` | 工作区未提交 |

Paddle 为纯 Python 改动，无需重新编译；以源码方式（或把这 4 个文件覆盖到 site-packages 下的 paddle）生效即可。

## 1. 开关一览

| 环境变量 | 所在 | 生效侧 | 消除的差异源 |
|---|---|---|---|
| `MUON_FSDP_BCAST_GRAD=1` | Paddle（4 处文件） | **两侧都要开** | 反向归约的 reduce root / 求和顺序；归约换成无算术 `broadcast(src=rank0)` |
| `MUON_FSDP_NS_SINGLE=1` | Paddle `muon.py` | **两侧都要开** | 批量 Newton-Schulz 的跨批量依赖 |
| `MUON_FSDP_STAGE1_CLIP_EPS=1` | trainer | 仅 FSDP 侧有效（stage1 开了也无副作用） | stage1 裁剪公式多出的 `+1e-6` |
| `MUON_FSDP_EXPERT_NO_ACCUM_SCALE=1` | trainer | 仅 FSDP 侧有效 | FSDP 专家 buffer 多除了一次 `accum` |

配套确定性 flag（两侧都设）：

```bash
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_deterministic_rng=1
```

## 2. 改动总览

| # | 文件 | 位置 | 开关 | 与参考文档 |
|---|---|---|---|---|
| A | `Paddle/.../fsdp/fully_shard_fusion.py` | `TensorFusionBuffer.do_reduce_scatter` | BCAST | ⚠️ 变了：广播分支**不再 `scale_`**，缩放推迟到 A2 |
| A2 | 同上 | `FSDPCommManager.finish_grads_sync` sharded 分支 | BCAST | 🆕 在累积后的 shard 上一次性 `scale_(grad_scale)` |
| B | 同上 | `finish_grads_sync` replicated（Muon）分支 | BCAST | 同参考文档 |
| H | 同上 | `FullyShardFusion` 的 `comm_hook`（fused kernel fp32 grad 列表） | BCAST | 🆕 多次 fused 调用（主干 + MTP head）逐分量 flush |
| C | `Paddle/.../fleet/utils/tensor_fusion_helper.py` | `_comm_grads` REDUCE_SCATTER 分支 | BCAST | 同参考文档 |
| D | 同上 | `__init__` / 通信前 / 通信后 | BCAST | 同参考文档 |
| G | 同上 | `_comm_grads` REDUCE 分支 | BCAST | 同参考文档 |
| F | `Paddle/.../fleet/meta_optimizers/muon_sharding_optimizer.py` | `_reduce_2d_grads` | BCAST | 同参考文档 |
| E | `Paddle/python/paddle/optimizer/muon.py` | `_muon_update_group` 入口 | NS_SINGLE | 同参考文档 |
| T1 | `ernie5/src/trainers/pretraining_trainer.py` | 模块级 `_install_stage1_clip_eps_hack()` | STAGE1_CLIP_EPS | 🆕 |
| T2 | 同上 | `_set_fsdp_grad_accum_steps()` 末尾 | EXPERT_NO_ACCUM_SCALE | 🆕 |

另：`tensor_fusion_helper.py` 与 `fully_shard_fusion.py` 各新增一个 `_muon_align_log(tag, msg)`，
按 tag 去重、只在 rank0 打一次 ACTIVE 标记；`muon_sharding_optimizer.py` 从 `tensor_fusion_helper` import 它。

## 3. Paddle 侧

### 改动 A ⚠️ —— FSDP AdamW（sharded）路径：broadcast，且不在这里缩放

```python
def do_reduce_scatter(self):
    tmp_buffer = self.get_tmp_buffer()
    shard = tmp_buffer._slice(0, self.data_buffer.shape[0])
    if os.environ.get("MUON_FSDP_BCAST_GRAD") == "1":
        _muon_align_log("fsdp_rs",
            "[MUON_FSDP_BCAST_GRAD] FSDP:   reduce_scatter -> broadcast(src=0) ACTIVE")
        paddle.distributed.broadcast(tmp_buffer, src=self.fsdp_group.ranks[0],
                                     group=self.fsdp_group, sync_op=True)
        shard_size = self.data_buffer.shape[0]
        if self.shard_start:                      # 手动 scatter：把本 rank 的片搬到头部
            paddle.assign(
                tmp_buffer._slice(self.shard_start, self.shard_start + shard_size),
                shard,
            )
        return TensorFusionBuffer._CompletedTask()  # 基线已有该占位类，上层会 .wait()
    tmp_buffer.scale_(self.grad_scale)            # 原有路径：先 scale 再 reduce_scatter
    return paddle.distributed.reduce_scatter(shard, tmp_buffer, ...)
```

**与参考文档的差别**：参考文档在广播前就 `scale_(grad_scale)`。这里改为不缩放，原因是
stage1 在累积结束后缩放一次，即 `(Σ g_i)·s`；每次 flush 都缩放得到 `Σ (g_i·s)`。
两者只有在 `s` 为 2 的幂时才逐位相同（如 accum=16 → 1/128），accum=3（1/24）之类就会差末位。

### 改动 A2 🆕 —— 在累积后的 shard 上缩放一次

`FSDPCommManager.finish_grads_sync` 的 sharded 分支（原来是 `if tmp is None: continue`，需改写成下面的结构，
否则 tmp 为空时会跳过缩放）：

```python
if grads_buffer.tmp_data_buffer is not None:
    grads_buffer.do_reduce_scatter().wait()
    grads_buffer.accumulate_reduced_grad()
if (os.environ.get("MUON_FSDP_BCAST_GRAD") == "1"
        and grads_buffer.grad_scale != 1.0
        and not grads_buffer.grad_scaled):
    grads_buffer.data_buffer.scale_(grads_buffer.grad_scale)
    grads_buffer.grad_scaled = True
```

A 与 A2 必须成对出现：只改 A 不加 A2，FSDP 的 1D 梯度会大 `nranks·accum` 倍。

### 改动 B —— FSDP Muon（replicated）路径

`finish_grads_sync` 中 `group.muon_owner_rank is not None` 分支：BCAST 时 `reduce(dst=owner, SUM)` →
`broadcast(src=fsdp_group.ranks[0])`，打印 `FSDP:   reduce(owner) -> broadcast(src=0) ACTIVE`。
其后原有的"每个累积周期 `scale_(grad_scale)` 一次"逻辑不变。代码与参考文档 §3 相同。

### 改动 H 🆕 —— fused kernel 多分量 fp32 grad 逐分量 flush

背景：基线 `51af813a03` 允许 fused linear CE（主干 + 各 MTP head）把同一参数的 fp32 梯度以**列表**形式
挂在 `param._fused_kernel_fp32_grad` 上。非 FSDP 路径把每个分量直接加进跨 micro-batch 累加器，
即 `(acc + a) + b`；FSDP 的 tmp buffer 则先形成 `a + b` 再累加，舍入不同。

在 `FullyShardFusion` 注册的 `comm_hook` 里，`param.get_main_grad(...)` 之前插入：

```python
fusion_buf = getattr(param, "_fusion_buffer", None)
if (len(fp32_grad) > 1
        and os.environ.get("MUON_FSDP_BCAST_GRAD") == "1"
        and fusion_buf is not None
        and fusion_buf.is_sharded):
    for g in fp32_grad[:-1]:            # 前 n-1 个分量各自 flush 进累加器
        param.get_main_grad(g.shape)
        param.main_grad.add_(g)
        fusion_buf.do_reduce_scatter()  # BCAST 下是同步、无算术的 broadcast
        fusion_buf.accumulate_reduced_grad()
        param.main_grad = None
    fp32_grad = fp32_grad[-1:]          # 最后一个分量走原有路径
```

依赖 A/A2：只有 flush 同步且不带缩放时，逐分量 flush 才等价于非 FSDP 的逐次 `add_`。
仅对 sharded（AdamW）buffer 生效；Muon replicated buffer 不经过 `do_reduce_scatter`。

### 改动 C / D / G / F —— stage1 侧（同参考文档）

- **C**（`tensor_fusion_helper.py` REDUCE_SCATTER 分支，1D AdamW）：`scale_(1/nranks)` + 置 `_bcast_scale_applied=True`
  + `broadcast(src=ranks[0], sync_op=False)`，`self._task = task; return`。**`÷nranks` 必须有**，见参考文档 §4。
- **D**：`__init__` 加 `self._bcast_scale_applied = False`；通信前预缩放条件加 `not _bcast`；
  `scale_grads` 通信后缩放条件加 `not self._bcast_scale_applied`。
- **G**（同文件 REDUCE 分支，2D Muon 融合路径，默认走这条）：与 C 同构。
- **F**（`muon_sharding_optimizer.py::_reduce_2d_grads`，2D Muon 非融合路径）：循环内
  `g_var.scale_(1/comm_group.nranks)` + `broadcast(src=comm_group.ranks[0], sync_op=True)` + `continue`。

### 改动 E —— 逐矩阵 Newton-Schulz（同参考文档）

`muon.py::_muon_update_group` 入口：`MUON_FSDP_NS_SINGLE=1` 且 `len(group_params_grads) > 1` 时，
逐个 `[_item]` 自递归后 `return`。

## 4. trainer 侧（`ernie5/src/trainers/pretraining_trainer.py`）

### 改动 T1 🆕 —— 在 FSDP 侧复刻 stage1 裁剪的 `+1e-6`（`MUON_FSDP_STAGE1_CLIP_EPS`）

- stage1（PaddleFleet `MoEHybridParallelClipGrad`，自带 `__call__`）：`clip = max / (maximum(norm, max) + 1e-6)`
- FSDP（paddle `ClipGradByGlobalNorm`）：`clip = max / maximum(norm, max)`

`norm <= max` 时 stage1 仍会把梯度乘 `max/(max+1e-6)`（实测 `0.9999990463256836`）。不改更新方向，
但改 fp32 moment 末位，经 bf16 回写后在量化边界放大成可见差异。

实现：模块级函数 `_install_stage1_clip_eps_hack()`，在 `logger = logging.getLogger(__name__)` 之后定义并
**import 时立即调用**。开关打开时 monkey-patch `paddle.nn.clip.ClipGradByGlobalNorm.__call__`：
先调原实现，再对 `need_clip is not False` 的梯度乘 fp32 计算的 `factor = mx / (mx + eps)`
（按需 `astype(g.dtype)`）。用 `ClipGradByGlobalNorm._stage1_eps_hack` 防重复 patch。
stage1 不经过 `ClipGradByGlobalNorm.__call__`，所以两侧脚本都 export 也没问题。

生产上正确的修法是去掉 stage1 的 `+1e-6`，而不是让 FSDP 跟着缩。

### 改动 T2 🆕 —— 专家 buffer 去掉多余的 `/accum`（`MUON_FSDP_EXPERT_NO_ACCUM_SCALE`）

位置：`_set_fsdp_grad_accum_steps(accum_steps)` 在 `ctx.set_grad_accum_steps(accum_steps)` 之后追加：

```python
if os.environ.get("MUON_FSDP_EXPERT_NO_ACCUM_SCALE") == "1":
    n = 0
    for group in ctx.buffer_manager.buffer_groups:
        gb = group.grads_buffer
        if gb is None or not group.is_expert_param:
            continue
        gb.grad_scale = 1.0 / gb.grad_div
        n += 1
    print(f"[ALIGN] expert no-accum-scale hack ACTIVE on {n} buffers", flush=True)
```

原因：FSDP 下专家组 nranks=1，`set_grad_accum_steps` 仍把专家 buffer 的 `grad_scale` 设成
`1/(grad_div·accum)`，而 stage1 的专家梯度只在 loss 侧除过一次 accum → FSDP 专家梯度小 accum 倍。
把专家 buffer 的 `grad_scale` 改回 `1/grad_div` 后，对任意 accum（含非 2 的幂）与 stage1 逐位一致。
必须在 `set_grad_accum_steps` **之后**执行，否则会被覆盖。

### 改动 T3 🆕 —— AdamH（Hyperball）参数在 FSDP 下按整矩阵处理（无开关，常开）

背景：Paddle 合入 develop（`d3b5b0c255` FSDP Support Muon / `19745a0160` hyperball）后，`Muon` 开始真正读取
`MuonParamInfo.use_hyperball`。此前 trainer 只是 setattr 了这个字段，Paddle 不认，两侧都是普通 Muon/AdamW，所以能对齐。

问题：`_mark_muon_params_for_fsdp` 只按 `use_muon` 打 `param.use_muon`，AdamH 的 `lm_head` 被放进 AdamW 元素切片融合
buffer；Muon 优化器对 `fuse_params_*` 查不到 info_map，直接走普通 `_adamw_update`（无 Hyperball 投影），
而 stage1 按 `keep_whole = use_muon or use_hyperball` 整块交给 owner 做 `_adamw_hyperball_update`。
表现：step 3 起 FSDP loss 明显更低（step 20：11.12 vs 12.11）。

修法：`param.use_muon = info.use_muon or info.use_hyperball`。FSDP 的 `use_muon` buffer 语义就是“整矩阵、owner 更新”，
取回后 Muon 优化器按 info_map（use_muon=False, use_hyperball=True）走 AdamH。这是语义修正而非诊断 hack，生产也应保留
（更彻底的做法是 Paddle FSDP 分桶 key 直接带上 use_hyperball）。日志标记：
`[FSDP] marked param.use_muon on 88 Muon + 1 AdamH (keep-whole) params`。

## 5. 启用与校验

两侧脚本（单机参考：`run_sharding_test.sh` / `run_fsdp_test.sh`）：

```bash
# 两侧都要
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_deterministic_rng=1
export MUON_FSDP_BCAST_GRAD=1
export MUON_FSDP_NS_SINGLE=1
export MUON_FSDP_STAGE1_CLIP_EPS=1          # 只在 FSDP 侧起作用
# 仅 FSDP 侧
export MUON_FSDP_EXPERT_NO_ACCUM_SCALE=1
```

注意：`run_fsdp_1node.sh` 里的 `MUON_FSDP_EXPERT_GRAD_MUL=16` 是早期遗留变量，**代码中没有任何地方读取**，
它也没有设 `MUON_FSDP_EXPERT_NO_ACCUM_SCALE`；照搬时以 `run_fsdp_test.sh` 为准。

rank0 日志里应看到的标记（缺哪条就是哪处改动没生效）：

```
# stage1 侧
[MUON_FSDP_BCAST_GRAD] stage1: reduce(owner) -> broadcast(src=0) ACTIVE       # G（与 F 二选一）
[MUON_FSDP_BCAST_GRAD] stage1: reduce_2d(owner) -> broadcast(src=0) ACTIVE    # F
[MUON_FSDP_BCAST_GRAD] stage1: reduce_scatter -> broadcast(src=0) ACTIVE      # C
# FSDP 侧
[MUON_FSDP_BCAST_GRAD] FSDP:   reduce_scatter -> broadcast(src=0) ACTIVE      # A
[MUON_FSDP_BCAST_GRAD] FSDP:   reduce(owner) -> broadcast(src=0) ACTIVE       # B
[ALIGN] stage1 clip-eps hack ACTIVE (factor=max/(max+1e-6))                   # T1
[ALIGN] expert no-accum-scale hack ACTIVE on <N> buffers                       # T2，N 应 > 0
```

stage1 侧走 Fleet logger（`logger.info`），FSDP 侧与 trainer 侧是 `print`。E、H 无日志标记。

对比方式：两侧 yaml 除 `sharding: stage1|fsdp`（及 FSDP 要求的 `split_param` 等相关项）外保持一致，
用 `compare_loss.py` 逐 step 比 loss，要求逐位相同。**step 1/2 相同不说明问题**（warmup lr≈1e-6，
Muon-NS 与 AdamW 首步对梯度幅值不敏感），至少看到 step 3 之后。

## 6. 复现步骤

```bash
# 1) 导出 Paddle 改动（改动是 staged 状态，所以用 --cached）
cd erniebot_nzs/Paddle
git diff --cached > /path/to/paddle_align_hack.patch

# 2) 在目标机器的 Paddle（同一基线 51af813a03）上应用
git checkout 51af813a03
git apply /path/to/paddle_align_hack.patch

# 3) trainer：整个文件带上（同时包含 FSDP 功能支持与 T1/T2）
cd erniebot_nzs
git diff HEAD -- ernie5/src/trainers/pretraining_trainer.py > /path/to/trainer_fsdp.patch
```

若基线不同导致 `git apply` 失败，按第 3、4 节逐处手工移植；关键是 A+A2 成对、C/G/F 的 `÷nranks`、
T2 在 `set_grad_accum_steps` 之后。
