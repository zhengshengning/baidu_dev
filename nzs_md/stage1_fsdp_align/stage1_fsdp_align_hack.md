# stage1 <-> FSDP 逐位对齐 hack（Muon）

纯诊断用，非生产语义。所有改动默认关闭，只在设置对应环境变量时生效。代码注释里提到的 `stage1_fsdp_align_hack.md` 就是本文。

## 文件

| 文件 | 内容 |
|---|---|
| `Paddle.patch` | Paddle 仓库 4 个文件的改动（`git diff`，331 行） |
| `PaddleFleet.patch` | `third_party/PaddleFleet` 的 `src/` 改动（18 行新增） |

基线 commit：

- Paddle：`77bc01519ce5a2d1b82df8c8ec4a8acd30037a70`（develop，`[FSDP] Defer fp32 grad scaling to the reduced shard (#79938)`）
- PaddleFleet：`e562577245607729c9e771418545f1fef019b3cb`（fsdp-sonic-moe）

PaddleFleet 里 `DeepSelect`、`flash-linear-attention` 两个子模块指针的变动与对齐无关，未收进 patch。

## 使用

```bash
cd Paddle && git apply ../nzs_md/stage1_fsdp_align/Paddle.patch            # 之后需要重新编译/安装 Paddle 的 python 包
cd third_party/PaddleFleet && git apply ../../nzs_md/stage1_fsdp_align/PaddleFleet.patch
```

基线不同时先 `git apply --check` 看冲突，或用 `git apply -3`。

## 环境变量

| 变量 | 作用侧 | 作用 |
|---|---|---|
| `MUON_FSDP_BCAST_GRAD=1` | stage1 + FSDP | 梯度通信全部换成 `broadcast(src=rank0)` + `1/nranks`，两侧都只看到 rank0 的本地梯度，消除 reduce 的舍入差异和 shard 归属差异 |
| `MUON_FSDP_NS_SINGLE=1` | stage1 + FSDP | Muon 的 Newton-Schulz 逐参数计算，不再按 shape 打 batch |
| `MUON_FSDP_STAGE1_CLIP_EPS=1` | 只在 FSDP 侧起作用 | 补齐 clip 公式里 stage1 多出的 `+1e-6`（实现在 erniebot 仓库，见下文） |

另外配合确定性 flag：`FLAGS_embedding_deterministic=1`、`FLAGS_cudnn_deterministic=1`、`FLAGS_deterministic_rng=1`。启动脚本参考 `run_align_1node.sh`：`bash run_align_1node.sh <stage1|fsdp> <sonic: true|false> [max_steps]`。

## 各处改动与原因

### 1. 梯度通信改 broadcast（`MUON_FSDP_BCAST_GRAD`）

stage1 和 FSDP 的梯度规约方式不同（reduce 到 owner / reduce_scatter / all_reduce），各 rank 拿到的 shard 也不同，规约的求和顺序和舍入就不同。改成从 rank0 广播后，没有任何算术参与通信，两侧梯度完全一致。

- `paddle/distributed/fleet/utils/tensor_fusion_helper.py`（stage1 的 `FusedCommBuffer`）
  - `REDUCE`、`REDUCE_SCATTER` 两个分支换成 `broadcast(src=rank0)`，并在广播前 `scale_(1/nranks)`。
  - 广播前不再走原来的 pre-scale；用 `_bcast_scale_applied` 标记，防止 `_scale_after_comm` 路径在 wait 后再除一次 `nranks`。这一步最容易漏，漏了梯度会小 `nranks` 倍。
  - 新增 `_muon_align_log`，每个 tag 只在 rank0 打一次日志，用来确认 hack 生效。
- `paddle/distributed/fleet/meta_optimizers/muon_sharding_optimizer.py`
  - `_reduce_2d_grads`：2D 参数的 reduce-to-owner 换成 `scale_(1/nranks)` + 同步 broadcast。
- `paddle/distributed/fsdp/fully_shard_fusion.py`
  - `TensorFusionBuffer.do_reduce_scatter`：换成整块 broadcast，再把本 rank 的 shard（`shard_start` 起 `data_buffer.shape[0]` 个元素）挪到 buffer 头部，即 reduce_scatter 原本放结果的位置。`grad_scale` 推迟到 `finish_grads_sync`，与 stage1 一样是 `(sum g_i) * s`。
  - `FSDPCommManager` 中非 sharded buffer：`all_reduce`（`whole_shard_numel`）和 `reduce(owner)`（`whole_owner_rank`）都换成 broadcast；owner 路径后续的非 owner 清理逻辑保持不变。
  - `FullyShardFusion` 中一个参数有多个 fp32 grad 分量时：除最后一个外逐个经累加器 reduce_scatter + `accumulate_reduced_grad`，使舍入顺序为 `(acc + a) + b`，与非 FSDP 一致，而不是 `acc + (a + b)`。

### 2. Newton-Schulz 逐参数（`MUON_FSDP_NS_SINGLE`）

batched Newton-Schulz（`paddle.stack`/concat + `baddbmm`）的结果依赖 batch 组成。stage1 和 FSDP 给每个 owner rank 分到的参数集合和顺序不同，batch 不同，结果就会有差异。改为逐参数调用（单矩阵走 `paddle.addmm`）后与 batch 组成无关。

- `paddle/optimizer/muon.py`：`Muon._muon_update_group` 在 group 内参数数 > 1 时递归为单参数调用。
- `PaddleFleet/src/paddlefleet/optimizers/muon_optimizer.py`：`MuonOptimizer` 遍历 `muon_groups` 时逐个调用 `_update_shape_group([item], lr)`，跳过 `_split_group_by_bytes`；首次生效打印 `[ALIGN] fleet ...: per-param Newton-Schulz (MUON_FSDP_NS_SINGLE) ACTIVE`。

### 3. clip eps（`MUON_FSDP_STAGE1_CLIP_EPS`，不在本目录 patch 内）

实现在 erniebot 仓库 `ernie5/src/trainers/pretraining_trainer.py` 的 `_install_stage1_clip_eps_hack()`。stage1 的 `MoEHybridParallelClipGrad` 用 `max / (maximum(norm, max) + 1e-6)`，FSDP 的 `ClipGradByGlobalNorm` 用 `max / maximum(norm, max)`。该 hack monkey-patch `ClipGradByGlobalNorm.__call__`，在 FSDP 侧再乘 `max / (max + 1e-6)`。只在未触发裁剪时与 stage1 一致，触发裁剪时两边仍不同。

## 生效确认

rank0 日志里应能看到：

```
[MUON_FSDP_BCAST_GRAD] stage1: reduce_scatter -> broadcast(src=0) ACTIVE   # 或 reduce(owner) / reduce_2d(owner)
[MUON_FSDP_BCAST_GRAD] FSDP:   reduce_scatter -> broadcast(src=0) ACTIVE   # 或 reduce(owner)
[ALIGN] fleet MuonOptimizer: per-param Newton-Schulz (MUON_FSDP_NS_SINGLE) ACTIVE
[ALIGN] stage1 clip-eps hack ACTIVE (factor=max/(max+1e-6))
```
