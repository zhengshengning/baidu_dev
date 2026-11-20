# ERNIE-5.5-Omni-Flash stage1 × FSDP 逐位对齐：全部修改清单

> **用途**：让 `conf/online/ERNIE-5.5-Omni-Flash/ernie5_5_omni_flash-pretrain-64k-bzz3-2node.yaml`（stage1）与
> `...-2node-fsdp.yaml`（FSDP）在同配置下 loss 逐位一致。
> **性质**：除 §4.4 的 encoder 专家梯度修复（生产 bug，必须保留）外，其余都是诊断 hack，
> 只在设置环境变量（启动脚本里统一为 `ALIGN=1`）时生效，默认关闭。**开启后训练语义是错的，不可用于生产。**
> 前身文档：同目录 `stage1_fsdp_align_hack.md`（Paddle 侧原始 hack）、`align_hack2.md`（ERNIELite 版）。

## 0. 验证结果（2026-11-19）

| 模型 | 配置 | 步数 | loss | grad_norm |
|---|---|---|---|---|
| ERNIE-5.5-Flash ladder C4_T4（h1280/L20，无 mHC/MTP/encoder） | 纯文本 `EB5Pretrain-260930-...-10pct`，2 机 | 100 | 逐位一致 | 2 步末位差 1 ulp |
| ERNIE-5.5-Omni-Flash 2node（L43 + HyperBody encoder + MTP + mHC + DSA） | 多模小清单 `omni_flash_text260904_mm1111hs_64k.1per_src`，fakegate，2 机 | 100 | 逐位一致 | 31 步末位差（最大 rel 2.9e-7） |

grad_norm 末位差来自 FSDP 在融合 buffer 上求平方和、加法顺序与 stage1 不同，只影响显示；对齐时裁剪已关，不进入更新。
逐参数梯度（第 1 步，各 rank 平方和相加后比较）：1647 个参数最大相对差 1.84e-7。

日志：`erniebot/outputs/20261119_1953_*-2node-fsdp`、`20261119_1954_*-2node`（Omni 100 步）；
`20261119_1656_*ladder*`（ladder 100 步）。

## 1. 代码基线

| 组件 | 位置 | 基线 | 对齐相关改动状态 |
|---|---|---|---|
| Paddle（训练实际加载） | `FSDP/fsdp_venv/lib/python3.12/site-packages/paddle` | nightly `3.5.0.dev20261116+10a99906f58` | 已打 `Paddle.patch`；原文件备份 `FSDP/align_patch_backup_20261119/` |
| Paddle 源码 | `FSDP/Paddle` | `10a99906f5` | 已打 `Paddle.patch`（4 文件 staged，未 commit），并已拷到 `build/python`（原文件备份 `FSDP/align_patch_backup_build_20261119/`） |
| PaddleFleet | `erniebot/third_party/PaddleFleet` | `630b660e fsdp_support_onim` | §4.4 已在 `630b660e`；§4.1–4.3 未提交 |
| erniebot | `erniebot`（`dev`） | `4d8c029f` | §5.3 已提交（Gerrit 123373396）；§5.1/5.2/5.4 未提交 |

**两侧必须加载同一份 Paddle**（都用 venv，或都在 PYTHONPATH 里加 `FSDP/Paddle/build/python`）。
`build/python` 的 `libpaddle.so` 与 venv 不同，混用即使 Python 补丁一致也不保证逐位一致。
`run_omni_fsdp.sh` 曾把 PYTHONPATH 指到 build（未打补丁），导致 FSDP 侧 hack 全部没生效，现已注释掉。

## 2. 补丁文件（本目录）

| 文件 | 应用位置 | 内容 |
|---|---|---|
| `Paddle.patch` | Paddle 源码 `python/paddle/`（venv 用 `patch -p2`） | §3 全部 |
| `PaddleFleet.patch` | PaddleFleet | §4.1（原版逐参数 NS） |
| `PaddleFleet_align_full.patch` | PaddleFleet（基线 `630b660e`） | §4.1 + §4.2 + §4.3，相对 HEAD 的完整 diff |
| `PaddleFleet_encoder_expert_grad_fix.patch` | PaddleFleet（基线 `251e9a3f`） | §4.4，`git show 630b660e -- fusion_layer_utils.py` |
| `erniebot_align.patch` | erniebot | §5.1 + §5.2（`hybrid_parallel_optimizer.py`） |

应用：

```bash
# venv 里的 Paddle（纯 Python，无需编译）
cd FSDP/fsdp_venv/lib/python3.12/site-packages && patch -p2 < .../Paddle.patch
# Paddle 源码
cd FSDP/Paddle && git apply .../Paddle.patch
# PaddleFleet / erniebot
cd erniebot/third_party/PaddleFleet && git apply .../PaddleFleet_align_full.patch
cd erniebot && git apply .../erniebot_align.patch
```

基线不同先 `git apply --check`，失败按 §3–§5 手工移植。

## 3. Paddle 侧（`Paddle.patch`，开关 `MUON_FSDP_BCAST_GRAD` / `MUON_FSDP_NS_SINGLE`）

与 `stage1_fsdp_align_hack.md` 相同，此处只列位置与作用。

| # | 文件（`python/paddle/` 下） | 位置 | 作用 |
|---|---|---|---|
| A | `distributed/fsdp/fully_shard_fusion.py` | `TensorFusionBuffer.do_reduce_scatter` | reduce_scatter → `broadcast(src=rank0)`，再把本 rank 的 shard 挪到 buffer 头部；不在此处缩放 |
| A2 | 同上 | `finish_grads_sync` | 用 `grad_scale_pending` 在累加后的 shard 上缩放一次，得到 `(Σg)·s`，与 stage1 一致 |
| B | 同上 | `finish_grads_sync` 非分片分支 | `all_reduce` / `reduce(owner)` → broadcast |
| H | 同上 | `FullyShardFusion` 的 `comm_hook` | fused kernel 的多个 fp32 梯度分量（主干 + MTP head）逐个 flush，舍入为 `(acc+a)+b` |
| C/D/G | `distributed/fleet/utils/tensor_fusion_helper.py` | `FusedCommBuffer` REDUCE / REDUCE_SCATTER | stage1 侧改 broadcast + `÷nranks`，并防止通信后重复缩放 |
| F | `distributed/fleet/meta_optimizers/muon_sharding_optimizer.py` | `_reduce_2d_grads` | stage1 2D Muon 非融合路径改 broadcast |
| E | `optimizer/muon.py` | `_muon_update_group` | `MUON_FSDP_NS_SINGLE=1` 时逐参数做 Newton-Schulz |

broadcast 之后两侧都只用 rank0 的本地梯度，通信不再引入任何算术差异。

## 4. PaddleFleet 侧

### 4.1 逐参数 Newton-Schulz（`muon_optimizer.py::_update_parameters`，`MUON_FSDP_NS_SINGLE`）

即原 `PaddleFleet.patch`。同 shape 的参数不再按字节打 batch，逐个调用 `_update_shape_group([item])`。
日志：`[ALIGN] fleet <Optimizer>: per-param Newton-Schulz (MUON_FSDP_NS_SINGLE) ACTIVE`。

### 4.2 🆕 3D 专家权重逐专家做 NS（`muon_optimizer.py::_update_shape_group`，`MUON_FSDP_NS_SINGLE`）

4.1 只拆开了不同参数，3D 专家权重 `[E, H, I]` 内部仍是一个 batched GEMM。stage1 的 owner 拿到全部专家
（如 `[96, 640, 2304]`），FSDP 按专家维切分后每个 rank 只有一部分（如 `[24, 640, 2304]`），batch 数不同，
kernel / tile 选择和舍入随之不同。修法：3D 参数的 `ortho_fn` 按专家维逐片调用再 concat；有
`split_concat_func` 时对拆出的每一片同样生效。**ladder 上缺这一条时，第 3 步起 loss 分叉。**

### 4.3 🆕 hyperball 逐专家投影（`hyperball.py::hyperball_apply`，`MUON_FSDP_NS_SINGLE`）

3D 参数（`reduce_group is None` 时）按专家逐片调用 `hyperball_apply` 再 concat，每个专家的 Frobenius norm
与本 rank 持有多少专家无关。只影响 `muonh` / `adamh` 等 `*h` 优化器。

### 4.4 🆕 encoder 专家梯度修复（`transformer/moe/fusion_layer_utils.py`，**生产修复，无开关**）

- **现象**：FSDP 下 HyperBody encoder 的 22 个专家权重（11 层 × w1 `[64,1280,1792]` / w2 `[64,896,1280]`）
  梯度全 0，关掉所有对齐 hack 也一样。encoder 专家在 FSDP 下一直没有更新。
- **原因**：encoder MoE 的 EP=1，`moe_layer.py` 的 EP=1 分支不给参数设 `is_moe_param`，FSDP 把它们当 dense
  参数分片。fused MoE 反向不经 autograd、直接写 `weight.main_grad`，遇到 None 就自己 `paddle.zeros`；
  FSDP 只给 `is_moe_param` 绑定 `main_grad`，dense 参数靠 autograd grad hook，于是梯度写进了 FSDP 不知道的
  私有 tensor，从未被 reduce。stage1（`main_grad` 常驻）和 decoder 专家（EP=16，有 `is_moe_param`）不受影响。
- **修法**：
  - 新增 `_bind_fsdp_main_grad(weight)`：权重有 `_fusion_buffer` 时，先 `comm_manager._maybe_zero_grads()` /
    `_ensure_grads_writable()`，再 `rebind_main_grad`，把 `main_grad` 绑到 FSDP grads buffer 里的槽位。
  - `_ensure_weight_grad` 对 FSDP 管理的权重改走上面的绑定，不再自己分配。
  - `backward()` 入口先对所有专家权重绑定一次（group_gemm 路径在 expert node 里懒分配 `main_grad`）。
  - 没有 `_fusion_buffer` 的权重行为完全不变。
- 已在 PaddleFleet `630b660e`。

## 5. erniebot 侧

### 5.1 🆕 关掉梯度裁剪（`ernie5/src/trainers/dygraph_optimizer/hybrid_parallel_optimizer.py`，`ALIGN_NO_CLIP=1`）

`HybridParallelClipGrad._comm_and_clip` 里把 `clip_var`（及 multimax 的 `clip_var_multimax`）置为 1；norm 照常算和打印。
Omni 每步 grad_norm 在 7–1000，`max_grad_norm=1.0` 每步都会触发裁剪；两侧 norm 差 1 ulp 时裁剪系数也不同，必须关。
日志：`[ALIGN] no-clip hack ACTIVE (clip_var = 1)`。
两侧 clip 都是 ernie5 的 `HybridParallelClipGrad`（FSDP 侧经 `distributed_optimizer_maybe_hack` 替换），公式里都有
`+1e-6`，所以 `align_hack2.md` 的 `MUON_FSDP_STAGE1_CLIP_EPS` 不需要。

### 5.2 🆕 逐参数梯度 dump（同文件，诊断工具，`GRAD_SUMSQ_DUMP_DIR`）

- `_dump_grad_sumsq`：clip 末尾把本 rank 每个参数梯度的平方和写到
  `$GRAD_SUMSQ_DUMP_DIR/step{n}_rank{r}.txt`（参数名、dtype、shape、float32 hex），前 `GRAD_SUMSQ_DUMP_STEPS`（默认 1）步。
- `_dump_fsdp_shard_sumsq`：FSDP 下 AdamW 类参数合在 `fuse_params_*` 里按元素切分，buffer 级的值对不上 stage1；
  这里用 `param_shard` 取出本 rank 与每个原参数重叠的那段，写 `*.shard.txt`。
- 比较方法：两侧都按参数名把所有 rank 的值相加再比（stage1 与 FSDP 的切分方式不同，不能逐 rank 比）。

### 5.3 trainer 的 FSDP 支持（`pretraining_trainer.py`，已提交 `4d8c029f`）

- `sharding=fsdp` 时断言 `calculate_per_token_loss=false` 和 `hyperencoder_batched_patch_embed=false`
  （两者在 FSDP 下会算错或读到已释放的分片，见 commit message）。
- `fully_shard(reshard_policy=os.environ.get("FSDP_RESHARD_POLICY", "after_forward"))`。多模数据下 `after_forward`
  会在前端非 unit 参数处 hang，`run_omni_fsdp.sh` 默认 `after_step`。
- per-token 统计取 hcg 时，优化器上没有 `_hcg`（FSDP 不包 HybridParallelOptimizer）就用
  `fleet.get_hybrid_communicate_group()`。

### 5.4 启动脚本（`run_omni.sh` / `run_omni_fsdp.sh` / `run_ladder_align.sh`）

`ALIGN=1` 时打开全部对齐开关，并经共享盘 env 文件传给各节点（mpirun 不透传环境变量）：

```bash
if [[ "${ALIGN:-0}" == "1" ]]; then
  export FLAGS_embedding_deterministic=1 FLAGS_cudnn_deterministic=1 FLAGS_deterministic_rng=1
  export MUON_FSDP_BCAST_GRAD=1 MUON_FSDP_NS_SINGLE=1 ALIGN_NO_CLIP=1
  # routed_scaling_factor gather 反向改成无 atomic 的实现（默认路径 atomic 累加，跑两次不可复现）
  export FLEET_MOE_ROUTER_SCALE_FAST=1
  # DSA / CSA 反向的 cudnn kernel 用 atomic 累加（不可复现），对齐时换 tilelang（逐位稳定，约慢 14 倍）
  ALIGN_KWARGS="mqa_sparse_attn_backward_backend=tilelang csa_indexer_backend=tilelang csa_sparse_attn_backend=tilelang"
fi
[[ -n "${GRAD_SUMSQ_DUMP_DIR}" ]] && export GRAD_SUMSQ_DUMP_DIR GRAD_SUMSQ_DUMP_STEPS
...
    --kwargs ... ${ALIGN_KWARGS} ${EXTRA_KWARGS}
```

另外：
- `declare -p ... ALIGN OMNI_MASTER_PORT GRAD_SUMSQ_DUMP_DIR GRAD_SUMSQ_DUMP_STEPS` 写进 env 文件。
- `run_omni_fsdp.sh` 加了与 `run_omni.sh` 相同的 `OMNI_MASTER_PORT`（61001–65000 随机），否则 aq 分到的 master 端口
  落在临时端口范围内会 `Address already in use`。
- `run_ladder_align.sh <stage1|fsdp>`：ladder 版，stage1 侧用 kwargs 改成 pp1 / acc16，与 FSDP 版并行配置一致。

## 6. 各差异源与对应的修改

按排查顺序。前面几条不处理，后面的问题会被掩盖。

| # | 差异源 | 现象 | 修改 |
|---|---|---|---|
| 1 | 梯度通信的求和顺序 / shard 归属 | 第 1 步梯度两侧不同 | §3（BCAST） |
| 2 | 不同参数打 batch 做 NS | 第 2 步更新后分叉 | §3-E、§4.1 |
| 3 | 3D 专家内部 NS / hyperball 的 batch 数不同 | ladder 第 3 步起 loss 分叉 | §4.2、§4.3 |
| 4 | 梯度裁剪每步触发，norm 差 1 ulp 系数就不同 | Omni 每步都可能分叉 | §5.1 |
| 5 | 反向 atomic kernel：DSA dkv、CSA indexer / sparse attention、router scale gather | **stage1 自己跑两次都不一致**（第 1 步 2434/3035 个 (参数,rank) 不同） | §5.4 的 tilelang 后端 + `FLEET_MOE_ROUTER_SCALE_FAST` |
| 6 | encoder 专家 FSDP 下梯度全 0 | 第 1、2 步 loss 一致，第 3 步起分叉（第 2 步是首次更新） | §4.4 |
| 7 | 真实路由的 top-k 翻转 | 放大任何微小差异 | `moe_router_force_load_balancing=true`（fakegate） |

## 7. 配置要求

两侧 yaml 除 `sharding: stage1|fsdp`、`split_param`、输出目录外必须一致，并且：

- `calculate_per_token_loss: false`、`hyperencoder_batched_patch_embed: false`（FSDP 不支持，stage1 也要关掉才能对比）。
- `moe_router_force_load_balancing=true`（命令行 `EXTRA_KWARGS` 覆盖）。
- `gradient_accumulation_steps: 1`（`align_hack2.md` 的 `MUON_FSDP_EXPERT_NO_ACCUM_SCALE` 针对 accum>1；我们的 trainer
  在 FSDP 下已跳过手动 1/accum，ladder acc16 下专家梯度也已逐位一致，所以不需要；该变量当前代码里也没有人读）。
- 不需要的旧 hack：`MUON_FSDP_STAGE1_CLIP_EPS`（见 §5.1）、`align_hack2.md` 的 T3（我们的分组里没有 AdamH）。

## 8. 运行与校验

```bash
cd erniebot
KW="moe_router_force_load_balancing=true max_steps=100"
ALIGN=1 AQ_SUBMIT=1 EXTRA_KWARGS="$KW" bash ./run_omni_fsdp.sh s100
sleep 70   # 输出目录按分钟命名，同一分钟启动会写进同一个目录、互相覆盖
ALIGN=1 AQ_SUBMIT=1 EXTRA_KWARGS="$KW" bash ./run_omni.sh s100
```

末尾的 `s100` 脚本不读，只用来让 aq 区分两次命令行相同、环境变量不同的提交（aq 按命令行去重）。

rank0 日志应能看到（缺哪条就是哪处没生效）：

```
# stage1
[MUON_FSDP_BCAST_GRAD] stage1: reduce(owner) -> broadcast(src=0) ACTIVE
[MUON_FSDP_BCAST_GRAD] stage1: reduce_scatter -> broadcast(src=0) ACTIVE
# FSDP
[MUON_FSDP_BCAST_GRAD] FSDP:   reduce_scatter -> broadcast(src=0) ACTIVE
# 两侧
[ALIGN] fleet MuonOptimizer: per-param Newton-Schulz (MUON_FSDP_NS_SINGLE) ACTIVE
[ALIGN] no-clip hack ACTIVE (clip_var = 1)
```

envlog 里应有 `ALIGN_NO_CLIP=1`、`FLEET_MOE_ROUTER_SCALE_FAST=1`；模型配置（`pp_layers.py:947` 那行）里三个 backend 应为 `'tilelang'`。

对比：逐步比 loss（要求完全相同）。**第 1、2 步相同不说明问题**（第 1 步 lr=0，第 2 步才首次更新），至少看到第 3 步之后。
grad_norm 允许末位差（见 §0）。

## 9. 排查方法

1. **先测自身确定性**：同一侧同配置跑两次，用 §5.2 的 dump 逐参数比。若自己都不一致，问题在 kernel，不在 FSDP。
2. **再比两侧**：按参数名把各 rank 的平方和相加后比（FSDP 用 `*.shard.txt`）。按参数类型（expert / dense /
   mHC / router / DSA）分组看哪类先出现差异。
3. **定位到 step**：梯度一致而 loss 分叉，问题在优化器更新（如 §4.2）；梯度就不一致，问题在前向 / 反向 / 通信。
