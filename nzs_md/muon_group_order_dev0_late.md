# Muon 分组顺序导致 rank0 晚到：分析、改法与验证（2026-11-04）

## 1. 现象

来源：`timeline/ernielite-T1-fsdp_n0.nsys-rep`（T1 h512 L8，n0 的 8 卡，step 6–9，丢掉第 1 个采集步后取 3 个稳态步）。

- 步时约 1565 ms。通信 kernel 每步约 237 ms，其中没有和计算重叠的约 189 ms（12.1%）。
- 没重叠的部分里，约 100 ms 是在等最慢的卡，真正搬数据的主要是 DeepEP dispatch/combine（约 86 ms）。
- 等待里有两项明显是在等 dev0：
  - step 末那次 AllReduce_f32（每步 7 次）：dev0 晚到约 14 ms/步。
  - 下一步第一个 FSDP AllGather：dev0 晚到约 22 ms/步。

## 2. 根因

### 2.1 问题 1：优化器各组的执行顺序在 rank0 上不同（已确认）

trace 证据：优化器窗口里 8 卡的 kernel 数（2938）和 kernel 总耗时（31.86 ms）完全相同，dev0 的窗口却是 69.4 ms，别的卡约 52 ms。对比 kernel 序列，只有一块 628 个 kernel 的位置不同：Concat、batched NS GEMM、Split，也就是 3D expert 组。这块在 dev1–7 上排在序号 82，在 dev0 上排在序号 2258。

dev0 先下发约 2200 个小 kernel，主机下发跟不上，GPU 大部分时间空等；最后再纯 GPU 执行 expert 组，没有东西和它重叠，所以多出约 17 ms。

代码链路有三处叠加：

1. `Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py` 的 `_assign_whole_owners`：把 dense 的 keep_whole 组从大到小排序，分给当前负载最小的 rank。排序是稳定的，所以第 r 层的 dense 组归 rank r，rank0 拿到第 0 层。
2. 同文件 buffer_groups 的 `sort_key` 是 `(unit_id, is_expert_param, keep_whole)`，同一层里 dense 组排在 expert 组前面。`whole_params_grads` 里，dense 组只在 owner 上出现，expert 组每个 rank 都有自己那一片。
3. `PaddleFleet/src/paddlefleet/optimizers/muon_optimizer.py` 的 `_update_parameters` 用 `defaultdict` 按 `(split_func_key, shape)` 分组，组的执行顺序就是 key 第一次插入的顺序。

结果是 rank0 先插入自己拥有的 dense shape，expert 组排到最后；其余 rank 先插入 expert 组。

运行时确认（`MUON_ORDER_DEBUG=1`，T1 两机 16 卡）：

```
rank=0 [([2048, 512], 1), ([512, 192], 1), ... ([512, 512], 1), ([72, 256, 1024], 8)]
rank=1 [([72, 256, 1024], 8), ([2048, 512], 1), ...]
```

T4 上现象相同：rank0 以 `[5120, 1280]` 开头，rank1、rank2 以 `[72, 640, 2304]×20` 开头。

### 2.2 问题 2：rank0 在 step 之间多 20–25 ms 的主机侧工作（未复现）

在原 trace 里，step 末同步之后到下一步第一个 AllGather，dev0 比其他卡多 20–25 ms。三张卡的 CUDA API 调用数都是 3251，完全相同，但 dev0 有两段没有任何 CUDA 调用的空档（约 14 ms 和 4–11 ms），夹在 D2H `cudaMemcpy` 和 `cudaEventSynchronize` 之间。看起来是 rank0 专属的 Python 逻辑（日志、metrics 或 callback），原 trace 用了 `--sample=none`，定位不到具体哪一行。

在下午的新 trace 里（两机，AllGather 分在两条流上），A 和 B 两组的 AllGather 上 dev0 晚到都是 0，这个问题没有复现，没有继续追。

## 3. 代码改动

### 3.1 `PaddleFleet/src/paddlefleet/optimizers/muon_optimizer.py`

两个环境变量开关，默认都是关，不影响现有行为：

- `MUON_ORDER_BY_COST=1`：各组按 NS 计算量从大到小执行。
- `MUON_ORDER_DEBUG=1`：每个 rank 打印一次分组顺序。

```diff
 import math
+import os
 from collections import defaultdict
@@
 __all__ = ["MuonOptimizer"]
 
 
+def _ns_cost(shape, n_params):
+    """Relative Newton-Schulz FLOPs of a group: matrices x m * n * min(m, n)."""
+    m, n = shape[-2], shape[-1]
+    n_mat = shape[0] if len(shape) == 3 else 1
+    return float(n_params) * n_mat * m * n * min(m, n)
+
+
 class MuonOptimizer(InnerOptimizer):
@@ def _update_parameters(self, params_grads, lr):
             muon_groups[(split_func_key(scf), param_shape)].append(
                 (param, grad)
             )
-        for grp in muon_groups.values():
+        groups = list(muon_groups.items())
+        if os.getenv("MUON_ORDER_BY_COST", "0") == "1":
+            # Heaviest Newton-Schulz groups first, so their GPU time overlaps
+            # the host launching the many small groups; also makes the order
+            # independent of which whole params this rank owns.
+            groups.sort(key=lambda kv: -_ns_cost(kv[0][1], len(kv[1])))
+        if os.getenv("MUON_ORDER_DEBUG", "0") == "1" and not getattr(
+            self, "_order_logged", False
+        ):
+            self._order_logged = True
+            print(
+                f"[MUON_ORDER] rank={paddle.distributed.get_rank()} "
+                f"{[(list(k[1]), len(v), f'{_ns_cost(k[1], len(v)):.3g}') for k, v in groups]}",
+                flush=True,
+            )
+        for _, grp in groups:
             for sub_grp in self._split_group_by_bytes(grp):
                 self._update_shape_group(sub_grp, lr)
```

各组的更新互相独立，hyperball 也是逐参数做的，所以改执行顺序不改变数值语义。

### 3.2 启动脚本

- `run_fsdp_test.sh`：
  - `declare -p` 增加 `MAX_STEPS MUON_ORDER_BY_COST MUON_ORDER_DEBUG`，两机运行时传到远端节点。
  - export 这两个开关，默认 0。
  - `max_steps=${MAX_STEPS:-10}`。
- `run_fsdp_test_v2.sh`：`declare -p` 增加 `MUON_ORDER_BY_COST MUON_ORDER_DEBUG`，并 export，默认 0。
- `run_muon_order_test.sh`（新增）：从 `run_fsdp_test.sh` 复制，用 `Paddle/build_reshard`，透传 `MUON_ORDER_*`、`CONFIG`、`TAG`、`MAX_STEPS`。

单机 8 卡的跑法：不改脚本，在命令行上跳过 mpirun 派发：

```bash
OMPI_COMM_WORLD_RANK=0 POD_INDEX=0 TRAINER_INSTANCES=10.95.241.104 \
MAX_STEPS=20 MUON_ORDER_BY_COST=1 MUON_ORDER_DEBUG=1 \
CONFIG=conf/experiment/ERNIE-5.5-Flash/ScalingLadder/<cfg>.yaml TAG=<tag> \
bash run_fsdp_test.sh
```

## 4. 验证

### 4.1 nsys：机制确认（T1，两机 16 卡，n0 的 8 卡，`/tmp/dev0_late.py`）

step 末那次 AllReduce_f32（stream 37，每步 7 次）：

- dev0 晚到：A（原顺序）9.24 ms/步，B（按计算量排序）0.25 ms/步。
- 其他卡等待：A 11.15 ms/步，B 4.27 ms/步。

### 4.2 端到端 A/B

每次 20 步，丢掉前 2 步取中位数。optimizer-step 是 rank0 日志里框架自报的字段，只能当线索。

| 配置 | 卡数 | A1 / A2 步时 | B1 / B2 步时 | B 均值 − A 均值 | A1 与 A2 之差 | optimizer-step A → B |
|---|---|---|---|---|---|---|
| T1 | 单机 8 卡 | 1.5559 / 1.5914 s | 1.5756 / 1.5783 s | +3 ms | 35 ms | 70 → 58 ms |
| T2 | 单机 8 卡 | 5.5196 / 5.5257 s | 5.5045 / 5.5317 s | −5 ms | 6 ms | 154 → 142 ms |
| T3 | 单机 8 卡 | 12.7597 / 12.7874 s | 12.7086 / 12.7797 s | −29 ms | 28 ms | 296 → 286 ms |
| T4 | 两机 16 卡 | 16.5217 / 16.4976 s | 16.5222 / 16.4933 s | −2 ms | 24 ms | 353 → 342 ms |
| T1 | 两机 16 卡 | 1.5671 / 1.5750 s | 1.5724 s（只跑了一次 B） | — | 8 ms | — |

- optimizer-step 在所有配置上都稳定少了 10–12 ms，方向和 nsys 一致。
- 端到端：B 和 A 的差都不比 A1 与 A2 之间的差大，没有可确认的收益。
- 数值：基线本身不确定，A1 和 A2 从第 3 步起 loss 就有 1e-5 量级的差。B 和 A 的差在同一量级，做不了逐位比对，只能说在运行间噪声范围内一致。

## 5. 结论与后续

- 根因已确认，改法在机制上生效：dev0 的晚到基本消除，optimizer-step 少约 10 ms。
- 端到端收益小于运行间噪声，T1 约 0.7%、T4 约 0.06% 的理论上限都测不出来。
- 改动保留在开关后面，默认关闭。
- 后续可选：
  1. 打开确定性开关（`FLAGS_*_deterministic`）做逐位对齐。
  2. 加长步数或多做几组重复对，压低噪声后再判断端到端收益。
  3. 如果要默认开启，可以直接让 3D expert 组先执行，不再依赖环境变量。

## 6. 本次相关的其他改动

### 6.1 `fsdp_reshard_policy` / `fsdp_reshard_filter` 支持

起因：T4 配置里写了这两个字段，启动时报"未知 flat config"。参考 `lizhenxing/ladder/1103` 补了三层：

- `PaddleFleet/src/paddlefleet/trainer/training_args.py`：新增 `fsdp_reshard_policy`（默认 `after_forward`）和 `fsdp_reshard_filter`（默认 `all`）。
- `ernie5/src/trainers/pretraining_trainer.py`：`fully_shard(...)` 透传 `reshard_policy` 和 `reshard_filter`。
- `Paddle/python/paddle/distributed/fsdp/fully_shard.py`：签名、文档和 `_fully_shard_manual_parallel` 都加上这两个参数。
- `Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`：`FSDPCommManager` 增加 reshard 策略校验、`keep_gathered`、`_keep_after_forward`、`_keep_after_backward`，以及释放和计数时跳过保留的 buffer。
  - 注意：这个文件现在和参考文件逐行完全一致，还带进了参考里的其他改动（只有 padding 的 shard 不建优化器状态、`sync_grads_each_microbatch`、`_backward_cnt`）。

两个 Paddle 文件已同步到 `Paddle/build/python`。

### 6.2 `Paddle/build_reshard`

原 `Paddle/build/python` 有其他进程在用，所以把 `Paddle/build/python/paddle` 完整复制到 `Paddle/build_reshard/python/paddle`，`diff -rq` 无差异。当前 `run_fsdp_test.sh` 和 `run_fsdp_test_v2.sh` 都指向 `build_reshard`。

## 7. 数据位置

- 原始 trace：`timeline/ernielite-T1-fsdp_n0.nsys-rep`
- A/B trace：`timeline/muon-order-{A,B}_n{0,1}.nsys-rep`
- 日志：
  - `log_muon_order/`：T1 两机
  - `log_muon_order_t4/`：T4 两机
  - `log_t123_ab/`：T1–T3 单机 A/B/A/B
  - `log_t123_1node/`：T1–T3 单机基线
- 分析脚本：`nzs_md/scripts/t1_comm.py`（通信重叠），`nzs_md/scripts/dev0_late.py`（dev0 晚到）。用法：`python dev0_late.py <x.sqlite>`，`python t1_comm.py <x.sqlite> <起始ms> <结束ms> <步数>`，sqlite 用 `/root/paddlejob/paddlefleet-public/lizhenxing/FSDP/nsys/bin/nsys export --type sqlite` 导出。
