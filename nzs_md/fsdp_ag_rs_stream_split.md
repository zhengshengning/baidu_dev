# FSDP AllGather / ReduceScatter 通信流拆分：实现与 A/B 记录

本文记录一次参考 Megatron-FSDP 对 Paddle FSDP（`FullyShardFusion`）做的通信流优化：把 AllGather（AG）和 ReduceScatter（RS）从同一条 stream 上拆开，并给 AG 单独建一个 NCCL communicator。包括问题定位、参考实现、代码改动、nsys A/B 数据和结论。

数据分两批：

- §1–§4：2026-11-04，单机 8 卡 T1/T2/T3，只测拆 stream / 拆 communicator（`FSDP_AG_RS_MODE`）。
- §5：2026-11-05，加测反向里调换 AG 预取和 RS 的提交顺序（`FSDP_AG_BEFORE_RS`），以及在调换基础上叠加拆 stream / 拆 communicator，覆盖单机 T1/T2/T3 和两机 T4。
- §6：对照 Megatron-FSDP 的实现，看两边的差异能不能解释没有收益。

证据文件路径见 §10。

## 1. 问题：RS 挡住 AG 预取

### 1.1 现象

ERNIE-5.5-Flash ScalingLadder T1/T2/T3 的 baseline nsys 里，FSDP 的 AG 和 RS 都在 s46 这一条 stream 上（`ab_analysis/steps.py` 打印的 stream 分布）。

| tier | AG 次数/步 | RS 次数/步 | AG busy ms/步 | RS busy ms/步 | AG∩RS | 紧跟在 RS 后面排队的 AG 次数/步 |
|---|---|---|---|---|---|---|
| T1 | 57 | 48 | 29.0 | 18.1 | 0 | 2.0 |
| T2 | 477 | 270 | 65.9 | 93.6 | 0 | 62.9 |
| T3 | 1445 | 726 | 185.1 | 251.6 | 0 | 216.8 |

（8 卡均值；AG∩RS = AG 与 RS 同时在跑的时间；"排队"= 同 stream 上前一个 kernel 是 RS，且 RS 结束后 20 us 内 AG 就开始。）

计算流在等 AG 的时间里，有一部分其实是 AG 排在 RS 后面等出来的（`ab_analysis/stall_unblocker.py`，把计算流每个 > 5 us 的空隙记到"计算恢复前最后结束的通信 kernel"头上）：

| tier | 计算流等 AG（AG 本身） | 等 AG（AG 排在 RS 后） | 等 RS |
|---|---|---|---|
| T1 | 23.5 | 6.2 | 3.4 |
| T2 | 35.8 | 11.6 | 4.7 |
| T3 | 54.7 | 46.6 | 5.7 |

单位 ms/步，8 卡均值。T3 一步 12.9 s，"AG 排在 RS 后"这一项是 46.6 ms，占 0.36%。这是本次优化能拿到的上界量级，不是收益。

### 1.2 原因

1. stream 是按 fsdp group 建的：`FSDPBufferManager._comm_stream(group)` 以 `group.id` 为 key，params buffer（发 AG）和 grads buffer（发 RS）拿到的是同一条 stream。
2. 反向里 `all_gather_params(is_backward=True)` 的顺序是：先 `_flush_expert_grads_after_unit` 发出后面 unit 的 RS，再等当前 unit 的 AG，最后发下一个 unit 的 AG 预取。所以 RS(U+1) 在 host 上先于 AG(U-1) 提交。
3. 同一条 stream 上 kernel 按提交顺序串行执行。AG(U-1) 必须等 RS(U+1) 跑完才能开始。RS 是 fp32 梯度，AG 是 bf16/fp8 参数，同一个 unit 的 RS 字节数是 AG 的 2–4 倍。

## 2. 参考：Megatron-FSDP 的做法

本地代码 `Megatron-LM/megatron/core/distributed/fsdp/src/` 里有两处相关设计：

- **AG 和 RS 用不同的 stream。** AG 走 `side_stream_for_param_gather`，RS 走单独的梯度 stream。
- **AG 可以用独立的 communicator。** `parallel_state.create_all_gather_groups()`（`megatron/core/parallel_state.py:1743`，由 `--create-all-gather-group` 打开，默认关闭）在同一批 rank 上再建一个 group（`DATA_PARALLEL_GROUP_WITH_CP_AG`），专门给 AG 用。

更完整的逐项对照（包括反向提交顺序、RS 精度、NCCL 选项）见 §6.1。

`docs/runtime_schedule.md` 第 41–46 行解释了为什么只拆 stream 不够：同一个 NCCL communicator 上的操作按 host 提交顺序执行，即使它们在不同的 CUDA stream 上。复现脚本是 `docs/scripts/nccl_same_pg_two_streams_{sync,async}.py`。

同一文档第 93–108 行也写了反方向的代价：单条通信 stream 可以减少 allocator 碎片，也让通信顺序更确定；拆开后 AG 和 RS 可以重叠，但"收益取决于负载和系统，需要实测"。

本仓库已有的 `megatron_fsdp_overlap_prefetch.md` §2.2 对此有更长的总结。

## 3. 实现

改动只在 `Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`（`git diff` 为 +58/−7 行），A/B 期间同步拷到 `Paddle/build/python/...`（训练时 `PYTHONPATH` 用的是 build 目录）。`auto_parallel/fully_shard_fusion.py` 有自己的 `TensorFusionBuffer`，不受影响。完整 diff 见文末附录。

11-04 的 A/B 结束后代码曾还原到 HEAD；11-05 重新打上补丁，并加了 `FSDP_AG_BEFORE_RS` 开关（§5.1）。11-05 实验结束后已放弃该优化，代码再次还原（§9）。

### 3.1 三档开关

环境变量 `FSDP_AG_RS_MODE`：

| 取值 | AG stream | RS stream | AG communicator |
|---|---|---|---|
| `shared` | 每个 fsdp group 一条，与 RS 共用 | 同左 | fsdp group |
| `split_stream` | 每个 fsdp group 一条 AG 专用 | 每个 fsdp group 一条 RS 专用 | fsdp group（与 RS 同一个） |
| `split_comm` | 同上 | 同上 | fsdp group 的复制品（同一批 rank） |

`shared` 与改动前的行为完全一致，也是补丁里的默认值（不设环境变量时走 `shared`）。

### 3.2 代码改动

- `TensorFusionBuffer.__init__` 新增参数 `comm_group=None`，保存为 `self.comm_group = comm_group or fsdp_group`。`do_all_gather` 改用 `self.comm_group`；RS 仍用 `self.fsdp_group`，不变。
- `FSDPBufferManager._comm_stream(group, kind)`：`shared` 下 key 是 `group.id`（与原来一样）；另外两档 key 是 `(group.id, "ag" | "rs")`，AG 和 RS 各拿一条 stream。params buffer 传 `"ag"`，grads buffer 传 `"rs"`。
- `split_comm` 下，`FSDPBufferManager._create_ag_groups` 在建 buffer 前为每个 nranks > 1 的 fsdp group 建一个同 rank 的复制 group，params buffer 用它发 AG。

`_create_ag_groups` 有两个约束，都写在代码注释里：

1. **所有 rank 必须以相同顺序建全部复制 group，包括自己不在的。** `paddle.distributed.new_group` 每次调用都会从全局计数器取一个 ring id，各 rank 调用次数或顺序不一致，ring id 就对不上。做法是先 `all_gather_object` 收集各 rank 本地的 group rank 列表，取并集、排序后统一建。
2. **只对 rank 列表本身有序的 group 成立。** `new_group` 会把 ranks 排序，而 AG 的输出按 group 内 rank 顺序拼接。原 group 若 ranks 无序，复制 group 的拼接顺序就和原来不一样。代码里对此加了 assert，触发时提示改用 `split_stream`。本次 T1/T2/T3 的 fsdp group 都是有序的。

### 3.3 启动脚本

`script/train_gpu.sh:33-35` 会把名字里含 `PADDLE` 或 `ENDPOINT` 的环境变量全部 unset。开关最早叫 `PADDLE_FSDP_AG_RS_MODE`，传不到训练进程：第一次标成 `shared` 的 T1 采集实际跑成了默认档（nsys 里 AG 在 s50、RS 在 s51），已作废。后来改名为 `FSDP_AG_RS_MODE`，并且每次采集都用 `ab_analysis/steps.py` 打印的 AG/RS stream 号核对实际档位，不依赖环境变量本身。

`run_fsdp_test.sh` 把 `FSDP_AG_RS_MODE` 写进 env 文件传给各节点；`timeline/agrs/run_ab.sh "T1:shared T3:split_comm ..."` 按顺序逐个采集。

## 4. A/B 采集

### 4.1 条件

- 单机 8 卡 H 系列（CC 9.0），`run_fsdp_test.sh`，`max_steps=10`，nsys 用 `cudaProfilerApi` 只抓 step 6–9。
- 每条 trace 取 3 个完整步：步边界 = dev0 上 s33 每簇 AllReduce_f32 的起点，丢掉采集开始时的半步（`ab_analysis/steps.py`）。步时取 3 步中位数。
- 每次只改 `FSDP_AG_RS_MODE` 一个变量，配置、数据、步数相同。
- 档位以 trace 里的 stream 号为准：`shared` 下 AG 和 RS 都在 s46；`split_stream` 下 AG 在 s46、RS 在 s47；`split_comm` 下 AG 在 s50、RS 在 s51。
- "base" 是改代码之前采的 baseline（`ernielite-T*-fsdp`）。它和新代码的 `shared` 行为相同，两者的差就是同配置重复跑的噪声。

### 4.2 步时

| tier | base | shared | split_stream | split_comm | 同配置噪声 \|base − shared\| |
|---|---|---|---|---|---|
| T1 | 1580.0 | 1581.6 | 1591.2 | 1589.3 | 1.6 |
| T2 | 5570.0 | 5599.8 | — | 5584.4 | 29.8 |
| T3 | 12936.1 | 12869.7 | 12909.1 | 12873.2 | 66.4 |

单位 ms，3 步中位数。三步明细：

- T1：base [1573.1, 1580.0, 1582.0]，shared [1599.3, 1581.6, 1580.4]，split_stream [1591.2, 1595.2, 1582.9]，split_comm [1575.0, 1589.3, 1593.3]
- T2：base [5568.6, 5570.0, 5598.8]，shared [5599.8, 5562.2, 5610.7]，split_comm [5560.7, 5584.4, 5605.3]
- T3：base [12879.2, 12956.2, 12936.1]，shared [12869.7, 12914.2, 12856.9]，split_stream [12858.6, 12928.3, 12909.1]，split_comm [12873.2, 12927.8, 12845.0]

相对同代码的 `shared`，`split_comm` 的差值是 T1 +7.7、T2 −15.4、T3 +3.5 ms，都小于同配置噪声（T2 29.8、T3 66.4）。每档只有一对重复，判别门槛没法严格算出来，但已经能确定：**三个 tier 上都没有测出步时收益**。

### 4.3 机制：依赖拆掉了，计算等 AG 的时间没变

T3（8 卡均值，ms/步）：

| 指标 | shared | split_stream | split_comm |
|---|---|---|---|
| AG busy | 175.2 | 178.1 | 361.2 |
| RS busy | 252.7 | 252.1 | 263.2 |
| AG∩RS（同时在跑） | 0 | 0 | 224.1 |
| 同 stream 上紧跟 RS 开始的 AG（次/步） | 215.9 | 0 | 0 |
| 任意 stream 上 RS 结束后 20 us 内开始的 AG（次/步） | 235.2 | 221.4 | 45.1 |
| 计算流等 AG | 95.1 | 99.1 | 108.5 |
| 计算流等 RS | 5.8 | 4.9 | 5.1 |
| FSDP 暴露通信 | 261.3 | 266.7 | 246.1 |

T2：

| 指标 | shared | split_comm |
|---|---|---|
| AG busy / RS busy | 71.9 / 93.7 | 119.3 / 96.2 |
| AG∩RS | 0 | 56.5 |
| 任意 stream 上紧跟 RS 开始的 AG（次/步） | 83.0 | 1.8 |
| 计算流等 AG | 48.0 | 53.2 |
| FSDP 暴露通信 | 122.4 | 117.8 |

T1：

| 指标 | shared | split_stream | split_comm |
|---|---|---|---|
| AG busy / RS busy | 31.3 / 17.3 | 32.3 / 17.4 | 38.4 / 18.4 |
| AG∩RS | 0 | 0 | 6.0 |
| 任意 stream 上紧跟 RS 开始的 AG（次/步） | 2.0 | 2.0 | 0 |
| 计算流等 AG | 32.3 | 34.5 | 31.1 |
| FSDP 暴露通信 | 43.3 | 44.4 | 45.3 |

（"计算流等 AG"= stall_unblocker 里 AG 本身与"AG 排在 RS 后"两项之和。）

从三张表可以读出三件事：

1. **只拆 stream 没用，和 Megatron 文档说的一致。** T3 `split_stream` 下 AG 和 RS 已在不同 stream，但 AG∩RS 仍为 0，每步仍有 221.4 次 AG 在某个 RS 结束后 20 us 内才开始（`shared` 是 235.2 次）。同一个 NCCL communicator 仍然按提交顺序把 AG 排在 RS 后面，计算等 AG 的时间（99.1）和 `shared`（95.1）一样。
2. **拆 communicator 后 AG 确实不再等 RS。** T3 `split_comm` 下 AG 和 RS 每步重叠 224.1 ms，紧跟 RS 开始的 AG 从 235.2 次降到 45.1 次。
3. **但和 RS 并发的 AG 变慢了，计算等 AG 的时间没有减少。** 按提交顺序把 `split_comm` 的第 k 个 AG 和 `shared` 的第 k 个 AG 配对（`ab_analysis/ag_concurrent_slowdown.py`，8 卡 × 3 步）：

| tier | 与 RS 重叠 > 50% 的 AG 个数 | shared 下这些 AG 总时长 | split_comm 下总时长 | 总时长比 | 单个比值中位数 | 不与 RS 重叠的 AG 总时长比 |
|---|---|---|---|---|---|---|
| T1 | 42 | 26.2 ms | 157.7 ms | 6.01 | 5.98 | 1.06 |
| T2 | 4035 | 152.0 ms | 1302.3 ms | 8.57 | 2.42 | 0.97 |
| T3 | 14256 | 899.1 ms | 5334.4 ms | 5.93 | 2.19 | 0.99 |

不和 RS 重叠的 AG 耗时不变（0.97–1.06）；和 RS 并发的 AG 耗时变成原来的 2–6 倍。AG 提前开始了，但和 RS 分享同一组 NVLink 链路，结束得并不更早。T3 计算流等 AG 的时间从 95.1 变成 108.5 ms，T2 从 48.0 变成 53.2 ms。省下的排队时间被并发变慢吃掉了。

### 4.4 数值与显存

- loss（rank 0，step 1–10）：所有档位 step 1–2 逐位一致；从 step 3 起各次运行之间相差 1e-5 量级，`base` 与 `shared`（行为相同）之间的差也是这个量级（T3 step 10：12.1517601 vs 12.15174866）。说明这份配置本身跨运行不逐位确定，拆流没有引入额外的数值差。AG 只搬数据不做计算，换 communicator 不改变结果。
- max_mem_reserved：T1 71.62–71.94 GB，T2 67.79–68.69 GB，T3 67.85–67.97 GB，各档位之间没有一致方向的差。Megatron 文档提到的"多一条 stream 增加 allocator 碎片"在这里没有测出来。

明细：`timeline/agrs/loss_mem_summary.txt`。

## 5. 追加实验（2026-11-05）：调换提交顺序，叠加拆流，两机 T4

### 5.1 改动：`FSDP_AG_BEFORE_RS`

§1.2 的第 2 点是 RS 挡住 AG 的直接原因：反向进入 unit U 时，`all_gather_params(is_backward=True)` 先调 `_flush_expert_grads_after_unit(U)`，把所有 unit > U 的专家梯度 RS 一次发出，然后才等 AG(U)、发 AG 预取。

开关 `FSDP_AG_BEFORE_RS=1` 把这次 flush 挪到函数末尾，也就是 AG 预取之后（`fully_shard_fusion.py:1034-1049, 1145-1146`）：

```python
flush_unit_id = None
if is_backward:
    ...
    if self._ag_before_rs:
        flush_unit_id = unit_id          # 先记下，不发
    else:
        self._flush_expert_grads_after_unit(unit_id)   # 原行为
    self._release_params_after_unit(unit_id)
...  # 等 AG(U)，发 AG 预取
if flush_unit_id is not None:
    self._flush_expert_grads_after_unit(flush_unit_id)
```

默认 `0`，与原行为相同。这个改动不需要第二个 communicator，AG 和 RS 仍在同一条 stream、同一个 communicator 上，只是 AG 预取在 host 上先提交。

### 5.2 条件

- 单机 T1/T2/T3：`run_fsdp_test_1node.sh`，`script/train_gpu.sh` 里 `LAUNCH_CMD` 用 `range(0,1)`。
- 两机 T4：`run_fsdp_test_2node.sh`，`range(0,2)`，16 卡。T4 有两份 yaml：
  - **T4**：`..._hzz1_fsdp_T5d.yaml`，mb2 / ga24。
  - **T4b**：`..._hzz1_fsdp.yaml`，mb3 / ga16。这份 yaml 显存峰值约 73 GB。
- `max_steps=20`，nsys 抓 step 6–9，取 3 个完整步，步时取中位数。分析脚本与 §4 相同，两机只分析 node 0 的 8 卡。
- 采集脚本 `timeline/agrs/run_order_ab.sh "T1:0 T1:1 T3:1:split_comm ..."`，每项是 `tier:FSDP_AG_BEFORE_RS[:FSDP_AG_RS_MODE]`。
- 拆流的两档（`split_stream`、`split_comm`）都叠在 `FSDP_AG_BEFORE_RS=1` 上。
- 档位仍以 trace 里的 stream 号核对：
  - 单机：shared 下 AG 和 RS 都在 s46；split_stream 下 AG s46、RS s47；split_comm 下 AG s50、RS s51。
  - T4：有两个 fsdp group，shared 下 AG 和 RS 都在 s50 和 s51；split_stream 下 AG s50/s52、RS s51/s53；split_comm 下 AG s58/s60、RS s59/s61。

下文 `0` 表示原顺序，`1` 表示 AG 预取先于 RS。

### 5.3 步时

| tier | 0（原顺序） | 1（AG 先发） | 1 + split_stream | 1 + split_comm |
|---|---|---|---|---|
| T1 | 1573.5 | 1582.8（+0.6%） | 1583.5（+0.6%） | 1573.3（0.0%） |
| T2 | 5554.2 | 5565.2（+0.2%） | 5560.7（+0.1%） | 5600.6（+0.8%） |
| T3 | 12890.5 | 12894.5（0.0%） | 12884.8（0.0%） | 12872.2（−0.1%） |
| T4b（mb3） | 16954.4 | 16916.4（−0.2%） | 16913.7（−0.2%） | 两次都崩（§5.5） |
| T4（mb2） | 19556.1 | 19435.8（−0.6%） | 未测 | 19848.1（+1.5%） |

单位 ms，3 步中位数，括号内是相对 `0` 的变化。三步明细：

- T1：0 [1592.0, 1573.5, 1571.2]，1 [1571.5, 1582.8, 1588.5]，1+split_stream [1583.5, 1595.3, 1578.5]
- T2：0 [5554.2, 5547.4, 5583.3]，1 [5551.1, 5565.2, 5600.9]
- T3：0 [12866.6, 12918.7, 12890.5]，1 [12958.9, 12894.5, 12844.8]
- T4b：0 [16917.2, 16977.4, 16954.4]，1 [16916.4, 16872.4, 16957.5]
- T4：0 [19600.4, 19556.1, 19533.0]，1 [19435.8, 19459.3, 19419.9]，1+split_comm [19848.1, 19678.4, 19899.1]

单机三个 tier 的差都在 §4.2 的同配置噪声以内（T2 29.8 ms，T3 66.4 ms），T2 split_comm 的 +46 ms 略超。

两机有两个值得单看的数：

- **T4 `1` 的 −0.6%（−120 ms）不能算到调换顺序头上。** 按阶段拆开，前向从 6754 降到 6660 ms（−94 ms），反向只从 12239 降到 12208 ms（−31 ms）。开关只改反向的提交顺序，前向的差只能是两机运行之间的波动。同一改动在 T4b 上是 −0.2%，反向 11088 → 11042 ms。
- **T4 split_comm 的 +1.5% 是稳定变慢。** 它的 3 步都比 `1` 的 3 步慢（相对 `1` 是 +2.1%，+412 ms），见 §5.5。

### 5.4 机制：排队没了，反向暴露的 FSDP 通信没变

列含义（node 0 的 8 卡均值，ms/步）：

- **AG / RS**：busy 时间。
- **AG∩RS**：两者同时在跑的时间。
- **排队**：每步紧跟 RS 开始的 AG 次数（定义同 §1.1）。
- **等 AG / 等 RS**：stall_unblocker 算出的计算流等待时间，"等 AG"含"AG 排在 RS 后"一项。
- **FSDP 暴露**：FSDP 通信没被计算盖住的时间，括号内是反向部分。

| run | AG | RS | AG∩RS | 排队 | 等 AG | 等 RS | FSDP 暴露（反向） |
|---|---|---|---|---|---|---|---|
| T1-0 | 32.7 | 17.5 | 0 | 2.0 | 29.9 | 3.3 | 45.1（20.4） |
| T1-1 | 34.8 | 12.1 | 0 | 0 | 24.0 | 4.9 | 41.4（16.1） |
| T1-1+split_stream | 36.9 | 12.9 | 0 | 0 | 26.6 | 4.9 | 43.9（17.4） |
| T1-1+split_comm | 32.0 | 18.0 | 6.4 | 0 | 21.3 | 3.6 | 38.5（16.7） |
| T2-0 | 64.1 | 92.3 | 0 | 62.8 | 45.4 | 3.7 | 113.7（82.8） |
| T2-1 | 123.5 | 46.4 | 0 | 0 | 57.8 | 3.9 | 122.3（79.7） |
| T2-1+split_stream | 115.2 | 46.0 | 0 | 0 | 48.3 | 3.6 | 113.8（81.3） |
| T2-1+split_comm | 266.5 | 95.7 | 57.2 | 0 | 198.2 | 3.8 | 261.7（78.2） |
| T3-0 | 187.0 | 252.2 | 0 | 215.6 | 99.5 | 5.6 | 270.1（222.0） |
| T3-1 | 402.1 | 95.8 | 0 | 0 | 131.8 | 90.5 | 308.4（227.4） |
| T3-1+split_stream | 364.7 | 94.2 | 0 | 0 | 93.8 | 82.0 | 272.7（226.9） |
| T3-1+split_comm | 384.9 | 225.0 | 190.4 | 0 | 107.5 | 5.0 | 265.8（208.0） |
| T4b-0 | 2078.1 | 4162.2 | 352.0 | 152.8 | 103.9 | 178.9 | 1049.8（729.3） |
| T4b-1 | 2440.7 | 4190.5 | 600.8 | 0 | 165.6 | 90.9 | 999.6（708.2） |
| T4b-1+split_stream | 2478.3 | 4163.1 | 602.0 | 0 | 149.6 | 90.4 | 992.4（697.3） |
| T4-0 | 2829.7 | 6586.7 | 634.5 | 258.6 | 243.1 | 151.2 | 2093.6（1142.4） |
| T4-1 | 3545.4 | 6735.6 | 760.1 | 0 | 198.5 | 83.6 | 2119.7（1143.2） |
| T4-1+split_comm | 3334.8 | 6293.0 | 644.6 | 0 | 319.6 | 96.1 | 2178.0（1078.1） |

T4/T4b 有两个 fsdp group（两条 stream、两个 communicator），所以 shared 档下 AG∩RS 也不为 0：一个 group 的 AG 可以和另一个 group 的 RS 并发。

从表里能读出：

1. **调换顺序确实消掉了排队。** 所有 tier 上"紧跟 RS 开始的 AG"都降到 0（T3 215.6 → 0，T4 258.6 → 0）。
2. **但反向的 FSDP 暴露通信基本不变。** `0` → `1`：T1 20.4 → 16.1，T2 82.8 → 79.7，T3 222.0 → 227.4，T4b 729.3 → 708.2，T4 1142.4 → 1143.2。同一个 communicator 上的通信总量没变，只是先后换了。原来 AG 等 RS，现在 RS 晚发、晚结束，计算流在 RS 队列满时等 RS（T3 等 RS 从 5.6 涨到 90.5，T4b 的等 AG 从 103.9 涨到 165.6）。
3. **AG / RS busy 在 T2、T3 上此消彼长**（T3 AG 187 → 402，RS 252 → 96）。推测原因是 NCCL kernel 的时长包含等其他 rank 到齐的时间，同一 communicator 上先跑的那个吸收了 rank 间的偏差。之前 RS 先跑，现在 AG 先跑。这一点没有单独验证。
4. **split_stream 和单纯调换没有区别。** 两者 AG∩RS 都是 0（T1–T3），步时和暴露通信都在噪声内。原因和 §4.3 一样，同一个 communicator 按提交顺序串行。
5. **split_comm 让 AG 和 RS 真正并发，但 AG 变慢。** T3 AG∩RS 190.4 ms，等 RS 回到 5.0，但 FSDP 暴露（265.8）和 `0`（270.1）差不多。T2 的等 AG 从 45.4 涨到 198.2，步时 +0.8%，与 §4.3 的"和 RS 并发的 AG 慢 2–6 倍"一致。

### 5.5 两机 T4 split_comm

**T4（mb2）跑完了，比 `1` 慢 2.1%，比 `0` 慢 1.5%。** 慢主要在前向：

| run | 前向 ms | 反向 ms | 前向 EP 暴露 | 前向 FSDP 暴露 |
|---|---|---|---|---|
| T4-0 | 6754.1 | 12239.2 | 575.6 | 911.8 |
| T4-1 | 6660.5 | 12208.3 | 511.6 | 940.3 |
| T4-1+split_comm | 7018.7 | 12209.9 | 705.0 | 1054.7 |

- 反向时间与 `1` 相同（12209.9 vs 12208.3），反向 FSDP 暴露反而少了 65 ms（1078.1 vs 1143.2），但没有转化成反向时间的缩短。
- 前向多了 358 ms：DeepEP 暴露 +193 ms（511.6 → 705.0），FSDP 暴露 +114 ms（940.3 → 1054.7）。"无通信等待"（host / launch）一项也从 1747.4 涨到 1974.5 ms。前向里会逐层收尾上一步残留的 RS（`fully_shard_fusion.py:1052-1056`），split_comm 下这些 RS 和前向的 AG 预取、DeepEP 的 dispatch/combine 同时在跑。推测是三者争抢 IB 和 NVLink 带宽，没有进一步拆分验证。

**T4b（mb3）split_comm 两次都在 step 1–2 崩溃。** 两次的现象相同：node 2 上多个 rank 报 `DeepEP timeout for dispatch senders`，随后在 `GradNodePyLayer_FusionMoePyLayer` 的反向里报 CUDA 719。第二次重跑前残留进程占着 GPU（79 GB / 100%），清理后再跑仍然失败。同一份 yaml 的 `0`、`1`、`1+split_stream` 都正常跑完，T4（mb2）的 split_comm 也正常跑完。推测是 mb3 显存峰值约 73 GB，多一个 communicator 的 NCCL buffer 再加上 DeepEP，触发了超时。这一点没有定位。

### 5.6 数值与显存

- loss：每组 `0` / `1` / 拆流之间相差 1e-5 到 1e-6，与 §4.4 的跨运行噪声相同。T4 step 20：12.0508337 / 12.05083847 / 12.05084229（0 / 1 / 1+split_comm）；T4b step 20：12.05090427 / 12.05090618 / 12.05090332（0 / 1 / 1+split_stream）。
- max_mem_reserved（GB，0 / 1）：T1 71.90 / 71.75，T2 67.79 / 67.63，T3 68.79 / 68.79，T4 60.46 / 60.50（split_comm 60.64），T4b 73.28 / 73.04（split_stream 72.90）。调换顺序没有抬高显存峰值。11-04 后续方向里担心的"AG 预取提前导致峰值上升"没有出现。

## 6. 与 Megatron-FSDP 实现对比：是不是实现不同导致没有收益

路径相对 `Megatron-LM/`。`MF` = `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`，`PGB` = 同目录 `param_and_grad_buffer.py`，`RS_DOC` = `megatron/core/distributed/fsdp/src/docs/runtime_schedule.md`。

### 6.1 逐项对照

| 项目 | Megatron-FSDP 默认 | Megatron 可选 | Paddle 本次测过的 |
|---|---|---|---|
| AG / RS stream | 两条全局 side stream：`side_stream_for_param_gather` 和 `side_stream_for_buffer_copy_and_grad_accum`（MF:449-450）。不按 group 建 | v2 `overlap_allgather_reduction=False` 合成一条，用来降峰值显存（`experimental/runtime_policy.py:20-41`） | shared（每 group 一条，AG/RS 共用）、split_stream |
| AG communicator | 与 RS 共用 dp_cp group | `--create-all-gather-group`，**默认关**（`megatron/training/arguments.py:3110-3112`）。建 `DATA_PARALLEL_GROUP_WITH_CP_AG` 和 `EXPERT_DATA_PARALLEL_GROUP_AG`（`megatron/core/parallel_state.py:1743`），只给 model weight buffer 发 AG 用（PGB:2718-2722）。v2 不用它 | split_comm |
| 反向提交顺序 | 进入 module i 时先发 AG(i) 和 i-1… 的预取，再等 AG(i)（MF:508-569, 980-994）。RS(i) 由 post-accumulate-grad hook 在 module i 算完后发（MF:1205, PGB:4115）。host 上是 AG(i-1) → RS(i) | — | 原顺序是 RS(>U) → 等 AG(U) → AG 预取；`FSDP_AG_BEFORE_RS=1` 改成 AG 预取 → RS |
| 预取 / RS 队列大小 | 通信单元默认 = 2 × 平均 FSDP unit 元素数，上限 1e9；AG 预取 ≈ 1 个 unit，RS 队列 ≈ 2 个 unit（MF:463-487） | `suggested_communication_unit_size` | 按 size 预取（`_prefetch_plan`），未改 |
| RS 精度 | `--megatron-fsdp-grad-comm-dtype auto` = main grad dtype，main grad 默认 = 模型梯度 dtype（`arguments.py:3615-3623`），bf16 模型即 bf16 RS | fp32 | fp32（`grad_comm_dtype` 未设） |
| 通信 SM / 带宽 | 无特殊设置 | `--use-nccl-ub`：NCCL user buffer + symmetric 注册，AG group 也注册（`arguments.py:3094`，PGB:2124-2169）；按 PG 设 `max_ctas` / `min_ctas` / `cga_cluster_size` / 高优先级 stream（`parallel_state.py:172`）；Blackwell 上 dp_cp、ep_dp 自动高优先级（`arguments.py:1354-1361`） | 无 |

### 6.2 这些差异能不能解释没有收益

**反向提交顺序：不能。** Megatron 的顺序就是 AG 预取先于 RS。`RS_DOC:38-45` 写明"反向里 AG(i-1) 要在 RS(i) 之前预取，否则共用 communicator 时 AG(i-1) 会排在 RS(i) 后面"，这正是 Paddle 原实现的问题。`FSDP_AG_BEFORE_RS=1` 把 Paddle 改成和 Megatron 一样的顺序，排队次数降到 0（§5.4），但步时不变（§5.3）。

**stream 和 communicator：不能。** Megatron 默认就是"两条 stream + 同一个 communicator"，相当于本文的 `1 + split_stream`。拆 communicator 在 Megatron 里也是默认关闭的可选项，v2 实现里干脆没用。Megatron 文档对它的收益只有一句"Anecdotal measurements on some workloads have shown performance gains… workload- and system-dependent and should be measured"（`RS_DOC:105-108`），没有给数字。本文测了 Megatron 默认和可选的两种组合：

- `1 + split_stream`（= Megatron 默认）：T1–T3、T4b 与 `0` 的差都在噪声内。
- `1 + split_comm`（= Megatron 打开 `--create-all-gather-group`）：单机持平或 T2 +0.8%，两机 T4 +1.5%，T4b 崩溃。

也就是说，Paddle 改成 Megatron 的做法以后，结果仍然是没有收益。没有收益不是因为实现和 Megatron 不同。

### 6.3 没有收益的实际原因

1. **排队只是换了位置，通信总量没变。** 同一个 communicator 上 AG 和 RS 总是串行，调换顺序只改变计算流等谁。反向 FSDP 暴露在 `0` 和 `1` 之间几乎相同（T3 222.0 / 227.4，T4 1142.4 / 1143.2 ms）。§1.1 估算的"AG 排在 RS 后"上界（T3 46.6 ms，0.36%）并没有变成步时收益，因为 RS 晚发后由计算流在 RS 队列上等回来了。
2. **并发的代价抵掉了并发的好处。** 拆 communicator 后 AG 和 RS 走同一组 NVLink（单机）或同一组 IB 网卡（两机），并发时和 RS 重叠过半的 AG 慢 2–6 倍（§4.3）。两机上还多了 DeepEP 争带宽（§5.5）。
3. **能抠的空间本来就小。** 反向 FSDP 暴露占步时：T1 1.3%，T2 1.5%，T3 1.7%，T4b 4.3%，T4 5.8%。T1–T3 主要的暴露通信是 DeepEP 的 dispatch/combine（每步暴露 143–1195 ms，占步时 9–10%），不是 FSDP。
4. **T4 的大头是 fp32 RS 的量。** T4 第二个 fsdp group 的 RS（s51）每步 busy 6577.5 ms、暴露 1152.3 ms，是 FSDP 里最大的一项（busy 里含等其他 rank 的时间，不全是传输）。调换顺序和拆流都不减少字节数。

所以，要降 FSDP 暴露通信，应该减少字节数或提高每字节的效率，而不是继续调整 AG 和 RS 的先后与并发。Megatron 有、Paddle 这次没测的几项都属于这一类（§8）。

## 7. 结论

1. **调换反向 AG/RS 提交顺序（`FSDP_AG_BEFORE_RS=1`）：没有步时收益。** 排队降到 0，但反向 FSDP 暴露不变。单机 T1–T3 的差在噪声内（+0.6% / +0.2% / 0.0%），两机 T4b −0.2%。T4 的 −0.6% 来自前向，开关不改前向，不能算作收益（§5.3）。
2. **只拆 stream（`split_stream`）：没有收益。** 不论在原顺序上（§4）还是在调换后的顺序上（§5），AG∩RS 都是 0。同一个 NCCL communicator 按 host 提交顺序执行，和 Megatron `RS_DOC:41-46` 的说法一致。
3. **拆 communicator（`split_comm`）：没有收益，两机变慢。** 单机 T1–T3 持平到 +0.8%。两机 T4（mb2）+1.5%，前向 DeepEP 和 FSDP 暴露各多 100–200 ms。T4b（mb3）两次在 step 1–2 崩溃（DeepEP timeout + CUDA 719）。
4. **Paddle 改成和 Megatron 一样的做法后仍然没有收益（§6.2）。** Megatron 默认就是 AG 预取先于 RS、两条 stream、同一个 communicator，对应本文的 `1 + split_stream`。`--create-all-gather-group` 默认关闭，文档里只说收益因负载而异，没有数字。没有收益不是因为实现不同。数据显示，调整顺序和并发都没有减少反向的 FSDP 暴露通信，所以推测它主要由通信量和链路带宽决定（§6.3）。这一点还没有用减少字节数的实验直接验证（§8 第一条）。
5. **数值与显存不受影响。** loss 的差在跨运行噪声以内，max_mem_reserved 没有一致方向的变化（§4.4、§5.6）。

两个开关都不建议默认打开。`FSDP_AG_BEFORE_RS=1` 和 Megatron 的顺序一致、代码改动小、没有副作用，如果想对齐 Megatron 的调度可以保留为可选项。`split_comm` 在两机上有害，还会让 T4b 崩溃，不建议保留。

这些结论只对本机 H 系列、单机 8 卡 T1/T2/T3 和两机 16 卡 T4/T4b 这几份 ERNIE-5.5-Flash 配置成立。两机只分析了 node 0。

## 8. 后续方向

11-04 版列的三条里，"调换反向提交顺序"和"多机再测 split_comm"已在 §5 测过，都没有收益。下面几条都**没有实现或测过**，按 §6.3 的判断排序：

- **RS 改用 bf16 通信。** Megatron 默认按模型梯度 dtype 做 RS（§6.1），Paddle 这几份配置用的是 fp32。`FSDPBufferManager` 已有 `grad_comm_dtype` 参数。这样字节数减半，T4 上 RS 占比最大，可能收益最多。需要单独验证 loss 是否受影响。
- **NCCL user buffer / symmetric memory 注册。** 对应 Megatron `--use-nccl-ub`。README 说它启用 SM 占用更少的 NCCL 算法，FSDP 通信同时跨 NVLink 和 IB 时配合 SHARP 更有效，代价是要开 double buffer、多占显存（`megatron/core/distributed/fsdp/src/README.md:142, 164-165`）。需要先查 Paddle 的 NCCL 封装是否支持 `ncclCommRegister` / `ncclMemAlloc`。
- **按 communicator 限制 CTA 数或设高优先级 stream。** 对应 Megatron 的 `get_nccl_options`（`max_ctas`、`min_ctas`、`is_high_priority_stream`）。11-04 版提的"限制 RS 带宽占用"属于这一条。Paddle 的 `new_group` 是否暴露 `ncclConfig_t` 还没查。
- **T4b split_comm 崩溃的原因。** 只有在需要保留 `split_comm` 时才值得查。目前不建议保留，优先级最低。

## 9. 复现与还原

**2026-11-05 已放弃该优化，代码已还原。** `Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py` 已 `git checkout` 回 HEAD `abadcba928`，build 目录副本同步，与源码逐字节相同。`run_fsdp_test_{1node,2node}.sh` 里为本实验加的 env 传递、`TAG` / `CONFIG` 覆盖也已去掉。下面的两个开关现在都不存在，要复现需先打 patch。

两个开关的完整 diff 存在 `timeline/agrs/fsdp_ag_before_rs_and_mode.patch`。

| 环境变量 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `FSDP_AG_BEFORE_RS` | `0` / `1` | `0` | `1`：反向里先发 AG 预取，再发专家梯度 RS（§5.1） |
| `FSDP_AG_RS_MODE` | `shared` / `split_stream` / `split_comm` | `shared` | AG/RS 的 stream 与 communicator（§3.1） |

- 两个默认值都等于原行为。变量名不能带 `PADDLE`，否则会被 `script/train_gpu.sh` 清掉（§3.3）。启动脚本需要把两个变量写进 env 文件（`declare -p ... FSDP_AG_BEFORE_RS FSDP_AG_RS_MODE`）并在各节点 export，才能传到所有节点。用 `run_order_ab.sh` 时还要让脚本支持 `TAG` / `CONFIG` 覆盖。这些改动已还原，复现时需重新加。
- 批量采集：`bash timeline/agrs/run_order_ab.sh "T1:0 T1:1 T3:1:split_comm T4:1"`。脚本按 tier 改 `script/train_gpu.sh` 里的 `range(0,1)` / `range(0,2)`，并选对应的启动脚本。
- 判断实际生效档位：看 nsys 里 AG 和 RS 的 stream 号（`ab_analysis/steps.py` 会打印），对照 §5.2。
- 在干净的 Paddle checkout 上复现：`git -C Paddle apply ../timeline/agrs/fsdp_ag_before_rs_and_mode.patch`，再拷到 build 目录。
- 还原：`git -C Paddle checkout -- python/paddle/distributed/fsdp/fully_shard_fusion.py`，再用 `/bin/cp -f` 把源码文件拷到 build 目录（本机的 `cp` 带 `-i` 别名）。

## 10. 证据文件

路径相对 `erniebot/timeline/`，除非写了绝对路径。

**nsys sqlite**（每个文件对应一次采集；同名 `.nsys-rep` 全部在 `timeline/` 下，包括 sqlite 放在 `/root/agrs_sqlite/` 的那几次，可以从 rep 重新导出）：

| tier | base | shared | split_stream | split_comm |
|---|---|---|---|---|
| T1 | `ernielite-T1-fsdp_n0.sqlite` | `/root/agrs_sqlite/agrs-T1-shared_n0.sqlite` | `/root/agrs_sqlite/agrs-T1-split_stream_n0.sqlite` | `agrs-T1-split_comm_n0.sqlite` |
| T2 | `ernielite-T2-fsdp_n0.sqlite` | `/root/agrs_sqlite/agrs-T2-shared_n0.sqlite` | — | `/root/agrs_sqlite/agrs-T2-split_comm_n0.sqlite` |
| T3 | `ernielite-T3-fsdp_n0.sqlite` | `agrs-T3-shared_n0.sqlite` | `/root/agrs_sqlite/agrs-T3-split_stream_n0.sqlite` | `agrs-T3-split_comm_n0.sqlite` |

`/root/agrs_sqlite/` 在容器本地 overlay 盘上，因为采集时 PFS 已满（97–100%），往 PFS 导 sqlite 报 disk I/O error。容器重建后这些 sqlite 会丢，但可以从 `timeline/` 下的 rep 用 `nsys export` 重新生成。

**分析输出**（`agrs/`）：

- `analysis-<tier>-<mode>.txt`、`analysis-ernielite-T*-fsdp.txt`：`ab_analysis/analyze_all.sh` 的完整输出，§4.2 步时和 §4.3 机制表的来源
- `slowdown-T*-split_comm-vs-shared.txt`：§4.3 并发变慢表的来源。`slowdown-T{1,3}-split_comm.txt`（对比 base）是早期版本，结论方向相同，文中不引用
- `loss_mem_summary.txt`：§4.4
- `<tier>-<mode>.driver.log`、`run_ab.*.log`：每次采集的启动日志

**脚本**（`ab_analysis/`）：`steps.py`（步边界、窗口、AG/RS stream 号）、`ag_rs_concurrency.py`、`stall_unblocker.py`、`ag_concurrent_slowdown.py`、`comm_overlap.py`、`phase_overlap.py`、`loss_mem.py`、`analyze_all.sh`。采集用 `agrs/run_ab.sh`。

**11-05 追加实验（§5）：**

- nsys：`timeline/order-<tier>-<run>_n0.nsys-rep`，sqlite 在 `/root/agrs_sqlite/order-<tier>-<run>_n0.sqlite`。`<run>` 取 `0`、`1`、`1-split_stream`、`1-split_comm`，tier 取 T1、T2、T3、T4、T4b。T4 没有 `1-split_stream`，T4b 没有 `1-split_comm`。两机只导出了 node 0 的 sqlite，node 1 的 rep 是同名 `_n1.nsys-rep`。
- 分析输出：`agrs/analysis-order-<tier>-<run>.txt`，§5.3–§5.5 各表的来源。
- 启动日志：`agrs/<tier>-order<run>.driver.log`；批量驱动日志 `agrs/run_order_ab.log`（T1–T4 的 0/1）、`run_order_ab_T4b.log`、`run_order_ab_split.log`、`run_order_ab_split2.log`、`run_order_ab_T4split.log`。
- 采集脚本：`agrs/run_order_ab.sh`。
- T4b split_comm 的失败日志：`agrs/T4b-order1-split_comm.run1-fail.driver.log`（step 2 崩溃）、`agrs/T4b-order1-split_comm.run2-stuck.driver.log`（残留进程占 GPU）、`agrs/T4b-order1-split_comm.driver.log`（step 1 崩溃）。
- T2 `0` 在 step 17 以 `Pod failed` 结束（rank 3 CUDA 719，7 张卡报 Xid 43）。采集窗口 step 6–9 已完整写出，没有重跑。

**作废的采集，不要引用：**

- `agrs-T1-INVALID-labelled-shared-ran-split_comm_n0.sqlite`：环境变量被 `train_gpu.sh` 清掉，标成 `shared` 实际跑的是当时的默认档 `split_comm`（§3.3）。对应的 `.nsys-rep` 已被后来有效的重跑覆盖。
- `/root/agrs_sqlite/agrs-T1-split_stream-run1-rank0stall_n0.sqlite`：rank 0 主机侧卡住约 1.4 s（3311–4716 ms 区间 dev0 没有 GPU 活动），其他 7 卡的 AG 在等它，单次 AG 长达 1.4 s。与开关无关，已重跑。
- T2 `split_comm` 第一次运行在初始化阶段以 `Exit code -15` 退出，没有产出 trace，原因未定位。重跑成功。

## 附录：代码 diff（11-04 版，仅 `FSDP_AG_RS_MODE`）

基线是 `Paddle` 仓库 HEAD `4017e4b879`，文件 `python/paddle/distributed/fsdp/fully_shard_fusion.py`。同一份 diff 另存为 `timeline/agrs/fsdp_ag_rs_mode.patch`。当前 HEAD `abadcba928` 已经有 `import os`，这份 patch 的第一个 hunk 会冲突。在当前 HEAD 上请用 §9 的 `fsdp_ag_before_rs_and_mode.patch`，它同时包含两个开关。

```diff
diff --git a/python/paddle/distributed/fsdp/fully_shard_fusion.py b/python/paddle/distributed/fsdp/fully_shard_fusion.py
index 521e2fe8cc..2c07a1253d 100644
--- a/python/paddle/distributed/fsdp/fully_shard_fusion.py
+++ b/python/paddle/distributed/fsdp/fully_shard_fusion.py
@@ -14,6 +14,7 @@
 
 import contextlib
 import math
+import os
 import re
 from collections import Counter, OrderedDict
 from dataclasses import dataclass, field
@@ -145,11 +146,16 @@ class TensorFusionBuffer:
         shard_align_numel=None,
         grad_comm_dtype=None,
         comm_stream=None,
+        comm_group=None,
     ):
         # Calculate total buffer size needed (with padding)
         self.unique_key = unique_key
         self.fsdp_group = fsdp_group
         self.comm_stream = comm_stream
+        # Group the stream collective runs on; params buffers may use a
+        # duplicate of fsdp_group so all_gather is not ordered behind
+        # reduce_scatter by the shared NCCL communicator.
+        self.comm_group = comm_group or fsdp_group
         self.fsdp_degree = fsdp_group.nranks
         self.is_sharded = shard_buffer and fsdp_group.nranks > 1
         self.is_params = is_params
@@ -343,7 +349,7 @@ class TensorFusionBuffer:
                 paddle.distributed.stream.all_gather(
                     full,
                     shard,
-                    group=self.fsdp_group,
+                    group=self.comm_group,
                     sync_op=True,
                     use_calc_stream=True,
                 )
@@ -427,6 +433,19 @@ class FSDPBufferManager:
         self.model = model
         self.grad_comm_dtype = grad_comm_dtype
         self._comm_streams = {}
+        # How all_gather (AG) and reduce_scatter (RS) are issued:
+        #   shared:       one stream per fsdp group; RS_i blocks the AG_{i-1}
+        #                 prefetch queued behind it in backward.
+        #   split_stream: AG / RS on separate streams, same communicator.
+        #   split_comm:   separate streams, and AG on a duplicate communicator
+        #                 over the same ranks (Megatron-FSDP
+        #                 --create-all-gather-group), since NCCL serializes ops
+        #                 of one communicator even across streams.
+        self._ag_rs_mode = os.getenv("FSDP_AG_RS_MODE", "shared")
+        assert self._ag_rs_mode in ("shared", "split_stream", "split_comm"), (
+            f"FSDP_AG_RS_MODE={self._ag_rs_mode!r}, expected "
+            "shared / split_stream / split_comm"
+        )
         self.hcg = fleet.get_hybrid_communicate_group()
         self._fsdp_group = self.hcg.get_sharding_parallel_group()
         use_ep = (
@@ -508,6 +527,12 @@ class FSDPBufferManager:
             )
             keyed_params.setdefault(key, []).append(param)
 
+        self._ag_groups = (
+            self._create_ag_groups({key[4] for key in keyed_params})
+            if self._ag_rs_mode == "split_comm"
+            else {}
+        )
+
         def sort_key(item):
             (
                 _,
@@ -594,7 +619,8 @@ class FSDPBufferManager:
                 is_params=True,
                 shard_buffer=shard_buffer,
                 shard_align_numel=shard_align,
-                comm_stream=self._comm_stream(group.fsdp_group),
+                comm_stream=self._comm_stream(group.fsdp_group, "ag"),
+                comm_group=self._ag_groups.get(group.fsdp_group.id),
             )
             if fp8_numel:
                 init_fp8_buffer(group.params_buffer, params)
@@ -616,7 +642,7 @@ class FSDPBufferManager:
                     shard_buffer=grads_shard_buffer,
                     shard_align_numel=shard_align,
                     grad_comm_dtype=self.grad_comm_dtype,
-                    comm_stream=self._comm_stream(group.fsdp_group),
+                    comm_stream=self._comm_stream(group.fsdp_group, "rs"),
                 )
             group.whole_shard_numel = (
                 group.params_buffer.shard_align_numel
@@ -643,12 +669,37 @@ class FSDPBufferManager:
         self._assign_whole_owners()
         self._set_clip_attrs()
 
-    def _comm_stream(self, group):
+    def _comm_stream(self, group, kind):
         if group.nranks <= 1:
             return None
-        if group.id not in self._comm_streams:
-            self._comm_streams[group.id] = paddle.device.Stream()
-        return self._comm_streams[group.id]
+        key = group.id if self._ag_rs_mode == "shared" else (group.id, kind)
+        if key not in self._comm_streams:
+            self._comm_streams[key] = paddle.device.Stream()
+        return self._comm_streams[key]
+
+    @staticmethod
+    def _create_ag_groups(fsdp_groups):
+        # new_group draws a global ring id, so every rank must create every
+        # duplicate group, in the same order, including groups it is not in.
+        local = sorted(
+            {tuple(g.ranks) for g in fsdp_groups if g.nranks > 1}
+        )
+        for ranks in local:
+            # new_group sorts ranks; all_gather output follows group rank
+            # order, so the duplicate is only equivalent for sorted groups.
+            assert list(ranks) == sorted(ranks), (
+                f"fsdp group ranks {ranks} are not sorted; "
+                "use FSDP_AG_RS_MODE=split_stream"
+            )
+        gathered = []
+        paddle.distributed.all_gather_object(gathered, local)
+        dup = {
+            ranks: paddle.distributed.new_group(list(ranks))
+            for ranks in sorted({r for part in gathered for r in part})
+        }
+        return {
+            g.id: dup[tuple(g.ranks)] for g in fsdp_groups if g.nranks > 1
+        }
 
     def _set_clip_attrs(self):
         for group in self.buffer_groups:
```
