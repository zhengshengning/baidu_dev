# FSDP 功能点重新拆解：comm_unit_size（T2）放到最后

`fsdp_perf_opt_breakdown.md` 第二部分按 T0 → T1 → T2 → T3 → T4 的顺序累积，T2（`comm_unit_size`）记了 +15.0%。
但在已经打开 fp8 param gather 的配置上，单独关掉 `comm_unit_size`，吞吐几乎没有变化（见 2026-11-04 的 2338 / 2342 两次运行）。
本文把 T2 挪到累积链的最后重新测一遍，看每个功能点在新顺序下的边际收益。

## 结论

- **`comm_unit_size` 在 T4 之上没有收益**：21430.3 → 21416.4（-0.1%，在噪声内），显存多 0.35 GB。叠在 T5d 上反而 -0.7%（21899.8 → 21749.5）。
- **旧文档里 T2 的 +15% 和 fp8 param gather 解决的是同一段暴露通信**，两者不能相加，谁排在前面就记在谁头上：
  - 旧顺序 T1 → T2 → T3：T2 +15.0%，T3 +3.6%，合计 T1 → T3 +19.1%。
  - 新顺序 T1 → T3：fp8 gather 单独 +19.2%，之后 T2 为 0。
  - 本次复测 T1 + T2（不开 fp8 gather）仍是 +15.3%，与旧文档一致。所以 T2 本身的效果没有变，变的是它和 fp8 gather 的先后。
- **`prefetch_units=2`（T2ref）在 T4 之上同样无收益**（剔除一步异常后 -0.3%）。
- **当前最优是 T5d 不带 `comm_unit_size`**：21899.8 tokens/s/card，相对 T0 +51.1%。旧文档的 T5d（21735.6）带着 `comm_unit_size`，和本次 T5d_T2（21749.5）一致。
- **累积链总收益基本不变**：T0 → T4 为 +47.8%（旧文档 +47.1%）。

建议：`fsdp_comm_unit_size` 和 `fsdp_prefetch_units` 保持默认（不设 / 1），只开 `fsdp_reshard_policy: after_step` + `fsdp_reshard_filter: all`，并把 `grad_comm_dtype=bfloat16` 接进 trainer（目前只能靠测试开关打开，见下文）。

## 测试方法

与旧文档第二部分相同的部分：ERNIELite T4（h1280 / L20，34.93B-A1.232B），2 机 16 卡，`sharding: fsdp`，seq 8192，mb2 / ga24，`max_steps=20`，`internal_medicine_monitor_interval=0`，统计 rank0 `workerlog.0` step 4–20（n=17）。

不同之处：

- **yaml：** `conf/experiment/ERNIE-5.5-Flash/ScalingLadder/ernie5_5_flash_ladder_v90_T4_h1280_L20_34.93BA1.232B-pretrain-8k-hzz1_fsdp_T5d.yaml`，mb2 / ga24 写在 yaml 里，三个 FSDP 参数都注释掉（即默认 `comm_unit_size=None` / `prefetch_units=1` / `after_forward` / `all`）。
- **代码：** 当前 Paddle 源码 HEAD `abadcba928`（已含 #79898 / #79901 / #79897 / #79899 / #79887 / #79903），`Paddle/build` 下的 `fully_shard_fusion.py` 与源码一致（md5 `364750cd…`）。
- **开关分两类：**
  - PR 已有训练参数的（`fsdp_comm_unit_size`、`fsdp_prefetch_units`、`fsdp_reshard_policy`、`fsdp_reshard_filter`）直接用 `--kwargs` 传，已逐阶段核对日志里打印的 TrainingArguments。
  - 没有训练参数的三项，在 `perf_stages3/final/fully_shard_fusion.py` 里加测试开关，从 `perf_stages3/knobs.json` 读取，启动时打印 `[perf_stages3] effective: ...`，已逐阶段核对：

| knobs.json | 对应 PR | 设上时的行为（PR 前的等价路径） |
| --- | --- | --- |
| `legacy_comm: true` | #79898 | `paddle.distributed.stream.all_gather/reduce_scatter(sync_op=False, use_calc_stream=False)`，走 ProcessGroup 自己的通信流 |
| `no_fp8_gather: true` | #79897 / #2371 | `FSDPBufferManager` 初始化时清掉 `_fsdp_fp8_quantizer`，专家权重 bf16 gather，Fleet `FP8QuantWeightCallback` 回退到每步逐层 gather 后量化 |
| `grad_comm_dtype: "bfloat16"` | #79899 | 覆盖 `FSDPBufferManager` 的 `grad_comm_dtype`（trainer 不传 `mp_policy`，没有别的打开方式） |

T0 / T1 与旧文档几乎一致（14498.1 对 14525.4，17597.6 对 17618.4，显存都是 61.46 GB），说明测试开关复现的 PR 前路径与旧测试等价。

## 结果

T0 → T1 → T3 → T4 → T2 为累积叠加；T2ref、T5d、T5d_T2 都建在 T4 上；T1_T2 是对旧顺序的复现。

| 阶段 | 配置 | tokens/s/card | 相对上一阶段 | 相对 T0 | fwd-bwd (ms) | opt-step (ms) | max_mem_alloc (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | 全部关闭 | 14498.1 ± 23 | — | — | 26550 | 362 | 61.46 |
| T1 | + side stream（#79898） | 17597.6 ± 34 | **+21.4%** | +21.4% | 21767 | 359 | 61.46 |
| T3 | + fp8 param gather（#79897 + #2371） | 20970.0 ± 48 | **+19.2%** | +44.6% | 18327 | 351 | 59.03 |
| T4 | + bf16 reduce-scatter（#79899） | 21430.3 ± 57 | +2.2% | +47.8% | 17914 | 352 | 59.03 |
| T2 | + `comm_unit_size=703647744`（#79901） | 21416.4 ± 48 | **-0.1%** | +47.7% | 17930 | 349 | 59.38 |
| T2ref | T4 + `prefetch_units=2` | 21363.9 ± 114（n=16，见下） | -0.3%（相对 T4） | +47.4% | 17967 | 346 | 59.08 |
| T5d | T4 + `after_step` / `all`（#79887） | 21899.8 ± 49 | **+2.2%（相对 T4）** | **+51.1%** | 17533 | 348 | 60.29 |
| T5d_T2 | T5d + `comm_unit_size=703647744` | 21749.5 ± 64 | -0.7%（相对 T5d） | +50.0% | 17643 | 347 | 60.29 |
| T1_T2 | T1 + `comm_unit_size=703647744`（旧顺序） | 20296.9 ± 66 | +15.3%（相对 T1） | +40.0% | 18779 | 359 | 62.20 |

- T2ref 的 step 12 掉到 19031.7，其余 16 步正常；含这一步的 17 步均值是 21226.7 ± 576。表中用剔除后的均值。
- 所有阶段 step 1 loss 都是 12.24116135，step 20 loss 在 12.0508327–12.0508404 之间（差 < 1e-5），未见精度回退。
- 所有阶段日志里都没有 `Recovered` / `AlreadyExists` / `error 700`。
- 每阶段 1 次运行，标准差约 0.2–0.4%，±0.5% 以内的差异不应视为确定结论。T2 的 -0.1% 和 T2ref 的 -0.3% 都在这个范围内，结论是"无收益"而不是"负收益"。T5d_T2 的 -0.7% 略超出噪声范围，但只有一次运行。

## 分析

**为什么 T2 在 T1 上有 +15%，在 T4 上为 0。** 本模型每个 unit 有 dense buffer 22.2M 元素、expert buffer 212.3M 元素，专家权重占约 90%。

- T1 时专家权重按 bf16 gather，每个 unit 约 469 MB。按层预取（深度 1）盖不住，fwd-bwd 21.8 s。加深到 2 个 unit（`comm_unit_size=703647744` 的实际效果）后，fwd-bwd 降到 18.8 s。
- T3 起专家权重按 fp8 gather，每个 unit 约 212 MB fp8 加 44 MB bf16 dense，gather 量约减半；Fleet callback 也不再每步逐层 gather bf16 再量化。fwd-bwd 直接降到 18.3 s，已经低于 T1_T2。
- 此时深度 1 的预取已能盖住 gather，再加深只多占显存（+0.35 GB），fwd-bwd 不再缩短（17914 → 17930 ms）。

以上机制是根据 fwd-bwd 和通信量推断的，本次没有采 nsys 验证 gather 是否完全被计算覆盖。

**为什么 T5d 上加 T2 会略降。** `after_step` 下 gather 出的参数都标了 `keep_gathered`，每层每步只 gather 一次，预取深度本来就没有作用。`comm_unit_size` 还会把梯度 reduce-scatter 的在途队列从按层改为按元素数（上限 703.6M 元素），可能是 -0.7% 的来源，本次未单独验证。

**与 2026-11-04 两次 6 步运行的关系。** 那两次没有打开 bf16 reduce-scatter，相当于本表 T3 再加 reshard / comm_unit_size，所以绝对值偏低（21273 / 20939），结论方向与本次一致。

## 复现

文件都在 `perf_stages3/`：

| 文件 | 作用 |
| --- | --- |
| `final/fully_shard_fusion.py` | 当前源码加三个测试开关（`legacy_comm` / `no_fp8_gather` / `grad_comm_dtype`），从 `knobs.json` 读取 |
| `backup/fully_shard_fusion.py` | `Paddle/build/python/paddle/distributed/fsdp/` 下的原文件 |
| `specs/all.txt` | 各阶段定义，每行 `<name>\|<knobs-json>\|<额外 --kwargs>` |
| `run_stage.sh` | `run_fsdp_test_v2.sh` 的精简副本：`max_steps=20`，多拼 `extra_kwargs.txt` |
| `run_stages.sh` | 装入 `final/`，逐行写 `knobs.json` / `extra_kwargs.txt` 并运行，结果追加到 `results.log`，退出时恢复 `backup/`；每阶段前检查两台机器 GPU 无残留 |
| `parse.py` | `python parse.py <workerlog> 4 20`，输出吞吐均值 / 中位数、fwd-bwd、opt-step、显存、loss |
| `results.log` | 本次原始结果，含每阶段的 `outputs/` 目录和 `[perf_stages3] effective` 行 |

```bash
# 多机环境（IS_STANDALONE=0），任一节点执行
bash perf_stages3/run_stages.sh perf_stages3/specs/all.txt
```

本次各阶段输出目录为 `outputs/20261105_0002_*`（T0）到 `outputs/20261105_0114_*`（T1_T2），均以 `_fsdp_T5d` 结尾。运行结束后已确认 `Paddle/build` 下的文件已恢复（md5 与源码一致）。
