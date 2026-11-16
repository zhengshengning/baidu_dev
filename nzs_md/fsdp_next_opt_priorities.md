# Paddle FSDP 后续优化点（对照 Megatron-FSDP，按优先级）

日期：2026-11-05。只读调研，未改代码。

对照版本：
- Megatron-LM HEAD `8817e0fd`（2026-11-04），`megatron/core/distributed/fsdp/` 近 3 个月提交。
- Paddle HEAD `abadcba928`，`python/paddle/distributed/fsdp/fully_shard_fusion.py`（下称 FSF）。
- PaddleFleet 本地副本 `PaddleFleet/src`。

数据来源：`fsdp_perf_opt_breakdown.md`（T0–T6）、`fsdp_ag_rs_stream_split.md`（stream 拆分 A/B）、`fsdp_zero_sm_ag_plan.md`（零 SM AG 阶段 0–2）。

## 0. 总体判断

- **单机（T1–T3）上 FSDP 不是瓶颈。** 反向 FSDP 暴露只占步时 1.3–1.7%，主要暴露是 DeepEP（9–10%）。
- **两机（T4）上 FSDP 是大头。** FSDP 暴露通信 2094 ms/步，占步时 19556 ms 的 10.7%，其中前向 940 ms、反向 1142 ms。dense sharding 组 16 卡跨机，每个 micro-batch 的 AG/RS 都走 IB。
- 因此优先级靠前的是：减少跨机字节数、显存稳定性。

## 1. 优先级总表

| 优先级 | # | 优化点 | 依据 | 工作量 |
|---|---|---|---|---|
| P0 | 1 | trainer 暴露 `grad_comm_dtype`，RS 走 bf16 | T4 实测 +1.8%，线上仍是 fp32 RS | 几行 |
| P0 | 2 | HSDP / HFSDP | T4 FSDP 暴露 10.7%，全部跨机 | 大 |
| P0 | 3 | FSDP 常驻 buffer 池（双 buffer） | mb3 随机崩溃；mb3 比 mb2 快约 13%。已实现，T4 mb3 未复现崩溃、吞吐持平（§2.3） | 中 |
| P1 | 4 | dense / expert 分别配置梯度同步策略 | T6 mb1 +13%，但全局开 OOM | 中 |
| P1 | 6 | 录制执行顺序替代静态 gid ±1 | 开 MTP / 共享层前的正确性风险 | 中 |
| P2 | 5 | NCCL symmetric 注册 + 零 SM AG | 已实现，默认关闭。C2 相对 C1：T1 +0.3%、T2 −0.15%、T3 +0.7%（均在波动内），T4 快 0.18%（§3.2） | 已完成；启用需重编 Paddle + 所有节点换 NCCL 2.31.2 |
| P2 | 7 | 去掉 step 前 `paddle.device.synchronize()` | 未测 | 小 |
| P2 | 8 | 统计并减少 padding | 未测 | 小→中 |
| P2 | 9 | 初始化显存峰值（meta init） | FSF:195 TODO | 中 |
| P2 | 10 | PP 支持 | 模型放不下时才需要 | 大 |
| P2 | 11 | 补多卡测试 | 测试缺口 | 小 |

## 2. P0：收益明确，先做

### 2.1 trainer 暴露 `grad_comm_dtype`（bf16 RS）

- **现状：** Paddle 已支持（[#79899](https://github.com/PaddlePaddle/Paddle/pull/79899)，`MixedPrecisionPolicy.grad_comm_dtype`）。PaddleFleet 没暴露：`PaddleFleet/src/paddlefleet/trainer/trainer.py:5787-5794` 只在 `amp_master_grad=False` 时设 `reduce_dtype`。生产 yaml 是 `amp_master_grad: true`，所以线上 RS 是 fp32。
- **Megatron：** `--megatron-fsdp-grad-comm-dtype auto` = main grad dtype，bf16 模型默认 bf16 RS。
- **实测：** T4 上 +1.8%（`fsdp_perf_opt_breakdown.md` T4 阶段），loss 到 step 20 一致到 1e-5。
- **做法：** training_args 加 `fsdp_grad_comm_dtype`，构造 `MixedPrecisionPolicy` 时传入。

### 2.2 HSDP / HFSDP

- **现状：** 未实现。FSF 只用 sharding / moe_sharding 两个组（FSF:431-440），没有外层 DP 规约。
- **Megatron：**
  - v1 `outer_dp_sharding_strategy` ∈ {no_shard（HSDP）, optim（HFSDP）}，`num_distributed_optimizer_instances>1` 触发（[#6773](https://github.com/NVIDIA/Megatron-LM/pull/6773)，29694c26e）。
  - v2 非最后 micro-batch 只做 inner 规约，outer 规约在最后 micro-batch、放在单独的流（[#6973](https://github.com/NVIDIA/Megatron-LM/pull/6973)，037b883c9）。
  - HFSDP 外层 AG 在 step 后第一次前向做。
- **预期：** 每个 micro-batch 的 AG/RS 留在 NVLink，跨机通信每 step 一次。T4 的 10.7% FSDP 暴露主要靠这一项降。
- **代价：** 参数、梯度在 8 卡上分片（原来 16 卡），显存上升；HFSDP 让优化器状态仍在 16 卡上分片，抵消一部分。
- **工作量：** 最大。hcg 加外层轴；FSF 加两级规约；加 outer 流。
- **expert 组：** T4 的 expert sharding 组是 2 卡、每机 1 卡（按 dp16 / EP8 推断），本身就跨机，inner 为 1。expert 怎么处理见 §3.1。

### 2.3 FSDP 常驻 buffer 池

- **现状：** 每次 gather `paddle.empty`（FSF:318-333），梯度 tmp buffer `paddle.zeros`，fp8 `fulls` 每次重新分配（`_fp8_utils.py:110-113`）。
- **问题：** T4 mb3/ga16 在 step 6–12 随机挂，报 VMM `Duplicate underlying allocation base ... in VMM V2 registry`，大量 `Recovered ... after memory defragmentation`，出现过一次 CUDA 700。A/B 只能退到 mb2/ga24。另一组实验中 T4b（mb3/ga16）步时 16954 ms，T4（mb2/ga24）19556 ms，mb3 快约 13%。
- **Megatron：** `fsdp_double_buffer` 两块常驻 buffer 交替分给相邻 unit；非对称模型用 MaxPool（`megatron_fsdp_max_pool_double_buffer`）；分配失败可回退常驻（`fsdp_db_use_persist_buf_on_alloc_fail`）。初始化后不再分配。
- **先验证：** mb3 崩溃是否由 FSDP 反复分配引起，还是 VMM allocator 本身的 bug。
- **已实现：** `FSDPBufferPool`（FSF），按 (dtype, numel) 精确复用 block，block 不还给 allocator。环境变量 `FSDP_BUFFER_POOL`：`0`（默认，不开）/ `params`（只池化参数 AG buffer，含 fp8 `fulls`）/ `all`（再加梯度 RS tmp buffer）。2 卡测试在 `params`、`all` 下 loss md5 与 stage1 一致。
- **T4 实测（2 机 16 卡，mb3/ga16，100 步，每种模式交替跑 2 次）：**

  | 模式 | max_mem_reserved | max_mem_allocated | 步时中位数 | tokens/s/card 中位数 | tokens/s/card 均值 | 慢步次数 |
  |---|---|---|---|---|---|---|
  | 0 | 73.35–73.41 GB | 70.95 GB | 16615 ms | 23671.6 | 23480 | 3 |
  | params | 73.95–73.98 GB（+0.6） | 71.47 GB（+0.52） | 16622 ms | 23671.6 | 23558（+0.3%） | 2 |
  | all | 74.71–74.81 GB（+1.4） | 72.35 GB（+1.40） | 16622 ms | 23652 | 23631（+0.6%） | 0 |

  - 步时和 tokens/s/card 从第 6 步起统计，取两次运行的平均。慢步指步时超过中位数 1.2 倍的步，每次多 5–19 s。
  - 吞吐：正常步三种模式相差不到 0.1%。均值的差别全部来自慢步次数，6 次运行共 5 次慢步，样本太少，不能说明池能减少慢步。
  - loss：15 对运行两两比较，最大差 5e-5 到 5.5e-4（相对差最多约 5e-5），出现在第 92–100 步，大小和模式无关。两次 0 模式之间就差 1.7e-4，所以池没有改变计算结果。
  - 6 次运行都没有 defrag、VMM Duplicate、CUDA 700 或 OOM。这次 baseline 本身没有复现 mb3 崩溃，所以无法判断池能不能解决崩溃。
- **结论：** T4 上没有可测到的收益，代价是显存 +0.6 GB（params）或 +1.4 GB（all）。代码保留，默认关闭。下一步只在能复现 defrag 或崩溃的配置上（显存更紧、更长序列、更大模型）比较 0 和 all；如果那里也没有差别，可以考虑删掉这部分代码。
- **与零 SM AG 的关系：** 两者可以合并实现，但互不依赖。
  - 零 SM AG 需要的是 `ncclCommWindowRegister` 注册过的内存，不需要常驻双 buffer。Megatron v2 用 caching MemPool 按需 resize，同样可以注册。
  - 照 v1 做法用 Paddle allocator 分配的双 buffer 没有注册，零 SM AG 仍用不上。
  - v1 双 buffer 限制同时在飞的 unit ≤2，与 `reshard_policy=after_step` 冲突（`fsdp_zero_sm_ag_plan.md` §3.2 对比表）。
  - 如果 #3 改用 `nccl_mem_alloc` 分配，即 `fsdp_zero_sm_ag_plan.md` §3.2 的 `_symm_pool`，可以同时满足两边。

## 3. P1：有收益，依赖 P0 或要先测

### 3.1 dense / expert 分别配置梯度同步策略

- **现状：** `sync_grads_each_microbatch` 是全局开关（FSF:1202-1207），trainer 未暴露。
- **实测：** mb1/ga48 开启 +13%，显存 +16.3 GB；mb2/ga24 OOM，原因是所有组的 fp32 全量梯度整个 step 常驻。
- **Megatron：** dense 与 expert 策略独立（[#7728](https://github.com/NVIDIA/Megatron-LM/pull/7728)，51e299ea9；v1 `expert_zero_dp_strategy`）。
- **建议：** 只对 expert 组（2 卡、跨机）推迟到最后 micro-batch 做 RS，配合 `reshard_filter=expert`。需测显存能否放下。

### 3.2 NCCL symmetric 注册 + 零 SM AG（降为 P2）

- 即 `fsdp_zero_sm_ag_plan.md`，阶段 0 结果见该文档 §8，实现和阶段 1、2 结果见 §9。
- **已实现（2026-11-06）：** `SymmBufferPool` 建在 #3 的 `FSDPBufferPool` 上，块用 `nccl_mem_alloc` 分配并注册为 window，AG 改 in-place；<1 MiB 的块不注册，留在 SM。开关 `FSDP_ZERO_SM_AG=1`，需同时设 `FSDP_BUFFER_POOL`，通信组用 `conf/nccl/fsdp_zero_sm.json`（`cta_policy: 2`）。
- **实测（C1 = 池，C2 = 池 + 零 SM AG，都用 `cta_policy=2` 和 NCCL 2.31.2）：**

  | | C1 步时中位数 | C2 | C2 相对 C1 |
  |---|---|---|---|
  | T1 | 1579 ms | 1584 | +0.3% |
  | T2 | 5533 ms | 5525 | −0.15% |
  | T3 | 13302 ms | 13391 | +0.7%（只有 3 步） |
  | T4（2 机，mb3/ga16） | 16555 ms | 16526 | −0.18% |

  - T1–T3 的差别都在波动以内。T3 nsys 确认 AG 换成了 CE（AG kernel 6115 → 825 ms，剩下的是 <1 MiB 的块），RS 不变。
  - T4 快 0.18%，与阶段 0 同位置口径的 0.22% 相当，低于外推的 0.7–0.9%。T4 没开 nsys，跨机 AG 是否全部走了 CE 未直接核对。
  - loss 与 C0 一致到 1e-5。window 内存（T4 5165 MiB）不计入 `max_mem_reserved`，总占用与 C1 相同。
- **阶段 0 时的分析（保留作参考）：**
  - 单机上 AG 让计算 kernel 变慢，折合步时 ≤0.2%（T1 0.01%、T2 0.05%、T3 0.17%）。这只是计算侧的一项。
  - 没算的一项：microbench 里 SM AG 与计算并发时自身慢 1.7–2.4 倍，CE AG 基本不变，计算流因此等 AG 的时间未统计。这项会让收益变大。
  - 反方向：CE 有约 85 us 固定延迟，≤16 MiB 单独跑时比 SM 慢 2–4 倍。T3 平均每次 AG 约 128 us，落在 CE 不占优的区间。这项会让收益变小甚至为负。
  - T4：同位置口径实测 0.22%；0.7–0.9% 是用 24–32% 的样本外推的粗估。T4 的 FSDP AG 全部跨机，机间走 RMA CPU proxy，未测。
- **RS 方向待测：** Megatron README（`src/README.md:127`）说注册后 RS 可能用 bf16 传输、高精度累加，但取决于走 NVLink 还是 IB。symmetric RS 是否少占 SM、能否放宽 DeepEP `num_sms` 约束，也没测。阶段 0 只测了 ZERO 组上未注册的 RS 与默认组一样快。
- **部署成本：** 需用 NCCL 2.31.2 重编 Paddle，所有节点和训练镜像都要换。
- Megatron 对应 `--use-nccl-ub`、`--fsdp-manual-registration`、`--disable-symmetric-registration`（v1），v2 `use_symmetric_memory`。v1 注册后仍走 SM kernel；zero-CTA AG 只在 v2 单测里验证过。
- **结论：** 0.18% 抵不过重编 Paddle、所有节点换 NCCL 2.31.2 的部署成本，保持 P2，代码默认关闭。
- **回到 P1 的条件：** 在 AG 消息更大、跨机比例更高的配置（更大的 dense 组、更大模型）上 C2 相对 C1 ≥0.5%。如果那里也在 0.5% 以下，可以考虑删掉这部分代码。

### 3.3 录制执行顺序

- **现状：** 预取按静态 gid ±1（FSF:863-887）；`_backward_cnt` 依赖反向 unit_id 单调递减（FSF:979-992）。
- **Megatron：** v2 `PrefetchOrder.RECORDED` 录制首轮实际顺序（[#8478](https://github.com/NVIDIA/Megatron-LM/pull/8478)，2d7ea246f；[#8485](https://github.com/NVIDIA/Megatron-LM/pull/8485)，2fcef8359）。
- **风险：** ladder 配置关了 MTP，现在没问题；生产蓝本有 MTP，开 MTP 或共享层后预取可能 gather 错块、梯度计数出错。上 MTP 前必须先查。

## 4. P2：小改动，或场景未到

| # | 优化点 | 现状 | Megatron |
|---|---|---|---|
| 7 | 去掉 step 前全设备同步 | `comm_sync_and_reset_status` 末尾 `paddle.device.synchronize()`（FSF:1972） | 等各通信流 event |
| 8 | 统计并减少 padding | 每个参数 padding 到 `align × degree` 倍数（FSF:175-177），degree 越大越浪费 | 打印每组 padding MB；v2 unpadded 分片（[#8116](https://github.com/NVIDIA/Megatron-LM/pull/8116)，47715bb02） |
| 9 | 初始化显存峰值 | GPU 上建全量 buffer 再切分（FSF:195 TODO） | `--init-model-with-meta-device` |
| 10 | PP 支持 | FSF 无 PP 逻辑；ernie5 trainer 断言走 pipeline wrapper（`pretraining_trainer.py:3774-3777`） | v2 PP（[#6486](https://github.com/NVIDIA/Megatron-LM/pull/6486)，2539bc339），1F1B（[#7112](https://github.com/NVIDIA/Megatron-LM/pull/7112)，ba5e63cff） |
| 11 | 补多卡测试 | 缺 EP+fp8、`reshard_filter` 端到端、`sync_grads_each_microbatch` 组合 | `tests/unit_tests/distributed/mfsdp_v2/` 覆盖 overlap、memory、expert_parallel 等 |

## 5. 暂不建议做

| 项目 | 原因 |
|---|---|
| AG/RS 拆 stream 或拆 communicator | `fsdp_ag_rs_stream_split.md` §7：改成 Megatron 做法后仍无收益；拆 communicator 两机慢 1.5%，T4b 崩溃 |
| recompute 权重预取（Megatron [#8515](https://github.com/NVIDIA/Megatron-LM/pull/8515)，90e40493e） | 只对 unit 级 full recompute 有用；当前是 unit 内 selective recompute，反向 pre-hook 已先 gather 整个 unit |
| MXFP8 按 plane gather（[#7907](https://github.com/NVIDIA/Megatron-LM/pull/7907)、[#8432](https://github.com/NVIDIA/Megatron-LM/pull/8432)） | 只在 Blackwell 上有意义，当前是 H 系列 |
| CUDA graph | 不在当前路线上 |

## 6. 已与 Megatron 持平

- Muon owner P2P 计算（Paddle `keep_whole` + owner；Megatron [#7432](https://github.com/NVIDIA/Megatron-LM/pull/7432)、[#7974](https://github.com/NVIDIA/Megatron-LM/pull/7974)）
- fp8 参数 gather（Paddle [#79897](https://github.com/PaddlePaddle/Paddle/pull/79897)，blockwise e4m3）
- 按大小预取，fp8 / 被保留 buffer 不计入预算（与 Megatron [#8575](https://github.com/NVIDIA/Megatron-LM/pull/8575) 同思路）
- reshard_policy（`after_forward` / `after_backward` / `after_step`）
- bf16 RS 底层能力（[#79899](https://github.com/PaddlePaddle/Paddle/pull/79899)）
- flex_checkpoint 改并行度加载

## 7. 建议执行顺序

1. §2.1：改动小，直接拿 +1.8%。
2. §2.3：已实现，T4 上无收益，默认关闭；等有能复现崩溃的配置再验证。
3. §2.2：多机主要收益。
4. §3.1 依赖 §2.2 的分组结构，放在后面。
5. §3.2：已实现，T4 快 0.18%，默认关闭；只在更大 AG、更高跨机比例的配置上再测。

## 8. 未验证项

- mb3 崩溃是否由 FSDP 反复分配引起。T4 mb3/ga16 上 4 次不开池的运行（20 步 2 次、100 步 2 次）都没复现，需要能复现的配置才能验证。
- §2.2 的收益由 T4 暴露通信推算，未实测。
- T4 的 expert 组布局（每机 1 卡）按 dp16 / EP8 推断，未在 trace 里核对。
- T4 跨机 AG 是否全部走了 CE（未开 nsys 核对）；T3 开零 SM 后 `AllReduce_f32` 从 21603 ms 涨到 24629 ms 的原因。
- 注册后的 symmetric RS 的精度和 SM 占用。
