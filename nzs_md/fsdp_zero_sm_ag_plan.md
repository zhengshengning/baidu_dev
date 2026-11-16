# Paddle FSDP 零 SM AllGather（Zero-CTA / Copy-Engine）实现方案

状态：阶段 0（§8）、阶段 1、阶段 2（§9）已完成。已实现，默认关闭；T1–T3 无可测收益，T4 快 0.18%。日期：2026-11-06。

参考：Megatron-LM（本地 `Megatron-LM/`，HEAD `8817e0fd`）。Megatron 里有两套 FSDP，并存在 `main` 上，用 `--megatron-fsdp-version {1,2}` 切换，默认 1：

- **v1**（`megatron_fsdp/`）：生产版。`--nccl-ub` 把参数、梯度缓冲分配在 NCCL mem pool 里并做对称注册，强制常驻双 buffer（`megatron/training/arguments.py:1069-1078`）。它用注册内存走 NCCL symmetric kernel，仍占 SM，没有 zero-CTA 的入口。
- **v2**（`megatron_fsdp/experimental/`，2026-06 引入）：设计文档（`src/docs/mfsdp_design.md:380-393`）定位为长期版本，目标是在各模型上先追平 v1 的性能再迁移用户。仓库里没有 v1/v2 的性能对比数据。`nccl_ub` 在 v2 里映射为 `RuntimePolicy.use_symmetric_memory`（`megatron/training/training.py:2542-2556`），staging 缓冲从 PyTorch symmetric MemPool 分配。zero-CTA AG 只在单测里验证过，由测试另建 ZERO 组（`tests/unit_tests/distributed/mfsdp_v2/test_symmetric_memory.py:171`），没有接入训练配置。

本方案的内存模型接近 v2，对比见 §3.2、§3.3。

## 1. 背景

### 1.1 机制

NCCL 2.28 起，同一 (MN)NVL 域内的 AllGather 可以由 Copy Engine 搬运、不占 SM，条件有两个：

1. communicator 用 `ncclConfig_t.CTAPolicy = NCCL_CTA_POLICY_ZERO` 创建；
2. 一次调用的输入、输出缓冲都在 `ncclCommWindowRegister` 注册的对称 window 里。

NCCL 2.30.7 加了跨机的分层 zero-SM AG / All2all（机内 Copy Engine，机间 RMA CPU proxy）。

RS 要做规约，Copy Engine 做不了，仍是 SM kernel（Megatron 单测里 RS 是 `ncclSymkDevKernel_ReduceScatter_LDMC/LL`）。**所以收益只覆盖 AG。**

### 1.2 为什么要做

`conf/experiment/ERNIE-5.5-Flash/ScalingLadder/*T3*fsdp.yaml` 第 85–88 行：DeepEP `num_sms` 从 132 降到 104，原因是 DeepEP 占满 SM 时与 FSDP 的 NCCL AG/RS（24 channel）互等，偶发 hang。AG 移出 SM 后能减少计算和通信的 SM 争用。但 RS 仍占 SM，`104 + 24 <= 132` 这条约束不会自动解除。

## 2. 现状

### 2.1 Paddle 已有的底层能力

HEAD（`abadcba928`）里已有两个提交：`b8eef5f55b`（#79677）、`4ac7a7572b`（#79686）。

| 能力 | 位置 |
|---|---|
| `cta_policy` 透传到 `ncclCommInitRankConfig` | `paddle/phi/core/distributed/nccl_config.cc:75-109`，`nccl_comm_context.cc:82-96` |
| Python 侧 `create_nccl_config({"cta_policy": 2})` | `python/paddle/distributed/fleet/base/topology.py:69-122` |
| sharding / moe_sharding 组透传 nccl_config | `topology.py:360-369`，`:942-955` |
| window 注册 / 注销（4096 对齐，按指针缓存，接受子区间） | `nccl_comm_context.cc:112-191` |
| `core.nccl_mem_alloc`、`ProcessGroupNCCL.register_comm_buffer` | `paddle/fluid/pybind/distributed_py.cc:1276-1420` |
| `_nccl_symmetric_empty` / `_register_comm_buffer` / `_ZeroSMPool` | `python/paddle/distributed/collective.py:464-707` |
| `paddle.distributed.zero_sm.{empty, all_gather, alltoall_single}` | `python/paddle/distributed/communication/zero_sm.py` |

以上代码全部受 `NCCL_VERSION_CODE >= 23007`（2.30.7）编译期保护。

### 2.2 当前 build 用不上

- `Paddle/build` 按 `NCCL_ROOT=/usr` 编译，`/usr/include/nccl.h` 是 2.29.7。`libpaddle.so` 里没有 `nccl_mem_alloc`，`cta_policy >= 0` 会直接报错。
- 运行时 `FLAGS_nccl_dir=Paddle/build/python/nvidia/nccl/lib` 不存在，实际加载系统 `/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7`。
- 机器上可用的新版本：pip `nvidia-nccl-cu13` 2.31.2（`site-packages/nvidia/nccl/include/nccl.h`，`lib/libnccl.so.2`）。

### 2.3 FSDP 没接入

训练用的是 `Paddle/build/python` 下的 `FullyShardFusion`（`run_fsdp_test*.sh` 把它放在 PYTHONPATH 最前），源码在 `Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`（下称 FSF）。

| 缓冲 | 位置 | 现状 |
|---|---|---|
| AG 输出（full params） | FSF:318-333 `get_tmp_buffer` | 每次 gather `paddle.empty`，`clear_tmp_buffer`（FSF:391-396）释放 |
| AG 输入（param shard） | FSF:221-233 | `data_buffer`，Paddle allocator，常驻，也是优化器参数 |
| AG 调用 | FSF:335-352 `do_all_gather` | `stream.all_gather(full, shard, use_calc_stream=True)`，在 `comm_stream` 上 |
| FP8 shard / fulls | `_fp8_utils.py:50-53`，`:107-114` | `fulls` 每次 gather 重新 `paddle.empty` |
| RS 输入（full grad） | FSF:329-332 | `paddle.zeros`，fp32，跨 micro-batch 累加 |
| 通信组 | FSF:431-440 | dense 用 `hcg.get_sharding_parallel_group()`，expert 用 `get_moe_sharding_parallel_group()` |
| 通信流 | FSF:653-658 | 每个 group 一条，AG 和 RS 共用 |

AG 缓冲不在 window 里，就算 group 开了 ZERO，AG 也会静默退回 SM kernel。

另外，PaddleFleet 的 `parse_nccl_config_file`（`third_party/PaddleFleet/src/paddlefleet/trainer/trainer_utils.py:1906-1975`）只拷贝已知字段，会丢掉 `cta_policy`。

## 3. 设计

### 3.1 通信组：原 sharding 组开 ZERO（方案 A，推荐）

- 用 `nccl_comm_group_config` JSON 设置 `{"sharding": {"cta_policy": 2}, "moe_sharding": {"cta_policy": 2}}`。
- 已注册的 AG 走 Copy Engine；RS、all_reduce 等未注册的调用退回 SM 路径。
- AG 和 RS 仍在同一个 communicator 上按提交顺序执行，和现在一样，不会同时抢 NVLink。

方案 B：复制一个同 rank 的 group，只在它上面开 ZERO 给 AG 用。`fsdp_ag_rs_stream_split.md` §4.3 测过，AG 与 RS 并发时 AG 慢 2–6 倍，所以不作默认。只有阶段 0 测出 ZERO 策略拖慢 RS 时才考虑。

### 3.2 FSDP 专用注册内存池（新文件 `fsdp/_symm_pool.py`，约 80 行）

- 每个 `(group, dtype)` 一个池。
  - `acquire(numel)`：字节数向上取整到 4096，按大小从空闲链表取块；没有就 `nccl_mem_alloc` + `register_comm_buffer` 新分配一块。
  - `release(block)`：放回空闲链表。块分配后不归还 CUDA。
- 不复用 `_ZeroSMPool` 的原因：
  - 按 2 的幂分桶，最多浪费一倍显存；
  - 每桶只有 8 块，靠弱引用回收；`reshard_policy=after_step` 下同尺寸 full buffer 超过 8 个会直接报错；
  - FSDP 的释放发生在状态机里确定的位置，不依赖 GC，可以用确定性的空闲链表。
- 跨 rank 对称：注册是集合操作，各 rank 必须同序注册。同一 fsdp group 内 acquire/release 序列由预取计划决定，各 rank 相同，首个 step 注册顺序一致，之后不再注册新块。dense 组和 expert 组各一个池。
- 复用安全：`run_on_stream` 发 AG 前做 `comm_stream.wait_stream(current_stream)`，新 AG 写入时旧块的计算读者都已执行完。前提是所有读者都在计算流上，代码里加注释说明。
- 一块内存只注册一个 window（window 起点 = 分配起点），不让多个 window 共用一段底层内存，避开 NCCL issue #2198（跨机 RMA proxy 在共用场景下偏移算错，静默数据错误）。

与 Megatron 的对比：

| | Megatron v1（`nccl_ub`） | Megatron v2（`use_symmetric_memory`） | 本方案 |
|---|---|---|---|
| full param 缓冲 | 常驻 2 块轮换（`FixedPoolAllocator` / `MaxPoolAllocator`，`param_and_grad_buffer.py:2616-2668`） | symmetric MemPool，gather 前 resize 回原大小，用完 resize 到 0，内存回到池里缓存（`experimental/dbuffer.py:190-210`） | 常驻空闲链表，容量 = 实际同时存活的块数 |
| 预取深度 | 同时在飞的 unit ≤2，`_enforce_double_buffer_limit`（`:4191`）超限时等前面的 RS | 无硬上限 | 无硬上限 |
| 布局不一致的 unit | 退回动态分配、不走注册；或开 `fsdp_db_use_persist_buf_on_alloc_fail` 改常驻 | 统一走 MemPool | 按块大小取，无此问题 |
| AG 输入（shard） | 已注册 | 已注册（`model_weight` 在 pool 里分配，`experimental/parameter_group.py:333`） | 不注册，见 §3.3 |
| RS 输入 | 已注册 | 已注册（`allocate_partial_grad_buffer`，`:619`） | 不注册，见 §6 |

不照搬 v1 的双 buffer：≤2 个 unit 的限制会压住 `reshard_policy=after_step` 下的预取。思路与 v2 一致，区别是用确定性的空闲链表，不依赖 caching allocator。

### 3.3 AG 改为 in-place

- full buffer 从池里取。先把本 rank 的 shard（`data_buffer`）`copy_` 到 `full[rank*L:(rank+1)*L]`，再发 AG，输入就是这段切片。
- 这样一次调用的输入、输出都在已注册内存里，满足 NCCL "同一次调用要么全注册、要么全不注册"。现有分片保证 `piece_len * deg == total`，切片位置直接可算。
- 代价：每次 AG 多一次 1/deg 大小的 D2D 拷贝。预期走 `cudaMemcpyAsync`（Copy Engine，不占 SM），需在 nsys 里确认。
- 去掉这次拷贝的办法是让 `data_buffer` 常驻 window。但它是优化器参数，加载 ckpt 时有被替换的风险，留到后面。
- Megatron v2 就是这么做的：shard 分配在 symmetric pool 里，AG 直接以它为输入，没有这次拷贝。阶段 0 实测 shard `copy_` 每次 6–10 us（§8 0-3），在 ≤16 MiB 的消息上会叠加到 CE 约 85 us 的固定延迟上。因此这是阶段 1 之后首个值得做的优化，前提是处理好 ckpt 加载时替换参数的路径。

### 3.4 FP8 参数路径

- `get_fp8_tmp_buffer` 改为从 fp8 池、fp32 池取 `fulls`，`clear_fp8_tmp_buffer` 里 release，同样 in-place AG。
- scale 的 AG 很小，Copy Engine 启动延迟可能比 SM kernel 高。是否注册按阶段 0 测出的阈值定；不注册时在同一 communicator 上走 SM 路径，也合法。

### 3.5 开关与降级

- 环境变量 `FSDP_ZERO_SM_AG=1`。名字不能含 `PADDLE`，否则被 `script/train_gpu.sh:33-35` 清掉。默认关，关时代码路径与现在完全一致。
- NCCL < 2.30.7 或注册返回 0 时打一条 warning，退回 `paddle.empty` 路径。各 rank NCCL 版本相同，退回决定也对称。
- `restart_process_group` 时连同池一起丢弃。
- 已知 bug，本次不修：`comm_context_manager.cc:162` 重建 communicator 时丢掉 `cta_policy`。

## 4. 改动清单

| 文件 | 改动 |
|---|---|
| `Paddle/python/paddle/distributed/fsdp/_symm_pool.py` | 新增注册内存池 |
| `Paddle/.../fsdp/fully_shard_fusion.py` | `get_tmp_buffer`、`clear_tmp_buffer`、`do_all_gather` 加池分支（约 +50 行） |
| `Paddle/.../fsdp/_fp8_utils.py` | `get_fp8_tmp_buffer`、`clear_fp8_tmp_buffer` 加池分支 |
| `third_party/PaddleFleet/.../trainer/trainer_utils.py` | `parse_nccl_config_file` 透传 `cta_policy`（约 1 行，子模块内） |
| `conf/nccl/fsdp_zero_sm.json` 和实验 yaml | sharding / moe_sharding 设 `cta_policy: 2` |
| Paddle 重编 | 不改 C++ 源码；`NCCL_ROOT` 指向 pip `nvidia-nccl-cu13` 2.31.2，运行时 `FLAGS_nccl_dir` 指向同一份库 |

Python 改动同步到 `Paddle/build/python`，做法与之前相同。

## 5. 验证：分三阶段，达标才进下一阶段

### 阶段 0：可行性和收益上界（不改 FSDP）

1. 用现有 baseline nsys 统计计算 kernel 与 AG 重叠 / 不重叠时的耗时差，作为 SM 争用的收益上界。不用编译即可先做。
2. 重编 Paddle，用 `NCCL_DEBUG=VERSION` 确认加载的是 2.31.2。
3. 单机 8 卡 microbench，消息 1–256 MiB：
   - (a) Copy Engine AG 与 SM AG 延迟对比，in-place AG 正确性；
   - (b) ZERO 组上发未注册的 RS，确认 kernel 和延迟与默认组相同（方案 A 的前提）；
   - (c) 计算流跑 GEMM 时并发 AG，比较 GEMM 被拖慢的程度。
4. 停止条件：FSDP 实际消息尺寸上 Copy Engine AG 明显更慢，或 (c) 里 GEMM 拖慢没有减少。

### 阶段 1：正确性

- 开关开 / 关各跑 10 步，step 1–2 loss 逐位一致，AG 后参数 checksum 一致。
- nsys 里 AG 不再出现 `ncclDevKernel_AllGather`，换成 memcpy；RS 不变。

### 阶段 2：性能 A/B

- T1/T2/T3 单机 8 卡，方法同 `fsdp_ag_rs_stream_split.md`：3 步中位数，nsys 抓 step 6–9。
- 指标：步时、与 AG 重叠的计算 kernel 耗时、计算流等 AG 时间、池占用字节 + `max_mem_reserved`。池内存不计入 Paddle 显存统计，需单独打日志。

## 6. 风险

- **收益可能很小。** split_comm 实验算出的 "AG 排队" 上界只占 0.36%。这次收益来自 SM 争用，量级未知，阶段 0 第 1 步先估。
- **小消息变慢。** T3 每步 1445 次 AG，平均每次约 128 us；Copy Engine 在这个尺寸上启动延迟更高的话，计算等 AG 会变长。
- **部署。** 多机作业所有节点都要用重编的 Paddle 和 NCCL 2.31.2。生产 sharding 组若跨机，走 RMA CPU proxy，需单独做多机正确性测试；现有数据都是单机。
- **显存。** 池内块不归还，峰值 = 同时存活的 full buffer 总量，这部分不能再给激活复用。
- **后续可选项，不在本次范围：RS 输入注册。** Megatron v1、v2 都把 RS 输入放在注册内存里，以用上 NCCL symmetric RS kernel（`ncclSymkDevKernel_ReduceScatter_LDMC/LL`）。这与 zero-SM 无关，RS 仍占 SM，可能改善的是 RS 自身耗时。代价是 fp32 full grad 跨 micro-batch 累加，常驻注册会增加显存。是否值得做需单独测 RS 延迟。

## 7. 待确认

1. 是否接受重编 Paddle 并换用 pip NCCL 2.31.2（含训练镜像）？
2. 通信组默认方案 A（原 sharding 组开 ZERO）？
3. 先做阶段 0，拿到数据再决定是否做阶段 1–2？

## 8. 阶段 0 结果

### 0-1 收益上界（`timeline/ab_analysis/ag_sm_contention.py`，baseline nsys，8 卡 × 3 步，每卡每步）

| | T1 | T2 | T3 | T4（mb2/ga24） | T4b（mb3/ga16） |
|---|---|---|---|---|---|
| 步时 ms | 1580 | 5570 | 12936 | 19556 | 16954 |
| FSDP AG 忙 / RS 忙 ms | 29 / 18 | 66 / 94 | 185 / 252 | 2830 / 6587 | 2078 / 4162 |
| 只与 AG 重叠的计算 kernel 耗时 ms | 2.8 | 25.0 | 97.0 | 1374 | 1491 |
| 其中比无重叠基线多出的时间 ms | ~0.1 | 2.9 | 21–22 | +43（仅同位置，覆盖 32%） | +38（仅同位置，覆盖 24%） |
| 占步时 | 0.01% | 0.05% | 0.17% | 0.22%（外推 ~0.7%） | 0.22%（外推 ~0.9%） |

- 基线：同一 kernel（名字 + launch 配置）在无任何通信重叠时的耗时中位数。同步内同位置的干净样本只覆盖 T3 的 13%，其余用同签名其他位置的干净样本，噪声较大（同法算 T2 的 RS 为 -17%）。
- 结论：单机上 AG 抢 SM 让计算多花的时间约为步时的 0.2% 以下，即零 SM AG 在步时上的收益上界。与 DeepEP 等其他通信同时重叠的计算（T3 348 ms）不在此口径内。

T4 补充（2 机 16 卡，dp16 `sharding_first`，EP8，DeepEP `num_sms: 120`；trace 为 `order-T4-0` / `order-T4b-0`，即 `FSDP_AG_BEFORE_RS=0` 的默认顺序，只分析 n0 的 8 卡；sqlite 在 `/root/agrs_sqlite/`）：

- T4 用 `*T4*_fsdp_T5d.yaml`，T4b 用 `*T4*_fsdp.yaml`，两者只差 mb / ga。
- T4 的 FSDP 通信全部跨机：dense sharding 组 16 卡，expert sharding 组 2 卡（按 dp16 / EP8 推断为每机 1 卡，未在 trace 里核对）。AG 忙时是 T3 的 11–15 倍，与 AG 重叠的计算有 1.4–1.5 s/步，是 T3 的 14–15 倍。
- 同位置基线覆盖 24–32% 的样本，测得 AG 让计算慢 11–12%，折合步时 0.22%。
- 同签名其他位置的回退基线在 T4 上给出负值（T4 -173 ms，T4b -119 ms，即 AG 重叠的实例反而比"干净"实例快 10–16%），说明同签名的 kernel 在不同位置耗时差异大，这个回退在 T4 上不可用，表中没有采用。
- 把同位置测得的 11–12% 外推到全部 AG 重叠计算，得到 137–159 ms，约步时的 0.7–0.9%。这是粗估，依据只有 1/4 到 1/3 的样本。
- 与 DeepEP 等其他通信同时重叠的计算（T4 1395 ms，T4b 911 ms）同样不在此口径内。
- 跨机 AG 走零 SM 时，机间部分由 RMA CPU proxy 完成（§1.1），尚无数据，收益能否兑现要看阶段 0-3 的多机 microbench。另外 RS 与计算重叠时让计算慢 11–16%（同位置口径，约步时 0.9–1.0%），比 AG 大；零 SM AG 不覆盖这部分。

### 0-2 重编

`erniebot/Paddle/build_nccl231`：`NCCL_ROOT` = pip `nvidia-nccl-cu13` 2.31.2（cmake 识别 v23102），`CUDA_ARCH_BIN=90`，`WITH_CUDNN_FRONTEND=OFF`（与原 build 一致；开着会卡在 GitHub 下载）。

- 编译通过；`paddle.version.nccl()` = 23102，`core.nccl_mem_alloc` 存在，`ncclCommWindowRegister` 在 `libphi_core.so` 里。
- 运行时：`paddle/__init__` 会把 `FLAGS_nccl_dir` 改写成 `<paddle>/../nvidia/nccl/lib`，手动设的值无效，不存在时退回系统 2.29.7（`cta_policy` 直接报错）。做法：`build_nccl231/python/nvidia/nccl` 软链到 pip 包。日志确认 `NCCL version 2.31.2+cuda13.3`。
- 本机 PDC 环境变量描述的是 2 机集群，单机跑 launch 要先清掉 `PADDLE_*` / `TRAINER*` / `POD_IP`，见 `timeline/zero_sm_bench/run_bench.sh`。

### 0-3 microbench（单机 8 卡 H 系列，`timeline/zero_sm_bench/`）

nsys 核对（`nsys_ce_check.sqlite`）：CE 路径没有任何 NCCL kernel，只有 P2P memcpy；shard 的 `copy_` 是一次 DtoD memcpy；SM 路径是 `ncclDevKernel_AllGather_RING`（24 CTA）。CE 结果与 SM 结果逐位一致。

(a) bf16 AG 单独跑，100 次均值，us（大小 = AG 输出）：

| MiB | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| SM | 24 | 23 | 35 | 46 | 69 | 122 | 205 | 388 | 723 |
| CE（in-place） | 89 | 90 | 82 | 133 | 137 | 164 | 224 | 368 | 661 |
| CE + shard copy_ | 95 | 109 | 91 | 95 | 129 | 173 | 239 | 394 | 701 |
| CE / SM | 3.6 | 3.9 | 2.4 | 2.9 | 2.0 | 1.34 | 1.09 | 0.95 | 0.91 |

- CE 有约 85–90 us 的固定延迟，≤16 MiB 慢 2–4 倍，64 MiB 以上持平或略快。shard copy_ 本身 6–10 us（256 MiB 为 26 us）。
- 同一 ZERO 组上未注册的 AG 与默认组相同（SM@ZERO 列，未列出），即未注册调用正常退回 SM。

(b) fp32 RS，未注册缓冲：ZERO 组与默认组耗时比 0.99–1.02（2–256 MiB），方案 A 的前提成立。

(c) 计算流与侧流 AG 并发（20 次，us / 次，括号内为相对单独运行的变化）：

| 计算 | AG MiB | 计算 + SM AG | 计算 + CE AG | SM AG 被拖慢 | CE AG 被拖慢 |
|---|---|---|---|---|---|
| GEMM 8k³ bf16（单独 1399） | 8 / 32 / 128 | +22% / +21% / +27% | +10% / +11% / +13% | +140% / +122% / +105% | -5% / +1% / +2% |
| 逐元素 256 MiB（单独 181） | 8 / 32 / 128 | +4% / +8% / +18% | 0% / +1% / +4% | +100% / +87% / +74% | 0% / +3% / +15% |

- CE AG 让 GEMM 的拖慢减半，但没有归零（推测是功耗 / 频率或 HBM、L2 争用，未核实）；对访存型 kernel 基本无影响。
- 更大的差别在 AG 自身：SM AG 与计算并发时慢 1.7–2.4 倍，CE AG 基本不变。32 MiB 时并发下 SM AG 280 us，CE AG 141 us。0-1 的口径只算了计算被拖慢，没算 AG 被拖慢后计算流等 AG 的时间。

### 阶段 0 小结

- 停止条件（§5 阶段 0 第 4 条）：GEMM 拖慢有减少，不触发；CE AG 单独跑时在小消息上明显更慢，但在与计算并发的场景下更快。是否继续取决于 FSDP 实际 AG 尺寸分布和计算流等 AG 的时间，下一步需要从 T3/T4 trace 里统计（nsys 里没有字节数，需按参数 shape 推算或加日志）。
- 本节数据都是单机。T4 的 FSDP 通信跨机，走 RMA CPU proxy，尚未测。

## 9. 实现与阶段 1、2 结果

### 9.1 实现

改动在 `Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`（建在 `FSDPBufferPool` 上），和 §3 的设计相比，注册池没有单独成文件：

- `SymmBufferPool(FSDPBufferPool)`：每个 fsdp group 一个。新块用 `_nccl_symmetric_empty` 分配，再用 `_register_comm_buffer` 注册到该组，每块单独一个 window（避开 NCCL #2198）。小于 `FSDP_ZERO_SM_AG_MIN_KB`（默认 1024）的块不注册，走 SM，fp8 scale 因此留在 SM。注册返回 0 时整个池退回 SM 并 warning。rank 0 每注册一个 window 打一行 `FSDP_ZERO_SM_AG: group <id> window <n> ... total <MiB>`，因为 window 内存不进 Paddle 显存统计。
- `do_all_gather`：full 是 window 时，先在 comm stream 上把 shard `copy_` 到本 rank 的切片，再 in-place AG。
- `FSDPBufferManager._init_symm_pools`：`FSDP_ZERO_SM_AG=1` 时把 sharded 参数 buffer 的池换成 `SymmBufferPool`，必须同时设 `FSDP_BUFFER_POOL`。梯度 buffer 不变。fp8 fulls 经 `params_buffer.pool` 自动用上。
- `restart_process_group` 后 window 不在新 communicator 上，gather 退回 SM，结果仍正确。

配套：

- PaddleFleet `trainer_utils.py:parse_nccl_config_file` 透传 `cta_policy`（+3 行）。
- `conf/nccl/fsdp_zero_sm.json`：`{"sharding": {"cta_policy": 2}, "moe_sharding": {"cta_policy": 2}}`，训练时 `nccl_comm_group_config=conf/nccl/fsdp_zero_sm.json`。
- 运行时要用 `erniebot/Paddle/build_nccl231/python`（§8 0-2）。

### 9.2 阶段 1：正确性

- `C2_CASE=moe`，8 卡（dense sharding 8 卡、moe_sharding 4 卡、EP2），3 组池配置：C1、C2 的 loss MD5 都与 stage1 逐位一致；dense 池 5–6 个 window，expert 池 2–6 个；第二轮训练池不再增长。
- `C2_CASE=fp8`，2 卡：3 组 (keep_transpose, muon) 下 C2 与 C1 的 loss 和 fp8 cache 完全一致，注册 11–17 个 window。fp8 用例只能 2 卡（expert buffer 196608 元素，8 卡时不能按 32768 对齐分片）。
- T3 nsys（step 7–10，8 卡合计）确认 AG 换成了 CE：

  | | `ncclDevKernel_AllGather` 次数 / ms | P2P memcpy（kind 10）次数 / ms / GB | RS kernel ms |
  |---|---|---|---|
  | C1 | 46240 / 6115 | 无 | 8367 |
  | C2 | 15904 / 825 | 455040 / 3146 / 619 | 8199 |

  剩下的 15904 次 AG kernel 是 <1 MiB、未注册的块。RS 不变。C2 的 `AllReduce_f32` 从 21603 ms 变成 24629 ms，原因未查。

### 9.3 阶段 2：性能 A/B

三组都用 `cta_policy=2` 和 build_nccl231：C0 不开池；C1 `FSDP_BUFFER_POOL=params`；C2 = C1 + `FSDP_ZERO_SM_AG=1`。零 SM 收益 = C2 相对 C1。

- T1、T2：单机 8 卡，20 步，统计 step 6–20（15 步），不开 nsys。
- T3：单机 8 卡，13 步，nsys 抓 step 7–10，统计 step 11–13（3 步）。
- T4：2 机 16 卡，mb3/ga16，20 步，统计 step 6–20，不开 nsys。按 C0、C1、C2、C0 顺序各跑一次。

| | C0 步时中位数 | C1 | C2 | C2 相对 C1 | C2 window（个 / MiB） |
|---|---|---|---|---|---|
| T1 | 1587 ms | 1579 | 1584 | +0.3% | dense 18 / 477 |
| T2 | 5526 ms | 5533 | 5525 | −0.15% | dense 17 / 726 |
| T3 | 13291 ms | 13302 | 13391 | +0.7% | 未记录（日志在 T3 之后才加） |
| T4 | 16558 / 16547 ms | 16555 | 16526 | **−0.18%** | expert 20 / 4050，dense 8 / 1115（见 §10.1） |

- **T1–T3 没有可测收益。** T1、T2 的差别在 15 步内的波动以内（T2 每步标准差约 80 ms）。T3 只有 3 步，而且是在 nsys 抓完之后；C2 在被抓的 step 7–9 也最慢（中位数 13565 ms，C1 13394 ms）。
- **T4 快 0.18%。** 均值 C1 16557、C2 16528 ms，每步标准差 15–25 ms；两次 C0 的均值只差 5 ms。和 §8 0-1 同位置口径的 0.22% 相当，低于外推的 0.7–0.9%。T4 没开 nsys，跨机 AG 是否全部走了 CE 没有直接核对，只确认了 window 注册成功、没有退回 SM 的 warning。
- **loss：** 每个 T 内各组相对 C0 的最大差 1.5e-5–2.2e-5，step 20 的 loss 到 1e-5 一致。
- **显存：** C2 的 `max_mem_reserved` 比 C1 低（T4 73.98 → 68.82 GB），差值就是 window 大小（T4 5165 MiB），实际总占用和 C1 相同。
- **慢步：** T1 C2 有一步 3116 ms、T2 C2 有一步 6053 ms（约中位数的 2 倍和 1.1 倍），其他组没有。各只有一次，不足以判断是否与 C2 有关。
- 所有运行都没有 VMM Duplicate、CUDA 700 或 OOM。

### 9.4 结论

- 代码保留，默认关闭（不设 `FSDP_ZERO_SM_AG`）。
- 单机上没有收益。CE 每次约 85 us 的固定延迟和小块 AG 的 SM 回退抵消了 SM 争用的减少。
- T4 的 0.18% 是唯一的正收益，量级太小，抵不过重编 Paddle、所有节点换 NCCL 2.31.2 的部署成本。
- 后续只在 AG 消息更大、跨机比例更高的配置（更大的 dense 组、更大模型）上再测；那里如果也在 0.5% 以下，可以考虑删掉这部分代码。

## 10. 对照 Megatron 的后续优化点（2026-11-06）

只读调研，收益为按 buffer 大小估算，未实测。行号指 `fully_shard_fusion.py`（PR #79930 版本）；PGB 指 Megatron `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`。

### 10.1 T4 C2 的 buffer 构成（rank 0 window 日志）

| group | 内容 | 块大小 | 块数 |
|---|---|---|---|
| 28（moe_sharding，2 卡，每机 1 卡） | 每层专家 fp8 full | 202.5 MiB（2.12 亿元素） | 20，每层一块 |
| 84（sharding，16 卡） | 每层 dense | 40.5 + 3.8 MiB | 各 3 |
| 84 | embedding、lm_head | 491.2 MiB | 2 |

- group 84 含 embedding，所以是 dense 组；§9.3 表里原先把两组标反了，已更正。
- 分配顺序显示：前向中专家 fp8 full 每层新开一块、从不复用，前向末尾 20 层同时在显存（约 4 GB）。
- lm_head 新开了一块 491 MiB，没有复用 embedding 那块：embedding 的 full 在整个前向都没释放。

### 10.2 优化点

| # | 类型 | 现状 | Megatron | 估算 | 代价 / 风险 |
|---|---|---|---|---|---|
| 1 | 性能 | ga>1 时每次 RS 前在计算流上对整块 fp32 梯度 `scale_`，再 SUM（:447-451；系数 `1/(grad_div*steps)`，:2095） | `premul_sum` 把系数并进 NCCL（PGB:5185-5217） | 专家每层 810 MiB fp32，一次读写约 0.57 ms；20 层 × 16 micro-batch 加 dense、embedding 约 220 ms/步（约 1.4%）。改为 RS 不预缩放，`finish_grads_sync` 里对 shard 乘一次，约 6 ms/步 | 改动小；fp32 下数值等价，ga=1 / AVG 路径不变 |
| 2 | 配置 A/B | C0–C2 都是 `after_forward`；`breakdown` T5d（`after_step`/`all`）实测 +1.7%、+0.9 GB；moe_sharding 组每机 1 卡，CE 不能跨 IB，4 GB window 在这组上没有 CE 可用 | — | 不改代码：mb3/ga16 上测 `after_step`/`all`；C2 去掉 `moe_sharding` 的 `cta_policy` 再测 | 开 `after_step` 后 AG 每步只做 1 次，零 SM 收益空间更小 |
| 3 | 正确性 | RS 输出写在 `full._slice(0, n)`（:460、:475）；非 0 rank 输出与输入第 0 段重叠，但不是 NCCL 定义的 in-place（`recvbuff == sendbuff + rank*n`），ring 下碰巧正确 | 输出写本 rank 段（PGB:4250-4519） | 无性能变化；是 RS 接 symmetric 注册的前提 | 两行；换 NVLS / symmetric kernel 时现写法有风险 |
| 4 | 显存 | fp8 buffer 不计 gather 预算、前向不释放（:946-949、:979-981） | fp8（uint8）同样进 2 槽双 buffer（PGB:679-1215） | 让 fp8 也按 `reshard_policy` 走（新选项，默认不变），峰值约 −3.4 GB | 反向每层每 micro-batch 多一次跨机专家 AG（每卡收 101 MiB）；用于以显存换 mb4 或关 `full_attn` recompute |
| 5 | 显存 | 释放 READY buffer 按 gid 顺序扫描（:974），unit 为 None 的 embedding 排最后（:633），`_release_params_after_unit` 跳过它（:1361） | 按 unit 顺序回收 | 约 −0.48 GB | 需确认 embedding 反向 hook 不再 gather；若会，每 micro-batch 多一次 491 MiB AG |

另：AG 前把 shard 拷进 full（:426-431），Megatron 把常驻 shard 直接分配在注册池里省掉这次拷贝（PGB:2956-3150）；T4 约 20 ms/步，不单独做。

### 10.3 执行顺序

每步完成后确认再进行下一步。

1. #1 延迟缩放：实现 + 2 卡测试 + T4 A/B。
2. #2 配置 A/B。
3. #3 RS 输出切片。
4. #4、#5：做成可选项，需要显存换 mb / recompute 时再开。

### 10.4 #1 延迟缩放结果

改动（`fully_shard_fusion.py`）：`do_reduce_scatter` 在 SUM 且 `grad_comm_dtype is None` 时不再预缩放，只置 `grad_scale_pending`；`finish_grads_sync` 等完所有 RS 后对 shard 做一次 `scale_`；`_maybe_zero_grads` 清标志。AVG 路径、bf16 `grad_comm_dtype`（仍预缩放）、unsharded buffer 不变。顺带修了 `buffer_pool="all"` 时 `grads_buffer is None`（冻结参数组）报 `AttributeError` 的问题。

- 2 卡：`dygraph_group_fsdp.py`（`ACCUM_STEPS=2`，`2*2 != 2`，走延迟路径，系数 1/4）loss md5 与 stage1 逐位一致；`dygraph_group_fsdp_fp8.py` 通过。NCCL 2.29.7、2.31.2 两种编译都通过。
- T4（2 机 16 卡，C2 配置，按 C2、C2D、C2D、C2 顺序各跑 20 步，统计 step 6–20）。运行时的 build_nccl231 副本用环境变量 `FSDP_DEFER_GRAD_SCALE=1` 切换，只用于 A/B，原文件备份为 `fully_shard_fusion.py.bak_before_defer`。

  | | 步时中位数 ms | 均值 / 标准差 | tok/s/card |
  |---|---|---|---|
  | C2 | 15985 / 15999 | 15987 / 42，16012 / 45 | 24615 / 24576 |
  | C2D | 15783 / 15794 | 15806 / 47，15798 / 24 | 24930 / 24891 |

  步时 −204 ms（**−1.27%**），与 §10.2 估算的约 214 ms 相当。四组两两 loss 最大差 1.3e-5–2.2e-5，同配置两次之间也是 1.9e-5。显存不变。
- 本批 C2 比 §9.3 的 C2（16526 ms）快约 3%，原因未查；只在同一批内比较。
- 不开池、不开 zero-SM 的复测：`Paddle/build/python`（NCCL 2.29.7，不设 `nccl_comm_group_config`），脚本 `timeline/zero_sm_bench/run_t4_defer.sh`，同样用 `FSDP_DEFER_GRAD_SCALE` 切换（原文件备份为 `fully_shard_fusion.py.bak_before_defer`），按 B、D、D、B 顺序各跑 20 步：

  | | 步时中位数 ms | tok/s/card 中位数 |
  |---|---|---|
  | B（优化前） | 16010 / 16013 | 24576 / 24537 |
  | D（延迟缩放） | 15842 / 15841 | 24812 / 24812 |

  −170 ms（**−1.06%**）。Da、Bb 各有一次 step 13 约 22.2 s 的慢步，两组都有，与改动无关；中位数不受影响。loss 两两最大差 1.2e-5–1.9e-5，同配置之间 1.6e-5–1.9e-5。显存不变（reserved 73.3 GB）。

## 参考

- [NCCL 2.30.7 Release Notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-7.html)
- [NCCL v2.28.3 Released · Discussion #1869](https://github.com/NVIDIA/nccl/discussions/1869)
- [NCCL API Supported Flags (2.30.3)](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/user-guide/docs/api/flags.html)
- [NCCL issue #2198: RMA proxy path uses window-relative offsets](https://github.com/NVIDIA/nccl/issues/2198)
- [Megatron-LM issue #5655](https://github.com/NVIDIA/Megatron-LM/issues/5655)
