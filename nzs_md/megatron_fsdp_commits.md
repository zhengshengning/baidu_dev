# Megatron-LM FSDP 相关 commit 汇总

- 仓库：`Megatron-LM/`，HEAD `70f7a269d`
- 筛选条件：commit message 含 `fsdp`（不区分大小写），或改动了 `megatron/core/distributed/fsdp/`，去重后共 351 个
- 时间范围：2024-11-13 ~ 2026-11-06，按提交时间倒序
- 标注“（间接相关）”的 commit 只是顺带提到 FSDP，主体不是 FSDP 改动
- 第三部分是对 Paddle FSDP（`Paddle/python/paddle/distributed/fsdp/`）的对照检查和下一步优化建议

## 一、主题归纳

| 主题 | 内容 | 代表 PR |
|---|---|---|
| MFSDP v2 重写 | `experimental/` 下新实现：DBuffer、Layout/owner planning、ExecutionOrder、SchedulePolicy/RuntimePolicy、Context 与模块元数据拆分 | #6369 #7895 #7962 #8207 #8479 #8585 #8586 |
| 通信调度 | AG 与 RS 分流（可选合并为单流省显存）、DP-outer 梯度归约独立流、按大小 prefetch、前后向独立 prefetch 预算、按记录的执行顺序 prefetch、recompute 权重在 backward 中 prefetch、optimizer step 后预取下一步 AG | #6563 #6973 #6905 #8478 #8485 #8515 #8575 #5985 |
| HSDP / HFSDP | 节点内 shard + 节点间 replicate 或 optim 分片；最后一个 micro-batch 才做 DP-outer 归约；DP-outer 与 DP-inner gather 流水 | #5743 #5972 #6773 #6973 |
| 显存与通信 buffer | double buffer 池（FixedPool / MaxPool）、NCCL user buffer 与 symmetric memory 注册、unpadded shard、meta-device 初始化 | #5462 #4492 #8116 #5369 #6799 |
| 低精度 | FP8 / MXFP8 param gather，按阶段只 gather 需要的 MXFP8 plane（前向 rowwise、反向 colwise），recompute 时全 plane；BF16 recipe | #7588 #7907 #8432 #7751 |
| 优化器 | Muon（分片 momentum）、多组 owner 负载均衡、per-token 梯度归一化、grad_comm_dtype | #8313 #7974 #7711 #8185 |
| MoE / EP | dense 与 expert 独立分片策略、EP 下 checkpoint、ShortcutMoEBlock 逐块 fully_shard、delayed wgrad 同步 | #7728 #8230 #8312 #7854 #8637 |
| Checkpoint | DCP / fsdp_dtensor 格式、torch_dist → fsdp_dtensor 转换、v2 optimizer DCP | #6215 #7674 |
| CUDA graph | CG 安全模式、HSDP 下去掉多余 stream wait、CG 测试 | #2941 #7729 |

## 二、commit 明细

| commit | 日期 | 标题 | 一句话概括 |
|---|---|---|---|
| 1ec6a7551 | 2026-11-06 | fix(fsdp): preserve delayed-wgrad markers on MFSDP v2 parameters (#8637) | 修复 MFSDP v2 参数上 delayed-wgrad 标记丢失的问题，确保延迟权重梯度计算的标记被保留。 |
| 2fcef8359 | 2026-11-02 | Unify preset FSDP execution order traversal (#8485) | 统一 MFSDP v2 预设 execution order 的遍历逻辑，重构 context、module 与 execution_order 中的相关实现。 |
| bfab98f4a | 2026-11-01 | [MFSDP] Extract V1 and V2 wrappers from the MCore adapter (#8471) | 将 MFSDP V1 与 V2 的 wrapper 从 MCore adapter 中拆分出来，独立成单独的实现。 |
| 037b883c9 | 2026-11-01 | MFSDP v2: overlap the DP-outer gradient reduction on its own stream (#6973) | 在 MFSDP v2 中将 DP-outer 维度的梯度规约放到独立 stream 上执行，以与计算重叠。 |
| 88312648a | 2026-11-01 | Add SchedulePolicy.unshard_all_planes_in_backward for MXFP8 recomputation in MFSDP v2 (#8432) | 为 MFSDP v2 新增 SchedulePolicy.unshard_all_planes_in_backward 选项，支持 MXFP8 recomputation 时在 backward 中 unshard 所有权重 plane。 |
| 2d7ea246f | 2026-11-01 | Record MFSDP execution order for prefetch (#8478) | 记录 MFSDP 的模块 execution order，用于后续 all-gather prefetch。 |
| 3249e4008 | 2026-10-31 | [MFSDP] Group execution settings in SchedulePolicy (#8585) | 将 MFSDP 的执行相关配置项归并到 SchedulePolicy 中统一管理。 |
| f44e54a36 | 2026-10-31 | [MFSDP] Wrap each ShortcutMoEBlock with fully_shard (#8312) | 对每个 ShortcutMoEBlock 单独调用 fully_shard 进行包装分片。 |
| c94825af6 | 2026-10-30 | Separate MFSDP construction scope from runtime context (#8479) | 将 MFSDP 的构建期 scope 与运行时 context 分离。 |
| 90e40493e | 2026-10-29 | Megatron-FSDP: prefetch activation-recompute weights in backward (#8515) | Megatron-FSDP 在 backward 中为 activation recompute 所需的权重提前 prefetch all-gather。 |
| 6dca79cfd | 2026-10-29 | Megatron-FSDP: exclude buckets that need no all-gather from the prefetch size (#8575) | Megatron-FSDP 计算 prefetch 大小时排除不需要 all-gather 的 bucket。 |
| ccea6622f | 2026-10-28 | [MFSDP] Group context-level options in RuntimePolicy (#8586) | 将 MFSDP context 级别的选项归并到 RuntimePolicy 中。 |
| b9d4e2b2c | 2026-10-28 | Enable BF16 in DeepSeek recipes and reject mixed-dtype MFSDP fusion (#7751) | 在 DeepSeek recipe 中启用 BF16，并禁止 MFSDP 将不同 dtype 的参数融合到同一 buffer。 |
| 47715bb02 | 2026-10-27 | Add unpadded shards for MFSDP v2 RowAtomic/BlockAtomic layouts (#8116) | 为 MFSDP v2 的 RowAtomic/BlockAtomic layout 增加无 padding 的 shard 支持。 |
| 17fc7ba10 | 2026-10-27 | Simplify MFSDP unshard readiness and storage ownership (#8464) | 简化 MFSDP 中 unshard 就绪状态判断与存储所有权管理逻辑。 |
| 3ddfe0fe0 | 2026-10-27 | fix(fsdp): preserve retention parameter weight decay (#8516) | 修复 FSDP 中 retention 参数的 weight decay 设置丢失的问题。 |
| 19895dafb | 2026-10-26 | docs: Muon design for MFSDP v2 (#8512) | 新增 MFSDP v2 上 Muon 优化器的设计文档。 |
| 18bc49d2f | 2026-10-23 | Gather only the MXFP8 weight planes each compute phase needs in MFSDPv2 (#7907) | MFSDP v2 中每个计算阶段只 all-gather 其所需的 MXFP8 权重 plane，减少通信量。 |
| 38a47abb3 | 2026-10-23 | Standardize MFSDP v2 adapter tests on pytest fixtures (#8354) | 将 MFSDP v2 adapter 测试统一改为使用 pytest fixtures。 |
| d61d8d44e | 2026-10-21 | Split MFSDP v2 adapter tests into focused files (#7710) | 将 MFSDP v2 adapter 测试拆分为多个按功能划分的文件。 |
| 4b9d4f145 | 2026-10-21 | chore: bump Core to 0.21.0 and FSDP to 0.8.0rc0 (#8374) | 将 Megatron Core 版本升级到 0.21.0、Megatron-FSDP 版本升级到 0.8.0rc0。 |
| 91d774de7 | 2026-10-20 | MFSDP v2: add Muon with sharded momentum (#8313) | 为 MFSDP v2 新增 Muon 优化器支持，momentum 以分片形式存储。 |
| 74c9420e3 | 2026-10-19 | Fix MFSDP v2 checkpoint test setup and teardown (#8326) | 修复 MFSDP v2 checkpoint 测试的 setup 与 teardown 逻辑。 |
| d37b3a8fd | 2026-10-19 | fix: Defer frozen MFSDP cleanup without input gradients (#8113) | 修复冻结参数模块在输入无梯度时的 MFSDP 清理时机，将其延后执行。 |
| 60547a3f4 | 2026-10-19 | Replace IndexedOrder with ExecutionOrder for MFSDP v2 (#8207) | 在 MFSDP v2 中用 ExecutionOrder 替换 IndexedOrder 来描述模块执行顺序。 |
| db3a6cd8a | 2026-10-18 | Fix MFSDP v2 checkpointing for expert-parallel MoE models (#8230) | 修复 MFSDP v2 在 expert-parallel MoE 模型上的 checkpoint 保存与加载问题。 |
| 31265d3c1 | 2026-10-17 | Support per-token gradient normalization in MFSDP v2 (#7711) | MFSDP v2 支持按 token 数进行梯度归一化。 |
| e6913b8f3 | 2026-10-17 | Support multi-axis all-gathers in QuantizedDBuffer (#7727) | QuantizedDBuffer 支持沿多个轴进行 all-gather。 |
| 4ab3af12d | 2026-10-16 | Balance MFSDP v2 owner planning around multiple groups (#7974) | MFSDP v2 的 owner 规划在多个 group 之间做负载均衡。 |
| 36731acc3 | 2026-10-16 | MoonEP style expert duplication for MoE load balancing (#7450) | （间接相关）新增 MoonEP 风格的 expert 复制以实现 MoE 负载均衡，并在 MCore FSDP adapter 中做了相应适配。 |
| 1fc08e50e | 2026-10-16 | Finalize gradient windows by declared multiplicity (#7581) | MFSDP v2 根据参数声明的使用次数（multiplicity）判断梯度累积窗口何时结束并触发规约。 |
| 502078a65 | 2026-10-16 | Clean up MFSDP parameter-group annotations (#7636) | 清理 MFSDP parameter group 相关的类型注解与标注。 |
| adcd90f4b | 2026-10-16 | Pad the scale buffers before calling TE's quantize (#8247) | 在调用 TE 的 quantize 前对 scale buffer 做 padding，避免 MXFP8 量化时尺寸不匹配。 |
| a666bd60e | 2026-10-16 | refactor(fsdp): Use DBuffer owner APIs and direct P2P receives (#8178) | 重构 FSDP 代码，改用 DBuffer 的 owner API，并直接使用 P2P receive 收取数据。 |
| fb011f244 | 2026-10-16 | Use weak-key membership for MFSDP context roots (#8222) | MFSDP context 的根模块集合改用 weak-key 方式记录，避免持有强引用导致内存泄漏。 |
| b597bd1fa | 2026-10-15 | refactor(fsdp): Remove unused owner gather/scatter plans (#8177) | 删除 FSDP 中未使用的 owner gather/scatter plan 代码。 |
| 0adca6528 | 2026-10-15 | Make partial_grad respect grad_comm_dtype in MFSDPv2 (#8185) | MFSDP v2 中 partial_grad 改为遵循 grad_comm_dtype 指定的通信精度。 |
| 913159367 | 2026-10-14 | Fix MFSDP v2 backward with shared-layer MTP (#7757) | 修复 MFSDP v2 在共享层 MTP（multi-token prediction）场景下 backward 出错的问题。 |
| 0c7d417d4 | 2026-10-14 | test: quarantine hybrid MFSDP CUDA-graph metric validation (#8219) | 暂时隔离 hybrid MFSDP CUDA graph 的指标校验测试。 |
| 51e299ea9 | 2026-10-14 | Support independent dense and expert MFSDP sharding policies (#7728) | 支持为 dense 参数和 expert 参数分别配置独立的 MFSDP 分片策略。 |
| f9ff95873 | 2026-10-13 | refactor(fsdp): Extract global layout builders (#7962) | 重构 FSDP 代码，将 GlobalLayout 的构建逻辑抽取为独立的 builder。 |
| 850bf4db5 | 2026-10-13 | Add owner-compute P2P gather/scatter (#7432) | 新增基于 owner 计算的 P2P gather/scatter 通信。 |
| d71578256 | 2026-10-12 | Reuse GlobalLayout in owner planning (#7966) | owner 规划中复用 GlobalLayout 结构。 |
| 736353b7e | 2026-10-11 | Add DSA indexer weights projection precision controls (#6842) | （间接相关）为 DSA indexer 的权重投影增加精度控制选项，并相应调整 MCore FSDP adapter。 |
| a8a5977f7 | 2026-10-11 | Support DCP checkpointing for the Megatron-FSDP v2 optimizer (#6215) | 支持对 Megatron-FSDP v2 optimizer 使用 DCP（torch distributed checkpoint）保存与加载。 |
| a3e1f8259 | 2026-10-11 | Extract MFSDP context and centralize module metadata (#6369) | 抽取 MFSDP context，并集中管理模块元数据。 |
| 9a81a5433 | 2026-10-10 | test(mfsdp): separate parameter, optimizer, and context tests (#6633) | 将 MFSDP 的 parameter、optimizer 与 context 测试拆分为独立文件。 |
| 4603a8362 | 2026-10-09 | Expose rank and tensor ranges in GlobalLayout (#7963) | 在 GlobalLayout 中对外暴露每个 rank 及每个 tensor 对应的区间范围。 |
| 43ecc88de | 2026-10-09 | refactor: Centralize MFSDP context construction (#7895) | 重构代码，将 MFSDP context 的构建集中到一处。 |
| fde43b6d6 | 2026-10-09 | Accept iterable inputs in GlobalLayout constructor (#7993) | GlobalLayout 构造函数支持接收任意 iterable 输入。 |
| 724886e5c | 2026-10-08 | MFSDP v2 owner planning: reword from "plan" to "layout" (#7977) | 将 MFSDP v2 owner 规划相关命名从 "plan" 统一改为 "layout"。 |
| 441a98740 | 2026-10-08 | fix: Synchronize delayed weight gradients in MFSDP v2 (#7854) | 修复 MFSDP v2 中延迟计算的 weight gradient 缺少同步的问题。 |
| 76ab5004f | 2026-10-08 | Avoid repeated dataclass creation in MFSDP layout validation (#7853) | 避免 MFSDP layout 校验过程中重复创建 dataclass，降低开销。 |
| cfb2ab5ff | 2026-10-03 | Build TensorAtomic layouts in MFSDP from context owner assignments (#7672) | MFSDP 根据 context 中的 owner 分配结果构建 TensorAtomic layout。 |
| d36b3f500 | 2026-10-02 | Add TensorAtomic placement support and uneven collectives to DBuffer (#7671) | DBuffer 新增 TensorAtomic placement，并支持非均匀切分的集合通信。 |
| 1b3e04143 | 2026-10-02 | Support multi-axis views and all-gathers in DBuffer (#7633) | DBuffer 支持多轴视图以及沿多轴的 all-gather。 |
| 1b638343e | 2026-10-01 | Fix MFSDP v2 CUDA graph test optimizer cleanup (#7729) | 修复 MFSDP v2 CUDA graph 测试中 optimizer 的清理逻辑。 |
| c16f739b1 | 2026-09-30 | Fix convert-torch-dist-to-fsdp-dtensor for hybrid Gated-DeltaProduct checkpoints (#7674) | 修复将 torch_dist checkpoint 转换为 FSDP DTensor 格式的工具在 hybrid Gated-DeltaProduct 模型上的问题。 |
| bab851bfe | 2026-09-30 | Use default profiler activities throughout MFSDP v2 tests (#7726) | MFSDP v2 测试统一使用 profiler 的默认 activities 配置。 |
| fed130fe2 | 2026-09-30 | Add explicit rank segments and per-placement builders to GlobalLayout (#7670) | GlobalLayout 新增显式的 rank segment 描述以及按 placement 区分的 builder。 |
| 7f8dcee97 | 2026-09-24 | feat: Enable MXFP8 through the MFSDP v2 MCore adapter (#7588) | 通过 MFSDP v2 的 MCore adapter 启用 MXFP8 训练。 |
| c9d5abb53 | 2026-09-24 | fix(mfsdp): reuse generated subclasses across module instances (#7612) | 修复 MFSDP 为每个模块实例重复生成子类的问题，改为在同类实例间复用。 |
| b81ee09f7 | 2026-09-23 | test: Exercise rank-specific inputs in MFSDP SGD parity (#7566) | MFSDP SGD 精度对齐测试改为给不同 rank 输入不同数据。 |
| ffb3c4f96 | 2026-09-23 | Rename FSDP Flat placement to RowAtomic (#7539) | 将 FSDP 的 Flat placement 重命名为 RowAtomic。 |
| 2539bc339 | 2026-09-23 | Enable Pipeline Parallelism Support for M-FSDP v2 (#6486) | MFSDP v2 支持与 pipeline parallelism 组合使用。 |
| 64053f7cc | 2026-09-22 | test: enable optimizer CUDA graphs in hybrid MFSDP v2 test (#7446) | 在 hybrid MFSDP v2 测试中启用 optimizer 的 CUDA graph。 |
| 85cda9247 | 2026-09-22 | Integrate MXFP8 weights into MFSDP (#7114) | 将 MXFP8 权重集成到 MFSDP 中进行分片与 all-gather。 |
| ba5e63cff | 2026-09-20 | Integrate MFSDP v2 with fine-grained 1F1B scheduling (#7112) | 将 MFSDP v2 与细粒度 1F1B 调度集成。 |
| e648932ff | 2026-09-18 | Add QuantizedDBuffer for MXFP8 plane storage (#7265) | 新增 QuantizedDBuffer，用于存储 MXFP8 的各个数据 plane。 |
| b427ebfad | 2026-09-18 | Fix one-shot stream causing FSDP memory leaks (#6885) | 修复一次性创建的 stream 导致 FSDP 内存泄漏的问题。 |
| 2c897b8bf | 2026-09-17 | Add shard plan/owner-based compute planning API for MFSDP v2 (#6597) | 为 MFSDP v2 新增 shard plan 及基于 owner 的计算规划 API。 |
| 29694c26e | 2026-09-16 | Megatron-FSDP: Hybrid FSDP over replicated weights, single-rank reduction and bucket padding fixes (#6773) | Megatron-FSDP 支持对复制权重做 Hybrid FSDP，并修复单 rank 规约与 bucket padding 的问题。 |
| 8ea4bb7a6 | 2026-09-16 | Default expert sharding strategies to their non-expert counterparts (#7277) | expert 参数的分片策略默认沿用对应的非 expert 参数策略。 |
| 52f7a9b16 | 2026-09-16 | Rename DBuffer.get_local_tensor to get_tensor_view (#7261) | 将 DBuffer.get_local_tensor 重命名为 get_tensor_view。 |
| 75e901f86 | 2026-09-15 | Fix inferred MFSDP communication unit cap (#7134) | 修复 MFSDP 自动推断通信单元大小上限时的错误。 |
| d046b0de5 | 2026-09-15 | Fix MFSDP v2 parameter metadata preservation (#7230) | 修复 MFSDP v2 分片后参数元数据未被保留的问题。 |
| 2f582e696 | 2026-09-14 | Separate DBuffer layout construction from allocation (#7257) | 将 DBuffer 的 layout 构建与显存分配拆分为两个步骤。 |
| 88cf15953 | 2026-09-14 | Add block-atomic DBuffer placement (#6978) | DBuffer 新增 block-atomic placement。 |
| 845d9be3d | 2026-09-12 | Mark non-final microbatches for MFSDP v2 hybrid data parallelism (#7186) | 在 MFSDP v2 hybrid data parallelism 下标记非最后一个 microbatch，避免每个 microbatch 都触发梯度规约。 |
| 976f723ea | 2026-09-12 | Allow context parallelism with MFSDP v2 (#7220) | 允许 MFSDP v2 与 context parallelism 组合使用。 |
| eb5b23a68 | 2026-09-11 | Run mFSDP v1 functional tests nightly (#7244) | 将 MFSDP v1 的功能测试改为每晚运行。 |
| 5e5325d68 | 2026-09-11 | Support independent HSDP instances for expert parameters* (#7012) | 支持为 expert 参数创建独立的 HSDP 实例。 |
| 5762bacc3 | 2026-09-11 | Deduplicate MFSDP context post-backward hooks (#7196) | 去除 MFSDP context 中重复注册的 post-backward hook。 |
| c9e7ea765 | 2026-09-10 | Fix MFSDP v2 precision-aware optimizer gradient clipping (#7074) | 修复 MFSDP v2 搭配 precision-aware optimizer 时梯度裁剪的错误。 |
| 8190837c2 | 2026-09-10 | Support full-iteration CUDA graphs with MFSDP v2 (#7075) | MFSDP v2 支持整个 iteration 级别的 CUDA graph。 |
| d4550898c | 2026-09-10 | Refactor MFSDP parameter group initialization (#7175) | 重构 MFSDP parameter group 的初始化逻辑。 |
| b9c22eb38 | 2026-09-09 | Derive MFSDP v2 checkpoint chunk metadata from the parameter layout (#6214) | MFSDP v2 checkpoint 的 chunk 元数据改为从参数 layout 推导。 |
| c9b53d0a8 | 2026-09-08 | Fix FSDP mixed-precision reset for fixed buffers (#6924) | 修复 FSDP 对固定 buffer 重置 mixed-precision 设置时的错误。 |
| b3393bbb0 | 2026-09-07 | Expose MFSDP v2 custom-schedule lifecycle (#6484) | 对外暴露 MFSDP v2 自定义调度的生命周期接口。 |
| c171f66f1 | 2026-09-04 | Golden deepseek_proxy_mfsdp_v2_ep2 against its own output; enable determinism (#7069) | 用 deepseek_proxy_mfsdp_v2_ep2 测试自身的输出重新生成 golden 值，并开启确定性计算。 |
| 13c5581fe | 2026-09-04 | test: clarify DeepSeek proxy MFSDP v1 names (#7073) | 重命名 DeepSeek proxy 功能测试用例，明确标注其使用的是 MFSDP v1。 |
| 91149c04c | 2026-09-03 | Add a MFSDP v2 port of the deepseek_proxy_fsdp_ep2_fsdp2 functional test (#7041) | 将 deepseek_proxy_fsdp_ep2_fsdp2 功能测试移植到 MFSDP v2 上运行。 |
| 1030379d1 | 2026-09-03 | Add per-module MFSDP prefetch budgets (#6905) | 为 MFSDP 增加按 module 配置的 prefetch 预算，控制 all-gather 预取的内存开销。 |
| 06a1d8e6f | 2026-09-03 | Handle Transformer Engine delayed gradients in MFSDP (#6697) | 让 MFSDP 正确处理 Transformer Engine 延迟计算（delayed wgrad）的梯度，确保其完成后再做 reduce-scatter。 |
| 121a603ab | 2026-09-03 | Megatron-FSDP: shard expert and non-expert parameters by different strategies (#6594) | 允许 Megatron-FSDP 对 expert 参数和非 expert 参数分别采用不同的分片策略。 |
| 0859a5816 | 2026-09-02 | Reuse persistent main-grad storage (#7001) | MFSDP v2 在各迭代间复用持久化的 main_grad 存储，不再反复释放和重新分配。 |
| d6bf0c9b4 | 2026-09-01 | fix(fsdp): reuse persistent model-weight storage (#6962) | MFSDP v2 复用持久化的模型权重存储（DBuffer），减少重复分配并修正内存测试。 |
| 3134ff987 | 2026-09-01 | docs: update MFSDP public shard placement (#6979) | 更新文档中 MFSDP 对外暴露的 Shard placement 说明。 |
| 48754af09 | 2026-09-01 | Exclude cuBLAS workspaces from the MFSDP peak-memory bound (#6936) | MFSDP 峰值显存测试的上界计算中排除 cuBLAS workspace 的占用。 |
| ed4e79c1b | 2026-09-01 | Don't assert an address across DBuffer storage release and reallocate (#6968) | 去掉 DBuffer 存储释放后重新分配时对内存地址不变的断言。 |
| 2790a532a | 2026-08-31 | Pre-Fetch DP-Outer AG post-optimizer step (#5985) | 新增选项，在 optimizer step 之后提前发起 DP-outer 维度的 all-gather 预取（HSDP 场景）。 |
| 6c440e61f | 2026-08-31 | Refactor FSDP gradient readiness countdown (#6957) | 将 MFSDP v2 中判断梯度是否就绪的计数逻辑抽取为独立的 countdown 模块。 |
| a3344a6fa | 2026-08-31 | Fix MFSDP v2 meta-device initialization (#6799) | 修复 MFSDP v2 在 meta device 上初始化模型时的问题。 |
| 216fb9007 | 2026-08-31 | Add MFSDP v2 expert sharding strategy (#6798) | 为 MFSDP v2 增加 expert 参数专用的分片策略配置。 |
| 713ea6ad3 | 2026-08-31 | feat(fsdp): expose HSDP and HFSDP through the MFSDP v2 adapter (#6873) | 通过 MFSDP v2 的 MCore adapter 支持 HSDP 和 HFSDP。 |
| c5bd8330d | 2026-08-29 | fix(optimizer): support MFSDP v2 gradient clipping across dense and expert meshes (#6489) | 使 MFSDP v2 的梯度裁剪能跨 dense 和 expert 两套 device mesh 正确计算全局梯度范数。 |
| 09fde85ea | 2026-08-22 | Use Shard for MFSDP public placements (#6751) | MFSDP 对外 API 改用 DTensor 的 Shard placement 表示参数分片。 |
| b7d476030 | 2026-08-21 | Move MFSDP placements to the API boundary (#5971) | 将 MFSDP 的 placement 表示上移到 API 边界处，内部实现不再直接依赖它。 |
| 0d970bfb1 | 2026-08-21 | Use DTensor placements for DBuffer (#6744) | DBuffer 改用 DTensor placement 描述其分片布局。 |
| 101966cc0 | 2026-08-21 | Remove MFSDP DTensor availability shim (#6752) | 移除 MFSDP 中用于兼容 DTensor 不可用情况的 shim 代码。 |
| 6a7bc998c | 2026-08-21 | [FSDP] Reject post-wrap load_state_dict(assign=True) in MFSDP V2 (#6203) | MFSDP v2 在模型 wrap 之后拒绝调用 load_state_dict(assign=True)，避免替换掉已分片的参数。 |
| cf3b8ee60 | 2026-08-20 | Share MFSDP gradient completion hook (#6695) | 在 MFSDP v2 中让多处共用同一个梯度完成 hook，精简重复代码。 |
| 93924d59c | 2026-08-20 | Fix fused MLA and MTP checkpointing for Megatron-FSDP. (#6271) | 修复 Megatron-FSDP 下 fused MLA 和 MTP 层的 checkpoint 保存与加载。 |
| fe1e8f807 | 2026-08-19 | Support mixed-precision optimizers in MFSDP v2 (#6506) | MFSDP v2 支持混合精度 optimizer（如 FP32 master weight 配合低精度模型权重）。 |
| 01dc20b4a | 2026-08-19 | Support no-shard, ZeRO-1, and ZeRO-2 in MFSDP v2 (#6137) | MFSDP v2 新增 no-shard、ZeRO-1、ZeRO-2 三种分片模式。 |
| 29361d658 | 2026-08-18 | Remove core teams from FSDP owners (#6623) | （间接相关）修改 CODEOWNERS，将 core 团队移出 FSDP 代码的 owner 列表。 |
| 540f4203e | 2026-08-19 | Add opt-in MFSDP unified communication stream (#6563) | 为 MFSDP 增加可选的统一通信 stream，让 all-gather 与 reduce-scatter 共用一条 stream。 |
| 342faa43c | 2026-08-18 | [MFSDPv1] Fix gradient sharding / allocation schedule with ZeRO-2, and fix wgrad accumulate fusion for ZeRO-1. (#6357) | 修复 MFSDP v1 在 ZeRO-2 下的梯度分片与分配时序，以及 ZeRO-1 下的 wgrad accumulate fusion。 |
| 54c7918b9 | 2026-08-18 | test(mfsdp): separate memory tests (#6568) | 将 MFSDP 的显存相关测试拆分到单独文件。 |
| dcb2bfd97 | 2026-08-18 | test(mfsdp): separate overlap test (#6591) | 将 MFSDP 的通信计算重叠测试拆分到单独文件。 |
| 35ae46c5b | 2026-08-18 | test(mfsdp): cover v2 training memory (#6567) | 增加 MFSDP v2 训练过程显存占用的测试。 |
| 43c0e027e | 2026-08-17 | fix(core): warn on invalid RANK env values (#4993) | （间接相关）RANK 环境变量非法时改为打印警告，并统一 megatron_fsdp utils 中的 rank 获取逻辑。 |
| 62a617338 | 2026-08-17 | test(mfsdp): cover v2 training-step peak memory (#6551) | 增加 MFSDP v2 单个训练 step 峰值显存的测试。 |
| e60e42fa1 | 2026-08-15 | Support HFSDP deferred DP-outer gradient reduction (#5972) | HFSDP 支持延迟 DP-outer 维度的梯度归约，在梯度累积时减少跨节点通信。 |
| 78ad10898 | 2026-08-14 | Track DBuffer allocation stream (#6441) | DBuffer 记录分配内存时所在的 CUDA stream，保证跨 stream 使用时的内存安全。 |
| 3d0956fe5 | 2026-08-13 | test(fsdp): verify optimizer CUDA graphs in v2 (#6498) | 增加测试，验证 MFSDP v2 下 optimizer step 可被 CUDA graph 捕获。 |
| 281075d81 | 2026-08-13 | feat(fsdp): support expert parallelism in MFSDP v2 adapter (#6450) | MFSDP v2 的 MCore adapter 支持 expert parallelism。 |
| 3ce4b4ac0 | 2026-08-13 | test(fsdp): mark default overlap check flaky (#6493) | 将默认配置下的 MFSDP 通信重叠检查测试标记为 flaky。 |
| 49a49a9b0 | 2026-08-11 | Add MFSDP v2 design document (#6431) | 新增 MFSDP v2 的设计文档。 |
| 80cda5db7 | 2026-08-11 | Refine MFSDP v2 configuration validation (#6333) | 完善 MFSDP v2 的配置合法性校验。 |
| 03318a83c | 2026-08-11 | Validate MFSDP module phase transitions (#6359) | 为 MFSDP module 的运行阶段（forward/backward 等）切换增加合法性校验。 |
| 1d402ad6e | 2026-08-11 | fix(mfsdp): refresh V2 compute weights after optimizer step (#6336) | 修复 MFSDP v2 在 optimizer step 后未刷新计算用权重的问题。 |
| db57c152e | 2026-08-11 | Gate expensive work behind log levels (#5805) | （间接相关）将耗时的日志字符串构造放到日志级别判断之后，涉及 megatron_fsdp 的 param_and_grad_buffer 等多个模块。 |
| 78901d8a7 | 2026-08-10 | Add construction-scoped MFSDP v2 fully shard options (#6127) | 为 MFSDP v2 的 fully_shard 增加仅在构造阶段生效的配置选项。 |
| 6518b75ec | 2026-08-09 | Add PyTorch DCP checkpoint save/load for MFSDP v2 (#6024) | MFSDP v2 支持用 PyTorch DCP（Distributed Checkpoint）保存和加载 checkpoint。 |
| f8ec15e4a | 2026-08-07 | Divide expert gradients by the expert-parallel size (#5956) | MFSDP v2 中将 expert 梯度除以 expert-parallel size，修正梯度缩放。 |
| 07a23196d | 2026-08-08 | Fix FSDP activation recompute prefetch (#6153) | 修复开启 activation recompute 时 FSDP 参数 prefetch 的错误。 |
| cde3b5c25 | 2026-08-07 | Encapsulate symmetric-memory AVG in DBuffer (#6169) | 将基于 symmetric memory 的 AVG 归约逻辑封装进 DBuffer。 |
| fa0758c3b | 2026-08-07 | Allocate main_grad on the reduce-scatter stream (#6187) | 在 reduce-scatter stream 上分配 main_grad，避免跨 stream 的内存复用问题。 |
| 59b72fa57 | 2026-08-06 | Replace lazy MFSDP context initialization with an explicit scope (#6190) | 将 MFSDP 上下文的惰性初始化改为显式的作用域管理。 |
| d180b7cca | 2026-08-05 | Document MFSDP optimizer design (#6186) | 新增 MFSDP optimizer 设计文档。 |
| 19c50e864 | 2026-08-05 | Release MFSDP storage after model deletion (#6230) | 模型删除后释放 MFSDP 持有的存储，避免显存泄漏。 |
| 42460a7af | 2026-08-02 | Clarify single communication stream benefits (#6177) | 在 runtime schedule 文档中补充使用单一通信 stream 的好处。 |
| 714c24a7e | 2026-08-01 | Fix tied parameters (#6164) | 修复 MFSDP v2 对 tied parameters（共享权重）的分组与分片处理。 |
| 1678db536 | 2026-08-01 | Add meta-parameter support (#5369) | MFSDP v2 支持对 meta device 上的参数执行 fully_shard。 |
| 313f6e95d | 2026-07-31 | Add MFSDP v2 MCore integration with FullyShardedOptimizer (#5865) | 将 MFSDP v2 接入 MCore，并提供 FullyShardedOptimizer。 |
| d3519d5e6 | 2026-07-30 | Add MFSDP runtime schedule design (#6031) | 新增 MFSDP 运行时通信与计算调度的设计文档。 |
| dc3c9ab35 | 2026-07-30 | Normalize MFSDP data-parallel axes at fully_shard (#6067) | 在 fully_shard 入口统一规范化 MFSDP 的 data-parallel mesh 维度。 |
| 9d416eed4 | 2026-07-30 | Rebind MFSDP v2 sharded grads once per gradient reduction (#6041) | MFSDP v2 每次梯度归约只重新绑定一次分片梯度，减少重复操作。 |
| ce8a8a736 | 2026-07-30 | Fix Megatron FSDP optimizer under-update (#5976) | 修复 Megatron-FSDP v1 fully_shard 下 optimizer 未完整更新参数的问题。 |
| efb3963f3 | 2026-07-29 | chore: AUT-1142 bump package versions (#6120) | （间接相关）自动升级 megatron_fsdp 和 megatron core 的包版本号。 |
| 72411e513 | 2026-07-24 | Fix FSDP2 SwiGLU checkpointing. (#5714) | 修复 Torch FSDP2 下 SwiGLU 权重的 checkpoint 保存与加载。 |
| 15d4b349d | 2026-07-23 | Move FSDP model weight sync to optimizer post-step (#5949) | 将 FSDP 模型权重同步移到 optimizer step 之后执行。 |
| d207685b4 | 2026-07-22 | Reuse profiler helpers in mFSDP v2 symmetric memory tests (#5873) | 在 MFSDP v2 symmetric memory 测试中复用已有的 profiler 辅助函数。 |
| 58bf14e9e | 2026-07-21 | Test mFSDP v2 overlap with default and symmetric memory (#5859) | 测试 MFSDP v2 在默认内存和 symmetric memory 两种模式下的通信计算重叠。 |
| 337c061c9 | 2026-07-20 | Add fully_shard_optimizer for mixed-precision FSDP (#5411) | 新增 fully_shard_optimizer 接口，支持混合精度 FSDP 训练。 |
| 5a88c57bd | 2026-07-21 | Support HSDP deferred DP-outer gradient reduction (#5743) | HSDP 支持延迟 DP-outer 维度的梯度归约，在梯度累积时减少跨节点通信。 |
| 53492eb1a | 2026-07-17 | Stabilize mfsdp_v2 overlap test by enlarging the model (#5846) | 加大测试模型规模，使 MFSDP v2 重叠测试结果更稳定。 |
| 4a1f74350 | 2026-07-17 | Avoid extra MFSDP v2 model-weight sync memcpy (#5834) | 去掉 MFSDP v2 模型权重同步时多余的一次 memcpy。 |
| 61f31145d | 2026-07-16 | Allow parameterless FSDP root modules (#5711) | 允许没有参数的 module 作为 MFSDP v2 的 root module。 |
| 9a67ea0ba | 2026-07-16 | test(mfsdp): AUT-881 mark test_overlaps_communication_and_compute flaky (#5848) | 将 test_overlaps_communication_and_compute 测试标记为 flaky。 |
| d981f66be | 2026-07-15 | Overlap FSDP communication with compute (#5719) | MFSDP v2 实现 FSDP 通信与计算的重叠。 |
| 38626c288 | 2026-07-15 | Pair frozen FSDP backward hooks (#5710) | 让冻结（不需要梯度）参数的 FSDP backward hook 成对触发，保证 unshard/reshard 正确配对。 |
| 5e45ccff5 | 2026-07-15 | Avoid FSDP unit terminology in MFSDP v2 (#5793) | 在 MFSDP v2 代码中不再使用 "FSDP unit" 这一术语。 |
| 4bf7fca05 | 2026-07-14 | Fix MegatronFSDP root module hook dispatch (#5808) | 修复 MegatronFSDP 在 root module 上 hook 分发的错误。 |
| f8c9911b2 | 2026-07-14 | Unset NCCL overrides for MFSDP v2 tests (#5794) | MFSDP v2 测试中取消对 NCCL 环境变量的覆盖设置。 |
| a3279434e | 2026-07-14 | Exercise nested MFSDP CUDA graph capture (#5796) | 增加嵌套 MFSDP module 下 CUDA graph 捕获的测试。 |
| 3005e9c1a | 2026-07-13 | Increase Megatron-FSDP overlap test dim to 8192 for reliable overlap (#5770) | 将 Megatron-FSDP 重叠测试的隐藏维度增大到 8192，使通信计算重叠更稳定可测。 |
| ce8865c6c | 2026-07-09 | Add forward all-gather overlap (#5513) | MFSDP v2 支持 forward 阶段 all-gather 与计算重叠。 |
| 328a77fd0 | 2026-07-09 | fix(fsdp): import os in safe_get_rank fallback (#4959) | 修复 megatron_fsdp 中 safe_get_rank 回退分支缺少 import os 的问题。 |
| 91fcdfe4c | 2026-07-08 | Add FSDP NVTX annotations (#5704) | 为 FSDP 的通信与计算阶段添加 NVTX 标注，方便 profiling。 |
| 72a78d62c | 2026-07-08 | Separate mFSDP v2 unit tests (#5640) | 将 MFSDP v2 单元测试拆分到独立目录。 |
| 6ee356a53 | 2026-07-08 | Fix Torch FSDP2 crash: add force_all_reduce kwarg to base finish_grad_sync (#4953) | 为基类 finish_grad_sync 增加 force_all_reduce 参数，修复 Torch FSDP2 调用时崩溃的问题。 |
| 5dbea4617 | 2026-07-07 | Add microbatch context helper (#5652) | 为实验版 fully_shard 增加 microbatch 上下文辅助接口，用于梯度累积时控制梯度同步。 |
| 25f61179c | 2026-07-02 | Add NCCL symmetric-memory staging to experimental FSDP (#5440) | 实验版 FSDP 支持通过 NCCL symmetric memory 缓冲区进行集合通信。 |
| adfe9e11d | 2026-07-01 | [Megatron-FSDP] MaxPoolAllocator for double-buffering hybrid architectures. (#5462) | 为 Megatron-FSDP 新增 MaxPoolAllocator，支持各层大小不一的混合架构使用 double buffer。 |
| f285ea5fa | 2026-06-30 | Fix TEGroupedMLP pre-backward unshard in fine-grained FSDP hooks for … (#4990) | 修复细粒度 FSDP hook 下 TEGroupedMLP 在 backward 前未正确 unshard 的问题。 |
| da42015c8 | 2026-06-25 | Add hybrid FSDP unit module support (#4329) | 支持将混合（hybrid）类型的 module 作为 FSDP unit。 |
| e7af86088 | 2026-06-23 | Add experimental Megatron-FSDP fully_shard implementation (#5387) | 新增实验版 Megatron-FSDP fully_shard 实现（MFSDP v2 的起点）。 |
| 47cb41364 | 2026-06-23 | Remove DBuffer mesh axis validation (#5441) | 移除 DBuffer 对 mesh 维度的校验。 |
| 000dc1c71 | 2026-06-16 | Add minimal DBuffer implementation (#4835) | 新增最小化的 DBuffer（分布式缓冲区）实现。 |
| 55638bc44 | 2026-06-09 | Fix wgrad race condition when using double buffers. (#5222) | 修复 Megatron-FSDP 开启 double buffer 时 wgrad 的竞态条件。 |
| d199bb9e9 | 2026-06-09 | Fix CUDA IMA in fsdp_double_buffer when an FSDP unit's bucket doesn't fit the pool (#4810) | 修复 fsdp_double_buffer 中 FSDP unit 的 bucket 超出内存池大小时出现的 CUDA 非法内存访问。 |
| 3e6e32b50 | 2026-06-08 | Fix bug with Megatron-FSDP zero counter not working with decoupled gradients. (#4802) | 修复 Megatron-FSDP 的 zero counter 在 decoupled gradient 模式下失效的问题。 |
| 7a7f7b413 | 2026-06-03 | Update MINOR version to 19 (#5096) | （间接相关）将 megatron_fsdp 和 megatron core 的 MINOR 版本号升到 19。 |
| 655a06419 | 2026-06-03 | Fix GDN DTensor splitting for FSDP checkpointing (#4843) | 修复 FSDP checkpoint 保存/加载时 GDN 层参数按 DTensor 切分不正确的问题。 |
| 54e199609 | 2026-06-02 | test: update DeepSeek FSDP2 GB200 memory golden (#5094) | 更新 DeepSeek 在 GB200 上 FSDP2 测试的显存 golden 值。 |
| 496e1ff32 | 2026-06-01 | test: skip mfsdp_fully_shard cases when world_size < mesh size (#4487) | 当 world_size 小于 mesh 大小时跳过 mfsdp_fully_shard 相关测试用例。 |
| 378d81fbd | 2026-05-30 | Preserve non-FSDP-unit buckets across AllGatherPipeline reset (#4717) | AllGatherPipeline reset 时保留非 FSDP unit 的 bucket 状态，避免其被错误重置。 |
| 67b2f3878 | 2026-05-27 | [Megatron-FSDP] Add conditional param.grad dereferencing logic to support full-iteration (FWD-BWD) CUDA graphability. (#4663) | 为 Megatron-FSDP 增加按条件解除 param.grad 引用的逻辑，以支持整轮迭代（FWD-BWD）的 CUDA graph 捕获。 |
| 80a2d39c9 | 2026-05-21 | fix no_shard training convergency and add unittest for no_shard (#3754) | 修复 no_shard 策略下训练不收敛的问题并补充对应单测。 |
| 657604004 | 2026-05-21 | Also persist asymmetrical units for the MXFP8 transpose weight buffer. (#4852) | MXFP8 转置权重 buffer 也对非对称 FSDP unit 做常驻保留。 |
| b5d143fe8 | 2026-05-21 | M-FSDP: Make fine_grained_param_gather configurable for MXFP8 to enable performance–memory trade-offs (#4181) | 让 MXFP8 下的 fine_grained_param_gather 可配置，以便在性能与显存之间取舍。 |
| e41b37002 | 2026-05-18 | Refactor CUDA graph API: decompose cuda_graph_scope into full_iteration impl, inference scope, and per-layer capture modules (#4292) | （间接相关）重构 CUDA graph API，将 cuda_graph_scope 拆为 full_iteration、inference scope 和逐层捕获模块，并同步调整 FSDP adapter。 |
| 7371e0134 | 2026-05-17 | add is_torch_min_version in fsdp src (#4812) | 在 Megatron-FSDP 源码中新增 is_torch_min_version 工具函数，用于 PyTorch 版本判断。 |
| e9d9a4c47 | 2026-05-16 | [Main][feat] Support A2A Overlap for Megatron-FSDP (#3797) | 让 Megatron-FSDP 支持 MoE A2A 通信与计算的 overlap。 |
| e35d4e50c | 2026-05-14 | NCCL UB fix: reduce memory cost and correctly deregister NCCL mem pool (#4492) | 修复 NCCL user buffer：降低显存开销并正确注销 NCCL mem pool。 |
| 434368c81 | 2026-05-11 | build(deps): bump nvidia-modelopt to 0.43 (#4723) | （间接相关）将 nvidia-modelopt 依赖升级到 0.43，顺带更新 megatron-fsdp 的 pyproject。 |
| 74687fe09 | 2026-05-12 | fix(fsdp): recognize legacy GDN TP metadata (#4664) | 让 FSDP 能识别旧版 GDN 层的 TP 元数据，保持 checkpoint 兼容。 |
| b23aa3f85 | 2026-04-29 | Unify and refactor Megatron-FSDP documentation. (#4418) | 统一并重构 Megatron-FSDP 文档。 |
| 0447347ce | 2026-04-27 | [Main] Fix FusedAdam.use_decoupled_grad mis-set for Megatron-FSDP. (#4427) | 修复 Megatron-FSDP 下 FusedAdam.use_decoupled_grad 被错误设置的问题。 |
| 384e6183f | 2026-04-22 | Fix: Auto enable manual registration and enhance the docummentation (#3295) | 使用 NCCL UB 时自动开启手动注册模式，并完善 README 文档。 |
| e5ec9ab91 | 2026-04-21 | Megatron-FSDP: log mcore detection only after imports succeed (#4400) | Megatron-FSDP 仅在 mcore 相关 import 成功后才打印检测到 mcore 的日志。 |
| 97f9ab6f2 | 2026-04-16 | Set megatron-fsdp to 0.5.0 | 将 megatron-fsdp 包版本号设为 0.5.0。 |
| 5f80f0ac6 | 2026-04-13 | Reset AG_pipeline bucket status after validation step. (#3155) | 在验证步骤后重置 all-gather pipeline 的 bucket 状态。 |
| 25129bf3d | 2026-04-14 | fix(megatron-fsdp): build expt_device_mesh only for MoE models (#3831) | 仅在 MoE 模型时构建 expert device mesh（expt_device_mesh）。 |
| ab43d43f0 | 2026-04-10 | Fix incorrectly set decoupled_grad and DistOpt mechanics for MFSDP. (#4133) | 修复 MFSDP 中 decoupled_grad 设置错误及与 DistOpt 机制相关的问题。 |
| 2ebfbb21f | 2026-04-10 | Enable AG/RS overlap with explicit process group passing (#3249) | 通过显式传入 process group 实现 all-gather 与 reduce-scatter 的 overlap。 |
| 8cbc45b6e | 2026-04-08 | [M-FSDP] Fix Tensor Parallel mode detection (#3191) | 修复 M-FSDP 对 Tensor Parallel 模式的检测逻辑。 |
| 76e4daa40 | 2026-04-04 | Fix FSDP checkpoint conversion and loading for Qwen3.5-VL (#3936) | 修复 Qwen3.5-VL 的 FSDP checkpoint 转换与加载问题。 |
| 69f3b3400 | 2026-04-03 | add support for DCP and FSDP async save (#4027) | 新增 DCP 格式与 FSDP 的异步 checkpoint 保存支持。 |
| 159e34720 | 2026-04-02 | [M-FSDP] Refactor uneven dtensor to full tensor and add UT (#3190) | 重构 uneven DTensor 转 full tensor 的逻辑并补充单测。 |
| 150e37ac5 | 2026-04-02 | [Megatron-FSDP] Fix compatibility with frozen parameters and add unit tests (#3287) | 修复 Megatron-FSDP 与冻结参数（requires_grad=False）的兼容性并补充单测。 |
| a52ceeb5c | 2026-04-02 | FIX: Use decoupled gradients for precision-aware M-FSDP grad norm (#3746) | precision-aware 优化器下 M-FSDP 计算 grad norm 改用 decoupled gradient。 |
| 41f3b6f30 | 2026-04-02 | Fix M-FSDP MXFP8 related BUGs (#3991) | 修复 M-FSDP 中与 MXFP8 相关的若干 bug。 |
| 3dc225122 | 2026-04-02 | Megatron-FSDP: Fix insufficient double buffers during gradient reduce (#4054) | 修复梯度 reduce 过程中 double buffer 数量不足的问题。 |
| 748ac49a2 | 2026-04-02 | Megatron-FSDP: Make _pre_forward_param_unshard and _register_post_backward_hook formal (#4029) | 将 _pre_forward_param_unshard 与 _register_post_backward_hook 规范化为正式接口。 |
| 8d7a3f8d6 | 2026-04-02 | feat(fsdp): use TE general_gemm for mixed-precision wgrad in FSDP path (#3822) | FSDP 路径下混合精度 wgrad 计算改用 TE 的 general_gemm。 |
| 606ac26ea | 2026-04-02 | Megatron-FSDP: Add MXFP8 transpose helper buffer for Hybrid FSDP (#3918) | 为 Hybrid FSDP（HSDP）新增 MXFP8 转置辅助 buffer。 |
| fe5291fa4 | 2026-04-01 | m-fsdp: wire use_precision_aware_optimizer from ddp_config to ParamAn… (#4024) | 将 ddp_config 中的 use_precision_aware_optimizer 传递给 ParamAndGradBuffer。 |
| 1284d25e1 | 2026-04-02 | Fix fsdp_dtensor conversion for pretrained-only checkpoints (#3912) | 修复只含预训练权重（无优化器状态）的 checkpoint 转换为 fsdp_dtensor 格式时的问题。 |
| 3499efef8 | 2026-04-02 | Modify mfsdp default data-parallel-sharding-strategy (#3691) | 修改 MFSDP 默认的 data-parallel-sharding-strategy。 |
| 97e36aaba | 2026-04-01 | [Main][feat] Support overlapping A2A Combine backprop with wgrad GEMM (#3795) | 支持 MoE A2A Combine 反向与 wgrad GEMM overlap（延迟 wgrad 计算），并适配 Megatron-FSDP 的梯度处理。 |
| 947bdbbd2 | 2026-03-23 | Pass Megatron-FSDP MixedPrecision args to DDPConfig. (#3992) | 将 Megatron-FSDP 的 MixedPrecision 参数传入 DDPConfig。 |
| 3f59f7170 | 2026-03-20 | Make args and kwargs optional positional arguments for the Module hooks. (#3976) | Megatron-FSDP 的 Module hook 中将 args 与 kwargs 改为可选位置参数。 |
| f45619970 | 2026-03-19 | Move Megatron-FSDP MixedPrecisionPolicy arguments from FSDP adapter t… (#3903) | 将 Megatron-FSDP 的 MixedPrecisionPolicy 参数从 FSDP adapter 移到 DistributedDataParallelConfig 中统一管理。 |
| ff70b2488 | 2026-03-16 | [Megatron-FSDP] Support 'auto' argument which defaults to pre-MixedPrecisionPolicy be… (#3810) | 为 Megatron-FSDP 精度参数新增 'auto' 取值，默认沿用引入 MixedPrecisionPolicy 之前的行为。 |
| 9a494f185 | 2026-03-16 | Fsdp dsv3 proxy (#3844) | 新增 DeepSeek-V3 proxy 模型在 GB200 上的 FSDP 功能测试配置与 golden 值。 |
| d99782091 | 2026-03-12 | Fix backward compatibility issue with MFSDP `--grad-reduce-in-bf16` (#3799) | 修复 MFSDP 中 `--grad-reduce-in-bf16` 参数的向后兼容问题。 |
| 7d526940a | 2026-03-11 | fix: skip FSDP DTensor boundary validation under fake process group (#3686) | 在 fake process group 下跳过 FSDP DTensor 边界校验。 |
| b969f7642 | 2026-03-02 | [Megatron-FSDP] Add dtype customization to Megatron-FSDP. (#3067) | 为 Megatron-FSDP 增加参数、梯度、通信等 dtype 的自定义能力。 |
| e8fe06865 | 2026-03-02 | M-FSDP: Cancel erroneous grad accumulation check (#3629) | 移除 M-FSDP 中错误的梯度累积检查。 |
| 310082a6d | 2026-02-27 | μP: Maximal Update Parameterization  (#3058) | （间接相关）新增 μP（Maximal Update Parameterization）支持，仅在 Megatron-FSDP 的 param_and_grad_buffer 中做一行适配。 |
| e7d80bd26 | 2026-02-25 | ci: Test docs build (#3583) | （间接相关）新增文档构建的 CI 测试，并修正包括 Megatron-FSDP 用户指南在内的文档问题。 |
| 773c11391 | 2026-02-20 | Fix Megatron-FSDP optimizer state DCP checkpointing, and fix DTensor deepcopy bug from PyTorch 26.01. (#3510) | 修复 Megatron-FSDP 优化器状态的 DCP checkpoint 问题，以及 PyTorch 26.01 中 DTensor deepcopy 的 bug。 |
| fdde15e96 | 2026-02-19 | [Megatron-FSDP] Add fsdp_all_gather_in_start_param_sync option in DDP Config (#3095) | 在 DDP Config 中新增 fsdp_all_gather_in_start_param_sync 选项，控制是否在 start_param_sync 中发起 all-gather。 |
| 7df15bd58 | 2026-02-12 | Add Qwen3-VL support with Megatron-FSDP (#2841) | 新增 Qwen3-VL 模型在 Megatron-FSDP 下的训练支持。 |
| 30695913b | 2026-02-09 | General README and pyproject fixes (#2907) | （间接相关）修正 README 与 pyproject，包括 megatron-fsdp 子包的 pyproject。 |
| 4ff7686c1 | 2026-02-09 | M-FSDP: Remove redundant stream waits in HSDP to prevent CG fail (#2941) | 移除 HSDP 中多余的 stream wait，避免 CUDA graph 捕获失败。 |
| 7cbbba2e6 | 2026-02-08 | Add check to prevent MFSDP from numeric issue in gradient accumulate fusion  (#2904) | 增加检查，防止 MFSDP 在 gradient accumulation fusion 下出现数值问题。 |
| 473e283cc | 2026-02-04 | Support EP with HSDP (#2840) | 支持 Expert Parallel 与 HSDP 组合使用。 |
| afe443bc4 | 2026-02-02 | Use global user buffer when the bucket size does not fit FixedPoolAllocator (#2857) | bucket 大小超出 FixedPoolAllocator 容量时回退使用全局 NCCL user buffer。 |
| b6b49e7e6 | 2026-01-28 | [M-FSDP] Fix double buffering not working with activation recompute (#2689) | 修复开启 activation recompute 时 M-FSDP double buffering 失效的问题。 |
| 964c902b4 | 2026-01-28 | Add end-to-end tests for M-FSDP and ND-Parallel (#3031) | 为 M-FSDP 与 ND-Parallel 组合新增端到端测试。 |
| 6cf285b23 | 2026-01-27 | Logging cleanup (only log on rank 0 if possible) (#3036) | （间接相关）全局日志清理，尽量只在 rank 0 打印，涉及 Megatron-FSDP 的 param_and_grad_buffer 日志。 |
| 528cb2e52 | 2026-01-26 | add all_gather process-group for overlapping in fsdp disributed training (#2663) | 为 FSDP 训练新增独立的 all-gather process group，以实现通信 overlap。 |
| bb42a0081 | 2026-01-27 | fix(fsdp): add CLI argument for outer_dp_sharding_strategy (#3053) | 为 HSDP 的 outer_dp_sharding_strategy 新增命令行参数。 |
| 3a7d74dac | 2026-01-24 | Keep FSDP's and DDP's finish_grad_sync API identical (#3070) | 统一 FSDP 与 DDP 的 finish_grad_sync 接口签名。 |
| ba456fdad | 2026-01-21 | [Megatron-FSDP] Fix incorrect gradient scaling target. (#3023) | 修复 Megatron-FSDP 中梯度缩放作用对象错误的问题。 |
| 517dfd435 | 2026-01-19 | [Megatron-FSDP] Test FP8 activations + parameter sharding with Megatron-FSDP fully-shard. Update README. (#2894) | 新增 Megatron-FSDP fully-shard 下 FP8 激活与参数分片的测试，并更新 README。 |
| 35129e71f | 2026-01-20 | Make Megatron-FSDP torch.compile compatible (#2425) | 让 Megatron-FSDP 兼容 torch.compile。 |
| 55103f9a5 | 2026-01-16 | docs(megatron-fsdp): add Megatron-FSDP user guide (#2396) | 新增 Megatron-FSDP 用户指南文档。 |
| 409ebf18b | 2026-01-14 | Revert "[dev] Add assertion for mxfp8 params without dp overlap (#2270)" (#2901) | 回滚 dev 分支上“MXFP8 参数未开启 DP overlap 时报错”的断言。 |
| ed29157a4 | 2026-01-08 | docs: fix broken images, links, and typos across documentation (#2794) | （间接相关）修复文档中失效的图片、链接与拼写错误，包括 FSDP 相关文档。 |
| 8f2f70067 | 2026-01-09 | FP8 params support for megatron-fsdp (MXFP8/Blockwise) (#2239) | 为 Megatron-FSDP 增加 FP8 参数支持（MXFP8 与 Blockwise）。 |
| 278be1535 | 2026-01-05 | DeepSeek V3 FSDP Fix for Precision-Aware Optimizer (#2466) | 修复 DeepSeek V3 在 FSDP 下使用 precision-aware 优化器时的问题。 |
| 876a04689 | 2025-12-23 | feat: manual registration mode for nccl-ub option when using megatron-fsdp (#2661) | 为 Megatron-FSDP 的 nccl-ub 选项新增手动注册模式。 |
| cc1b0b5cf | 2025-12-23 | Hsdp register submesh fix lifuz mirror (#2467) | 修复 HSDP 下 submesh 注册的问题。 |
| 732bb8d68 | 2025-12-16 | [Megatron-FSDP] Build default FSDP DeviceMesh, and remove model arg from fully_shard_optimizer(). (#2471) | 自动构建默认的 FSDP DeviceMesh，并移除 fully_shard_optimizer() 的 model 参数。 |
| 597e88a01 | 2025-12-15 | [Megatron-FSDP] Support both old and new DeviceMesh APIs. (#2575) | 让 Megatron-FSDP 同时兼容新旧版本的 DeviceMesh API。 |
| dcee1c581 | 2025-12-05 | fix: publish release wheel and github release version number (#2561) | （间接相关）修复 release wheel 发布流程与版本号，涉及 megatron-fsdp 的 package_info。 |
| e79d9a895 | 2025-12-05 | ci: Add release workflows (#2507) | （间接相关）新增 release CI workflow，并更新 megatron-fsdp 版本号。 |
| 3cacd5baf | 2025-12-03 | Add assertion for mxfp8 params without dp overlap (#2271) | 增加断言：MXFP8 参数必须配合 DP 通信 overlap 使用。 |
| ebb2e9119 | 2025-11-26 | Various small fixes for Megatron-FSDP. (#2346) | 对 Megatron-FSDP 的 uneven_dtensor、utils 和 fully_shard 测试做若干小修复。 |
| aa4ec9920 | 2025-11-13 | [Megatron-FSDP] Fix hang caused by non-deterministic reduce-scatter (#2218) | 修复 reduce-scatter 顺序不确定导致的 hang。 |
| 909c74631 | 2025-11-12 | Revert "remove `megatron.training` dependency from `megatron.core` for FSDP checkpoint with EP (#2113)" | 回滚“EP 下 FSDP checkpoint 去除 megatron.core 对 megatron.training 依赖”的改动。 |
| 45b40bbde | 2025-11-12 | remove `megatron.training` dependency from `megatron.core` for FSDP checkpoint with EP (#2113) | 移除 EP 场景下 FSDP checkpoint 代码中 megatron.core 对 megatron.training 的依赖。 |
| d82a6d825 | 2025-11-12 | Fix Megatron-FSDP checkpoint save failure (#2138) | 修复 Megatron-FSDP 保存 checkpoint 时失败的问题。 |
| c5ac86354 | 2025-10-28 | Update symmetric registration interface to sync-up with upstream pytorch change (#1924) | 跟进上游 PyTorch 接口变更，更新 Megatron-FSDP 与 nccl_allocator 中的 NCCL symmetric memory 注册接口。 |
| e36a8758f | 2025-10-27 | Remove M-FSDP testing under LTS environment (#1959) | 从 MoE 测试 recipe 中移除 LTS 环境下的 M-FSDP 测试用例。 |
| 268fda085 | 2025-10-22 | ADLR/megatron-lm!3955 - Megatron-FSDP Expert Parallel (DeepSeek-v3) Support | 为 Megatron-FSDP 增加 Expert Parallel 支持，可用于 DeepSeek-v3 等 MoE 模型训练。 |
| 4666de7a2 | 2025-10-21 | chore: Version bump 0.16.0 | 将 megatron-core 与 megatron_fsdp 包版本号升级到 0.16.0。 |
| 0ba847081 | 2025-10-15 | ADLR/megatron-lm!4226 - Use up-to-date model state dictionary when pre-processing the state dict for DCP. | fully_shard 在为 DCP 预处理 state dict 时改用最新的模型 state dict，避免使用过期参数。 |
| 2c1044266 | 2025-10-15 | ADLR/megatron-lm!4208 - Added attribute to support latest TE | 为 FSDP 参数设置 overwrite_main_grad 属性，使新版 TE 在 gradient accumulation fusion 下覆盖而非累加 main_grad。 |
| f3452948b | 2025-10-06 | chore: Version bump | 更新 megatron-core 与 megatron_fsdp 包的版本号。 |
| e65669313 | 2025-09-29 | chore: Version bump | 更新 megatron-core 与 megatron_fsdp 包的版本号。 |
| 03045f2d8 | 2025-09-26 | ADLR/megatron-lm!4062 - Fix mis-set of model_auto_sync and add basic gradient existence assertion to fully_shard tests. | 修复 model_auto_sync 设置错误，并在 fully_shard 单测中增加梯度存在性断言。 |
| cabf5d1a3 | 2025-09-25 | ADLR/megatron-lm!4047 - fix(FSDP): avoid redundant meta device materialization for Megatron FSDP | 避免 Megatron-FSDP 对 meta device 上的模型重复 materialize。 |
| d91daa0cd | 2025-09-23 | ADLR/megatron-lm!4039 - Added average in collective support | Megatron-FSDP 支持 average_in_collective，按 expert DP 与 DP 大小之比设置专家梯度缩放系数。 |
| dd7e13150 | 2025-09-23 | ADLR/megatron-lm!4038 - Add support for gradient accumulation fusion | 为 Megatron-FSDP 增加 gradient accumulation fusion 支持。 |
| c78fb0876 | 2025-09-22 | chore: Version bump | 更新 megatron-core 与 megatron_fsdp 包的版本号。 |
| 199113bfa | 2025-09-18 | ADLR/megatron-lm!3879 - Add unit tests, and refactor fully_shard into fully_shard_model and fully_shard_optimizer. | 将 fully_shard 拆分为 fully_shard_model 与 fully_shard_optimizer 两个接口，并补充单元测试。 |
| 9f72f4775 | 2025-09-18 | ADLR/megatron-lm!4018 - bugfix: Added support for FSDP grad accum fusion | 修复 FSDP 下 gradient accumulation fusion 的支持问题。 |
| 2fbece51e | 2025-09-15 | chore: Version bump | 更新 megatron-core 与 megatron_fsdp 包的版本号。 |
| ba97a7e28 | 2025-09-08 | chore: Version bump | 更新 megatron-core 与 megatron_fsdp 包的版本号。 |
| b615e7310 | 2025-09-07 | Merge branch 'yuya/m4_merge_into_pg_collection' into 'main' | （间接相关）合并 !3962：将 model_comm_pgs、grad_comm_pgs、grad_finalize_pgs 统一为 pgs_collection，涉及 mcore_fsdp_adapter 的适配。 |
| 8c1a3f5df | 2025-09-07 | ADLR/megatron-lm!3962 - Replay (!3892) M4: Merge `model_comm_pgs`, `grad_comm_pgs` and `grad_finalize_pgs` into `pgs_collection` and Custom PG hot fix for GPT model | （间接相关）重新提交 !3892，把多个通信 process group 集合合并为 pgs_collection，并修复 GPT 模型自定义 PG 问题。 |
| 2108cd8bb | 2025-09-06 | Merge branch 'boxiangw/megatron-fsdp-fix' into 'main' | 合并 !3964：修复 Megatron-FSDP 日志输出并移除多余配置。 |
| 696977fc3 | 2025-09-06 | ADLR/megatron-lm!3964 - Fix megatron-fsdp logging and remove extra config | 修复 Megatron-FSDP 的日志问题并删除多余的配置项。 |
| ca9797e95 | 2025-09-04 | Revert "ADLR/megatron-lm!3892 - M4: Merge `model_comm_pgs`, `grad_comm_pgs` and `grad_finalize_pgs` into `pgs_collection` and Custom PG hot fix for GPT model" | （间接相关）回滚 !3892 中 process group 集合合并为 pgs_collection 的改动。 |
| 76622edf3 | 2025-09-04 | ADLR/megatron-lm!3892 - M4: Merge `model_comm_pgs`, `grad_comm_pgs` and `grad_finalize_pgs` into `pgs_collection` and Custom PG hot fix for GPT model | （间接相关）将 model_comm_pgs、grad_comm_pgs、grad_finalize_pgs 合并为 pgs_collection，并修复 GPT 模型自定义 PG（后被回滚）。 |
| 1e9e94cc6 | 2025-09-03 | ADLR/megatron-lm!3881 - Megatron-FSDP NCCL symmetric register support | 为 Megatron-FSDP 的参数与梯度 buffer 增加 NCCL symmetric memory 注册支持。 |
| 2c7e98a4d | 2025-09-03 | Merge branch 'nvfsdp-symmetric' into 'main' | 合并 !3881：Megatron-FSDP NCCL symmetric 注册支持。 |
| d8714d902 | 2025-09-02 | chore: Version bump | 更新 megatron_fsdp 包的版本号。 |
| 1d0995d00 | 2025-08-28 | Merge branch 'cye/fix-fsdp-dist-shape' into 'main' | 合并 !3877：修复维度大于 2 的参数在 FSDP 中的分布式权重 shape。 |
| b74396f14 | 2025-08-28 | ADLR/megatron-lm!3877 - Fix FSDP distributed parameter weight shapes for paramaters > 2-D. | 修复维度大于 2 的参数在 FSDP 分布式参数（DTensor）中 shape 计算错误的问题。 |
| dcf7d3695 | 2025-08-25 | Merge branch 'ko3n1g/ci/push-megatron-fsdp' into 'main' | 合并 !3875：CI 自动发布 megatron-fsdp 包。 |
| 875ad2a6a | 2025-08-25 | ADLR/megatron-lm!3875 - ci: Auto-publish megatron-fsdp | 在 CI 中增加 megatron-fsdp 独立包的自动发布流程。 |
| 4dd2f2bc3 | 2025-08-21 | Merge branch 'nvfsdp_convergence' into 'main' | 合并 !3443：将 Custom FSDP 解耦为可独立安装的包。 |
| af28b5a55 | 2025-08-21 | ADLR/megatron-lm!3443 - [FSDP] Decouple Custom FSDP to make it independently installable | 将 Custom FSDP 从 Megatron-Core 解耦为可独立安装的 megatron_fsdp 包（即 Megatron-FSDP）。 |
| af962d8f1 | 2025-08-01 | Merge branch 'jianbinc/fix_cfsdp_doc_typo' into 'main' | 合并 !3752：修正 custom_fsdp.md 文档拼写错误。 |
| b0ccc17ed | 2025-08-01 | ADLR/megatron-lm!3752 - docs: fix typo in custom_fsdp.md | 修正 custom_fsdp.md 文档中的拼写错误。 |
| 0a1fdc1f2 | 2025-07-24 | Merge branch 'jianbinc/fix_cfsdp_conflict_w_act_offload' into 'main' | 合并 !3488：使 custom-fsdp 兼容 TE activation offloading。 |
| 0cf180a70 | 2025-07-24 | ADLR/megatron-lm!3488 - [FSDP] make custom-fsdp compatible with TE activation offloading | 解决 custom-fsdp 与 TE activation offloading 的冲突，使二者可同时使用。 |
| 44fa0ea40 | 2025-07-13 | Merge branch 'boxiangw/fsdp2-te2-fp8-warning' into 'main' | 合并 !3338：FSDP2 搭配 fp8-param-gather 时增加 TE 2.0 版本检查。 |
| d2c22105f | 2025-07-13 | ADLR/megatron-lm!3338 - Add TE 2.0 check for FSDP2 with fp8-param-gather | 在 PyTorch FSDP2 开启 fp8-param-gather 时检查 TE 版本是否不低于 2.0。 |
| 6f6968f07 | 2025-07-02 | Merge branch 'fix_fsdp_double_buffer' into 'main' | 合并 !3545：修复 FSDP double buffer 问题。 |
| 6d1e2d762 | 2025-07-02 | ADLR/megatron-lm!3545 - Fix FSDP-double-buffer | 修复 FSDP double buffer 功能中的问题。 |
| 5e0e2c7e0 | 2025-06-30 | Merge branch 'cye/fsdp-vp-stage-fix' into 'main' | 合并 !3440：为 FSDP wrapper 增加 vp_stage 属性。 |
| 7fd003f3d | 2025-06-30 | ADLR/megatron-lm!3440 - Add vp_stage attr to FSDP wrapper. | 为 FSDP wrapper 增加 vp_stage 属性，以支持 virtual pipeline 场景。 |
| aad967ffa | 2025-06-16 | Merge branch 'custom_fsdp_hsdp_support' into 'main' | 合并 !3081：为 Custom-FSDP 增加 HSDP 支持。 |
| 6dc605043 | 2025-06-16 | ADLR/megatron-lm!3081 - Add Hybrid Shard Data-Parallel Support for Custom-FSDP | 为 Custom-FSDP 增加 Hybrid Shard Data-Parallel（HSDP）支持。 |
| 549d63735 | 2025-06-02 | Merge branch 'fix_cfsdp_fp8_param_load' into 'main' | 合并 !3280：修复 custom FSDP 中 float8 tensor 的 set_item 问题。 |
| da3f0ffb7 | 2025-06-02 | ADLR/megatron-lm!3280 - Fix custom FSDP float8 tensor set_item | 修复 custom FSDP 对 float8 tensor 执行 set_item 时的错误，解决 FP8 参数加载问题。 |
| 20304a309 | 2025-05-22 | Merge branch 'fsdp-ubr' into 'main' | 合并 !2902：为 MCore-FSDP 增加 userbuffer 注册。 |
| 40df28b5c | 2025-05-22 | ADLR/megatron-lm!2902 - Userbuffer registration for MCore-FSDP | 为 MCore-FSDP 的通信 buffer 增加 NCCL userbuffer 注册支持。 |
| 1ff5a3789 | 2025-05-16 | ADLR/megatron-lm!3116 - Move fsdp 2 import from _composable to public | 将 PyTorch FSDP2 的导入路径从 torch.distributed._composable 改为公开 API。 |
| ed0d528d5 | 2025-05-16 | Merge branch 'boxiangw/public_fsdp_import' into 'main' | 合并 !3116：FSDP2 改用公开导入路径。 |
| 4429e8ebe | 2025-04-26 | Revert "Revert "ADLR/megatron-lm!2581 - Add support for ZeRO-2 with PyTorch FSDP2"" | 重新引入此前被回滚的 PyTorch FSDP2 ZeRO-2 支持。 |
| f7a25e5ac | 2025-04-25 | Merge branch 'custom_fsdp_dsv3' into 'main' | 合并 !2910：custom FSDP 支持 DeepSeek-v3 的 EP + FSDP 训练。 |
| 5fe1eeb83 | 2025-04-25 | ADLR/megatron-lm!2910 - [custom FSDP] Support EP + FSDP training for DeepSeek-v3 | 让 custom FSDP 支持 Expert Parallel 与 FSDP 组合训练 DeepSeek-v3。 |
| 1eaed211f | 2025-04-23 | Revert "ADLR/megatron-lm!2581 - Add support for ZeRO-2 with PyTorch FSDP2" | 回滚 PyTorch FSDP2 的 ZeRO-2 支持。 |
| 1cb506c28 | 2025-04-22 | Merge branch 'torch_fsdp_zero2' into 'main' | 合并 !2581：PyTorch FSDP2 支持 ZeRO-2。 |
| 0ad06b525 | 2025-04-22 | ADLR/megatron-lm!2581 - Add support for ZeRO-2 with PyTorch FSDP2 | 为 PyTorch FSDP2 增加 ZeRO-2（仅切分梯度与优化器状态）模式支持。 |
| 76d6bcf0f | 2025-04-16 | Merge branch 'fp8_patch_for_cfsdp' into 'main' | 合并 !3057：custom FSDP 的 FP8 primary weight 支持补丁。 |
| b799e3fa0 | 2025-04-16 | ADLR/megatron-lm!3057 - patch for fp8 primary weight custom fsdp support | 修补 custom FSDP 对 FP8 primary weight 的支持。 |
| 9ad5992f7 | 2025-04-11 | Merge branch 'yash/refactor_custom_fsdp' into 'main' | 合并 !3012：custom FSDP 改用 GradCommProcessGroups。 |
| 9d083ad6f | 2025-04-11 | ADLR/megatron-lm!3012 - Custom FSDP to use GradCommProcessGroups with fallback to global PG from parallel state | custom FSDP 改为使用 GradCommProcessGroups 传入通信组，未传入时回退到 parallel_state 的全局 PG。 |
| 31e857895 | 2025-04-10 | Revert "Revert "ADLR/megatron-lm!2786 - [custom FSDP] add aggregate communication and some modifications for better use in NeMo"" | 重新引入此前被回滚的 custom FSDP 聚合通信与 NeMo 适配改动。 |
| 3497231dc | 2025-04-10 | Revert "ADLR/megatron-lm!2786 - [custom FSDP] add aggregate communication and some modifications for better use in NeMo" | 回滚 custom FSDP 聚合通信与 NeMo 适配改动。 |
| 4f6fe2bd1 | 2025-04-10 | Merge branch 'fix_golden_value_for_cfsdp' into 'main' | 合并 !3059：更新 custom FSDP 测试的 golden values。 |
| a8aab72a9 | 2025-04-09 | Merge branch 'custom_fsdp_aggregate_comm' into 'main' | 合并 !2786：custom FSDP 聚合通信及 NeMo 适配。 |
| 9a5a6656e | 2025-04-09 | ADLR/megatron-lm!2786 - [custom FSDP] add aggregate communication and some modifications for better use in NeMo | 为 custom FSDP 增加聚合通信（合并多个 bucket 的 all-gather/reduce-scatter），并做了便于 NeMo 使用的修改。 |
| 85a54471d | 2025-03-31 | Merge branch 'yash/fsdp_pg_refactor' into 'main' | 合并 !2943：torch FSDP 构造函数的 process group 参数改为可选。 |
| 2683da02e | 2025-03-31 | ADLR/megatron-lm!2943 - Make process group an optional argument for torch FSDP class constructor | 将 torch FSDP 封装类构造函数中的 process group 改为可选参数。 |
| 2dc00235d | 2025-03-15 | Merge branch 'trintamaki/torch_dcp' into 'main' | 合并 !2757：FSDP 支持 Torch distributed checkpointing。 |
| cf3163843 | 2025-03-15 | ADLR/megatron-lm!2757 - Torch distributed checkpointing support for FSDP | 为 FSDP 增加 Torch distributed checkpointing（DCP）的保存与加载支持。 |
| 12bce5b4b | 2025-03-11 | ADLR/megatron-lm!2765 - Configurable FSDP modules | 支持配置 torch FSDP2 要包装的子模块，便于 VLM 和 HF 模型使用。 |
| 311bb1861 | 2025-03-11 | Merge branch 'trintamaki/vlm-hf-fsdp-modules' into 'main' | 合并 !2765：FSDP 包装模块可配置。 |
| 08c53e9cf | 2025-03-06 | Merge branch 'fsdp2-fp8ag' into 'main' | 合并 !2446：调整 torch FSDP2 FP8 all-gather 的参数检查条件。 |
| 7ef4f9038 | 2025-03-06 | ADLR/megatron-lm!2446 - Required argument condition changes for torch fsdp2 fp8 all-gather | 调整 torch FSDP2 开启 FP8 all-gather 时的必需参数检查条件。 |
| 892c4c854 | 2025-02-26 | Merge branch 'distopt_extended_fsdp' into 'main' | 合并 !2071：引入基于 distributed optimizer 的 MCore Custom FSDP。 |
| d165a8548 | 2025-02-26 | ADLR/megatron-lm!2071 - MCore Customized FSDP (Distopt-based FSDP) | 新增基于 distributed optimizer 扩展的 MCore Custom FSDP 实现（Megatron-FSDP 的前身）。 |
| 2224b04cd | 2025-02-24 | ADLR/megatron-lm!2570 - Add CPU init support for FSDP2 | 为 PyTorch FSDP2 增加在 CPU 上初始化模型的支持。 |
| 3a654fc47 | 2025-02-24 | Merge branch 'boxiangw/fsdp2-cpu-init' into 'main' | 合并 !2570：FSDP2 支持 CPU 初始化。 |
| 7debdd589 | 2025-02-10 | Merge branch 'trintamaki/vlm-fsdp-workaround' into 'main' | 合并 !2606：VLM 在 FSDP 下的临时规避方案。 |
| 062681d3d | 2025-02-10 | ADLR/megatron-lm!2606 - VLM FSDP workaround | 为 LLaVA 等 VLM 模型在 FSDP 下训练加入临时规避方案，修改 multimodal 示例与 llava_model。 |
| 6219d96ef | 2025-02-06 | Merge branch 'fsdp2_fp8_cache' into 'main' | 合并 !2508：使用 torch FSDP2 时禁用 FP8 transpose cache。 |
| 6213cff61 | 2025-02-06 | ADLR/megatron-lm!2508 - Disable the FP8 transpose cache when using torch FSDP2 | 使用 torch FSDP2 时禁用 TE 的 FP8 transpose cache。 |
| 4c4215fd7 | 2024-11-13 | Merge branch 'boxiangw/fsdp2' into 'main' | 合并 !2150：支持 PyTorch FSDP-2。 |
| e1993fa6f | 2024-11-13 | ADLR/megatron-lm!2150 - Add support for PyTorch FSDP-2 | 为 Megatron-LM 增加 PyTorch FSDP-2（fully_shard）训练支持。 |

## 三、Paddle FSDP 现状与下一步优化点

检查对象：`Paddle/` HEAD `abadcba928`（[FSDP] Fix FSDP adamh of mixed_opt），主实现 `python/paddle/distributed/fsdp/fully_shard_fusion.py`（下文行号均指此文件），对照用户配置 `ernie5_5_flash_ladder_v90_T4_..._fsdp.yaml`（EP8、gradient_accumulation_steps=16、selective recompute、expert fp8）。

### 已具备的能力

按 dtype / unit / expert / decay / keep_whole 等分组的 fused buffer；按层或按大小（`comm_unit_size`）prefetch；`reshard_policy`（after_forward / after_backward / after_step）与 `reshard_filter`；`grad_comm_dtype`；EP 下 expert 用独立 sharding 组；fp8 先在 shard 上量化再 all-gather；Muon / Hyperball / MixedInnerOptimizer 的 keep_whole 路径；flex_checkpoint 分片保存，不需要 all-gather。

### 建议优先做（P0）

**1. 步与步之间的空泡：每步一次全设备同步，首个 unit 的 all-gather 无法 overlap** （没有收益）
- `comm_sync_and_reset_status()` 末尾调 `paddle.device.synchronize()`（1972 行），CPU 要等 GPU 全部跑完才能下发 optimizer kernel。
- `reset_params_buffer_status()`（1104-1125 行）会丢弃 backward 末尾预取的结果，下一步 forward 的 embedding 和第一个 unit 只能同步 all-gather。
- Megatron 做法：`fsdp_all_gather_in_start_param_sync`，以及 #5985 在 optimizer step 后立即预取下一步的 AG。
- 建议：把全设备同步换成 compute stream 对各 comm stream 的 `wait_stream` 再释放 buffer；optimizer step 结束后立刻对执行顺序上的前 1~2 个 unit 发异步 AG。

**2. 非 unit 参数组（embedding、final norm、lm_head 等）的梯度 RS 全部堆在 backward 末尾串行执行** （完成✅）
- dense 和 expert 的 RS 都只在 `_flush_expert_grads_after_unit()` 中触发（1217-1227 行），触发时机是 backward 进入前一个 unit，且只处理 `fsdp_unit_id is not None` 的组。
- `fsdp_unit_id=None` 的组和最后一个 unit 都在 `finish_grads_sync()`（1293-1336 行）里逐组 `do_reduce_scatter().wait()`，不和计算 overlap。
- lm_head 的梯度在 backward 一开始就算完，但它整份 fp32 的未分片 tmp grad buffer 要一直占到 backward 结束（h=1280 时约 0.5GB，h=8192 时约 3.4GB）。
- `grads_use_cnt` / `grads_use_sum` 已经在计数（646、1183 行），但没有用来触发 RS。Megatron v2 用 `grad_accumulation_count` 判断一个 unit 的梯度何时就绪（含 delayed wgrad，#7854）。
- 建议：某组的梯度回调次数到齐就立即异步发 RS；或者把 lm_head / embedding 单独设为 unit。

**3. 缺少 HSDP（节点内 shard、节点间 replicate 或只分片 optimizer state）** （同事在做）
- `FSDPBufferManager` 只用 sharding group 和 moe_sharding group（431-440 行），没有 DP-outer 维度。
- 当前配置 ga=16 且默认 `sync_grads_each_microbatch=True`（753 行），每步要做 16 次 RS，sharding group 跨节点时每次都走 IB。
- Megatron 方案：HSDP / HFSDP（#5743 #5972 #6773）。节点内 NVLink 上做 AG/RS；DP-outer 归约只在最后一个 micro-batch 做，并放在独立的流上（#6973）；DP-outer 的 AG 在 optimizer step 后预取（#5985）。
- 多机扩展时这一项收益最大，工作量也最大，需要改 buffer 分组、grad scale 和 checkpoint 描述。

### 其次（P1）

**4. AG 与 RS 共用一条通信流**
- 同一个 fsdp_group 的 params_buffer 和 grads_buffer 共用 `_comm_stream(group)`（597、619、653-658 行）。
- backward 进入 unit k 时，先发 unit k+1 的 RS（990 → 1210 行），再发 unit k-1 的预取 AG（1071-1084 行）。AG 排在 RS 后面，而 RS 是 fp32，字节数是 bf16 AG 的 2 倍，所以 unit k-1 的权重要等 RS 跑完才开始传。
- Megatron v2 默认 `overlap_allgather_reduction=True`，AG 和 RS 各用一条流（`experimental/context.py`）；#6563 又加了可选的合并单流，可以省一份临时 buffer（峰值从 4 份 full buffer 降到 3 份）。
- 建议：做成开关，用 nsys 对比。要注意同一个 NCCL communicator 上的 kernel 会被串行，RS 想真正和 AG 并发，可能需要单独的通信组。把 `grad_comm_dtype` 设为 bf16 可以先把 RS 量减半。

**5. prefetch 顺序是 buffer 组下标的静态顺序，不是实际执行顺序**
- `_next_buffer_id` / `_extend_by_prefetch_size`（863-942 行）沿 gid ±1 遍历，gid 按 unit_id 排序（511-531 行）。
- 遇到 MTP、多模态 encoder、共享层或 full recompute 时会预取错对象。full recompute 下，backward 中重算 forward 会走 `all_gather_params(is_backward=False)`（2256 行），按前向方向去预取 unit k+1，而这个 unit 刚被 backward 释放。这一点还需要 nsys 确认。当前 yaml 用的是 unit 内部的 selective recompute，不会触发 unit 级 hook，所以暂时不受影响。
- Megatron 做法：#8478 记录执行顺序、#8207 ExecutionOrder、#8485 统一遍历；#6153 / #8515 处理 recompute 权重的 prefetch。

**6. fp8 expert buffer 不参与 gather 预算，forward 后一直留到本层 backward**
- `_count_gathered` / `_release_one_buffer_if_needed` / `_release_ready_buffers` 都跳过 fp8（818-825、849-853、897-900 行），所以 fp8 expert 实际上相当于 `reshard_after_forward=False`。
- 每个 fp8 buffer 同时 gather rowwise 和 transposed 两份（`_fp8_utils.py` 44、110-113 行）。
- 当前配置能放下，EP 变小或模型变大后会成为显存瓶颈。可以参考 Megatron #7907 / #8432：forward 只 gather 前向用的那份，backward 只 gather 转置那份，recompute 时才两份都要。通信量不变，换来 fp8 也能 reshard after forward。

**7. Muon keep_whole 路径全程同步，没有 overlap**
- `whole_params_grads()` / `broadcast_whole_params()` 里先用 `batch_isend_irecv` 把分片收集到 owner，再 scatter 回去，每一步都 `wait()`（1975-2043、2058-2064、2150-2181 行）。
- 未分片的 keep_whole 组在 `finish_grads_sync()` 里同步 reduce / all_reduce（1302-1318 行）。
- 可参考 Megatron #8313（Muon 分片 momentum）、#7974（多组 owner 均衡）、#8178（直接 P2P 接收）、#8512（设计文档）。建议在 backward 中某组梯度就绪后就异步发 reduce-to-owner，并让 NS 迭代和其他组的通信重叠。

**8. 通信 buffer 每次现分配，FSDP 要从 DeepGEMM 让出 8 个 SM**
- AG 的 full buffer 每次 `paddle.empty`（325 行），用完 `_clear_data`；`training_args.py:79` 为 FSDP 通信预留了 8 个 SM。
- Megatron 有 `fsdp_double_buffer`、MaxPoolAllocator（#5462）和 `nccl_ub` / symmetric memory。固定 buffer 注册后，NCCL 可以用更少的 SM（NVLS / CE 路径），也减少分配器抖动。需要先确认 BCCL 是否支持 user buffer 注册。

**9. MuonH 的 hyperball 更新按 slice 逐个下发小 kernel**
- 带 `split_concat_func` 的参数（fused QKV、gate_up、MLA heads 等）在 `Muon._hyperball_apply` 里按 slice 递归调用自身（`Paddle/python/paddle/optimizer/muon.py:738-745`）。每个 slice 约 19 个 kernel，其中 3 次 Frobenius norm。
- NSYS 显示 optimizer 阶段的 kernel 都很小，耗时受 Python 侧 launch 开销限制，GPU 大部分时间在等 kernel 下发。
- 建议：同形状的 slice 拼成一个 batch，一次算出各 slice 的 norm 并完成更新，或者把整个 hyperball 步写成一个融合 kernel。预计上限约 70 ms/步（T4 步时的 0.4%）。

### 低优先级（P2）

- **初始化显存峰值**：先在每张卡上构造完整模型，再拼 buffer 并切片（195 行有 TODO）。更大模型需要 meta-device 初始化（Megatron #5369 #6799）。
- **RS 输出位置**：`do_reduce_scatter` 把输出放在输入 buffer 的 offset 0（369、384 行）。rank>0 时输出与输入重叠，但不符合 NCCL 规定的 in-place 布局（`recvbuff == sendbuff + rank*recvcount`），按规范属于未定义行为，目前在 ring 算法下能正常工作。改成取 `rank*shard` 位置的 slice 就能消除这个风险，成本很低。
- **Host 开销**：每个 unit 的 pre-forward 都遍历全部 buffer 组检查 fp8（2248-2251 行）；`_release_dense_grad_buffer` 和 `finish_grads_sync` 都遍历 `model.parameters()`（1143、1338 行）。层数多的 MoE 模型可以在 nsys 里看 CPU launch 间隙，再决定要不要缓存。CUDA graph 暂不支持，Megatron 有 CG 模式。
- **配置层面可先试**：dense 用 `fsdp_reshard_policy: after_step`（yaml 注释实测 +1.26GB），每步 AG 从 2×16 次降到 1 次；`grad_comm_dtype=bf16`；`sync_grads_each_microbatch=False`（实验性，代价是整份未分片的 fp32 梯度常驻显存）。

### 建议的验证方式

用 `run_fsdp_test_1node.sh` / `run_fsdp_test_2node.sh` 的 `NSYS=1` 采集 6~10 步，重点看三处：步间空泡（第 1 项）、backward 末尾串行的 RS（第 2 项）、backward 中 AG 是否排在 RS 后面（第 4 项）。先量化每项的耗时再排期。

## 四、P0 实测结果（T4，2026-11-06）

### 结论

- **P0 第 2 项（非 unit 组梯度 RS 提前发出）**：已实现并保留。T4 每步快约 46 ms（-0.29%），tokens/s/card 提升约 0.32%。最确定的收益在 optimizer-step，每步少约 15 ms（-4.4%），优化前后的区间不重叠。显存基本不变，loss / gnorm 对齐。
- **P0 第 1 项（去掉 FSDP 步间同步）**：测不出收益，已撤回。

### 第 2 项的改动

文件：`Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`

| 位置 | 改动 |
|---|---|
| `BufferGroup` | 新增 `grads_ready_cnt`，记录该组参数梯度全部到齐的 micro-batch 数 |
| `reduce_scatter_grads` | lm_head、final norm、embedding 这几个非 unit 组，以及第一个 unit 的 dense 组，在最后一个 micro-batch 上组内所有参数的 grad hook 都触发（`grads_use_cnt >= grads_use_sum`）后立即发 RS |
| `all_gather_params` | 第一个 unit 的 expert 组：最后一个 micro-batch 的 backward 离开第一个 unit（进入 embedding）时发 RS |
| `_issue_reduce_scatter` | 把原 `_reduce_group_grads` 里发 RS 的那段代码抽出来，供两个触发点共用，逻辑不变 |
| `finish_grads_sync` | 剩余的 sharded RS 先全部发出，再统一 wait 和累加，不再逐个串行 |

两类组的触发方式不同，原因如下：

- **dense 组按 hook 计数触发。** 有残差连接时，个别 dense 权重的 grad hook 会在 backward 已经离开第一个 unit 之后才触发。最初的版本在离开第一个 unit 时把整个 unit 一起 flush，后到的梯度只能由 `finish_grads_sync` 补发，这个组一步里做了两次 RS，CI 的 `test_dygraph_group_fsdp`（`check_acc_sync`）因此失败。
- **expert 组按离开 unit 触发。** expert 梯度由融合 kernel 直接写入 main_grad，不经过 grad hook，计数永远凑不满。expert 组如果也按计数触发，RS 会全部退回到 `finish_grads_sync`，T4 实测 optimizer-step 回到 350.9 ms，收益消失。expert 梯度在 unit 内的 MoE backward 里已经写完，不会晚到。

修正后在 T4 上复测（2026-11-07，与优化前交替各跑 2 次）：optimizer-step 为 331.6 / 332.9 ms，优化前为 351.6 / 350.3 ms，与下表中的结果一致。step time 的差值小于跑间抖动。

正确性兜底：如果某个参数的梯度在 RS 发出之后才到（例如同一参数在一个 micro-batch 内 hook 触发多次），原有的 `_ensure_grads_writable` 会等这次 RS 完成，再把后到的梯度写入新的 tmp buffer，最后由 `finish_grads_sync` 补发。结果正确，只是这一组没有重叠。

### 测试环境

- 模型：T4（`ernie5_5_flash_ladder_v90_T4_h1280_L20_34.93BA1.232B-pretrain-8k-hzz1-fsdp.yaml`），hidden 1280，20 层，768 个路由专家，MLA，`sharding: fsdp`，`use_expert_parallel: true`
- 规模：2 机 16 卡，`bash run_fsdp_test_2node.sh`，共跑 6 步，取 step 3-6 平均，只看 rank 0 的日志
- 优化前后各跑 2 次，同一时段先后完成。每组开跑前确认两机 build 目录下文件的 md5 一致，并驱逐 BE 混部任务

### 性能与显存

| | 优化前 run1 | 优化前 run2 | 优化后 run1 | 优化后 run2 |
|---|---|---|---|---|
| step time | 16.072 s | 16.065 s | 16.025 s | 16.019 s |
| tokens/s/card | 24,468 | 24,468 | 24,547 | 24,547 |
| forward-backward | 15,651 ms | 15,643 ms | 15,619 ms | 15,613 ms |
| optimizer-step | 349.3 ms | 351.2 ms | 336.7 ms | 332.8 ms |
| optimizer-step 区间 | 348.6-350.1 | 349.9-353.1 | 335.4-340.2 | 331.9-335.3 |
| max_mem_allocated | 70.944 GB | 70.944 GB | 70.944 GB | 70.944 GB |
| max_mem_reserved | 73.166 GB | 73.182 GB | 73.213 GB | 73.041 GB |

两组平均对比：

| | 优化前 | 优化后 | 差值 |
|---|---|---|---|
| step time | 16.068 s | 16.022 s | -46 ms（-0.29%） |
| tokens/s/card | 24,468 | 24,547 | +79（+0.32%） |
| forward-backward | 15,647 ms | 15,616 ms | -31 ms（接近跑间波动） |
| optimizer-step | 350.3 ms | 334.8 ms | -15.5 ms（-4.4%） |

- optimizer-step 计时包含 `finish_grads_sync`，因为它在 `optimizer.step()` 内被调用。所以尾部串行 RS 的减少直接体现在这一项上。
- forward-backward 的差值和跑间波动在同一量级，不算作确定收益。
- 显存：max_allocated 完全一致，max_reserved 前后的波动范围重叠，没有可见变化。
- 精度：step 6 的 loss 在 4 次运行中都落在 12.21424-12.21427，gnorm 都落在 3.6895-3.6899，与以往跑间的非确定性差异一致。

### NSYS 时间线验证

node0 GPU0，抓 step 2-4，以最后一个 micro-batch（mb16）的 EmbeddingGrad 结束为 0 点，单位 ms。
优化前的时间线抓取时还带着第 1 项改动（后已撤回）；该改动不影响 RS 发出的时机。
优化后的时间线是最初版本（第一个 unit 整体在离开 unit 时 flush）。修正后第一个 unit 的 dense RS 改为等所有 hook 触发后才发，时间可能略晚，未重新抓取。

| | 优化前 | 优化后 |
|---|---|---|
| lm_head RS（s50，约 3 ms） | 尾部 12.5 / 27.75 | mb16 backward 开头，比第一个 flash bwd 早 23 ms |
| final norm RS（0.1 ms） | 尾部 15.2-15.4 | mb16 backward 开头，早 19 ms |
| 第一个 unit 的 expert RS（s51） | 16.4 开始，持续 10.1 ms | -0.4 开始，持续 13.7 ms，与 EmbeddingGrad 重叠 |
| 第一个 unit 的 dense RS（s50） | 在尾部串行 | -0.3 开始 |
| embedding RS（s50） | 在尾部 | 2.2 开始，持续 4.5 ms |
| 最后一个 RS 结束 | 约 30.4 | 13.3 |
| Muon SendRecv 开始 | 30.4 | 16.7 |
| clip all-reduce 开始 | 55.2 | 41.5 |

- mb16 的 backward 中，RS 数量从 72 个变为 74 个，多出来的 2 个是 lm_head 和 final norm。intra-step 的 mb8 仍是 72 个，说明只有最后一个 micro-batch 会触发早发。
- 现在尾部的关键路径是第一个 unit 的 expert RS（约 13 ms），它已经在结构上最早的时机发出。再往前压只能拆分第一个 unit，收益约 0.08%。
- 这个 expert RS 从 10.1 ms 变成 13.7 ms，推测是和 embedding RS 并发、争抢带宽造成的。即便如此，整体仍然更早结束。

### 已撤回：第 1 项（去掉 FSDP 步间同步）

尝试把 `comm_sync_and_reset_status` 里的 `paddle.device.synchronize()` 换成计算流等待各通信流（`wait_stream`）。

| | baseline | 去掉步间同步 |
|---|---|---|
| step time | 16.06 s | 16.15 s |
| optimizer-step | 354 ms | 351 ms |

测不出收益，已恢复原代码。NSYS 显示步边界整卡空闲约 178 ms（占步时 1.1%），主因不是这次同步。另外 trainer 里 `ernie5/src/trainers/pretraining_trainer.py:8496` 每步还有一次无条件的 `paddle.device.synchronize()`，所以 host 端照样会阻塞。

### 后续可做

- optimizer 阶段的小 kernel：MuonH 对每个 slice 单独调用 `_hyperball_apply`（每个 slice 约 19 个 kernel，其中 3 次 Frobenius norm），这一阶段受 Python 侧 launch 开销限制。预计上限约 70 ms/步（0.4%）。见第三节 P1 第 9 项。
- trainer 每步那次无条件 `synchronize`：未处理。

### 文件

- 改动备份：`/tmp/fsdp/bench/fully_shard_fusion.rs_count2.py`（当前版本），`/tmp/fsdp/bench/fully_shard_fusion.rs_early.py`（最初版本）；原版：`/tmp/fsdp/bench/fully_shard_fusion.build.orig.py`
- 测试日志：`/tmp/fsdp/bench/final_{base,base2,opt,opt2}.log`，修正后复测 `/tmp/fsdp/bench/re_{base1,base2,fix1,fix2}.log`；统计脚本：`/tmp/fsdp/bench/avg.py`
- NSYS：`timeline/ernielite-T4-fsdp_n{0,1}.nsys-rep`（优化后），`timeline/ernielite-T4-fsdp_n{0,1}.nosync.nsys-rep`（优化前）