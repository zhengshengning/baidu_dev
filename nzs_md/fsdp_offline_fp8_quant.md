# FSDP 支持 offline_quant_expert_weight

## 结论

FSDP（`sharding: fsdp`）下 `offline_quant_expert_weight=true` 已支持：每个 step 开始时把 MoE 专家权重量化成 fp8 一次，并缓存到参数对象上。forward、`moe_gate_up` 重计算和 dgrad 都直接用这份缓存，不再每个 micro-batch 重新量化。

测试脚本是 `run_fsdp_test.sh`，配置为 `ernielite_ladder_v90_T1_h512_L8_2.66BA0.182B-pretrain-8k-hzz1_fsdp.yaml`，单机 8 卡，EP8，跑 20 步。对比对象是"不预量化"，与之前在 callback 里直接 return 的行为等价，即 `offline_quant_expert_weight=False`。

| | 每卡 tokens/s（step 3-20 平均） | max_mem_allocated |
|---|---|---|
| 不预量化（旧行为） | 94938 | 50.08 GB |
| 预量化（新实现） | 96579（+1.7%） | 50.64 GB |

- 20 步 loss 逐位一致。两条路径用同一个量化 kernel、输入同样的 bf16 值。
- 显存多出约 0.56 GB，来自 fp8 缓存（非转置和转置两份，`using_cache_weight_quant_transpose: true`）在整个 step 内常驻。
- `run_sharding_test.sh`（stage1）这次没有补跑回归。stage1 分支的逻辑没改。

PaddleFleet 的同名回调（`PaddleFleet/src/paddlefleet/trainer/trainer_callback.py`，SFT 流程使用）原来也是在 FSDP 下直接 return，现在也已支持，见"实现 3"。测试脚本是 `dsv4/run_sharding_fsdp_align.sh`：DeepSeek-V4-Flash-Base 截前 4 层，Muon，EP8，单机 8 卡，跑 10 步，开启 stage1 和 FSDP 的逐位对齐诊断开关。

| | 每卡 tokens/s（step 3-10 平均） | max_mem_allocated |
|---|---|---|
| sharding stage1 | 6853 | 57.77 GB |
| FSDP 不预量化（旧行为） | 7064 | 58.63 GB |
| FSDP 预量化（新实现） | 7157（+1.3%） | 61.63 GB |

- 三组 10 步 loss 逐位一致。
- step 10 的 moe-mlp 从 250 ms 降到 228 ms。只跑了 10 步，吞吐数字有波动。
- 显存多出约 3 GB，来自 fp8 缓存常驻：4 层 × 每卡 32 个专家。PaddleFleet 固定 `quant_transpose=False`，只缓存非转置一份。

## 问题根因

stage1 下 `FP8QuantWeightCallback.on_step_begin` 做三件事：

1. `model.fp8_quant_weight(...)`：读取 `grouped_gemm_experts.weight1/weight2` 的 bf16 值，生成 `fp8_weight_stacked`、`fp8_scale_stacked` 和 `*_transpose`。
2. `optimizer.clear_param_storage("moe_expert" / "moe_weight_no_hook")`：释放 bf16 专家权重。step 开始时由 `reset_param_storage` 恢复，再由 master weight 写回。
3. 可选：开启 `HACK_OFFLOAD_OPTIMIZER` 时，把 master weight 放到 pinned memory。

FSDP 下这三件事都做不了：

- `fully_shard` 把参数展平到分片 buffer `fuse_params_*`。只有在所属 unit 的 forward/backward 里，才会 all_gather 并把完整权重绑定到参数上。optimizer step 开始时，`comm_sync_and_reset_status` 调用 `reset_params_buffer_status`，所有参数又变回没有存储的空壳，所以 `on_step_begin` 访问权重会报 "Tensor not initialized yet"。
- FSDP 不包装 optimizer，用的是普通的 AdamW 或 Muon，没有 `clear_param_storage`。
- master weight 的 key 是 `fuse_params_*`，不是专家参数名。

缓存缺失时，`fp8_utils.fused_stack_quant` 会回退到 `fused_stack_quant_without_cache`，所以直接 return 也能跑。代价是每层每个 micro-batch 要量化 5 次；有缓存时，每层每个 step 只需 4 次。

- 无缓存的 5 次：
  - forward 2 次：`fwd_gate_up_fp8` 量化 w1 转置版，`fwd_down_fp8` 量化 w2 转置版。
  - 重计算 1 次：`recompute_modules` 含 `moe_gate_up` 时，`backward_impl_fp8` 会再调一次 `fwd_gate_up`，又量化一次 w1 转置版。
  - dgrad 2 次：`bwd_down_input_fp8` 量化 w2 非转置版，`bwd_gate_up_input_fp8` 量化 w1 非转置版。
  - wgrad 走 `bf16_weight_grad`，不量化。
- 有缓存的 4 次：`quantize_weights` 对 w1、w2 各生成非转置版和转置版两份，共 4 份。上面 5 个调用点都直接读这份缓存，与 micro-batch 数无关。

## 实现

### 1. Paddle：`FullyShardFusion.unshard_params(params)`

文件：`Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`。测试实际加载的是 `Paddle/build/python/...`，改完要同步拷过去。

这是一个上下文管理器，在 forward/backward 之外临时把参数绑定到完整存储。用法：

```python
with fsdp_context.unshard_params(params):
    ...  # 块内 params 是完整 bf16 权重，可读
# 退出后 params 重新变回空壳
```

生命周期分三段，以 `yield` 为界：

- 进入（`yield` 之前）：在 `no_grad` 下调用 `comm_manager.all_gather_params(params)`，用 `_share_data_with` 把完整 buffer 绑定到参数上。期间临时关闭 `enable_overlap`，同步 gather，不预取别的 unit。
- 块内（`yield` 期间）：调用方读取完整权重。本方案里就是 `layer.fp8_quant_weight(...)` 进入 `quantize_weights`，把生成的 fp8 tensor 写到参数对象的属性上。
- 退出（`finally`）：调用 `shard_params(params)`，对参数执行 `_clear_data()`。进入前状态是 `FREED` 的 buffer 会释放 tmp buffer、恢复成 `FREED`，并回退 `buffer_cnt_in_using`，下一次 forward 看到的 double-buffer 状态不变。块内抛异常时也会走到这一步。

设计要点：

- 用上下文管理器，是为了让 gather 和 shard 成对出现，调用方不用自己管理 buffer 状态。
- 不归 FSDP 管理的参数会被忽略，所以调用方可以把整层参数直接传进来，不用自己过滤。
- fp8 缓存能留下来，是因为 `_clear_data()` 只断开参数的底层存储（`DenseTensor`），不动参数 Python 对象的 `__dict__`。fp8 tensor 是独立分配的，不依赖被释放的 bf16 buffer。

### 2. ERNIEBot：`ernie5/src/callbacks/fp8_quant_weight_callback.py`

FSDP 分支（`_fsdp_owns_param_storage(args)`）放在 `offline_quant_expert_weight and sharding_parallel_size > 1` 条件内部：

- `_fsdp_fp8_quant_weight`：逐个 `TransformerLayer` 用 `get_fsdp_context().unshard_params(...)` 包住 `layer.fp8_quant_weight(...)`。显存峰值只占一层，和 forward 一样。调用链为 `TransformerLayer.fp8_quant_weight` → `MoELayer.fp8_quant_weight` → `quantize_weights`，全部在 with 块内执行。
- `_fp8_quant_target_params(layer)`：选出这一层量化时要读的参数，即 `mlp.grouped_gemm_experts` 或 `mlp.experts` 的参数，加上开启 fp8 的非专家 Linear 的 `weight`。
- FSDP 下不调用 `clear_param_storage`，也不做 master weight 卸载。bf16 分片不能释放：它是 optimizer 的参数（每步由 fp32 master weight cast 回写），也是下一次 forward all_gather 的数据源。完整 bf16 副本不需要 callback 释放：上一个 step 的副本在 callback 之前已由 `reset_params_buffer_status` 释放；量化时临时 gather 的副本由 `unshard_params` 退出时释放。
- `FLAGS_shard_bypass_dygraph_optimizer` 和 `skip_count` 的语义与 stage1 一致。

时序上没有问题：缓存在 `on_step_begin` 生成，用的是上一个 step 更新后的分片值。到下一次 optimizer step 之前，没有任何操作会改动权重。

### 3. PaddleFleet：`PaddleFleet/src/paddlefleet/trainer/trainer_callback.py`

这个回调在 `cli/train/sft/workflow.py` 中注册（非 lora 时加入）。它和 ERNIEBot 版本有四点不同：

- 触发条件不看 `offline_quant_expert_weight`。只要 `sharding_parallel_size > 1` 且模型开启了 fp8，就会预量化。
- 参数固定为 `batch_mode=True, quant_transpose=False`。
- stage1 下除了 `moe_expert`，还会释放 `rms_linear`、`memory_attn`、`attn_out_project`、`shared_expert` 四类参数的 bf16 存储。
- master weight 卸载由 `offload_fp8_expert_master_weight` 控制：`on_step_begin` 卸到 CPU，`on_optimizer_begin` 再加载回来。

原来的写法是在 `on_step_begin` 和 `on_optimizer_begin` 开头遇到 FSDP 就直接 return。下面逐条说明改了什么、为什么这样改。

**(1) 两处 FSDP 早退都删掉。** `on_step_begin` 的早退就是性能问题的来源。`on_optimizer_begin` 的早退是为了和前者对称：没有卸载，也就不用加载。现在 `on_step_begin` 会走完整流程，对称关系必须保留，所以两处一起删。FSDP 下如果不开卸载，`moe_weights_name` 是空列表，reload 循环什么都不做，效果和原来的早退相同。

**(2) 量化改成逐层 unshard。** FSDP 分支用 `_fsdp_fp8_quant_weight(model, True, quant_transpose=False)` 替换 `model.fp8_quant_weight(...)`。原因和 ERNIEBot 一样：`on_step_begin` 时参数都是空壳，对整个模型一次性量化会读到未初始化的 tensor。改成每个 `TransformerLayer` 依次 unshard、量化、reshard，显存峰值只占一层。

- 入参沿用 stage1 分支的 `True, quant_transpose=False`，这样两条分支生成的缓存形态一致（只有非转置一份），forward 走的是同一条 kernel 路径。这也是 dsv4 上 stage1 和 FSDP 的 loss 能逐位对齐的前提。
- `_fp8_quant_target_params` 和 `_fsdp_fp8_quant_weight` 与 ERNIEBot 版本逐行相同。ERNIEBot 本来就依赖 PaddleFleet（`from paddlefleet.trainer.trainer_callback import TrainerCallback`），以后可以直接 import 这两个函数来去掉重复，本次没有做。

**(3) FSDP 下不调用 `clear_param_storage`。** 有两个原因：

- FSDP 不包 sharding optimizer，optimizer 上没有这个方法。
- bf16 分片暂时不能释放：fp32 master weight 仍在，数值上可以从它重建（stage1 的 `clear_param_storage` 就是先 `_create_master_weight` 再释放）；但 FSDP 每次 forward pre-hook 都会从 bf16 分片 all_gather 并绑定到参数，释放后 forward 会读到无效存储。要释放需先改 `all_gather_params`，见"未完成"一节。

代价是 stage1 下释放上述五类 bf16 参数带来的显存收益，在 FSDP 下拿不到，见"未完成"一节。

**(4) 卸载对象的名字要按 FSDP 的 master weight key 重新算。** stage1 原逻辑是从 `optimizer._inner_opt._parameter_list` 里挑出 `color == "moe_expert"` 的参数名，因为 stage1 的 master weight 按原参数名存。FSDP 的 key 规则不同，见 `fully_shard_fusion.py` 的 `init_optimizer_state` 里 `not (is_muon and group.use_muon)` 这个条件：

- 非 Muon 的 group：master weight 和累加器建在展平 buffer `group.params_buffer.data_buffer` 上，key 是 `fuse_params_*`。
- `use_muon` 的 group：仍然建在原参数上，key 是原参数名。

`_parameter_list` 里放的仍是原参数，所以如果照搬按 color 挑选的逻辑，非 Muon 的专家 group 一个 key 都匹配不上。`_fsdp_expert_master_weight_names()` 的做法是遍历 `buffer_manager.buffer_groups`，对 `is_expert_param` 的 group 同时收集 buffer 名和参数名，再由后面的 `if name in master_weights` 过滤掉不存在的 key。

两条分支用 `moe_weights_name = None` 作为哨兵区分：FSDP 分支先算好名字，stage1 分支原来的 color 循环完全不动。

**(5) 用 `_master_weights_of(optimizer)` 取 master weight。** 它沿 `_inner_opt` 一层层往里，找第一个有 `_master_weights` 属性的 optimizer。FSDP 下最外层可能还套着 wrapper，`_master_weights` 在最内层；Paddle 自己的 `init_optimizer_state` 也是这样往里剥的。stage1 原代码直接访问 `optimizer._master_weights` 是能取到的，所以第一层就会命中，行为不变。卸载和加载两处统一用这个函数，保证查的是同一张表。

**(6) 第一个 step 不会卸载。** Paddle optimizer 的 master weight 是在第一次 step 的 `_create_accumulators` 里懒创建的。第一个 step 的 `on_step_begin` 时表还是空的，循环什么都不做，从第二个 step 开始才生效。代码里加了注释说明。

**(7) `on_optimizer_begin` 里用 `getattr(self, "moe_weights_name", [])`。** `moe_weights_name` 只有 `on_step_begin` 进入分支时才会赋值。这只是防御性写法，正常流程下不会用到默认值。

另外，`SonicMoELayoutSwitchCallback._reject_fsdp` 的删除是另一项改动，不在本文范围内。

## 未完成：省显存和通信

和 stage1 相比，目前只拿到了计算上的收益。FSDP 的 forward pre-hook 仍然会 all_gather 并绑定 bf16 专家权重，stage1 那种"释放 bf16 专家权重"的显存收益没有拿到。

如果要做，需要在 `FSDPCommManager.all_gather_params` 里，对已有 fp8 缓存的专家 buffer 跳过 bf16 的 gather 和绑定。需要注意：

- wgrad 走 `bf16_weight_grad`，只读 `weights.shape` 和 `main_grad`，不读 bf16 值。但 `comm_hook`、`_bind_expert_main_grads` 和 `shard_params` 都依赖参数的绑定状态，改动面较大。
- 当前配置下 `moe_sharding_degree = world_size // (pp × ep) = 8 // 8 = 1`（日志 `ep_degree: 8, moe_sharding_degree: 1` 已确认）。专家参数的 FSDP group 是 `moe_sharding` group，只有 1 个 rank，`is_sharded = False`，专家 buffer 实际没有切分，gather 本身没有通信，收益有限。多机且 `moe_sharding_degree > 1` 时再做比较合适。
- 如果改成 `quant_transpose=False`，fp8 缓存显存减半，代价是 forward 时现场做 transpose。

## 复现

```bash
# 新实现（预量化）
bash run_fsdp_test.sh > /tmp/fsdp_fp8_test.log 2>&1
# 对照组：在 run_fsdp_test.sh 的 --kwargs 里追加 offline_quant_expert_weight=False
```

确认生效：日志里每个 step 都会出现一次 `offline quant expert weight from bf16 to fp8.`（20 步共 20 次）。

PaddleFleet（dsv4）：

```bash
cd /root/paddlejob/paddlefleet-public/ningzhengsheng/FSDP/dsv4
bash run_sharding_fsdp_align.sh   # 先后跑 stage1、fsdp，最后用 compare_loss.py 比较 loss
```

- 模型：`dsv4/DeepSeek-V4-Flash-Base-L4/`。权重软链到 `/root/paddlejob/pfsl3-pdc-public/model/DeepSeek/DeepSeek-V4-Flash-Base`（原始 FP8 发布版，43 层）；`config.json` 只保留前 4 层，`compress_ratios` 同步截成 `[0,0,4,128]`。单机 8×80G 放不下全量，而且只在 yaml 里设 `num_hidden_layers` 会被 `compress_ratios` 的长度校验拦下。
- 数据：`dsv4/data/sft_2000_erniekit.jsonl`，由 `glm45_ci/data/sft_data` 的 messages 格式转成 erniekit 的 src/tgt 格式。直接用 messages 文件时，erniekit 解析不出样本，会在 binpacking 里死循环。
- 脚本依赖 `MUON_FSDP_BCAST_GRAD` 诊断代码（stage1 侧在 `tensor_fusion_helper.py`、`muon_sharding_optimizer.py`、`optimizer/muon.py`，FSDP 侧在 `fully_shard_fusion.py`）。目前源码和 `Paddle/build/python` 两份都带这段 hack；缺了 FSDP 侧那段，loss 会从 step 2 开始对不上。日志里出现 `[MUON_FSDP_BCAST_GRAD] FSDP: reduce_scatter -> broadcast(src=0) ACTIVE` 说明已生效。提交 Paddle 代码前要把这段 hack 去掉（干净版本是 git HEAD `4b25c78487`）。
- 对照组（FSDP 不预量化）：在 `FP8QuantWeightCallback.on_step_begin` 的 FSDP 分支开头临时加一个 `return`，跑完要删掉。

未验证：

- FSDP 下 `offload_fp8_expert_master_weight=true`。dsv4 配置里这一项是 false，(4)～(6) 的卸载和加载路径还没有实际跑过。
- `moe_sharding_degree > 1` 的多机场景。
