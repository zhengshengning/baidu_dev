# FSDP fp8 param gather 与 grad_comm_dtype：代码走读与接入教程

本文逐段讲解三个 PR 新增的代码，说明每处改动做什么、为什么这样做，以及怎样给新的模块接入。

| PR | 仓库 | 本地 commit | 内容 |
|---|---|---|---|
| [#79897](https://github.com/PaddlePaddle/Paddle/pull/79897) | Paddle | `071c0f960e` | FSDP 对 MoE 专家权重做 fp8 参数 all-gather，fp8 组用 fp32 分片做主权重 |
| [#2371](https://github.com/PaddlePaddle/PaddleFleet/pull/2371) | PaddleFleet | `3e58437f` | `FP8QuantWeightCallback` 在 `fully_shard` 之前给专家权重挂上量化器 |
| [#79899](https://github.com/PaddlePaddle/Paddle/pull/79899) | Paddle | `b6473e2913` | 梯度 reduce-scatter 可以用 bf16 通信，累加仍是 fp32 |

代码以本仓库下 `Paddle/`（develop）和 `PaddleFleet/` 里的这几个 commit 为准。PR 描述和合入代码有两处不一致：

- #79897 的描述写的是 `fully_shard(fp8_param_gather=...)` 开关。评审中这个开关被删掉了，合入版本没有开关：参数上挂了 `_fsdp_fp8_quantizer` 就走 fp8 gather。
- #79899 的描述写的是 `fully_shard(grad_comm_dtype=...)`。评审中它被移进了 `MixedPrecisionPolicy`。

涉及的文件：

```
Paddle/python/paddle/distributed/fsdp/_fp8_utils.py          新增，fp8 逻辑全部在这里
Paddle/python/paddle/distributed/fsdp/fully_shard.py         fp8 主权重 dtype；grad_comm_dtype 参数与校验
Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py  只按 params_buffer.fp8 是否为 None 分支
Paddle/test/collective/fleet/dygraph_group_fsdp_fp8.py       新增，fp8 gather 与 checkpoint 的 2 卡测试
Paddle/test/collective/fleet/dygraph_group_fsdp.py           新增 run_grad_comm_dtype
PaddleFleet/src/paddlefleet/transformer/moe/fp8_utils.py     新增 mark_fsdp_fp8_params
PaddleFleet/src/paddlefleet/trainer/trainer_callback.py      FP8QuantWeightCallback.on_init_end
PaddleFleet/tests/single_card_tests/fp8/test_fp8_fsdp_callbacks.py
```

---

## 1. 背景：改动前 FSDP 下的 fp8 专家权重

`offline_quant_expert_weight=true` 时，MoE 的 grouped GEMM 不读 bf16 专家权重，只读挂在参数对象上的 fp8 缓存（`fp8_weight_stacked`、`fp8_scale_stacked` 及其 `_transpose` 版本，读取入口是 PaddleFleet `fp8_utils._get_fp8_weight_and_scale`）。

改动前 FSDP 的做法（见 `fsdp_offline_fp8_quant.md`）：

1. 每个 step 开始时，`_fsdp_fp8_quant_weight` 逐层调用 `unshard_params`，把完整 bf16 专家权重 all-gather 过来，每张卡对完整权重量化一次，fp8 缓存挂在参数上。
2. forward/backward 时，FSDP 仍按 unit 把 bf16 专家权重 all-gather 过来，但 GEMM 只读 fp8 缓存。

这样有三类开销：

- 通信：每个 step 为预量化多做一轮专家权重的 bf16 all-gather；forward/backward 里 gather 过来的 bf16 完整权重没有被计算用到。
- 计算：每张卡都量化完整权重，同一份量化在 d 张卡上重复 d 次（d 为 FSDP 组大小）。
- 显存：所有层的 fp8 缓存在整个 step 内常驻；bf16 分片和 fp32 master weight 也都常驻。

## 2. 新设计总览

每张卡只量化自己的分片，然后 all-gather fp8 数据和 scale，在 unit 内把 gather 结果以 view 的形式挂到参数属性上，unit 结束即释放。bf16 完整专家权重不再出现。

```
optimizer.step()              原地更新 fp32 主权重分片（params_buffer.data_buffer）
      │
第一个 unit 的 forward pre-hook（all_gather_comm）
      │  sync_ckpt_weight_to_main   只有 load checkpoint 之后才有事做
      │  quantize_fp8_shard         本卡分片 fp32 -> bf16 -> k 份 fp8 + k 份 fp32 scale，每 step 一次
      ▼
unit 的 all_gather_params
      │  do_all_gather              对每个 (full, shard) 发一次 all_gather，共 2k 次
      │  bind_fp8_param             在 full 上切 view，setattr 到 param.fp8_weight_stacked 等
      ▼
GEMM 只读 fp8 缓存；param 本身是 0 字节存储的空壳
      │
unit 释放
      │  clear_fp8_tmp_buffer       delattr 这些缓存属性，丢掉 full buffer
      ▼
backward 结束 reset_params_buffer_status：quantized=False，下个 step 重新量化
```

其中 k 是量化器 `attrs` 的长度：只缓存非转置一份时 k=1，同时缓存转置版时 k=2。

`fully_shard_fusion.py` 只在几个固定位置判断 `params_buffer.fp8 is not None`，具体逻辑都放在 `_fp8_utils.py`。不挂量化器的参数组 `fp8` 为 `None`，走原路径，行为不变。

---

## 3. Paddle #79897：fp8 param gather

### 3.1 接口：`param._fsdp_fp8_quantizer`

FSDP 和模型之间只通过一个挂在参数上的可调用对象约定，`fully_shard` 不新增参数：

```python
def quantizer(weights, transpose):
    # weights: [E_local, R, C]，dtype 为 param dtype（bf16），是本卡分片里的整数个专家
    # 返回 (data, scale)：
    #   data  fp8_e4m3fn，numel == weights.numel
    #   scale fp32，      numel == weights.numel // block**2
    ...

quantizer.block = 128
quantizer.attrs = (
    # (fp8 数据属性名, scale 属性名, 是否转置)
    ("fp8_weight_stacked", "fp8_scale_stacked", False),
    ("fp8_weight_stacked_transpose", "fp8_scale_stacked_transpose", True),
)
param._fsdp_fp8_quantizer = quantizer   # 必须在 fully_shard 之前挂上
```

对参数和量化器输出的要求：

- 参数是 3 维 `[E, K, N]`，`K` 和 `N` 都是 `block` 的整数倍，否则 `fp8_shard_numel` 抛 `ValueError`。读 shape 时优先用 `original_shape`，因为 FSDP 会把参数 flatten。
- `data.flatten()` 必须按专家依次排列，每个专家占 `K*N` 个元素；scale 同理，每个专家占 `K*N/block²` 个。FSDP 按这个假设把分片结果写进 fp8 分片 buffer 的对应位置。
- gather 之后，FSDP 按下面的 shape 把 view 挂到参数上，量化器输出的布局要和它一致：

| attrs 项 | data shape | scale shape |
|---|---|---|
| 非转置 | `[E*K, N]` | `[E*K/block, N/block]` |
| 转置 | `[E*N, K]` | `[E*N/block, K/block]` |

这正是 PaddleFleet grouped GEMM 读 `fp8_weight_stacked*` 时期望的布局。

### 3.2 分组：量化器进入 buffer group 的 key

`FSDPBufferManager` 按一个 key tuple 把参数分进 `TensorFusionBuffer`。这次在 key 末尾加了一项：

```python
getattr(param, "_fsdp_fp8_quantizer", None),
```

`sort_key` 和后面的解包相应多了一个 `_`。效果：

- 挂了量化器和没挂的参数不会落进同一个 buffer。
- 量化器按对象身份比较，挂了不同量化器对象的参数也会分开。PaddleFleet 对所有专家用同一个对象，所以同一层的 `weight1`、`weight2` 仍在一个组里。
- `fp8_shard_numel` 只读 `params[0]` 的量化器，正是因为同组参数的量化器一定相同。

### 3.3 分片对齐：每张卡的分片必须是整数个专家

```python
fp8_numel = fp8_shard_numel(params)          # lcm(block², 每个参数的 K*N)
shard_align = muon_align
if fp8_numel:
    shard_align = math.lcm(muon_align or 1, fp8_numel)
```

`shard_align` 同时传给 params buffer 和 grads buffer。`TensorFusionBuffer.__init__` 要求总长度能被 `d * shard_align` 整除，且每个参数的 padded size 能被 `shard_align` 整除；任一条件不满足，就把 `is_sharded` 置为 `False`。随后 `init_fp8_buffer` 发现组没有分片，抛 `ValueError("... is not sharded")`。fp8 组不会静默退回到不分片的方式。

要求分片边界落在专家边界上，原因有三个：

1. 本卡分片能 reshape 成 `[E_local, K, N]`，直接交给量化器。
2. 128×128 的量化块不会跨卡，分片各自量化后拼起来，与量化完整权重的结果逐位相同。测试 `run_fp8_gather` 用 md5 校验了这一点。
3. scale 在分片 buffer 里的偏移 `lo // block²` 是整数。

举例：一层的 `weight1` 为 `[E, H, 2I]`，`weight2` 为 `[E, I, H]`，忽略额外 padding。`shard_align = lcm(2HI, HI) = 2HI`，总长度为 `3E·HI`，需要 `3E` 被 `2d` 整除，且 `weight2` 的 `E·HI` 被 `2HI` 整除，即 `E` 为偶数。`E=4, d=2` 满足；测试里的 `OddExpertLayer` 用 `E=3`，会被拒绝。

### 3.4 主权重 dtype：fp8 组用 fp32 分片，不再另存 master

`fully_shard.py` 的 `_fully_shard_manual_parallel` 沿 `_inner_opt` 链找到最内层优化器，读 `_multi_precision`（属性不存在时按 `True` 处理）：

```python
multi_precision = getattr(inner, "_multi_precision", True)
FullyShardFusion(..., fp8_main_dtype=paddle.float32 if multi_precision else None)
```

`FSDPBufferManager` 给 fp8 组的 params buffer 选 dtype：

```python
main_dtype = (fp8_main_dtype or group.dtype) if fp8_numel else group.dtype
```

`TensorFusionBuffer` 初始化时把参数拷进 buffer，现在会先转成 buffer dtype：

```python
paddle.assign(param if param.dtype == dtype else param.astype(dtype), ...)
```

结果是：

- `multi_precision=True`：fp8 组的 `data_buffer` 是 fp32。非 Muon 组的优化器直接在 fused buffer 上 step，看到的是 fp32 参数，不会创建 master weight。原来的“bf16 分片 + fp32 master”变成“只有 fp32 分片”。测试断言了这些名字都不在 `opt._master_weights` 里。
- `multi_precision=False`：`data_buffer` 用 param dtype（bf16），与原路径一致。
- 参数对象本身的 dtype 不变，仍是 bf16（测试断言 `p.dtype == g.dtype`）。
- 非 fp8 组完全不受影响。

### 3.5 `init_fp8_buffer`：常驻的 fp8 分片

params buffer 建好后，fp8 组调用 `init_fp8_buffer(params_buffer, params)`，在 `params_buffer.fp8` 上挂一个 `SimpleNamespace`：

| 字段 | 含义 |
|---|---|
| `params` | 组内参数 |
| `quantizer` | 量化器 |
| `quantized` | 本 step 是否已量化过分片 |
| `shards` | 2k 个常驻分片：前 k 个 fp8，长度 `full/d`；后 k 个 fp32 scale，长度 `full/block²/d` |
| `fulls` | gather 目标 buffer，只在 gather 到释放之间存在 |
| `weight_copy` | 存 checkpoint 时导出的 param dtype 权重，见 3.12 |

`full` 是 `total_buffer_size`，含 padding。padding 位置在 fp8 分片里没有初始化，也不会被读到。

### 3.6 `quantize_fp8_shard`：每 step 量化一次本卡分片

```python
for param in fp8.params:
    shard = params_buffer.param_shard(param)    # 参数与本卡分片的交集，分片内偏移 (lo, hi)
    if shard is None:
        continue
    lo, hi, _ = shard
    _, rows, cols = param.original_shape or param.shape
    weights = view_slice(params_buffer.data_buffer, lo, hi)
    weights.get_tensor()._set_dims([(hi - lo) // (rows * cols), rows, cols])
    weights = weights.astype(param.dtype)       # fp32 主权重先舍入到 bf16
    for i, (_, _, transpose) in enumerate(quantizer.attrs):
        data, scale = quantizer(weights, transpose)
        assign(data.flatten(),  fp8.shards[i]._slice(lo, hi))
        assign(scale.flatten(), fp8.shards[k + i]._slice(lo // bn, hi // bn))
fp8.quantized = True
```

要点：

- 3.3 的对齐保证 `(hi - lo)` 是 `rows*cols` 的整数倍。
- 先 `astype(param.dtype)` 再量化。原路径量化的是 bf16 参数，而 bf16 参数等于 fp32 master 舍入后的值，所以两条路径输入的 bf16 值相同。
- 调用点有两个，都受 `quantized` 标志保护，每 step 只真正执行一次：
  - `register_tensor_fusion_hooks` 的 forward pre-hook `all_gather_comm`。每个 unit 的 pre-hook 都会遍历所有 fp8 组，所以第一个 unit 进来时就把所有 fp8 组都量化了。
  - `get_fp8_tmp_buffer`，即 `do_all_gather` 的入口。这样不论 gather 从哪条路径发起（pre-hook、prefetch、backward 重新 gather），发出前分片一定已经量化，不依赖调用顺序。
- `reset_params_buffer_status`（backward 结束、optimizer step 之前）把 `quantized` 置回 `False`，下个 step 用更新后的主权重重新量化。

### 3.7 `do_all_gather`：一个 buffer 发 2k 次 all-gather

```python
if self.fp8 is not None:
    pairs = get_fp8_tmp_buffer(self)       # [(full_i, shard_i)]，2k 对
else:
    pairs = [(self.get_tmp_buffer(), self.data_buffer)]
```

`gather()` 在 comm stream 上对每一对依次 `all_gather`，所有张量一起交给 `run_on_stream` 做跨流记录。prefetch 和 overlap 的调度逻辑不变，fp8 组照常参与预取。

### 3.8 `bind_fp8_param`：把 gather 结果挂到参数上

`all_gather_params` 原来会在 tmp buffer 上切出参数大小的 view 绑定给参数。fp8 组改为调用 `bind_fp8_param` 后直接 `continue`：

```python
begin = params_buffer.param_offsets[param.name]
end = begin + prod(shape)
for i, (d, s, transpose) in enumerate(attrs):
    rows, cols = (nn, kn) if transpose else (kn, nn)
    setattr(param, d, view(fp8.fulls[i]._slice(begin, end), [num * rows, cols]))
    setattr(param, s, view(fp8.fulls[k + i]._slice(begin // bn, end // bn),
                           [num * rows // block, cols // block]))
dims = param.shape
share_scratch(param, param.dtype)
param._clear_to_zero_allocation()
param.get_tensor()._set_dims(dims)
```

最后三行让参数保持 param dtype 和原 shape，但不占显存。`share_scratch` 先让参数共享一块 1 元素的 param dtype 临时存储，把 dtype 恢复成 param dtype（Muon step 期间参数可能被绑成 fp32 view），并保留 `stop_gradient`。

由此得到一条硬约束：fp8 组的参数在 forward/backward 里只能通过 fp8 缓存属性读取，不能读参数本身的值。

### 3.9 释放

- `TensorFusionBuffer.clear_tmp_buffer` 对 fp8 组先调用 `clear_fp8_tmp_buffer`：把参数上的 2k 个缓存属性 `delattr`，`fulls` 置 `None`，gather buffer 随之释放。unit 正常释放时就走这条路径。
- `reset_params_buffer_status` 对 fp8 组额外做三件事：`quantized=False`、`weight_copy=None`、`clear_fp8_tmp_buffer`。

### 3.10 预取计数：fp8 组不进 `comm_unit_size` 账本

按大小预取（`comm_unit_size`）时，`FSDPCommManager` 用 `gathered_numel` 记录已 gather 的元素数，超出预算时会提前释放 READY 状态的 buffer。这次改动让 fp8 组完全不参与这套记账：

- `_count_gathered`：fp8 组直接返回。
- `_free_params_buffer`：仍然清 tmp buffer、置 `FREED`，但不扣减计数。
- 两处按预算驱逐 READY buffer 的循环：跳过 fp8 组。

代码里没有注释原因。从实现看，这套计数按 `total_buffer_size` 计元素数，而 fp8 组的 gather 占 `k` 字节/元素，和 bf16 组的口径不同。结果是 fp8 组的 gather buffer 不计入预算，也不会因预算被提前释放，只在 unit 正常结束时释放。调 `comm_unit_size` 时要把这部分显存另算。

### 3.11 与 Muon 的配合

两处小改动：

1. `group.muon_shard_numel` 只在 `group.use_muon` 时取 `shard_align_numel`，否则为 `None`。原来对所有组都直接取值。现在非 Muon 的 fp8 组也有 `shard_align_numel`，不加这个判断会被误当成 Muon 分片组。
2. 新增 `FullyShardFusion._unbind_muon_shard_view(param, group)`，替换原来两处手写的恢复代码（创建 accumulator 之后、step 之后）：

```python
full = param._muon_full_shape
if group.params_buffer.fp8 is not None:
    share_scratch(param, group.dtype)
    param._clear_data()
param.get_tensor()._set_dims(full)
param.original_shape = list(full)
```

Muon 组在 step 时把参数绑成主权重分片的 view。fp8 组的主权重是 fp32，绑定后参数也变成 fp32。解绑时先 `share_scratch` 换回 param dtype 的存储，再 `_clear_data`，参数就恢复成 bf16、无存储、完整 shape 的空壳，和非 fp8 组一致。

### 3.12 Checkpoint

fp8 组的主权重可能是 fp32，但存档仍要和常规路径一样：模型权重是 param dtype，fp32 精度放在 master weight 里。

**保存模型**：`FullyShardFusion.sharded_state_dict` 对 fp8 组调用 `build_ckpt_weight(params_buffer, group.dtype)`：

```python
main = params_buffer.data_buffer
weight = main if main.dtype == dtype else main.astype(dtype)   # fp32 主权重导出 bf16 副本
params_buffer.fp8.weight_copy = weight
```

之后各参数的 `ShardedWeight` 都建在 `weight` 上。

**保存优化器**：优化器的 sharded state 里，对 fp32 主权重的 fp8 组，按 `_shard_descs` 为每个参数加一项 `f"{struct_name}.w_0"`，直接指向 fp32 主权重分片。这和常规 master weight 用的 key 规则相同（同一函数里普通 master weight 的 key 也是 `f"{struct_name}.w_0"`）。主权重是 bf16 的组（`multi_precision=False`）没有 master，不加这项。

**加载**：

- 模型权重写进 `weight_copy`；如果主权重本身就是 bf16，`weight_copy` 就是主权重，直接原地写入。
- `.w_0` 直接写进 fp32 主权重。
- 下一次 forward pre-hook 调用 `sync_ckpt_weight_to_main`：

```python
if weight is not main and not equal_all(main.astype(weight.dtype), weight):
    assign(weight.astype(main.dtype), main)
fp8.quantized = False
```

全量续训时，加载后的 bf16 权重等于 fp32 主权重舍入后的值，判断为相等，保留 fp32 精度。只加载模型权重时（没有 `.w_0`），两者不等，用 bf16 权重覆盖主权重。

测试 `check_fp8_checkpoint` 覆盖了 AdamW+multi_precision、Muon+multi_precision、AdamW 无 multi_precision 三种组合：导出的模型权重是 bf16，`.w_0` 是 fp32 且舍入后等于模型权重；保存再加载后下一步 loss 逐位一致；只加载模型权重时主权重恢复为存档的 bf16 值。测试没有覆盖 fp8 组与非 fp8 组之间互相加载对方的存档。

### 3.13 `unshard_params` 拒绝 fp8 组

fp8 组没有 bf16 完整权重可以 gather，`FullyShardFusion.unshard_params` 遇到 fp8 组直接抛：

```python
RuntimeError("unshard_params: fp8 groups have no bf16 weight")
```

所以任何依赖 `unshard_params` 读取专家 bf16 完整权重的代码，在 fp8 组上都不能用。PaddleFleet 原有的逐层预量化 `_fsdp_fp8_quant_weight` 就属于这种，所以 #2371 在专家已被标记时跳过它（见 4.2）。

### 3.14 一个 step 内的状态

| 时机 | fp8 分片 `shards` | `quantized` | gather buffer `fulls` / 参数上的缓存属性 |
|---|---|---|---|
| step 开始，第一个 unit 的 pre-hook | 用当前主权重重新写入 | `False` → `True` | 无 |
| unit forward | 不变 | `True` | 有，unit 结束释放 |
| unit backward（重新 gather） | 不变 | `True` | 有，unit 结束释放 |
| `reset_params_buffer_status` | 不变 | `True` → `False` | 清空 |
| `optimizer.step` | 不变（主权重被更新） | `False` | 无 |

---

## 4. PaddleFleet #2371：给专家权重挂量化器

### 4.1 `mark_fsdp_fp8_params`（`transformer/moe/fp8_utils.py`）

```python
def mark_fsdp_fp8_params(args, model):
    if not getattr(args, "offline_quant_expert_weight", False):
        return
    try:
        import paddle.distributed.fsdp._fp8_utils
    except ImportError:
        return                      # 老版本 paddle：保持逐 step 预量化
    if any(getattr(m, "use_ue8m0", False) for m in model.sublayers()):
        raise NotImplementedError("fsdp fp8 param gather: no ue8m0 support.")

    def quantizer(weights, transpose):
        return fused_stack_quant_without_cache(
            [weights[i] for i in range(weights.shape[0])], transpose=transpose
        )

    quantizer.block = 128
    keep_transpose = getattr(args, "using_cache_weight_quant_transpose", True)
    quantizer.attrs = (
        ("fp8_weight_stacked", "fp8_scale_stacked", False),
        ("fp8_weight_stacked_transpose", "fp8_scale_stacked_transpose", True),
    )[: 2 if keep_transpose else 1]
    for layer in model.sublayers():
        if isinstance(layer, GroupedMLPExpert) and not isinstance(layer, SonicMoEExpert):
            layer.weight1._fsdp_fp8_quantizer = quantizer
            layer.weight2._fsdp_fp8_quantizer = quantizer
```

逐条说明：

- 启用条件：`offline_quant_expert_weight=True`，且 paddle 里有 `_fp8_utils`。导入失败时什么都不做，PaddleFleet 可以和不含 #79897 的 paddle 搭配使用，自动退回原来的逐 step 预量化。
- ue8m0 scale 不支持，直接报错，避免静默走错路径。
- 量化器复用已有的 `fused_stack_quant_without_cache`：把分片里的每个专家拆成列表传进去。Blackwell（SM100）上它会自动使用 pow2 scale。
- 是否缓存转置版由 `using_cache_weight_quant_transpose` 决定，缺省为 `True`，即 k=2。
- 只标记 `GroupedMLPExpert` 的 `weight1`/`weight2`，`SonicMoEExpert` 除外。所有专家共用同一个 `quantizer` 对象，所以同层两个权重会被分进同一个 buffer group（见 3.2）。

### 4.2 `FP8QuantWeightCallback`（`trainer/trainer_callback.py`）

新增 `on_init_end`：

```python
def on_init_end(self, args, state, control, **kwargs):
    if args.using_sonic_moe or not _fsdp_owns_param_storage(args):
        return
    mark_fsdp_fp8_params(args, kwargs["model"])
```

`on_init_end` 在 Trainer 初始化结束时触发，早于 `_wrap_model` 里的 `fully_shard`，所以分组时量化器已经挂好。只在 `--sharding fsdp` 下生效。

`on_step_begin` 的 FSDP 分支里，只要有参数挂了量化器就跳过逐层预量化：

```python
if not any(getattr(p, "_fsdp_fp8_quantizer", None) is not None
           for p in model.parameters()):
    _fsdp_fp8_quant_weight(model, True, quant_transpose=False)
moe_weights_name = _fsdp_expert_master_weight_names()
```

这里必须跳过：`_fsdp_fp8_quant_weight` 依赖 `unshard_params`，而 fp8 组会让它抛错（3.13）。

`offload_fp8_expert_master_weight` 的逻辑没改，仍按 `_fsdp_expert_master_weight_names()` 去 `_master_weights` 里找。`multi_precision=True` 时 fp8 组没有 master weight，fp32 主权重就是参数本身，所以这些名字都找不到，不会 offload 任何东西，fp32 主权重留在显存里。

### 4.3 ERNIEBot 侧

ERNIEBot 用自己的回调 `ernie5/src/callbacks/fp8_quant_weight_callback.py`。当前工作区里它已有对应修改（尚未提交）：`on_init_end` 在 FSDP 下调用 `mark_fsdp_fp8_params`；`on_step_begin` 在 FSDP 分支里，参数挂了量化器时跳过 `_fsdp_fp8_quant_weight`。和 PaddleFleet 版的差别是没有判断 `using_sonic_moe`。

### 4.4 单测

`tests/single_card_tests/fp8/test_fp8_fsdp_callbacks.py` 新增：

- `test_marked_experts_skip_pre_quant`：有参数挂了量化器时，`on_step_begin` 既不调用 `unshard_params`，也不调用各层的 `fp8_quant_weight`。
- `TestMarkFSDPFP8Params`：
  - `on_init_end` 只在 FSDP 且非 sonic 时调用 `mark_fsdp_fp8_params`；
  - `offline_quant_expert_weight=False` 或 paddle 缺 `_fp8_utils` 时不标记；`use_ue8m0` 时报错；
  - 只标记 grouped 专家，不标记 sonic 专家；量化器把 `[E, ...]` 拆成专家列表调用 `fused_stack_quant_without_cache`；`using_cache_weight_quant_transpose=False` 时只有一项 attrs。

---

## 5. Paddle #79899：`grad_comm_dtype`

### 5.1 接口与校验（`fully_shard.py`）

`MixedPrecisionPolicy` 新增字段：

```python
@dataclass
class MixedPrecisionPolicy:
    param_dtype: paddle.dtype | None = None
    reduce_dtype: paddle.dtype | None = None      # 梯度累加 dtype
    output_dtype: paddle.dtype | None = None
    cast_forward_inputs: bool = True
    grad_comm_dtype: paddle.dtype | None = None   # 梯度 reduce-scatter 的通信 dtype
```

- `fully_shard` 入口校验：只接受 `None`、`paddle.float32`、`paddle.bfloat16`，其他值（包括字符串 `"bfloat16"`、`float16`）抛 `ValueError`。
- `_fully_shard_auto_parallel` 遇到非 `None` 抛 `NotImplementedError`，只支持手动并行路径。
- `FullyShardFusion` 把它传给 `FSDPBufferManager`，再传给每个 grads buffer 的 `TensorFusionBuffer`。

### 5.2 什么时候真正生效

`TensorFusionBuffer.__init__`：

```python
self.grad_comm_dtype = grad_comm_dtype if grad_comm_dtype != self.main_grad_dtype else None
self._rs_comm_buffer = None
```

通信 dtype 和梯度累加 dtype 相同时置为 `None`，走原路径。grads buffer 的累加 dtype 规则是：专家参数和 fp32 参数固定用 fp32，其余用 `mp_policy.reduce_dtype`（`None` 时为 fp32）。在 PaddleFleet trainer 里：

| trainer 配置 | 非专家组累加 dtype | 专家组累加 dtype | `grad_comm_dtype=bf16` 的作用范围 |
|---|---|---|---|
| `amp_master_grad=False`（trainer 会设 `reduce_dtype=bf16`） | bf16 | fp32 | 只有专家组 |
| `amp_master_grad=True`（`mp_policy=None`） | fp32 | fp32 | 所有分片的 grads buffer |

### 5.3 reduce-scatter 流程

`do_reduce_scatter`：

```python
tmp_buffer = self.get_tmp_buffer()          # fp32 完整梯度
# ... 先按原逻辑决定 AVG 或者先 scale_ 再 SUM（在 fp32 上做）
if self.grad_comm_dtype is None:
    self._rs_comm_buffer = tmp_buffer
else:
    self._rs_comm_buffer = paddle.cast(tmp_buffer, self.grad_comm_dtype)
    self.tmp_data_buffer = None             # buffer 不再持有 fp32 完整梯度
full = self._rs_comm_buffer
shard = full._slice(0, self.data_buffer.shape[0])
# comm stream 上：reduce_scatter(shard, full)
```

`accumulate_reduced_grad`（等通信完成后调用）：

```python
shard = self._rs_comm_buffer._slice(0, self.data_buffer.shape[0])
self._rs_comm_buffer = None
if shard.dtype != self.data_buffer.dtype:
    shard = shard.astype(self.data_buffer.dtype)   # bf16 -> fp32
self.data_buffer.add_(shard)                       # 在 fp32 梯度分片上累加
self.clear_tmp_buffer()
```

只有通信这一段是 bf16。scale 在 cast 之前、在 fp32 上做；多个 micro-batch 的结果在 fp32 分片上累加。`AVG` 分支的 1/n 由 NCCL 在 bf16 上做。

`FSDPCommManager._reduce_group_grads` 在发起 reduce-scatter 之后，立即调用 `_release_reduced_grad_views(grads_buffer)`，清掉参数上指向 fp32 完整梯度的 `main_grad` view。原来这一步要等通信完成（`_retire_grad_comm`）才做。

提前释放的原因：开了 `grad_comm_dtype` 后，`tmp_data_buffer` 已置空，但参数的 `main_grad` view 还引用着 fp32 完整梯度的存储。把 view 也清掉，这块 fp32 存储在 cast 之后就能释放，通信期间只剩 bf16 那份。reduce-scatter 发出时该组梯度已经算完，之后不会再往 `main_grad` 里累加，所以提前清 view 是安全的；不开 `grad_comm_dtype` 时，`_rs_comm_buffer` 仍引用 fp32 buffer，提前清 view 不影响通信。

### 5.4 精度

`run_grad_comm_dtype`（`dygraph_group_fsdp.py`）用同样的 MoE 模型和数据跑 `None` 与 `bfloat16` 两组，断言：

- 非法 dtype 被拒绝；
- 每个分片的 grads buffer 上 `grad_comm_dtype` 是否为 `None` 与配置一致；
- 第一个 optimizer step 前的 `ACCUM_STEPS` 个 loss 逐位相同（forward 不受影响）；
- 第一个 step 的梯度分片满足 `atol = 1e-2 * max|g|`，并且至少有一个分片和 fp32 版不完全相同，用来证明确实走了 bf16 通信；
- 全部 loss 的相对误差小于 1e-2；
- 训练结束后 `grad_reduce_queue` 为空，所有 `_rs_comm_buffer` 都已释放。

### 5.5 接入方式

PaddleFleet trainer 目前没有把 `grad_comm_dtype` 暴露成训练参数。`trainer.py` 里只在 `in_fsdp_mode and not amp_master_grad` 时构造 `MixedPrecisionPolicy(reduce_dtype=...)`。要用这个功能，需要在构造 `fsdp_mp_policy` 的地方传入：

```python
from paddle.distributed.fsdp.fully_shard import MixedPrecisionPolicy, fully_shard

model = fully_shard(
    model,
    optimizer=opt,
    mp_policy=MixedPrecisionPolicy(grad_comm_dtype=paddle.bfloat16),
)
```

---

## 6. 显存与通信量估算

下表按“每个专家权重元素”计，单位是字节。d 为 FSDP 组大小，k 为缓存份数，优化器为 AdamW 且 `multi_precision=True`。scale 每元素约 `4/128² ≈ 0.0002` 字节，忽略不计。“改动前”指 PaddleFleet/ERNIEBot 的逐 step 预量化路径，它只缓存非转置一份。

| 项目 | 改动前 | fp8 param gather |
|---|---|---|
| 常驻：参数分片 | bf16 分片 2/d | fp32 主权重 4/d |
| 常驻：master weight | 4/d | 无 |
| 常驻：Adam m、v | 8/d | 8/d |
| 常驻：fp8 | 所有层完整 fp8 缓存 1（不除以 d） | fp8 分片 k/d |
| 正在计算的 unit 内临时 | bf16 完整权重 2 | fp8 完整缓存 k |
| 每次 all-gather 通信量 | 2 | k |
| 每 step 额外的 gather | 预量化多一轮 2 | 无 |
| 每卡量化计算量 | 完整权重 | 1/d |

可以看出：

- 主要的显存收益来自去掉“所有层完整 fp8 缓存常驻”，以及 bf16 分片 + master 合并成一份 fp32 分片。#79897 的描述给出的数据是：两机 35B 参数 MoE 模型，峰值显存约降低 4 GB。
- k=2（`using_cache_weight_quant_transpose=True`，默认）时，单次 gather 的字节数和 bf16 相同，通信收益来自去掉预量化那一轮 gather。k=1 时单次 gather 减半。

`grad_comm_dtype=bf16` 对生效的 grads buffer：reduce-scatter 通信量从 4 字节/元素降到 2。cast 的瞬间 fp32 和 bf16 两份完整梯度同时存在，之后 fp32 那份释放（5.3），通信期间只占 2 字节/元素。

PR #79897 描述中的精度验证：单机 8 卡，h512 L8 模型，fsdp8 ep4，moe_sharding2，Muon/MuonH + AdamW + AdamH，开梯度裁剪和 fp8，mb2，在消除规约顺序和批量 Muon NS 差异后，FSDP 与 sharding stage1 的 20 步 loss 逐位一致。

---

## 7. 测试

### 7.1 怎么跑

```bash
# Paddle：2 卡分布式测试（CI 入口）
cd Paddle
python -m pytest test/collective/fleet/test_dygraph_group_sharded_api_for_eager.py -k fsdp

# 也可以直接起
python -m paddle.distributed.launch --devices 0,1 test/collective/fleet/dygraph_group_fsdp_fp8.py
python -m paddle.distributed.launch --devices 0,1 test/collective/fleet/dygraph_group_fsdp.py

# Paddle：单进程单测（fake buffer 补了 fp8=None 字段）
python -m pytest test/legacy_test/test_fsdp_fully_shard_fusion.py

# PaddleFleet：回调单测
cd PaddleFleet
python -m pytest tests/single_card_tests/fp8/test_fp8_fsdp_callbacks.py
```

### 7.2 `dygraph_group_fsdp_fp8.py` 测了什么

测试模型 `GroupedExpertModel` 每层有 `weight1 [E, H, 2I]`、`weight2 [E, I, H]`。`FP8ExpertGEMM` 只从 fp8 缓存反量化出权重来算，读不到缓存就会出错，所以测试本身就验证了“forward/backward 只读缓存”。

- `check_unaligned_fp8_raises`：3 个专家的层无法对齐分片，`fully_shard` 必须抛 `"is not sharded"`。
- `run_fp8_gather`：对 `(keep_transpose, muon, multi_precision)` 的 4 种组合，分别跑参照路径（`unshard_params` 后量化完整 bf16 权重）和 fp8 gather 路径：
  - 每层 forward 后记录缓存的 md5，与“把所有分片 all-gather 后量化完整权重”的期望值逐位比较；
  - 两条路径 step 0 的缓存和前 `ACCUM_STEPS` 个 loss 逐位相同。只比 step 0，是因为 optimizer step 后“bf16 参数 + fp32 master”和“fp32 主权重”在部分架构（如 Volta）上舍入可能不同；
  - 主权重 dtype 符合 `multi_precision`，没有 master weight，参数 dtype 仍是 bf16；
  - 训练结束后 `unshard_params` 必须抛 `RuntimeError`。
- `run_fp8_checkpoint`：见 3.12。

`test/legacy_test/test_fsdp_fully_shard_fusion.py` 的 fake params buffer 都补了 `fp8=None`。以后再写这类 fake 对象，也要带上这个字段，否则 `FSDPCommManager` 里新增的判断会报 `AttributeError`。

---

## 8. 限制与注意事项

- fp8 gather 只覆盖 3 维专家权重，`K`、`N` 须是 128 的整数倍，分片须能对齐到整数个专家（3.3），否则 `fully_shard` 报错。
- 不支持 ue8m0 scale，不支持 `SonicMoEExpert`。
- fp8 组的参数是 0 字节空壳。forward/backward 只能读 fp8 缓存属性；`unshard_params` 对 fp8 组直接报错，依赖它取专家 bf16 完整权重的工具都不能用。
- fp8 组的 gather buffer 不计入 `comm_unit_size` 预算，也不会被预算驱逐（3.10）。
- `multi_precision=True` 时 fp8 组没有 master weight，`offload_fp8_expert_master_weight` 对它们不起作用（4.2）。
- `grad_comm_dtype` 只支持手动并行路径，只接受 fp32/bf16，PaddleFleet trainer 还没暴露这个参数（5.5）。

---

## 9. 给新的模块接入 fp8 param gather

FSDP 侧不需要改代码，只要在模型侧满足 3.1 的约定。步骤：

1. **确认权重形状**：参数是 `[E, K, N]`，`K`、`N` 是 128 的整数倍。
2. **改计算路径**：forward、重计算和 dgrad 只从 `attrs` 里约定的属性读 fp8 数据和 scale，不读参数本身。wgrad 不需要权重，梯度照常交给 FSDP 的 grads buffer。
3. **写量化器**：输入 `[E_local, K, N]` 的 param dtype 张量，按 3.1 的布局返回 `(data, scale)`；设置 `block` 和 `attrs`。同一个 buffer group 里的参数共用一个量化器对象。
4. **在 `fully_shard` 之前挂上**：在 `on_init_end` 一类早于 `_wrap_model` 的时机设置 `param._fsdp_fp8_quantizer`。可以参照 `mark_fsdp_fp8_params` 扩展它的匹配条件。
5. **关掉旧的预量化**：参数挂了量化器后，不要再对它调用 `unshard_params`（4.2 的写法）。
6. **核对对齐**：用 3.3 的方法，按实际的 `E`、`K`、`N` 和 FSDP 组大小确认能分片。
7. **验证**：参照 `run_fp8_gather`，比较 fp8 gather 的缓存 md5 和量化完整 bf16 权重的结果，以及 step 0 的 loss；参照 `check_fp8_checkpoint` 验证存取。
