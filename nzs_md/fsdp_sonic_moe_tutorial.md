# FSDP 支持 SonicMoE：代码修改与原理讲解

本文记录让 `using_sonic_moe: true` 能在 `sharding: fsdp`（Paddle 原生 `fully_shard`）下训练的全部改动，并逐段解释代码。读完后应能回答三个问题：

1. 原来的 SonicMoE 流程为什么在 FSDP 下跑不了。
2. 新方案里权重、fp8 量化结果、梯度分别存在哪里、什么时候创建和释放。
3. 实现过程中遇到的段错误、显存泄漏、OOM 是怎么来的，怎么修的。

验证用的配置是 `conf/experiment/ERNIE-5.5-Flash/V0.2.1/ernie5_5_flash_ladder_v0.2.1_C1_T3_h1024_L16-pretrain-8k-hzz1-fsdp.yaml`（SM90、fp8 e4m3、`fp8_weight_quant_format=128x128`、ep8、`recompute_modules: ["moe_gate_up"]`），启动脚本是 `run_fsdp_test_1node.sh`。

## 1. 改动的文件

| 文件 | 改动 |
|---|---|
| `third_party/PaddleFleet/src/paddlefleet/transformer/moe/moe_expert.py` | 主体。`SonicMoEExpert` 新增 FSDP 前向/反向路径，新增 `_FSDPSonicGradBinder` |
| `third_party/PaddleFleet/src/paddlefleet/trainer/trainer_callback.py` | `SonicMoELayoutSwitchCallback` 去掉"FSDP 直接报错"，改为 FSDP 分支 |
| `ernie5/src/callbacks/sonicmoe_callback.py` | ERNIE 实际注册的 `SonicMoECallback`（`pretrain.py` 里注册），加同样的 FSDP 分支 |
| `third_party/PaddleFleet/tests/single_card_tests/fp8/test_fp8_fsdp_callbacks.py` | 新增 `TestSonicMoECallbackUnderFSDP` 4 个用例 |
| `third_party/PaddleFleet/tests/single_card_tests/moe/test_sonic_moe_fsdp_expert.py` | 用假 fusion buffer 和假 `run_sonic_moe` 测副本的生成、梯度回传、存储释放与回填 |
| `third_party/PaddleFleet/tests/multi_card_tests/moe/test_sonic_moe_fsdp.py` | 双卡真实 `fully_shard` + 真实 sonic kernel，与非 FSDP 路径对齐 loss 和专家梯度 |

## 2. 背景：两种权重布局和原来的流程

### 2.1 grouped 布局与 sonic 布局

`SonicMoEExpert` 的两个专家权重有两种存放方式：

| 权重 | grouped 布局（优化器、checkpoint 用） | sonic 布局（sonic kernel 用） |
|---|---|---|
| `weight1`（gate+up） | `[E, H, 2I]`，gate 和 up 前后拼接 | `[E, 2I, H]`，gate 和 up 按行交错 |
| `weight2`（down） | `[E, I, H]` | `[E, H, I]`，只是转置 |

两者之间的转换由 `_grouped_w1_to_sonic` / `_sonic_w1_to_grouped`（w1）和 `_transpose_w2_layout`（w2）完成。注意 w1 的转换不只是转置，还要重排 gate/up，所以 w1 的梯度从 sonic 布局回到 grouped 布局时必须做一次真正的数据搬运。

### 2.2 stage1 下原来的流程

非 FSDP（例如 sharding stage1）时，每个 step 的流程是：

```
on_step_begin:
    convert_weights_to_sonic_layout()   # 原地改写参数存储为 sonic 布局
    quant_weight()                       # 生成 weight.fp8
    optimizer.clear_param_storage(...)   # 释放 bf16 参数存储，只留 fp8
forward / backward:
    sonic kernel 直接用 self.weight1/weight2 (sonic 布局) 和 .fp8
    wgrad 写进 weight.main_grad
on_optimizer_begin:
    clear_fp8_weights()
    convert_weights_to_grouped_layout()  # 改回 grouped，交给优化器
```

关键点：**布局切换是在前向之外、直接原地改写参数存储**。

### 2.3 FSDP 为什么不允许这样做

`fully_shard` 把每个参数打平放进一块 fusion buffer，每张卡只存自己那一片。参数上会被挂一个 `_fusion_buffer` 属性。对 FSDP 参数来说：

- 在所属 unit 的 forward/backward 之外，`param` 没有存储，是一个空壳。callback 里根本拿不到完整权重。
- unit 前向前 FSDP 会 all-gather 出完整权重；对这份 gather 结果的原地改写，下一次 gather 时就被覆盖了。
- 优化器（Muon 的 2D 切片规则）和 `sharded_state_dict` 都假定权重是 grouped 布局。
- FSDP 下的优化器不是 sharding optimizer，没有 `clear_param_storage`。
- 专家参数的 `main_grad` 不是常驻的：只在反向时由 `FusionBackwardHook` → `_bind_sharded_expert_main_grads` 绑定到 buffer 里的 fp32 视图，reduce 完后 `_release_reduced_grad_views` 又把它清掉。

所以原来的 `SonicMoELayoutSwitchCallback` 在 FSDP 下直接抛 `NotImplementedError`。

## 3. 总体设计

思路是：**参数永远保持 grouped 布局，sonic 需要的东西全部放在一份"副本"上，副本在前向里、权重已经 gather 好的时候生成。副本的数据（bf16 和 fp8 结果）跟 FSDP 的 gather 一样，用时才有，用完就释放。**

```
              FSDP 参数 (grouped, 分片)
                     │ unit 前向时 all-gather
                     ▼
     第一个 micro-batch 的 forward：_fsdp_sonic_copies()
       w1 = grouped→sonic(weight1).contiguous()
       w2 = transpose(weight2).contiguous()
       量化 → w1.fp8, w2.fp8
       SM90: bf16 数据换成 stride-0 占位张量（只留 shape 和 .fp8）
       w1.main_grad / w2.main_grad = 无存储的 fp32 占位张量
                     │ 副本对象缓存在 self._fsdp_sonic_weights，本 step 所有 micro-batch 复用
                     │ 之后的 forward 先回填数据（_fsdp_refill_sonic_weights）
                     ▼
     run_sonic_moe(x, ..., w1, w2)   # sonic 在前向里抓住 w.main_grad 作为 wgrad 累加器
                     │                # sonic 的 ctx 也一直引用 w1/w2 对象，直到反向
     _fsdp_free_sonic_weights()      # 释放副本和 fp8 结果的数据，对象保留
                     │
     反向：_FSDPSonicGradBinder.backward（先于 sonic 反向执行，此时 FSDP 已 gather）
       用 gather 好的权重重新生成副本数据，填回原来的对象
       w1.main_grad ← 挂上新的 fp32 零存储
       w2.main_grad ← 直接共享 FSDP weight2 的 main_grad 存储
                     │
     sonic 反向算 wgrad，写进累加器，然后调用权重上的 backward hook
       flush_w2_grad: 已经写在 FSDP 存储里，只解除共享
       flush_w1_grad: sonic→grouped 转换后加到 FSDP weight1.main_grad，释放临时存储，
                      再释放两份副本的数据
                     ▼
     on_optimizer_begin: clear_fp8_weights() 丢掉副本对象，下个 step 重新生成
```

这样做的好处：

- 不改写任何 FSDP 参数存储，优化器和 checkpoint 看到的始终是 grouped 布局。
- 副本只在 gather 之后、forward/backward 内部有数据，符合 FSDP 的约束，各层不会同时常驻一份未切分的专家权重。
- 梯度最终进入 FSDP 的 main_grad，FSDP 的 reduce-scatter 流程不需要任何修改。

## 4. 代码逐段讲解：`moe_expert.py`

### 4.1 判断是否由 FSDP 管理：`_fsdp_owns_weights`

```python
def _fsdp_owns_weights(self):
    return all(
        getattr(w, "_fusion_buffer", None) is not None
        for w in (self.weight1, self.weight2)
    )
```

`fully_shard` 会给它打平的每个可训练参数设置 `_fusion_buffer`，这是判断"参数存储归 FSDP 管"最直接的标志。后面所有 FSDP 分支都靠这个函数分流。`SonicMoEExpert` 拿不到训练参数，所以层内看参数本身，callback 看 `args.sharding`。

冻结的专家是例外：`fully_shard` 照样分片，但不设 `_fusion_buffer`，反向也不 gather（`FusionBackwardHook` 只 gather trainable 参数），而 sonic 反向的 dgrad 和回填都要读权重。所以这种情况不支持。为了不让层内静默走非 FSDP 路径，两个 callback 在 `on_train_begin` 调用 `_check_sonic_moe_experts_under_fsdp`：`args.sharding` 含 FSDP 但某个专家的 `_fsdp_owns_weights()` 为 False 时直接抛 `ValueError`。用 `all` 而不是 `any`，是为了只冻结 weight1 或 weight2 的情况也能拦住。

### 4.2 关掉原地布局切换和预量化

```python
def _convert_layout(self, target_layout, weight1_converter, weight2_converter):
    if self._fsdp_owns_weights():
        # Weights stay grouped; forward uses separate sonic copies.
        return
    ...
```

`convert_weights_to_sonic_layout` 和 `convert_weights_to_grouped_layout` 都走 `_convert_layout`。FSDP 下直接返回，保证参数永远是 grouped 布局。这样外部代码（比如 `sharded_state_dict` 里调用的 `convert_weights_to_grouped_layout`）即使调用了也是安全的空操作。

```python
@paddle.no_grad()
def quant_weight(self):
    if self._fsdp_owns_weights():
        self.clear_fp8_weights()
        self._forward_counter = 0
        return
    self.convert_weights_to_sonic_layout()
    self.clear_fp8_weights()
    self._quantize_into(self.weight1, self.weight2)
    self._forward_counter = 0

def need_quant_weight(self):
    if self._fsdp_owns_weights():
        return False
    ...
```

FSDP 下参数在前向外是空壳，没法量化，所以 `quant_weight` 只清状态，`need_quant_weight` 永远返回 False，量化推迟到前向里对副本做。

### 4.3 把量化逻辑抽成 `_quantize_into`

原来的 `quant_weight` 把量化结果直接写到 `self.weight1.fp8` 上。现在拆成：

```python
def _quantize_into(self, weight1, weight2):
    """Attach the fp8 payload of sonic-layout ``weight1``/``weight2``."""
    if get_gpu_capability() == 9:
        ...
        payload = quantize_sm90_weights(weight1, weight2, block_size)
        weight1.fp8 = (w1_fp8, w1_scale)
        weight2.fp8 = (w2_fp8, w2_scale)
        return
    ...  # SM100: quantize_native_fp8_weights，同时生成 transposed_fp8
```

参数化之后，stage1 路径传 `self.weight1/self.weight2`，FSDP 路径传副本 `w1/w2`，两条路径共用同一套量化代码，量化结果逐位一致。这是后面"FSDP 与非 FSDP 结果逐位相同"能成立的前提。`_forward_counter = 0` 从量化函数里移回 `quant_weight`，因为它是 step 级状态，不属于"量化一对张量"这个动作。

### 4.4 `clear_fp8_weights` 同时丢掉副本

```python
def clear_fp8_weights(self):
    clear_all_fp8_weight_caches()
    self._fsdp_sonic_weights = None
    for weight in (self.weight1, self.weight2):
        weight.fp8 = None
        weight.transposed_fp8 = None
```

副本是从本 step 的权重算出来的，优化器更新后就过期了，所以跟 fp8 结果一起丢掉。`on_optimizer_begin` 会调用这个函数。`__init__` 里新增 `self._fsdp_sonic_weights = None` 作为初始状态。这里丢的是副本**对象**；副本的数据在每次前向、反向用完后就已经释放了（见 4.6）。

### 4.5 副本的生成：`_fsdp_sonic_copies`

这是整个方案的核心，分三部分看。

**(1) 生成并量化副本，对象每个 step 只建一次**

```python
def _fsdp_build_sonic_weights(self):
    with paddle.no_grad():
        w1 = self._grouped_w1_to_sonic(self.weight1).contiguous()
        w2 = self._transpose_w2_layout(self.weight2).contiguous()
        if self.sonic_moe_config.enabled:
            self._quantize_into(w1, w2)
            if get_gpu_capability() == 9:
                w1, w2 = (self._fp8_shape_holder(w) for w in (w1, w2))
    return w1, w2

def _fsdp_sonic_copies(self):
    if self._fsdp_sonic_weights is not None:
        self._fsdp_refill_sonic_weights()
        return self._fsdp_sonic_weights
    w1, w2 = self._fsdp_build_sonic_weights()
    ...  # 见 (3)
```

- 调用发生在 forward 内，此时 FSDP 已经 gather 好 `self.weight1/weight2`。
- `paddle.no_grad()` 让副本成为与 FSDP 参数没有 autograd 连接的叶子张量。梯度不能经 autograd 流回 FSDP 参数（那样会走 FSDP 的普通 grad 路径，形状和布局都不对），而是由下面的 hook 手动送回去。
- `.contiguous()` 保证副本有独立、连续的存储，sonic kernel 需要连续内存。
- 副本对象缓存到 `self._fsdp_sonic_weights`，同一个 step 的 8 个 micro-batch 共用同一组对象（hook 和 main_grad 占位只注册一次）。但对象的数据在上一次使用后已被释放，所以后续 forward 要先调用 `_fsdp_refill_sonic_weights` 重新生成数据（见 4.6）。重计算（`moe_gate_up`）时再跑一次 forward 也一样先回填。
- 生成逻辑抽成 `_fsdp_build_sonic_weights`，首次构建和回填共用。

**(2) SM90 上把 bf16 数据换成占位张量**

```python
@staticmethod
def _fp8_shape_holder(weight):
    holder = paddle.as_strided(
        paddle.zeros([1], dtype=weight.dtype),
        weight.shape,
        [0] * len(weight.shape),
    )
    holder.fp8 = weight.fp8
    holder.transposed_fp8 = getattr(weight, "transposed_fp8", None)
    return holder
```

最初的版本里副本数据整个 step 常驻，16 层的 bf16 副本会额外占一大块显存，第 2 步就 OOM（见第 7 节），占位张量就是为此引入的。现在副本数据在用完后会释放（4.6），但占位张量仍有用：SM90 上连生成和使用副本的这段时间也不需要 bf16 数据，量化完就可以丢。在 SM90 上：

- `sm90_funcs.segments_aligned` 永远返回 True，sonic 总是走 fp8 aligned 路径；
- 这条路径的前向用 `w.fp8`，反向用 `ctx._w1T_fp8`（来自 `.fp8`，SM90 上在反向里临时转置），`save_for_backward` 不保存 bf16 的 w1/w2（见 sonic `functional/__init__.py` 中 "w1 omitted — backward uses ctx._w1T_fp8 + metadata"）；
- 它只读副本的 `shape`、`dtype` 和 `place`（`ctx._w1_device = w1.device`）。

所以可以用"1 个元素、所有 stride 为 0"的视图替代副本：形状、dtype、place 都对，但只占 1 个元素的显存。

为什么不在量化后直接 `w1._clear_data()`？因为 sonic 前向会读 `w1.device`，而 Paddle 对没有存储的张量调用 `place` 会报 `Tensor not initialized yet when DenseTensor::place() is called`。stride-0 视图有真实存储，可以正常回答 `place`。4.6 里在前向**结束后**释放数据是安全的：sonic 已在前向里把 device 存进 ctx，反向开始前数据又会被回填。

SM100 上遇到未对齐的 segment 会回退到 bf16 计算，需要真实数据，因此只在 SM90 上替换。SM100 和不开 fp8 的 bf16 模式下，副本保留 bf16 数据，但同样在前向后释放、反向时回填。

**(3) 梯度占位张量和回传 hook**

先看 sonic 自己是怎么处理权重梯度的（`paddlefleet_ops/sonicmoe/functional/__init__.py`）：

```python
def _main_grad_accumulator(weight, shape=None):
    if weight is None or not hasattr(weight, "main_grad"):
        return None
    if weight.main_grad is None:
        weight.main_grad = paddle.zeros(...)
    ...
    return weight.main_grad

def _apply_weight_backward_hook(weight):
    if (weight is not None and hasattr(weight, "_apply_backward_hook")
            and not weight.stop_gradient):
        weight._apply_backward_hook()
```

- **前向**时，sonic 调用 `_main_grad_accumulator(w)`，把 `w.main_grad` 这个张量对象存进 ctx，作为反向 wgrad 的累加目标。w1 的累加器是 sonic 布局 `[E, 2I, H]`；w2 的累加器被 reshape 成 grouped 布局 `[E, I, H]`（Fleet 的 main_grad 本来就是 grouped 的）。
- **反向**时，sonic 把 wgrad 累加进这个张量，然后调用 `_apply_weight_backward_hook(w)` 触发权重上注册的 backward hook。

问题在于时序：sonic 在**前向**就抓住了 `main_grad` 对象，而 FSDP 专家参数的 main_grad 只在**反向**才绑定。所以不能在前向直接把 FSDP 的 main_grad 交给 sonic。解决办法是先给一个"空壳"，反向时再往空壳里装存储：

```python
def flush_w1_grad():
    if self._fsdp_sonic_weights is None:
        return
    grad = self._fsdp_sonic_weights[0].main_grad
    if grad._is_initialized():
        with paddle.no_grad():
            self._fsdp_main_grad(self.weight1).add_(
                self._sonic_w1_to_grouped(grad)
            )
        grad._clear_data()
    # w1's wgrad comes after w2's, so neither copy is needed anymore.
    self._fsdp_free_sonic_weights()

def flush_w2_grad():
    if self._fsdp_sonic_weights is not None:
        self._fsdp_sonic_weights[1].main_grad._clear_data()

for weight, grad_shape, hook in (
    (w1, w1.shape, flush_w1_grad),
    (w2, self.weight2.shape, flush_w2_grad),
):
    weight.main_grad = paddle.empty(grad_shape, dtype=paddle.float32)
    weight.main_grad._clear_data()
    weight.stop_gradient = False
    weight._register_backward_hook(hook)
self._fsdp_sonic_weights = (w1, w2)
```

逐行解释：

- `paddle.empty(...)` 后立刻 `_clear_data()`：得到一个形状正确、但没有显存的 fp32 张量。sonic 前向只抓它的引用，不读数据，所以空壳不会出错。
- w1 占位形状是 `w1.shape`（sonic 布局），w2 占位形状是 `self.weight2.shape`（grouped 布局），与 sonic 内部累加器的形状约定一致。
- `stop_gradient = False` 是为了让 `_apply_weight_backward_hook` 的条件 `not weight.stop_gradient` 成立。
- `_register_backward_hook(hook)` 注册的 hook 在 sonic 写完该权重 wgrad 后被调用：
  - `flush_w1_grad`：sonic 布局的 w1 梯度要经过 `_sonic_w1_to_grouped` 重排 gate/up，再 `add_` 到 FSDP `weight1` 的 main_grad 上。用 `add_` 而非赋值，是因为多个 micro-batch 要累加。加完后 `_clear_data()` 释放这块临时 fp32 显存。最后释放两份副本的数据：sonic 反向先跑 DownProjection（读 w2）、再跑 UpProjection（读 w1），w1 的 hook 最后触发，此时两份副本都不再需要。
  - `flush_w2_grad`：w2 的累加器直接共享了 FSDP 的存储（见 4.6），sonic 已经写到位，这里只需解除共享，避免占位张量继续引用 FSDP buffer。
  - `_is_initialized()` 检查用于跳过本次没有挂存储的梯度，`self._fsdp_sonic_weights is None` 检查用于副本已被清掉的情况。

**为什么 hook 不直接捕获 `w1`/`w2`，而是经过 `self._fsdp_sonic_weights` 查找？** hook 挂在 `w1` 上，如果闭包又引用了 `w1`，就形成"张量 → hook → 闭包 → 张量"的环。hook 保存在 C++ 侧，Python 的循环垃圾回收看不到这条边，这个环永远不会被回收。早期版本就是这么写的，结果每个 step 泄漏约 31.5 MB（单卡测试的规模），端到端训练第 2 步 OOM。现在闭包只引用 `self`，`clear_fp8_weights` 把 `_fsdp_sonic_weights` 置空后，副本就没有引用了。

`_fsdp_main_grad` 是一个小工具：

```python
@staticmethod
def _fsdp_main_grad(param):
    param._fusion_buffer.rebind_main_grad(param)
    return param.main_grad
```

`rebind_main_grad` 让 FSDP 把参数的 main_grad 重新绑到 fusion buffer 中当前有效的 fp32 视图，保证拿到的一定是 FSDP 这一刻认可的存储。

### 4.6 副本数据的释放与回填，反向时挂梯度存储

**(1) 为什么要释放副本数据，为什么只在 flush 后释放不够**

sonic 的 ctx 从前向到反向一直引用副本：`ctx._w1_original`/`_w2_original`，bf16 或未对齐路径的 `save_for_backward(x, w1, ...)`，以及 fp8 aligned 路径通过 `_get_transposed_fp8_weight` 拿到的 fp8 结果。如果副本数据一直保留，前向跑完到反向开始之间，每一层都持有一份未切分的专家权重（SM100 和 bf16 模式下是 bf16 的 w1 `[E,2I,H]` 和 w2 `[E,H,I]`，SM90 是 fp8 结果），这正是 FSDP 要省掉的显存。只在 flush 之后释放降不了峰值，因为峰值出现在所有层前向完、反向还没回到这些层的时候。

所以副本数据的生命周期跟 FSDP 的 gather 对齐：前向用完就释放，反向时在 FSDP 重新 gather 之后回填，flush 之后再释放。副本**对象**不能换，因为 sonic 的 ctx 引用的就是这些对象，只能清掉再填回它们的底层存储：

```python
@staticmethod
def _fsdp_copy_storage(weights):
    tensors = []
    for weight in weights:
        tensors.append(weight)
        for attr in ("fp8", "transposed_fp8"):
            tensors.extend(getattr(weight, attr, None) or ())
    return tensors

def _fsdp_free_sonic_weights(self):
    for tensor in self._fsdp_copy_storage(self._fsdp_sonic_weights):
        tensor._clear_data()

def _fsdp_refill_sonic_weights(self):
    fresh = self._fsdp_copy_storage(self._fsdp_build_sonic_weights())
    for dst, src in zip(
        self._fsdp_copy_storage(self._fsdp_sonic_weights), fresh
    ):
        dst.get_tensor()._share_data_with(src.get_tensor())
```

- `_fsdp_copy_storage` 按固定顺序列出持有数据的所有张量：副本本身加上它的 fp8 结果（SM90 是 `(fp8, scale)`；SM100 1x32 是 `fp8` 和 `transposed_fp8` 各两个张量；iso32 的 `transposed_fp8` 是 `fp8` 的转置视图）。
- iso32 的转置视图也要列进来：`_clear_data()` 只释放当前张量的 holder，视图还引用着同一块分配，只清 `fp8` 的话显存不会真正释放。
- 回填时用 gather 好的权重重新生成一份，再用 `_share_data_with` 把新数据挂到旧对象上。对象、shape、stride 都不变（转置视图回填后仍是转置 stride，和 `fp8` 共享同一块存储），所以 sonic ctx 里拿到的引用依然有效。同一份 gather 权重重新量化，结果与第一次逐位相同。
- 代价是每次回填都要重做一次布局转换和量化：每个 micro-batch 的前向和反向各一次。

**(2) 反向入口：`_FSDPSonicGradBinder`**
```python
class _FSDPSonicGradBinder(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, bind_fn=None):
        ctx.bind_fn = bind_fn
        return x

    @staticmethod
    def backward(ctx, grad):
        ctx.bind_fn()
        ctx.bind_fn = None
        return grad
```

这是一个对数值完全透明的 PyLayer：前向原样返回，反向原样传梯度，唯一的作用是"在反向经过这里时执行一次 `bind_fn`"。它包在 sonic 输出外面，所以反向时它一定**先于** sonic 的反向执行，而此时 FSDP 已经为这个 unit 做完 pre-backward 处理（all-gather 权重、绑定专家 main_grad）。`ctx.bind_fn = None` 用来及时解开对 `self`/副本的引用。

**(3) 回填副本、挂梯度存储：`_fsdp_bind_sonic_grads`**

```python
def _fsdp_bind_sonic_grads(self, w1, w2):
    self._fsdp_refill_sonic_weights()
    w1_storage = paddle.zeros(w1.shape, dtype=paddle.float32)
    w2_storage = self._fsdp_main_grad(self.weight2)
    w1.main_grad.get_tensor()._share_data_with(w1_storage.get_tensor())
    w2.main_grad.get_tensor()._share_data_with(w2_storage.get_tensor())
```

- 第一行回填副本数据。权重刚被 FSDP gather 好，正好用来重新生成。1F1B 这类交错顺序（多个前向之后才轮到反向）也没问题：每次反向都自己回填，不依赖前向留下的数据。
- w1：sonic 布局和 FSDP 的 grouped 布局不同，没法共享，所以分配一块临时的 fp32 零张量，sonic 写完后由 `flush_w1_grad` 转换并累加。
- w2：sonic 写出来的就是 grouped 布局，形状与 FSDP `weight2.main_grad` 一致，直接共享存储，省一次拷贝和一块临时显存。
- `_share_data_with` 作用在底层 `DenseTensor` 上，让占位张量对象（sonic 已经在 ctx 里持有它）指向新存储，对象本身不变，所以 sonic 写的就是新存储。

**为什么要先把 `w1_storage`/`w2_storage` 存成局部变量？** 早期写法是 `w1.main_grad.get_tensor()._share_data_with(paddle.zeros(...).get_tensor())`。`get_tensor()` 返回的是对底层 `DenseTensor` 的引用，并不持有它；`get_tensor()` 一返回，临时张量的引用计数归零被释放，引用随之悬空，结果是间歇性段错误（gdb 栈在 `phi::DenseTensor::ShareDataWith` → `check_memory_size`）。用局部变量把源张量留到调用结束即可。

### 4.7 FSDP 前向：`_fsdp_forward`

```python
def _fsdp_forward(self, hidden_states, topk_indices, topk_scores, **kwargs):
    w1, w2 = self._fsdp_sonic_copies()
    w1.stop_gradient = True
    w2.stop_gradient = True
    try:
        out = run_sonic_moe(hidden_states, topk_indices, topk_scores,
                            self.K, self.num_local_experts, w1, w2, **kwargs)
    finally:
        w1.stop_gradient = False
        w2.stop_gradient = False
    self._fsdp_free_sonic_weights()
    if not out.stop_gradient:
        out = _FSDPSonicGradBinder.apply(
            out, bind_fn=lambda: self._fsdp_bind_sonic_grads(w1, w2)
        )
    return out
```

这里 `stop_gradient` 先设 True、调用完再设回 False，看起来多余，其实是为了控制 hook 的触发时机：

- 如果调用 sonic 时副本 `stop_gradient=False`，Paddle 会在 sonic 的 PyLayer 和副本之间建立 autograd 边。这条边在 PyLayer 反向**返回**时就会触发副本上的 backward hook。
- 但 sonic 有一种情况会把 wgrad 推迟：EP 下 wgrad 可能被放到 combine 通信的槽位里（`fp8_combine_grad_handle`），晚于 PyLayer 返回才算。这时 hook 先于 wgrad 触发，`flush_w1_grad` 会把还没写入的累加器转换并清掉，梯度就丢了。
- 调用时设 True，就不会建立这条边；调用后恢复 False，sonic 反向里 `_apply_weight_backward_hook` 的条件 `not weight.stop_gradient` 又成立。这样 hook 只会由 sonic 在真正算完 wgrad 后显式触发一次。

`try/finally` 保证即使 sonic 抛异常，副本的 `stop_gradient` 也会被恢复。

sonic 返回后立刻 `_fsdp_free_sonic_weights()`，对应 FSDP 在 unit 前向结束后 reshard：sonic 的 ctx 还引用着副本对象，但它们已经不占显存，直到反向时 binder 回填。不需要反向时（下一段）同样释放，下一次前向会回填。

`if not out.stop_gradient` 用来跳过不需要反向的场景（比如 eval / `no_grad`）。此时不需要挂存储，也不包 binder。

`lambda` 捕获 `w1`/`w2` 没有问题：这个 lambda 存在 binder 的 ctx 里，ctx 随本次反向图释放，并且 `backward` 里已经置空 `ctx.bind_fn`，不会像 4.5 那样形成 C++ 侧不可回收的环。

### 4.8 `forward` 入口

```python
def forward(self, hidden_states, topk_indices, topk_scores, use_fp8=False,
            tokens_per_expert=None, fp8_scale=None, recompute_moe_gate_up=False,
            fp8_combine_grad_handle=None, sync_free_sizing=False):
    if self._fsdp_owns_weights():
        self.sonic_moe_config.recompute_z = recompute_moe_gate_up
        hidden_states = self._fsdp_forward(
            hidden_states, topk_indices, topk_scores,
            fp8=use_fp8,
            tokens_per_expert=tokens_per_expert,
            fp8_scale=fp8_scale,
            fp8_combine_grad_handle=fp8_combine_grad_handle,
            fp8_config=self.sonic_moe_config,
            release_fp8_weights=False,
            **self._sonic_activation_kwargs,
            **({"sync_free_sizing": True} if sync_free_sizing else {}),
        )
        self._forward_counter += 1
        return hidden_states
    ...  # 原有非 FSDP 路径不变
```

FSDP 分支放在最前面，原有路径一行没改。参数基本照搬原路径，区别有两个：

- `release_fp8_weights=False`：原路径会在最后一个 micro-batch 后让 sonic 释放 `weight.fp8`。FSDP 下 fp8 结果挂在副本上，数据由 `_fsdp_free_sonic_weights` 在每次用完后释放，对象由 `clear_fp8_weights` 统一丢掉，不需要 sonic 插手。
- 不调用 `convert_weights_to_sonic_layout` / `quant_weight`，这些工作已经由 `_fsdp_sonic_copies` 在副本上完成。

## 5. Callback 的改动

两个 callback 逻辑相同：PaddleFleet 的 `SonicMoELayoutSwitchCallback` 和 ERNIE 自己的 `SonicMoECallback`。ERNIE 训练实际注册的是后者，所以两个都要改。以 ERNIE 的为例：

```python
def on_train_begin(self, args, state, control, **kwargs):
    if args.using_sonic_moe and _fsdp_owns_param_storage(args):
        _check_sonic_moe_experts_under_fsdp(kwargs["model"])

def on_step_begin(self, args, state, control, **kwargs):
    global skip_count
    if args.using_sonic_moe:
        if _fsdp_owns_param_storage(args):
            skip_count += 1
            return
        ...  # 原 stage1 流程

def on_optimizer_begin(self, args, state, control, **kwargs):
    if args.using_sonic_moe:
        if _fsdp_owns_param_storage(args):
            self._apply_to_sonic_moe_experts(kwargs["model"], "clear_fp8_weights")
            return
        ...  # 原 stage1 流程
```

- `_fsdp_owns_param_storage(args)` 是 `trainer_callback.py` 里已有的工具函数，根据 `args.sharding` 判断是否是 Paddle 原生 FSDP。
- `on_train_begin`：`fully_shard` 之后运行一次，拒绝 FSDP 不托管的专家（冻结专家），见 4.1。
- `on_step_begin`：FSDP 下什么都不做。原流程里的布局切换、量化、`clear_param_storage` 在 FSDP 下要么不允许要么不存在，现在都由 forward 负责。`skip_count += 1` 保持和原流程相同的计数语义（它控制 `FLAGS_shard_bypass_dygraph_optimizer` 下是否跳过量化）。
- `on_optimizer_begin`：只调用 `clear_fp8_weights`，丢掉本 step 的副本和 fp8 结果，下一步 forward 用更新后的权重重新生成。不需要 `convert_weights_to_grouped_layout`，因为参数从来没离开过 grouped 布局。

PaddleFleet 版本的改动相同，同时删掉了原来的 `_reject_fsdp`（它会直接抛 `NotImplementedError`）。

单测 `TestSonicMoECallbackUnderFSDP` 用会"爆炸"的假模型和假优化器（任何整模型访问或 `clear_param_storage` 调用都抛错）验证：FSDP 下 `on_step_begin` 不碰权重，`on_optimizer_begin` 只调用 `clear_fp8_weights`，`on_train_begin` 遇到 `_fsdp_owns_weights()` 为 False 的专家抛错，`using_sonic_moe=False` 时这些钩子都是空操作。

## 6. 一个 step 的完整时序

以 gradient_accumulation_steps=8、某一层 MoE 为例：

| 时刻 | 发生了什么 | 显存里有什么（该层专家相关） |
|---|---|---|
| `on_step_begin` | callback 直接返回 | 只有 FSDP 分片 |
| micro-batch 1 前向，FSDP gather 本 unit | `_fsdp_sonic_copies` 生成副本、量化，SM90 换成 stride-0 占位 | gather 的 bf16 权重 + 副本数据（SM90 只有 fp8 结果） |
| micro-batch 1 前向 | sonic 抓住空壳 `main_grad`；返回后释放副本数据；输出包上 binder | 只有 FSDP 分片（unit 前向结束后 FSDP 释放 gather 结果） |
| micro-batch 1 反向，FSDP gather 并绑定专家 main_grad | binder 反向：回填副本数据；w1 挂临时 fp32 零存储，w2 共享 FSDP main_grad | gather 的 bf16 权重 + 副本数据 + w1 临时 fp32 梯度 |
| sonic 反向算完 wgrad | `flush_w2_grad` 解除共享；`flush_w1_grad` 转换并累加到 FSDP main_grad，释放临时存储和副本数据 | 回到只有 FSDP 分片 |
| micro-batch 2~8 | 副本对象复用，前向和反向各回填一次；其余重复上面几行 | 同上 |
| `on_optimizer_begin` | `clear_fp8_weights`：丢副本对象 | 只有 FSDP 分片 |
| 优化器更新 | 看到的是 grouped 布局的 main_grad | |

开启 `moe_gate_up` 重计算时，反向里会再跑一次前向，同样先回填副本，数值和第一次前向完全一致。

如果 EP 下 w1 的 wgrad 被推迟到 combine 通信的槽位（见 4.7），`flush_w1_grad` 也随之推迟，副本数据会多留一段时间，但仍在本层反向内释放。

## 7. 实现过程中踩过的坑

| 现象 | 根因 | 修法 |
|---|---|---|
| 间歇性段错误（exit -11），gdb 栈在 `DenseTensor::ShareDataWith` | 对临时张量调用 `get_tensor()`，返回的引用在临时张量被释放后悬空 | 源张量先存成局部变量（4.6） |
| 单卡测试每 step 显存涨约 31.5 MB；端到端第 2 步 OOM | hook 闭包捕获了它所在的张量，C++ 侧的 hook 引用对 Python GC 不可见，形成不可回收的环 | hook 通过 `self._fsdp_sonic_weights` 查找副本（4.5(3)） |
| 修完泄漏仍在第 2 步 `cuMemCreate` OOM | 16 层的 bf16 sonic 副本整个 step 常驻 | SM90 上量化后换成 stride-0 占位张量（4.5(2)） |
| `Tensor not initialized yet when DenseTensor::place() is called` | 第一次尝试用 `_clear_data()` 释放副本，但 sonic 前向会读 `w1.device` | 改用 stride-0 视图，保留 place |
| SM100/bf16 下每层仍常驻一份未切分的副本（review 指出） | 副本数据一直留到 `on_optimizer_begin`；只在 flush 后释放也不够，sonic ctx 从前向到反向都持有副本 | 前向后释放、binder 反向回填、flush 后再释放（4.6） |
| 冻结专家在 FSDP 下静默走非 FSDP 路径（review 指出） | `fully_shard` 只给 trainable 参数设 `_fusion_buffer`，层内判据和 callback 的 `args.sharding` 判据不一致 | `on_train_begin` 检查两者一致，不一致就抛错（4.1） |
| 改了代码但测试结果不变 | `PYTHONPATH` 用了 `$PWD` 相对写法，cwd 变化后指向不存在的路径，实际导入的是 site-packages 里安装的 paddlefleet | 用绝对路径设 PYTHONPATH，并打印 `module.__file__` 确认 |

一个通用的检查方法：怀疑改动没生效时，先 `python -c "import paddlefleet.transformer.moe.moe_expert as m; print(m.__file__)"`。

## 8. 如何验证

### 8.1 单卡逐位对比

脚本 `/tmp/sonic_fsdp_check.py`（E=4、K=2、H=1024、I=896、T=2048）构造两个权重相同的 `SonicMoEExpert`：一个走原路径（ref），另一个给参数挂一个假的 `_fusion_buffer`（只实现 `rebind_main_grad`）来模拟 FSDP。对比 out、dx 以及 w1/w2 的 main_grad，要求最大误差为 0；之后重复 3 个 step 检查显存增量。

```bash
R=/root/paddlejob/paddlefleet-public/ningzhengsheng/erniebot
export PYTHONPATH=$R/third_party/PaddleFleet/src:$R/Paddle/build/python
cd $R
python /tmp/sonic_fsdp_check.py e4m3 2 0   # fp8, 2 个 micro-batch, 不重计算
python /tmp/sonic_fsdp_check.py e4m3 4 1   # fp8, 4 个 micro-batch, 重计算
python /tmp/sonic_fsdp_check.py none 2 1   # bf16
```

结果：fp8/bf16、2/4 个 micro-batch、开关重计算的组合全部逐位一致，3 个 step 的显存增量都是 0.0 MB。

stride-0 占位张量里全是 0。如果 sonic 在这条路径上读了 bf16 数据，结果就会和 ref 对不上，所以逐位一致本身也证明了 SM90 fp8 路径不读 bf16 权重。

### 8.2 单测

```bash
cd $R/third_party/PaddleFleet
PYTHONPATH=$R/third_party/PaddleFleet/src:$R/Paddle/build/python \
  python -m pytest -q tests/single_card_tests/fp8/test_fp8_fsdp_callbacks.py \
    tests/single_card_tests/moe/test_sonic_moe_fsdp_expert.py
# 32 passed
```

`test_sonic_moe_fsdp_expert.py` 用假 fusion buffer 和假 `run_sonic_moe`（模仿 sonic 在前向抓 main_grad、在反向重新读权重并触发 hook）覆盖：前向和反向各自看到正确的 sonic 布局数据；前向后、反向后副本数据都已释放；两个前向之后再做两个反向的交错顺序；iso32 转置视图释放后回填，仍与 `fp8` 共享存储。

### 8.3 双卡对齐与峰值显存

```bash
cd $R/third_party/PaddleFleet
python -m paddle.distributed.launch --devices 0,1 \
  tests/multi_card_tests/moe/test_sonic_moe_fsdp.py
# 两个 rank 都是 Ran 3 tests ... OK
```

用真实 sonic kernel 和真实 `fully_shard`（sharding_degree=2，ep=1）训练一个小 GPT MoE 3 步，每步 2 个 micro-batch，bf16 和 fp8 各一遍，与非 FSDP 的参考对比每个 micro-batch 的 loss 和本 rank 的专家梯度分片；同时断言每次专家前向结束后、每个 micro-batch 反向结束后副本数据都已释放。`test_frozen_experts_rejected` 冻结专家后 `fully_shard`，确认 `on_train_begin` 抛错。

在 PDC 容器里直接跑 launch 会读到双机的 `PADDLE_*`/`TRAINER*` 环境变量，一直卡在 `master not ready`，需要先在当前 shell 里 unset 这些变量。

峰值显存对比（hidden=2048、moe_intermediate=2048、4 个专家、2 层、seq=1024、2 个 micro-batch，统计一个 step 前向+反向的峰值减去 step 开始时的占用；"副本常驻"是把释放函数换成空操作得到的）：

| case | 副本常驻 | 用完释放 | 节省 |
|---|---|---|---|
| bf16 | 1689.6 MiB | 1593.6 MiB | 96.0 MiB |
| fp8 (SM90) | 1566.6 MiB | 1520.1 MiB | 46.5 MiB |

这个配置下每层 bf16 副本约 96 MiB，节省量随层数和专家规模增长。

### 8.4 端到端

`bash run_fsdp_test_1node.sh`（单机 8 卡，max_steps=10），基线是同配置 `using_sonic_moe=False`：

| step | sonic loss | 基线 loss | sonic 峰值显存 | 基线峰值显存 | sonic 吞吐 | 基线吞吐 |
|---|---|---|---|---|---|---|
| 1 | 12.33850 | 12.33821 | 50.1 GB | 58.9 GB | – | – |
| 2 | 12.33289 | 12.33291 | 66.6 GB | 67.6 GB | 48470 | 40396 |
| 3 | 12.33662 | 12.33607 | 66.6 GB | 67.6 GB | 48890 | 40737 |
| 4 | 12.32935 | 12.32975 | 66.6 GB | 67.6 GB | 48811 | 40632 |
| 10 | 12.21098 | – | 66.6 GB | – | 48864 | – |

吞吐单位是 tokens/s/卡。两边 kernel 不同，loss 有微小差异，属于预期。日志里 `quack_replica` 的 sm_100a 报错来自 kernel 预热，在 SM90 上无害。

上表是副本数据"用完释放"之前测的。合入版本的端到端结果见 8.5。

### 8.5 与 stage1 逐位对齐

在确定性开关和对齐诊断开关都打开的条件下，FSDP 和 stage1 两侧都开 `using_sonic_moe=True`，各跑 100 步，逐步比较 loss。配置是 V0.2.1 C1_T3 的 stage1 / FSDP 两份 yaml（dp8、ep8，`per_device_train_batch_size=2`，`gradient_accumulation_steps=16`）。

```bash
bash run_align_1node.sh stage1 true 100 > align_stage1_true_100.log 2>&1
bash run_align_1node.sh fsdp   true 100 > align_fsdp_true_100_after_pr2515.log 2>&1
python3 compare_loss.py align_stage1_true_100.log align_fsdp_true_100_after_pr2515.log --diff-only --tps-skip 5
```

`run_align_1node.sh` 只负责设置环境变量，读这些变量的诊断代码要单独打上，三处都要有：

| 位置 | 环境变量 | 作用 |
|---|---|---|
| `Paddle/build/python`（`muon.py`、`tensor_fusion_helper.py`、`muon_sharding_optimizer.py`、`fully_shard_fusion.py`） | `MUON_FSDP_BCAST_GRAD`、`MUON_FSDP_NS_SINGLE` | 对齐梯度通信方式；Newton-Schulz 逐个参数算 |
| `ernie5/src/trainers/pretraining_trainer.py` | `MUON_FSDP_STAGE1_CLIP_EPS` | FSDP 侧按 stage1 的方式计算 clip 系数 |
| `third_party/PaddleFleet/src/paddlefleet/optimizers/muon_optimizer.py` | `MUON_FSDP_NS_SINGLE` | fleet Muon 的 Newton-Schulz 逐个参数算，不按 batch 拼接 |

PaddleFleet 那处不在 develop 上，要从 `hotfix_patches/20261112_215014/PaddleFleet.patch` 里抽出 `muon_optimizer.py` 部分再 `git apply`。日志里出现 `[ALIGN] fleet MuonHyperballOptimizer: per-param Newton-Schulz (MUON_FSDP_NS_SINGLE) ACTIVE` 和 `[ALIGN] stage1 clip-eps hack ACTIVE` 才说明生效。这些改动只用于诊断，不要提交。

结果（stage1 基准为 11-12 的数据；合入版本 11-16 重跑，跑前驱逐了本机的 BE pod）：

| | stage1 | FSDP 副本常驻版（11-12） | FSDP 合入版本（11-16） |
|---|---|---|---|
| loss | 基准 | 100 步逐位相同 | 100 步逐位相同 |
| step 100 的 max_mem_allocated | 53.21 GB | 54.48 GB | 52.51 GB |
| 吞吐（tokens/s/卡，跳过前 5 步） | 28408 | 28441 | 27808 |

"副本常驻版"是加入"副本用完释放"之前的代码。和它相比，合入版本峰值显存少约 2.0 GB，吞吐低 2.2%。吞吐下降来自每个 micro-batch 多做的两次布局转换和量化（前向、反向各回填一次）。

## 9. 已知限制

- **SM100**：尚未在 SM100 上测试。SM100 和 bf16 模式下副本保留 bf16 数据，但只在本层前向、反向期间存在。
- **w2 副本释放偏晚**：两份副本都在 `flush_w1_grad` 里一起释放，w2 在 UpProjection 反向期间多留一会儿。8.3 的配置下比分开释放少省约 32 MiB（bf16）/ 17.5 MiB（fp8），只影响单层反向内部，不随层数累积。
- **冻结专家不支持**：FSDP 下专家必须可训练，否则 `on_train_begin` 抛错。不经过 trainer 直接调用 `fully_shard` 时没有这层检查。
- **未覆盖**：FSDP 下 checkpoint 保存/加载（`sharded_state_dict`）、多机、`fsdp_reshard_policy` 除 `after_step` 以外的取值、全量重计算的专门测试。
- **依赖 sonic 内部约定**：方案依赖 sonic 的 `_main_grad_accumulator` 在前向抓取 `weight.main_grad`、反向后调用 `_apply_backward_hook`、反向先算 w2 再算 w1 的 wgrad、反向读的是 ctx 里同一批权重对象，以及 SM90 fp8 aligned 路径不读 bf16 权重。升级 `paddlefleet_ops` 后应重跑 8.1 的逐位对比和 8.3 的双卡测试。
