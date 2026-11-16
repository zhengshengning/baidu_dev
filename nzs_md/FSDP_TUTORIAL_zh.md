# Paddle 动态图 FSDP 实现教程（详解版）

> 基于 PR [#78577](https://github.com/PaddlePaddle/Paddle/pull/78577)（`【FSDP】Add FSDP in dynamic`，
> commit `f14c7231db`）及其后续 PR：`#79640`（EP 支持）、`#79713`（性能修复）、`#79724`（save/load）。
> 代码行号对应本仓库 `develop` 分支当前状态。

## 目录

- [零、阅读指南](#零阅读指南)
- [一、理论：从 DDP 到 ZeRO-3/FSDP](#一理论从-ddp-到-zero-3fsdp)
- [二、总体架构与一次 step 的全景](#二总体架构与一次-step-的全景)
- [三、TensorFusionBuffer 逐段精读](#三tensorfusionbuffer-逐段精读)
- [四、FSDPBufferManager：分组策略](#四fsdpbuffermanager分组策略)
- [五、FSDPCommManager：overlap 的全部机制](#五fsdpcommmanageroverlap-的全部机制)
- [六、Hook 体系与完整时序](#六hook-体系与完整时序)
- [七、Optimizer 集成](#七optimizer-集成)
- [八、Checkpoint：分布式 save/load](#八checkpoint分布式-saveload)
- [九、显存与性能定量分析](#九显存与性能定量分析)
- [十、使用、调参与精度对齐](#十使用调参与精度对齐)
- [十一、FAQ 与排查手册](#十一faq-与排查手册)
- [十二、与 PyTorch FSDP 的对比](#十二与-pytorch-fsdp-的对比)
- [十三、已知限制与源码索引](#十三已知限制与源码索引)

---

## 零、阅读指南

### 涉及文件

- `python/paddle/distributed/fsdp/fully_shard.py` — 用户 API `fully_shard`，按模式分发
- `python/paddle/distributed/fsdp/fully_shard_fusion.py` — **手动并行（动态图）实现主体，本文重点**
- `python/paddle/distributed/fsdp/_fsdp_context.py` — 全局 FSDP context 注册表
- `python/paddle/distributed/auto_parallel/fully_shard.py` — 自动并行分支（`FullyShardAuto`）
- `python/paddle/optimizer/optimizer.py:1702` — optimizer 侧唯一接管点
- `python/paddle/distributed/flex_checkpoint/dcp/sharded_weight.py` — `ShardedWeight` 描述结构
- `test/collective/fleet/dygraph_group_fsdp.py` — 与 `group_sharded_parallel` 对齐的精度测试

### 前置知识

1. **ZeRO / 数据并行**：知道 `all_reduce = reduce_scatter + all_gather`。
2. **Paddle 动态图 hook**：`Layer.register_forward_pre_hook` / `register_forward_post_hook`
   只在前向触发；`Tensor._register_grad_hook` 在该 tensor 的梯度算出后触发；
   `paddle.autograd.PyLayer` 可以在计算图中插入自定义 forward/backward 节点。
3. **Paddle 张量的“壳与数据分离”**：`Tensor` 对象（含 `name`/`shape`/`stop_gradient` 等属性）
   与其底层 `DenseTensor` 存储是可分离的，`_clear_data()` 只丢存储，
   `_share_data_with()` 可把存储换成别的张量的视图。**这是整套实现成立的物理基础。**

### 术语表

- **shard / 分片**：本 rank 常驻的那 $1/N$ 段数据。
- **unshard / gather**：把各 rank 的分片 all_gather 成完整数据。
- **reshard / free**：丢掉完整数据，回到只有分片的状态。
- **FSDP unit**：通信粒度，通常是一个 Transformer 层，整层参数一起 unshard/reshard。
- **buffer group**：一组被融合进同一块连续显存的参数（`BufferGroup`）。
- **main_grad**：fp32 梯度累加张量；本实现里它是 fused 梯度 buffer 上的一个视图。

---

## 一、理论：从 DDP 到 ZeRO-3/FSDP

### 1.1 训练显存由什么构成

设模型参数量 $\Psi$，数据并行度 $N$，典型混合精度配置（bf16 计算 + fp32 master weight + Adam）：

- bf16 参数：$2\Psi$
- bf16 梯度：$2\Psi$（本实现用 fp32 main_grad，见 3.5，此处先按经典 ZeRO 记法）
- fp32 master weight：$4\Psi$
- Adam 一阶动量 $m$：$4\Psi$
- Adam 二阶动量 $v$：$4\Psi$

$$M_{\text{DDP}} = \underbrace{2\Psi + 2\Psi}_{\text{param + grad}} + \underbrace{4\Psi + 4\Psi + 4\Psi}_{K\Psi,\ K=12} = 16\Psi$$

7B 模型即 $16 \times 7\text{e}9 = 112$ GB，单卡 80GB 放不下 —— 这就是必须切分的原因。
注意这 $16\Psi$ 里 **不含激活值**，激活值与 batch/seq 有关，由 recompute 等手段单独治理。

### 1.2 ZeRO 三级

- **ZeRO-1**：切 optimizer state → $2\Psi + 2\Psi + 12\Psi/N$
- **ZeRO-2**：再切 gradient → $2\Psi + 14\Psi/N$
- **ZeRO-3 / FSDP**：再切 parameter → $16\Psi/N$

Paddle 的 `group_sharded_parallel` 对应 `level="os" / "os_g" / "p_g_os"` 三级；
`fully_shard` 则是 ZeRO-3 的新实现（并且是本文的主角）。

### 1.3 FSDP = ZeRO-3 的“逐层实现”

ZeRO-3 的难点在于：**参数被切了，但计算需要完整参数**。FSDP 的答案是按层临时还原：

```
参数常态：只有 1/N 分片
计算某层前：all_gather 出该层完整参数
该层算完：立刻丢掉完整参数，回到分片状态
```

于是稳态显存是 $16\Psi/N$ 加上一个 **只与单层大小有关的常数峰值**，与模型总层数无关。
这就是 FSDP 能训超大模型的根本原因。

单个 step 的语义时序：

```
forward(layer_i) :  all_gather(param_i) -> compute -> free full param_i
backward(layer_i):  all_gather(param_i) -> compute grad_i -> reduce_scatter(grad_i) -> free
optimizer.step() :  只在本 rank 的 param shard / grad shard 上更新（不需要任何 all_gather）
```

最后一行值得强调：**因为参数、梯度、优化器状态在同一套 offset 上按相同规则切分，
optimizer 完全可以只看本 rank 的分片做逐元素更新**，这是 ZeRO-3 各状态“对齐切分”的红利。

### 1.4 通信量：为什么是 DDP 的 1.5 倍

每 step：参数 all_gather 两次（forward + backward），梯度 reduce_scatter 一次。
ring 算法下单次 all_gather 与 reduce_scatter 的单卡收发量均为 $\frac{N-1}{N}\Psi \cdot s$
（$s$ 为元素字节数）：

$$V_{\text{FSDP}} = 3 \cdot \frac{N-1}{N}\Psi s, \qquad
V_{\text{DDP}} = V_{\text{all\_reduce}} = 2 \cdot \frac{N-1}{N}\Psi s$$

即 $V_{\text{FSDP}} / V_{\text{DDP}} = 1.5$。

**这 1.5 倍的额外通信是用显存换来的，能否不掉吞吐完全取决于 overlap。** 粗略判据：

$$T_{\text{comm}}^{\text{layer}} \approx \frac{2 \cdot \frac{N-1}{N} \Psi_{\text{layer}} s}{B_{\text{link}}}
\quad\text{vs}\quad
T_{\text{compute}}^{\text{layer}} \approx \frac{\text{FLOPs}_{\text{layer}}}{\text{TFLOPS}_{\text{eff}}}$$

只要 $T_{\text{comm}} < T_{\text{compute}}$ 且预取深度足够（≥1 层），通信就能被完全隐藏。
序列越长、batch 越大，计算越占优，FSDP 越划算；反之小 batch 场景通信会露出来。

### 1.5 两个必须解决的工程问题

朴素实现即使正确也会很慢，原因有两条：

1. **小张量通信效率低**。一个 Transformer 层有几十个参数（q/k/v/o、mlp、norm…），
   逐个 all_gather 会被 kernel launch 开销和小消息延迟吃掉带宽。
   → 解法：**tensor fusion**，把一组参数摊平进一块连续显存，一次通信搞定。
2. **通信暴露在关键路径**。同步 all_gather 时 GPU 在干等。
   → 解法：**预取 + double buffer**，算第 $i$ 层时异步 gather 第 $i+1$ 层；
   梯度侧维护异步 reduce_scatter 队列。

这两条决定了 Paddle 实现的全部结构复杂度。理解了它们，代码就不难读了。

---

## 二、总体架构与一次 step 的全景

### 2.1 四层结构

```
FullyShardFusion            入口对象（fully_shard_fusion.py:796）
   │   注册 hook / 接管 sharded_state_dict / 与 optimizer 交互
   ├─ FSDPCommManager       通信调度（:445）
   │      double buffer 状态机、预取、梯度 reduce_scatter 队列
   └─ FSDPBufferManager     分组与 buffer 创建（:316）
          └─ TensorFusionBuffer  连续显存 + 分片 + 两个通信原语（:91）
```

对象关系（一个模型对应一个 `FullyShardFusion`，$G$ 个 buffer group）：

```
FullyShardFusion
 ├── buffer_manager.buffer_groups : list[BufferGroup]   # 按执行顺序排好
 │      每个 BufferGroup 持有:
 │        params        : 组内参数列表
 │        params_buffer : TensorFusionBuffer(is_params=True)   # 常驻 1/N 参数分片
 │        grads_buffer  : TensorFusionBuffer(is_params=False)  # 常驻 1/N 梯度分片
 │        fsdp_group / fsdp_unit_id / is_expert_param / is_tie / no_decay
 ├── buffer_manager.param_to_buffer_id : {param.name -> gid}
 └── comm_manager: 持有 BufferState 机与两个队列
```

每个 `TensorFusionBuffer` 内部只有两块显存：

- `data_buffer`：**常驻**，长度 = `total_buffer_size / N`（本 rank 分片）
- `tmp_data_buffer`：**临时**，长度 = `total_buffer_size`（完整），惰性分配、用完即释放

参数侧 `tmp_data_buffer` 是 all_gather 的目的地；梯度侧它是反向写 `main_grad` 的区域、
也是 reduce_scatter 的输入。

### 2.2 “壳与数据分离”是怎么串起来的

这是最容易卡住的一点，单独画一下参数张量在一个 step 中数据指针的变化：

```
初始化 (TensorFusionBuffer.__init__)
    param.data ──copy──> params_buffer.data_buffer[offset : offset+numel]
    param._clear_data()                     # param 变成“空壳”，只剩 name/shape
                                            # data_buffer 随后被切成 1/N

forward pre hook -> all_gather_params()
    tmp = view_slice(tmp_data_buffer, offset, offset+numel)   # 零拷贝视图
    param.get_tensor()._share_data_with(tmp)                 # 壳挂上完整数据
    → 此后 paddle.matmul(x, param) 读到的就是完整参数

forward post hook -> shard_params()
    param._clear_data()                     # 壳重新变空，tmp buffer 暂不回收(READY)

backward: FusionBackwardHook.backward -> all_gather_params(is_backward=True)
    → 壳再次挂上完整数据

param 的 grad hook (comm_hook)
    param.main_grad = view_slice(grads_buffer.tmp_data_buffer, offset, ...)
    param.main_grad.add_(grad); grad._clear_data()
    param._clear_data()                     # 参数视图释放
    反向进入前一个 unit 时 → reduce_scatter(grads tmp -> grads data_buffer)（见 5.8）

optimizer.step()
    不看 param，直接更新 (params_buffer.data_buffer, grads_buffer.data_buffer)
```

**关键收益**：`param` 这个 Python 对象从头到尾没变过，所以用户代码、
`model.parameters()`、optimizer 的 `_parameter_list`、`state_dict` 里的引用都不会失效；
变的只是它背后的数据指针。这让 FSDP 可以做到“原地改造模型、用户代码零改动”。

### 2.3 一次 step 的调用链总览

```
model(x)
 └─ Layer.__call__
     ├─ forward_pre_hook   → all_gather_comm()      → comm_manager.all_gather_params(params)
     ├─ 真正的 forward 计算（读到完整参数）
     └─ forward_post_hook  → shard_comm()           → comm_manager.shard_params(params)
                           → _forward_post_hook()   → FusionBackwardHook.apply(outputs)

loss.backward()
 ├─ FusionBackwardHook.backward   → all_gather_params(..., is_backward=True)
 │                                 （顺带 _flush_expert_grads_after_unit）
 ├─ 各 param 的 grad hook comm_hook → main_grad 累加 + shard_params + reduce_scatter_grads
 └─ FusionForwardHook.backward    → shard_params(...)   # 仅 tie 权重

opt.step()
 └─ Optimizer._apply_optimize
     ├─ fsdp_context.comm_sync_and_reset_status()   # 等完通信、状态复位
     ├─ fsdp_context.bind_decay_param_fun(self)     # 修正 weight decay
     ├─ params_grads = [(params_buffer.data_buffer, grads_buffer.data_buffer), ...]
     ├─ grad_clip（打开 should_comm_on_shard_dim）
     └─ _create_optimization_pass  → 在分片上做 AdamW
```

---

## 三、TensorFusionBuffer 逐段精读

`fully_shard_fusion.py:91`。一个实例管理**一组参数**的连续显存，由 `is_params` 决定形态。

### 3.1 构造参数

```python
def __init__(self, unique_key, params, fsdp_group, dtype,
             is_params=False, main_grad_dtype=None):
```

- `unique_key`：buffer group 的序号 `gid`，用来生成 `fuse_params_<gid>` 这个张量名
  （**后面 optimizer 状态名解析、weight decay 判定都依赖这个名字**）。
- `fsdp_group`：本 buffer 的通信组；`fsdp_degree = fsdp_group.nranks`。
  `is_sharded = nranks > 1` —— 单卡时退化为“只融合、不分片”，代码路径统一。
- `dtype`：参数 dtype。`main_grad_dtype`：梯度累加 dtype（默认与 `dtype` 相同，
  实际由 `FSDPBufferManager` 传 fp32）。

### 3.2 offset 表与对齐（`:126` / `get_padded_size` `:242`）

```python
for param in params:
    self.param_offsets[param.name] = self.total_buffer_size
    self.total_buffer_size += self.get_padded_size(param)
```

```python
def get_padded_size(self, param):
    size = np.prod(param.shape)
    align_size = alignment[get_current_device_type()] // align[param.dtype] * self.fsdp_degree
    return ((size + align_size - 1) // align_size) * align_size
```

`alignment["gpu"] = 256`（字节），`align[bfloat16] = 2`（字节/元素），
所以 bf16 + `fsdp_degree=8` 时 `align_size = 256 // 2 * 8 = 1024` 个元素。

举例：`Linear(1024, 1024)` 的 weight 有 $1048576$ 元素，已是 1024 的整数倍，不 pad；
而 `LayerNorm` 的 `bias`（1024 元素）也刚好；但 `Linear(1000, 1000)` 的 $10^6$ 元素
会被 pad 到 $1000448$（= 977×1024），浪费 448 个元素。

**为什么 `align_size` 里要乘 `fsdp_degree`？** 两个目的一起满足：

1. `total_buffer_size` 一定能被 $N$ 整除 → reduce_scatter 可等分（梯度 buffer 直接按
   `total // N` 分配，见 3.5）。
2. 每个 rank 的分片首地址 `rank * (total/N)` 仍然是 256 字节对齐的 → 通信与向量化访存高效。

代价是**内碎片**：参数越小越碎、$N$ 越大浪费越多。这也是为什么 FSDP 更适合把整层参数
（大张量为主）放一起，而不是把成千上万个小参数塞进同一个 buffer。

### 3.3 参数 buffer 的构造三步（`:135`）

```python
# ① 先按完整大小分配，把每个 param 的数据搬进对应 offset
self.data_buffer = paddle.zeros([self.total_buffer_size], dtype=dtype)
self.status = BufferState.FREED
for param in params:
    offset = self.param_offsets[param.name]
    stop_gradient, _shape = param.stop_gradient, param.shape
    param.stop_gradient = True          # 避免 flatten_/assign 建图
    param.flatten_()
    paddle.assign(param, self.data_buffer._slice(offset, offset + param._numel()))
    param._clear_data()                 # ② 释放 param 自身存储，只剩空壳
    param.stop_gradient = stop_gradient
    param.get_tensor()._set_dims(_shape)  # 形状信息保留，供后续 view 用
paddle.device.cuda.empty_cache()

if self.is_sharded:                     # ③ 切分：只常驻 [start, end)
    piece_len = (total_nums + self.fsdp_degree - 1) // self.fsdp_degree
    start = curr_rank * piece_len
    end = min(start + piece_len, total_nums)
    self.shard_start, self.shard_end = start, end
    self.data_buffer = paddle.slice(self.data_buffer, [0], [start], [end]).clone()

self.data_buffer.name = "fuse_params_" + str(unique_key)
self.data_buffer.stop_gradient = params[0].stop_gradient
self.data_buffer.optimize_attr = params[0].optimize_attr
```

几个细节：

- **`empty_cache()` 的位置**：所有参数原存储被释放后立刻归还给 allocator，
  给紧接着的 `slice(...).clone()` 腾地方。
- **`optimize_attr` 与 `stop_gradient` 从 `params[0]` 继承**：这就要求组内参数在
  learning rate scale、是否可训练上一致 —— 分组键为什么那样设计，答案的一半在这里。
- **末尾一次 `paddle.slice(...).clone()`**：`clone` 是必须的，否则 `data_buffer` 仍持有
  整块完整显存的引用，切分就白做了。

### 3.4 分片区间与 `param_shard`（`:254`）

`shard_start / shard_end` 是本 rank 在 **buffer 全局坐标系** 中的区间，`begin` 和 `end` 是该参数在分片上的区间。
`param_shard` 计算某个参数与本 rank 分片的交集：

```python
def param_shard(self, param):
    offset = self.param_offsets[param.name]
    numel = int(np.prod(param.shape))
    begin = max(offset, self.shard_start)
    end = min(offset + numel, self.shard_end)
    if end <= begin:
        return None                                  # 该参数完全不在本 rank
    return (begin - self.shard_start,                # 在 data_buffer 中的起点
            end - self.shard_start,                  # 在 data_buffer 中的终点
            slice(begin - offset, end - offset))     # 在参数 flatten 视角下的区间
```

三元组的两套坐标要分清：前两项用于对 `data_buffer` 做 `_slice`，第三项用于告诉
checkpoint “这是该参数 flatten 后的哪一段”。**一个参数可能被切到两个 rank 上（甚至完全
不在本 rank）**，这正是 FSDP 与“按参数粒度切分”的最大区别：这里切的是 **flatten 后的字节流**。

### 3.5 梯度 buffer 与 main_grad 注入（`:184`）

```python
self.data_buffer = paddle.zeros([self.total_buffer_size // self.fsdp_degree],
                                dtype=self.main_grad_dtype)     # 只按分片大小分配
if self.is_sharded:
    shard_size = self.total_buffer_size // self.fsdp_degree
    self.shard_start = paddle.distributed.get_rank(fsdp_group) * shard_size
    self.shard_end = self.shard_start + shard_size

for param in params:
    if param.trainable:
        param._fusion_buffer = self
        param._param_offsets = self.param_offsets
        param._main_grad_addr = None

        def get_grad_from_tmp_buf(param, shape=None):
            tmp_buffer = param._fusion_buffer.get_tmp_buffer()
            offset = param._param_offsets[param.name]
            main_grad = paddle._C_ops.view_slice(tmp_buffer, offset, offset + param._numel())
            main_grad.get_tensor()._set_dims(param.shape if shape is None else shape)
            param.main_grad = main_grad
            param._main_grad_addr = main_grad.data_ptr()
            return main_grad

        param.get_main_grad = get_grad_from_tmp_buf.__get__(param)
```

三点值得展开：

- **梯度 buffer 不需要完整常驻**：`data_buffer` 只有 $1/N$，完整梯度只在反向期间以
  `tmp_data_buffer` 形式短暂存在。
- **`main_grad` 是视图，不是独立张量**：所以 `param.main_grad.add_(grad)` 直接写进了
  fused 梯度 buffer 的正确位置，reduce_scatter 之前不需要任何 gather/copy 阶段。
- **`main_grad_dtype` 与参数 dtype 解耦**：bf16 参数用 fp32 累加，避免多 micro-batch 累加
  的精度损失；这也是 `MixedPrecisionPolicy.reduce_dtype` 的落点。

注意区分两处 `add_`：`main_grad.add_(grad)` 是**单次反向内**往完整 tmp 写局部梯度（共享层等
多次触发时才真正相加）；跨 micro-batch 的梯度累加发生在 RS 之后的 `data_buffer.add_(shard)`
（`accumulate_reduced_grad`），tmp 每次 RS 后即释放。两次 micro-batch 的完整流程：

```text
mb1 反向: tmp(新 zeros) ← main_grad.add_(g1)      # 完整, 局部
         RS(tmp) → shard1;  data_buffer += shard1;  释放 tmp
mb2 反向: tmp(新 zeros) ← main_grad.add_(g2)      # 新 buffer, 不含 g1
         RS(tmp) → shard2;  data_buffer += shard2;  释放 tmp
optimizer 读 data_buffer = shard1 + shard2
```

### 3.6 `rebind_main_grad`：tmp buffer 地址会变（`:225`）

`tmp_data_buffer` 在每次 reduce_scatter 后被释放（`clear_tmp_buffer`），下次再用时重新
分配，**地址会变**。若此时某个 `param.main_grad` 还指着旧地址且里面有未被 reduce 的数据
（典型场景：MoE expert 参数跨 unit 累加），直接丢弃就会丢梯度：

```python
def rebind_main_grad(self, param):
    stale_grad = getattr(param, "main_grad", None)
    if not (self.is_sharded and self.tmp_data_buffer is None):
        if stale_grad is not None and stale_grad._is_initialized():
            if stale_grad.data_ptr() == getattr(param, "_main_grad_addr", None):
                return                       # 地址没变，无需重绑
    main_grad = param.get_main_grad()        # 绑到新 tmp buffer
    if stale_grad is not None and stale_grad._is_initialized():
        if stale_grad.data_ptr() != main_grad.data_ptr():
            main_grad.add_(stale_grad)       # 把旧数据搬过来
            stale_grad._clear_data()
```

### 3.7 惰性 tmp buffer 与两个通信原语（`:268` 起）

```python
def get_tmp_buffer(self):
    if not self.is_sharded:
        return self.data_buffer          # 单卡：完整 == 分片，直接用
    if self.tmp_data_buffer is None:     # 惰性分配（显存只在需要时占用）
        self.tmp_data_buffer = paddle.zeros(
            [self.total_buffer_size],
            dtype=self.dtype if self.is_params else self.main_grad_dtype)
    return self.tmp_data_buffer

def do_all_gather(self):                 # 分片 -> 完整，异步
    return paddle.distributed.all_gather(
        self.get_tmp_buffer(), self.data_buffer, group=self.fsdp_group, sync_op=False)

def do_reduce_scatter(self):             # 完整 -> 本 rank 分片，异步
    tmp_buffer = self.get_tmp_buffer()
    shard = tmp_buffer._slice(0, self.data_buffer.shape[0])   # 输出借用输入前 1/N 段
    tmp_buffer.scale_(1.0 / self.fsdp_degree)                 # 先 scale 再 SUM == MEAN
    return paddle.distributed.reduce_scatter(
        shard, tmp_buffer, op=ReduceOp.SUM, group=self.fsdp_group, sync_op=False)

def accumulate_reduced_grad(self):
    shard = self.get_tmp_buffer()._slice(0, self.data_buffer.shape[0])
    self.data_buffer.add_(shard)          # 累加而非覆盖 → 支持梯度累加
    self.clear_tmp_buffer()
```

四个设计点：

1. **`sync_op=False`**：两个原语都返回 comm task，由 `FSDPCommManager` 决定何时 `wait()`
   —— overlap 的前提。
2. **输出复用输入显存**：`shard` 是 `tmp_buffer` 的前 $1/N$ 段，省一块临时显存
   （NCCL 的 reduce_scatter 允许 in-place 形式的这种布局）。
3. **`scale_ + SUM` 而不是 `ReduceOp.AVG`**：兼容性更好，且与 DDP 的 `all_reduce(mean)`
   数值路径一致（先缩放再求和，误差行为可预测）。
4. **`add_` 而不是 `assign`**：`data_buffer` 是跨 micro-batch 累加的落点。
   一个 step 内首次使用前由 `_maybe_zero_grads()` 清零，之后每次 reduce_scatter 的结果都
   累加上去 —— **梯度累加（`accumulate_steps > 1`）就是靠这一行成立的**。

---

## 四、FSDPBufferManager：分组策略

`fully_shard_fusion.py:316`。它回答一个问题：**哪些参数应该放进同一块 buffer？**
因为同一 buffer 内的参数会被 *一次通信、一次 optimizer 更新* 统一处理，
任何“会导致行为不一致”的维度都必须成为分组键。

### 4.1 通信组的选择（`:326`）

```python
self.hcg = fleet.get_hybrid_communicate_group()
self._fsdp_group = self.hcg.get_sharding_parallel_group()      # 普通参数
use_ep = (hasattr(self.hcg, "get_expert_parallel_world_size")
          and self.hcg.get_expert_parallel_world_size() > 1)
self._ep_fsdp_group = (self.hcg.get_moe_sharding_parallel_group() if use_ep
                       else self._fsdp_group)                   # expert 参数
```

**FSDP 的并行度来自 fleet 的 sharding 组**，即 `hybrid_configs["sharding_degree"]`，
而不是 `fully_shard(mesh=...)` 参数（手动并行分支目前不用 `mesh`）。

MoE 场景：expert 参数已被 EP 切开，在 EP 组内各 rank 持有的是**不同的专家**，
不能再在整个 sharding 组上分片，只能在 `moe_sharding` 组（即 sharding 组去掉 EP 维度）
上分片。逐参数还支持 `param.color = {"group": ...}` 显式指定通信组，优先级最高。

### 4.2 FSDP unit：通信粒度（`:345`）

```python
self.fsdp_unit_layers = fsdp_unit_layers or [
    'TransformerLayer', 'Qwen3VLTextDecoderLayer', 'Qwen3MoeDecoderLayer']

param_to_unit_id = {}
for unit_id, m in enumerate(self.model.modules()):
    if type(m).__name__ in self.fsdp_unit_layers:
        for p in m.parameters():
            param_to_unit_id[p.name] = unit_id
```

- **按类名匹配**，不是按 isinstance —— 好处是不依赖具体模型库，坏处是自定义层名不在
  列表里就享受不到“整层一次通信”。
- `unit_id` 取 `model.modules()` 的遍历序号，`modules()` 是**前序遍历**，
  所以 `unit_id` 天然近似前向执行顺序（这一点被 4.4 的排序直接利用）。
- 不属于任何 unit 的参数（embedding、lm_head、norm 等）`unit_id = None`，
  排序时视为 `+inf`，落在最后。

粒度权衡：unit 越大 → 通信消息越大、效率越高，但 unshard 的峰值显存越高；
unit 越小 → 峰值低但通信碎、overlap 机会少。以 Transformer 层为单位是经验上的平衡点。

### 4.3 七元分组键（`:379`）

```python
key = (
    param.dtype,                                     # ① 一块 buffer 只能一种 dtype
    param.trainable,                                 # ② 冻结参数不需要梯度 buffer
    param_to_unit_id.get(param.name),                # ③ 通信粒度：属于哪个 unit
    is_expert,                                       # ④ MoE expert 参数
    fsdp_group,                                      # ⑤ 通信组不同必须分开
    param.name == self.tie_param_name,               # ⑥ tie_weights 共享参数
    bool(getattr(param, "no_weight_decay", False)),  # ⑦ weight decay 行为必须一致
)
keyed_params.setdefault(key, []).append(param)
```

逐项的“如果不这样分会怎样”：

- ① dtype 混用 → 无法用一个 `paddle.zeros(dtype=...)` 承载，`align` 计算也不同。
- ② 冻结参数与可训练参数混用 → `data_buffer.stop_gradient` 继承 `params[0]`，
  会出现“buffer 说不需要梯度、组内却有参数要梯度”的矛盾；
  另外冻结参数不必参与 reduce_scatter，混在一起白跑通信。
- ③ 跨 unit 混用 → 一次 all_gather 会把两层的参数一起拉起来，
  显存峰值和预取逻辑都失效（预取以 unit 为计数单位）。
- ④/⑤ expert 与非 expert、不同通信组混用 → **通信语义直接错**（分片规则不同）。
- ⑥ tie 权重（embedding 与 lm_head 共享）会在一个 step 内被多处使用，
  释放时机与普通参数不同（见 6.4），必须独立成组。
- ⑦ weight decay：optimizer 只看 buffer 名字来判断是否 decay，
  一块 buffer 只能有一个答案（见 7.3）。混用会被显式报错拦住。

### 4.4 排序：预取能否生效的前提（`:391`）

```python
def sort_key(item):
    _, trainable, unit_id, is_expert_param, _, is_tie, _ = item[0]
    return (0 if is_tie else (1 if not trainable else 2),
            unit_id if unit_id is not None else float('inf'),
            is_expert_param)
```

排序后 `buffer_groups` 的下标 `gid` 就是**近似前向执行顺序**。这一点非常关键：
预取逻辑 `_next_buffer_id` 是纯粹按 `gid ± 1` 扫的，不做任何图分析。
**如果排序与实际执行顺序偏差大，预取就会取错 buffer，overlap 失效（但不会算错）。**

优先级安排：tie 权重放最前（它在 forward 最早的 embedding 处就要用，
且会被复用到最后的 lm_head）；冻结参数其次（只在 forward 参与，反向会被跳过）；
其余按 unit_id 递增。

### 4.5 创建 buffer 与索引（`:425`）

```python
for gid, group in enumerate(self.buffer_groups):
    group.params_buffer = TensorFusionBuffer(gid, params, group.fsdp_group,
                                             group.dtype, is_params=True)
    if not params[0].stop_gradient:
        group.grads_buffer = TensorFusionBuffer(
            gid, params, group.fsdp_group, group.dtype,
            main_grad_dtype=paddle.float32
                if group.is_expert_param or group.dtype == paddle.float32
                else self.main_grad_dtype)
    group.grads_use_sum = len(params)          # 该组“梯度到齐”的计数目标
    for param in params:
        self.param_to_buffer_id[param.name] = gid
```

- 冻结组不创建 `grads_buffer`（省 $1/N$ 显存）。
- **expert 参数强制 fp32 main_grad**：expert 梯度会跨 unit 多次累加，
  低精度累加的误差会被放大。
- `grads_use_sum = len(params)`：早期计数触发方案的分母，当前代码已不读取（见 5.7）。

---

## 五、FSDPCommManager：overlap 的全部机制

`fully_shard_fusion.py:445`。这一层不碰显存布局，只管**什么时候发通信、什么时候等、
什么时候释放**。它被清晰地分成两半，源码注释也写明了“两半不要互碰状态”：

- 参数侧：`buffer_cnt_in_using` / `double_buffer_limit` / `prefetch_units` + `BufferState` 机
- 梯度侧：`grad_reduce_queue` / `need_zero_grads`

### 5.1 BufferState 状态机（`:58`）

```
        ┌────────── 复用 (READY -> USING，省一次 all_gather) ──────────┐
        │                                                             │
     FREED ──_issue_async_gather──> SYNCING ──wait()──> USING ──shard_params──> READY
        ^                                                                        │
        └───────── _release_one_buffer_if_needed（仅在超出 limit 时释放）──────────┘
     FREED ──同步 all_gather──> USING            （overlap 关闭或没来得及预取时）
```

- `FREED`：只有分片，`tmp_data_buffer` 未分配，占 $1/N$。
- `SYNCING`：异步 all_gather 在飞，已占完整显存。
- `USING`：完整参数正在被计算使用。
- `READY`：完整参数仍在，但计算已结束 —— 可被复用，也可被回收。

**`READY` 是这套实现的精髓**：一层算完后只清参数的“壳”，不回收 tmp buffer。
好处有两个：反向紧接着用同一层时可 `READY -> USING` 直接复用（省一次 all_gather）；
tie 权重被多处复用时也不必反复通信。

### 5.2 显存与 overlap 的旋钮（`:464`）

```python
groups_per_unit = 2
self.double_buffer_limit = double_buffer_limit or (groups_per_unit * 2)   # 默认 4
self.prefetch_units = max(self.double_buffer_limit // groups_per_unit - 1, 0)  # 默认 1
self.buffer_cnt_in_using = 0        # 当前处于非 FREED（占完整显存）的 buffer 数
```

- `groups_per_unit = 2` 是经验值：一个 Transformer 层通常产生约 2 个 buffer group。
- `double_buffer_limit = 4` ⇒ 约允许 **2 个 unit 同时 unshard** ⇒ `prefetch_units = 1`，
  即“用当前层的计算掩盖下一层的通信”。
- 省显存就减小它（=2 时 `prefetch_units=0`，**预取完全关闭**，退化为同步通信）；
  加深 overlap 就增大它，代价是同时 unshard 的层数变多。

### 5.3 释放策略：`_release_one_buffer_if_needed`（`:477`）

```python
def _release_one_buffer_if_needed(self, keep=()):
    while self.buffer_cnt_in_using >= self.double_buffer_limit:
        found = False
        for gid_idx, group in enumerate(self.buffer_manager.buffer_groups):
            if gid_idx in keep:              # 本次请求要用的，不能释放
                continue
            if not group.params_buffer.is_sharded:
                continue                     # 单卡 buffer 没有 tmp，无需释放
            if group.params_buffer.status == BufferState.READY:
                group.params_buffer.status = BufferState.FREED
                group.params_buffer.clear_tmp_buffer()      # 真正归还显存
                self.buffer_cnt_in_using -= 1
                found = True
                break
        if not found:
            break                            # 都在 USING/SYNCING，放弃释放
```

只回收 `READY`（`USING` 正在用、`SYNCING` 在飞都不能动）；`found=False` 时**直接退出而不是
报错**，也就是说 `double_buffer_limit` 是**软上限** —— 极端情况下宁可多占显存也不死锁。

### 5.4 预取选择：`_next_buffer_id`（`:495`）

```python
def _next_buffer_id(self, req_gids, is_backward):
    if self.prefetch_units <= 0:
        return []
    step = -1 if is_backward else 1                 # 前向 gid+1，反向 gid-1
    requested = set(req_gids)
    next_gid = (min(req_gids) if is_backward else max(req_gids)) + step
    while next_gid in requested:
        next_gid += step

    gids, seen_units = [], set()
    while 0 <= next_gid < len(groups):
        group = groups[next_gid]
        if is_backward and not group.params_buffer.trainable:
            next_gid += step                        # 反向跳过冻结参数
            continue
        if group.fsdp_unit_id not in seen_units:
            if len(seen_units) >= self.prefetch_units:
                break                               # 已取够 prefetch_units 个 unit
            seen_units.add(group.fsdp_unit_id)
        gids.append(next_gid)
        next_gid += step
    return gids
```

- **方向由 `is_backward` 决定**：反向按层逆序执行，`gid` 递减才是“下一个要用的”。
- **以 unit 为计数单位**：`prefetch_units=1` 意味着“把下一个 unit 的所有 buffer 都取上”，
  这样下一层一进来就已是 `SYNCING/READY`。
- 反向跳过 `trainable=False` 的 buffer，避免为不需要梯度的参数白发通信。

### 5.5 主流程：`all_gather_params`（`:529`）

```python
# 阶段 0：收集本次请求涉及的 buffer id（去重、保序）
req_gids = [...]

# 阶段 0.5（仅反向）：跨 unit 时结算 expert 梯度
if is_backward:
    unit_id = buffer_groups[req_gids[0]].fsdp_unit_id
    if unit_id != self._last_backward_unit_id:
        self._last_backward_unit_id = unit_id
        self._flush_expert_grads_after_unit(unit_id)

# 阶段 1（overlap 开启）：给本次要用的 FREED buffer 发异步 gather，并预取后继 unit
if self.enable_overlap:
    keep = set(req_gids)
    for gid in req_gids:
        if status == FREED:
            self._release_one_buffer_if_needed(keep)
            self._issue_async_gather(gid)              # FREED -> SYNCING
    for next_gid in self._next_buffer_id(req_gids, is_backward):
        if status != FREED: continue
        self._release_one_buffer_if_needed(keep)
        if self.buffer_cnt_in_using >= self.double_buffer_limit:
            break                                      # 腾不出位置就不预取
        self._issue_async_gather(next_gid)

# 阶段 2：逐参数落地
for param in params:
    if status == READY:   status = USING                # 复用，零通信
    if status == SYNCING: status = USING; comm_task.wait(); comm_task = None
    tmp_buffer = params_buffer.get_tmp_buffer()
    if status == FREED:                                 # 没来得及预取 → 同步兜底
        params_buffer.fsdp_group.process_group.all_gather(
            params_buffer.data_buffer, tmp_buffer).wait()
        self.buffer_cnt_in_using += 1
        status = USING
    # 阶段 3：把参数的壳挂到完整数据上
    tmp_param = paddle._C_ops.view_slice(tmp_buffer, offset, offset + param._numel())
    tmp_param.get_tensor()._set_dims(param.shape)
    param.get_tensor()._share_data_with(tmp_param.get_tensor())
```

三条路径的性能含义：`READY` = 0 次通信；`SYNCING` = 通信已被计算掩盖，只付 wait 的残余；
`FREED` = 通信完全暴露。**profiling 若看到大量同步 all_gather，说明预取没跟上，
应检查 `double_buffer_limit`，以及 buffer 排序是否与真实执行顺序一致。**

### 5.6 释放与复位：`shard_params` / `reset_params_buffer_status`

```python
def shard_params(self, params, is_backward=False):     # :612
    for param in params:
        stop_gradient, _shape = param.stop_gradient, param.shape
        param._clear_data()                            # 只清壳，不回收 tmp buffer
        param.stop_gradient = stop_gradient
        param.get_tensor()._set_dims(_shape)
        affected_gids.add(gid)
    for gid in affected_gids:
        if group.params_buffer.status == BufferState.USING:
            group.params_buffer.status = BufferState.READY     # 惰性释放
```

```python
def reset_params_buffer_status(self):                  # :636
    self._last_backward_unit_id = None
    for group in ...:
        if status in (READY, USING):
            clear_tmp_buffer(); status = FREED
            if is_sharded and self.buffer_cnt_in_using > 0:
                self.buffer_cnt_in_using -= 1
```

后者在每个 `optimizer.step()` 前被调用（经 `comm_sync_and_reset_status`），把所有 buffer
拉回 `FREED`、释放全部 tmp 显存：**optimizer 更新时显存最宽松，下一个 step 从干净状态开始。**

### 5.7 梯度侧：懒清零、写保护、RS 发起

```python
def _maybe_zero_grads(self):                           # :655
    if not self.need_zero_grads: return
    self.need_zero_grads = False
    for group in ...:
        if group.grads_buffer is not None:
            group.grads_buffer.data_buffer.zero_()
```

`need_zero_grads` 由 `comm_sync_and_reset_status()` 在每个 step 末置回 `True`，所以
**清零发生在一个 step 内第一次写梯度时（懒清零），而不是 `clear_grad()` 里** ——
梯度累加的多个 micro-batch 之间因此不会被误清。

```python
def _ensure_grads_writable(self, param):               # :663
    grads_buffer = ...
    while grads_buffer in self.grad_reduce_queue:
        self._wait_for_grad_comm(queue_limit=len(self.grad_reduce_queue) - 1)
```

**为什么需要它？** reduce_scatter 的输入就是 `tmp_data_buffer`。若该 buffer 的 RS 还在飞，
此时又往里 `add_` 新梯度，就会与通信读发生竞争、结果错乱。这里等到它出队为止。
梯度累加、或同一参数在一个 step 内被多次使用（共享层）时会真实触发这条路径。

```python
def reduce_scatter_grads(self, param):                 # :766
    self._maybe_zero_grads()
    group.grads_use_cnt += 1
    param.main_grad = None                             # 视图即将失效，防误用
    if group.grads_buffer is not None and (
            group.is_expert_param or not group.grads_buffer.is_sharded):
        return
```

虽然名字叫 `reduce_scatter_grads`，但它在 grad hook 里**并不发起 RS**：只清掉 `main_grad`
视图引用并给 `grads_use_cnt` 计数。`grads_use_cnt` 目前只有自增和清零、没有任何读取，
是早期“组内梯度到齐即 RS”的计数触发方案的遗留 —— 该方案对共享层、MoE 专家等“一个
参数在一次反向里被写多次”的情况不可靠，已被 5.8 的结构性触发取代。

真正发起 RS 的是 `_reduce_group_grads`，调用方只有 5.8 的 `_flush_expert_grads_after_unit`：

```python
def _reduce_group_grads(self, group):                  # :778
    group.grads_use_cnt = 0
    grads_buffer = group.grads_buffer
    if (grads_buffer is None or not grads_buffer.is_sharded
            or grads_buffer.tmp_data_buffer is None       # 本 step 没人写过梯度
            or grads_buffer in self.grad_reduce_queue):   # 已在飞，勿重复发
        return
    self._wait_for_grad_comm()                            # 限流（默认队列 ≤ 2）
    grads_buffer.comm_task = grads_buffer.do_reduce_scatter()
    self.grad_reduce_queue.append(grads_buffer)
    if not self.enable_overlap:
        self._wait_for_grad_comm(queue_limit=0)           # 同步模式：立即等
```

四个守卫条件缺一不可，尤其 `tmp_data_buffer is None` —— 它同时覆盖“该组全是冻结参数”
和“该组本 step 未参与计算（如未被路由到的专家）”两种情况。

### 5.8 梯度 RS 的结构性触发（`:796`）

```python
# all_gather_params(params, is_backward=True) 中（:609）
if unit_id != self._last_backward_unit_id:       # 反向首次进入该 unit
    self._last_backward_unit_id = unit_id
    self._flush_expert_grads_after_unit(unit_id)
    self._release_params_after_unit(unit_id)

def _flush_expert_grads_after_unit(self, unit_id):
    if unit_id is None: return
    for group in reversed(self.buffer_manager.buffer_groups):
        if group.fsdp_unit_id is None: continue
        if group.fsdp_unit_id > unit_id:
            self._reduce_group_grads(group)
```

核心判断：**反向从最后一个 unit 往前走，一旦开始处理 unit $i$，所有 unit $> i$ 的梯度
就一定写完了**，此刻统一发起异步 RS。注意它**对 dense 组和 expert 组一视同仁**（没有
`is_expert_param` 过滤），函数名中的 “expert” 是历史命名。已发过 RS 的组会被
`_reduce_group_grads` 的守卫（`tmp_data_buffer is None` / 已在队列）跳过，重复调用无副作用。

与参数侧的 `_release_params_after_unit` 共用同一个“进入新 unit”的时机：unit $i+1$ 的梯度
RS、参数释放，与 unit $i$ 的反向计算重叠。

结构性触发覆盖不到的组，统一留给 5.9 的 `finish_grads_sync` 同步兜底：

- **最前面的 unit**：之后不再有更小的 unit 进入，没人 flush 它；
- **`fsdp_unit_id is None` 的组**：不属于任何 FSDP unit 的参数（如 tie weight、final norm）；
- **非分片 buffer**：不走 RS，在收尾时改用 `all_reduce` / `reduce`（muon）+ scale。

### 5.9 队列限流与收尾（`:726` / `:738`）

```python
def _wait_for_grad_comm(self, queue_limit=2):
    while len(self.grad_reduce_queue) > queue_limit:
        grads_buffer = self.grad_reduce_queue.pop(0)      # FIFO
        if grads_buffer.comm_task is None:
            grads_buffer.clear_tmp_buffer(); continue
        grads_buffer.comm_task.wait()
        grads_buffer.comm_task = None
        grads_buffer.accumulate_reduced_grad()            # 累加进分片 + 释放 tmp

def finish_grads_sync(self):
    self._wait_for_grad_comm(queue_limit=0)               # 等完所有在飞的
    for group in ...:                                     # 兜底：补做没触发的组
        group.grads_use_cnt = 0
        if grads_buffer.tmp_data_buffer is None: continue
        grads_buffer.do_reduce_scatter().wait()
        grads_buffer.accumulate_reduced_grad()
```

`finish_grads_sync` 的第二个循环是**正确性兜底**：某组若因 expert 触发条件、或部分参数
本 step 未参与计算而未触发 RS，这里同步补上。它保证“到 `optimizer.step()` 时，
所有梯度一定已落到 `grads_buffer.data_buffer`”。

---

## 六、Hook 体系与完整时序

### 6.1 能力边界与解法

FSDP 需要在四个时刻插入动作，但 Paddle 的 hook 能力并不一一对应：

| 需要的时刻 | 可用机制 |
| --- | --- |
| 某层前向前 unshard | `register_forward_pre_hook` ✅ |
| 某层前向后 reshard | `register_forward_post_hook` ✅ |
| 某层反向前 unshard | ❌ 没有直接 hook → 用 `FusionBackwardHook`（挂输出） |
| 某参数梯度算完后 reduce | `Tensor._register_grad_hook` ✅ |
| 某层反向后 reshard | 一般由 grad hook 顺带完成；tie 权重用 `FusionForwardHook`（挂输入） |

两个 `PyLayer` 的诀窍是：**PyLayer 在前向是恒等映射，但会在计算图上留一个节点，
反向经过该节点时就得到了一个精确的时机。** 挂在输出上 → 反向早于该层；
挂在输入上 → 反向晚于该层。

```python
class FusionBackwardHook(PyLayer):        # :754  挂在 layer 输出上
    @staticmethod
    def forward(ctx, *inputs, layer, comm_manager, recursive=False):
        ctx.layer, ctx.comm_manager, ctx.recursive = layer, comm_manager, recursive
        return inputs if len(inputs) > 1 else inputs[0]      # 恒等
    @staticmethod
    def backward(ctx, *args):
        trainable_params = [p for p in ctx.layer.parameters(include_sublayers=ctx.recursive)
                            if p.trainable]
        ctx.comm_manager.all_gather_params(trainable_params, is_backward=True)  # 反向前 unshard
        return args

class FusionForwardHook(PyLayer):         # :776  挂在 layer 输入上
    @staticmethod
    def forward(ctx, *inputs, layer, comm_manager, recursive=False):
        ...
        return inputs
    @staticmethod
    def backward(ctx, *args):
        ctx.comm_manager.shard_params(
            ctx.layer.parameters(include_sublayers=ctx.recursive), is_backward=True)  # 反向后 reshard
        return args
```

一句话记忆：**输出侧的 PyLayer 负责反向前 unshard，输入侧负责反向后 reshard。**

### 6.2 注册流程（`register_tensor_fusion_hooks`, `:1152`）

```python
# 1. 所有可训练参数注册 grad hook
for param in model.parameters():
    if param.trainable:
        self._register_grad_hook_once(param)

# 1.5 顶层前向前，为 expert 参数重绑 main_grad（tmp buffer 地址可能已变）
model.register_forward_pre_hook(_bind_experts_pre_forward)

# 2. 自顶向下递归注册 forward hook
def _register_recursive(layer):
    is_unit = type(layer).__name__ in self.buffer_manager.fsdp_unit_layers
    if is_unit:
        layer.register_forward_pre_hook(_pre_forward_hook(layer, recursive=True))
        layer.register_forward_post_hook(_post_forward_hook(layer, recursive=True))
        self._register_fusion_layer_hooks(layer, recursive=True)
        return                        # 命中 unit 后停止下探
    if layer.parameters(include_sublayers=False):     # 只为“自己直接持参”的层注册
        layer.register_forward_pre_hook(_pre_forward_hook(layer, recursive=False))
        layer.register_forward_post_hook(_post_forward_hook(layer, recursive=False))
        self._register_fusion_layer_hooks(layer, recursive=False)
    for child in layer.children():
        _register_recursive(child)
_register_recursive(model)
```

**递归策略的两个不变量**（破坏任一条都会出现重复 unshard/reshard，甚至参数被提前释放）：

1. 命中 `fsdp_unit_layers` 的层以 `recursive=True` 整层注册，并且**不再下探**；
2. 未命中的层只在“自己直接持有参数”时注册 `recursive=False`，
   所以每个参数恰好被一个层的 hook 管辖。

`_pre_forward_hook` / `_post_forward_hook` 生成的闭包很薄：

```python
@paddle.autograd.no_grad()
def all_gather_comm(*_):
    params = sublayers.parameters(include_sublayers=recursive)
    comm_manager.all_gather_params(params)
    self._bind_expert_main_grads(params)      # expert 参数的 main_grad 重绑

@paddle.autograd.no_grad()
def shard_comm(*_):
    comm_manager.shard_params(sublayers.parameters(include_sublayers=recursive))
```

`@paddle.autograd.no_grad()` 很重要：hook 内部的 `add_`、`_share_data_with` 等操作
绝不能进入计算图。

### 6.3 参数梯度 hook（`_register_grad_hook_once`, `:1125`）

```python
def _register_grad_hook_once(self, param):
    param.main_grad = None
    if not hasattr(param, "get_main_grad"): return       # 冻结参数没有这个注入
    if getattr(param, "_fsdp_grad_hooked", False): return  # 幂等，防重复注册
    param._fsdp_grad_hooked = True

    @paddle.autograd.no_grad()
    def comm_hook(grad):
        comm_manager._maybe_zero_grads()                 # 本 step 首次：清零梯度分片
        comm_manager._ensure_grads_writable(param)       # 若该 buffer 的 RS 在飞，先等
        if grad is not None and grad._is_initialized():
            param.get_main_grad(grad.shape)              # 绑到 fused buffer 视图
            param.main_grad.add_(grad)                   # 累加（bf16 grad -> fp32 buffer）
            grad._clear_data()                           # 立刻释放 autograd 产出的 grad
        comm_manager.shard_params([param], is_backward=True)   # 释放参数视图
        comm_manager.reduce_scatter_grads(param)               # 仅计数 + 清 main_grad，不发 RS（见 5.7）
    param._register_grad_hook(comm_hook)
```

`grad._clear_data()` 是显存优化的关键一步：autograd 刚产生的 bf16 梯度在累加进 main_grad
后立刻释放，避免“fused 梯度 buffer + 逐参数梯度”双份占用。

### 6.4 tie_weights 的特殊处理（`:1224` 起）

共享权重（embedding 与 lm_head 共用一个 `weight`）有两个麻烦：

1. 它在一个 step 内被 **两处** 前向使用 —— 第一处的 post hook 若把它 reshard 掉，
   第二处就要再 gather 一次。→ 靠 `READY` 复用缓解（5.1）。
2. 它的梯度来自 **两条路径** —— 叶子张量的 grad hook（`comm_hook`）要等两条路径都写完、
   即 embedding 反向结束后才触发一次。若只靠它 reshard，完整的 $W$（vocab × hidden）会从
   lm_head 反向一直以 `USING` 状态挂到 embedding 反向，期间既占显存又占 double buffer 名额
   （`_release_one_buffer_if_needed` 只回收 `READY`）。
   → 在 tie 层输入侧挂 `FusionForwardHook`：它的 backward 在“反向走完这一层”后执行，
   lm_head 反向一结束就 `shard_params`（`USING → READY`），embedding 反向时再由其
   `FusionBackwardHook` 复用或重新 gather；`comm_hook` 里的 `shard_params` 只是幂等收尾。
   （embedding 输入是 int token id，`stop_gradient=True`，其上的 FusionForwardHook 不会触发，
   实际生效的是 lm_head 上那个。）注册方式：

```python
layer.register_forward_post_hook(_forward_post_hook)     # 所有层：插 FusionBackwardHook
for param in layer.parameters(include_sublayers=False):
    if param.name == self.comm_manager.buffer_manager.tie_param_name:
        layer.register_forward_pre_hook(_forward_pre_hook)   # 仅 tie：插 FusionForwardHook
```

`tie_param_name` 的识别（`:351`）依赖模型实现 `get_input_embeddings()`：

```python
if hasattr(self.model, "get_input_embeddings"):
    input_embeddings = self.model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        self.tie_param_name = input_embeddings.weight.name
```

`_forward_post_hook` 还要处理输出结构：`dict` 时逐 value 包装（跳过 `stop_gradient` 的）、
`tuple` 时整体包装、单张量直接包装 —— 目的是保证 `FusionBackwardHook` 一定在反向路径上。

### 6.5 一个完整 step 的时间线

以 4 层（L0..L3）、`prefetch_units=1`、`double_buffer_limit=4` 为例：

```
FORWARD
  pre_hook(L0)   : gather(L0) 同步（首层无从预取）+ 发出异步 gather(L1)
  L0.compute                                        <- 与 gather(L1) 重叠
  post_hook(L0)  : L0 壳释放，状态 USING -> READY（显存暂留）
                   插入 FusionBackwardHook 到 L0 输出
  pre_hook(L1)   : wait(L1) -> USING + 异步 gather(L2)
                   若 buffer_cnt_in_using 到顶 -> 回收 READY 的 L0
  L1.compute                                        <- 与 gather(L2) 重叠
  ...
  loss = f(L3.out)

BACKWARD（逆序）
  FusionBackwardHook(L3).backward :
        _flush_expert_grads_after_unit(unit(L3))     # 后面已无 unit，无事可做
        gather(L3)（多半是 READY 复用）+ 异步 gather(L2)
  L3 各 param 的 comm_hook :
        main_grad.add_(grad); grad 释放; 参数壳释放（不发 RS）
  FusionBackwardHook(L2).backward :
        _flush_expert_grads_after_unit(unit(L2)) -> 异步 reduce_scatter(L3.grad)  <- 与 L2 反向重叠
        gather(L2) 已就绪 + 异步 gather(L1)
  ...
  L0 完成；queue 中最多 2 个 RS 在飞；L0 的梯度留给 finish_grads_sync 同步做

OPTIMIZER.STEP
  comm_sync_and_reset_status():
        finish_grads_sync()          -> 等完/补做所有 RS
        reset_params_buffer_status() -> 所有 buffer 回 FREED，tmp 全释放
        need_zero_grads = True; 所有 param.main_grad = None
  params_grads = [(fuse_params_i, fuse_grads_i) ...]
  grad_clip（跨 fsdp_group 求 global norm）-> AdamW 在分片上更新
```

---

## 七、Optimizer 集成

### 7.1 全局 context：为什么要有 `_fsdp_context.py`

```python
# python/paddle/distributed/fsdp/_fsdp_context.py
_g_fsdp_context = None
def register_fsdp_context(context): ...
def get_fsdp_context(): return _g_fsdp_context
```

optimizer 是用户自己构造的，FSDP 无法侵入它的构造过程，所以用一个全局注册表让
`Optimizer._apply_optimize` 能找到当前的 FSDP context。**PR #78577 把这个注册表从
`auto_parallel.fully_shard_fusion` 抽到独立模块**，使自动并行与手动并行两条实现共用一处，
optimizer 侧只需一个 `get_fsdp_context()` 分支。

### 7.2 唯一接管点（`python/paddle/optimizer/optimizer.py:1702`）

```python
fsdp_context = get_fsdp_context()
if fsdp_context is not None:
    if self._param_groups and isinstance(self._param_groups[0], dict):
        raise NotImplementedError("FSDP does not support optimizer parameter groups.")
    fsdp_context.comm_sync_and_reset_status()      # ① 等完梯度通信 + 状态复位
    if hasattr(fsdp_context, "bind_decay_param_fun"):
        fsdp_context.bind_decay_param_fun(self)    # ② 修正 weight decay 判定
    new_params_grads = []                          # ③ 换成 fused 分片
    for group in fsdp_context.buffer_manager.buffer_groups:
        if not group.params_buffer.data_buffer.stop_gradient:
            new_params_grads.append((group.params_buffer.data_buffer,
                                     group.grads_buffer.data_buffer))
    params_grads = new_params_grads
    if self._grad_clip is not None:                # ④ 告知 grad clip 这是分片
        self._grad_clip.should_comm_on_shard_dim = True
        self._grad_clip.fsdp_group = fsdp_context.buffer_manager._fsdp_group
```

关键含义：

- **替换发生在 step 内部**。用户仍然用 `parameters=model.parameters()` 构造 optimizer，
  但真正被更新的是 `fuse_params_<gid>` 这些分片张量 → 用户代码零改动。
- **optimizer 状态自动分片**。因为 `_create_accumulators` 是按传入的参数创建的，
  master weight / moment1 / moment2 的形状就等于分片形状，天然只有 $1/N$。
  顺带把逐参数的小 kernel 合成了每 buffer 一个大 kernel，**step 时间也显著下降**。
- **grad clip 必须知情**。每张卡只有梯度的一部分，`global_norm` 需要在 `fsdp_group` 上
  再做一次通信；否则 clip 系数算错，训练缓慢跑偏（不会报错，最难查的一类问题）。
- **param_groups 显式不支持**。一块 buffer 只能对应一套超参（lr scale 继承 `params[0]` 的
  `optimize_attr`），混进多套会静默算错，所以直接抛 `NotImplementedError`。

### 7.3 weight decay 的静默陷阱（`bind_decay_param_fun`, `:947`）

`AdamW` 用 `apply_decay_param_fun(name) -> bool` 按**参数名**决定是否施加 decay。
FSDP 把参数换成了 `fuse_params_3`，这个回调对所有名字都返回 `False`，
**weight decay 会被静默丢掉** —— 训练能跑、loss 曲线也不炸，但正则化没了。

```python
def bind_decay_param_fun(self, optimizer):
    fun = getattr(optimizer, "_apply_decay_param_fun", None)
    if fun is None or getattr(optimizer, "_fsdp_decay_fun_bound", False):
        return                                    # 幂等：每个 step 都会调用
    buffer_decay = {}
    for group in self.buffer_manager.buffer_groups:
        buffer_name = group.params_buffer.data_buffer.name
        decayed, exempt = [], []
        for param in group.params:
            (decayed if fun(param.name) else exempt).append(param.name)
        if decayed and exempt:
            raise ValueError(...)                 # 一块 buffer 只能有一个答案
        buffer_decay[buffer_name] = bool(decayed) and not group.no_decay

    def fsdp_apply_decay_param_fun(name):
        if name in buffer_decay: return buffer_decay[name]
        return fun(name)                          # 非 fused 参数走原逻辑

    optimizer._apply_decay_param_fun = fsdp_apply_decay_param_fun
    optimizer._fsdp_decay_fun_bound = True
```

若一块 buffer 内混了“要 decay”和“不要 decay”的参数，**直接抛错并给出修复指引**：
先给豁免参数打 `param.no_weight_decay = True`，再调 `fully_shard()`；
该标记是分组键第 ⑦ 项，打上后自然分到不同 buffer。

> 这类“融合导致按名字工作的机制失效”的问题，是所有 tensor fusion 实现共同的隐患。
> 自己接入新优化器/新正则时，第一件事就是检查它是否依赖参数名。

### 7.4 辅助接口

```python
def owns_optimizer_params(self, optimizer):     # :980
    """该 optimizer 是否在更新本 context 的 fused buffer（按 param.name 判断）"""

def init_optimizer_state(self, optimizer):      # :989
    """load 之前先在 fused buffer 上创建 accumulator，否则 state_dict 无处可填"""
    parameter_list = [g.params_buffer.data_buffer
                      for g in self.buffer_manager.buffer_groups
                      if not g.params_buffer.data_buffer.stop_gradient]
    optimizer._create_accumulators(
        paddle.base.framework.default_main_program().global_block(), parameter_list)
```

`init_optimizer_state` 解决的是恢复训练时的顺序问题：优化器状态是 lazy 创建的
（第一次 `step()` 才有），而 load checkpoint 发生在 step 之前，因此必须先手动建出来。

---

## 八、Checkpoint：分布式 save/load

FSDP 存 checkpoint 有两种思路：(a) all_gather 还原完整参数再存 —— 简单但要一份完整显存/内存；
(b) **各 rank 只存自己那段，并把“这是谁的哪一段”描述清楚**。Paddle 选了 (b)，
由 `flex_checkpoint` 负责跨 rank 拼装，好处是 **checkpoint 与并行度解耦**（换 $N$ 也能加载）。

### 8.1 描述结构 `ShardedWeight`

```python
ShardedWeight(
    key=struct_name,                 # 结构化参数名，如 "layers.0.linear.weight"
    local_tensor=data_buffer[begin:end],   # 本 rank 实际持有的那段
    local_shape=..., global_shape=..., global_offset=...,
    is_flattened=True,               # 关键：说明这是 flatten 后的片段
    flattened_range=slice(a, b),     # 在该参数 flatten 视角下的区间
)
```

### 8.2 先解决“壳没有数据”的鸡生蛋问题（`_base_sharded_state_dict`, `:832`）

模型自带的 `sharded_state_dict()` 需要读参数的元信息，但 fused 参数已被 `_clear_data()`。
做法是临时给它们指一块 scratch 显存，拿到描述后立刻还原：

```python
cleared, scratch = [], {}
for group in ...:
    for param in group.params:
        if param._is_initialized(): continue
        cleared.append((param, param.shape, numel))
        scratch[param.dtype] = max(scratch.get(param.dtype, 0), numel)   # 每 dtype 一块，取最大
scratch = {dtype: paddle.empty([numel], dtype=dtype) for dtype, numel in scratch.items()}
for param, shape, numel in cleared:
    view = paddle._C_ops.view_slice(scratch[param.dtype], 0, numel)      # 所有参数共用同一块
    view.get_tensor()._set_dims(shape)
    param.get_tensor()._share_data_with(view.get_tensor())
try:
    return self._model_sharded_state_dict(structured_name_prefix)
finally:
    for param, shape, _ in cleared:                                      # 一定还原
        param._clear_data(); param.get_tensor()._set_dims(shape)
```

注意 scratch 是**按 dtype 一块、取该 dtype 下最大 numel**，所有参数共用 —— 因为这里只要
元信息（名字/形状/dtype），数据内容无意义，所以可以极限复用显存。

### 8.3 生成分片描述（`sharded_state_dict`, `:881`）

```python
base = self._base_sharded_state_dict(prefix)          # 结构化名 -> ShardedWeight(模型视角)
static_to_struct = {}                                 # 静态名(param.name) -> [结构化名...]
for key, sw in sorted(base.items()):
    static_to_struct.setdefault(sw.local_tensor.name, []).append(key)

result = dict(base)
for group in self.buffer_manager.buffer_groups:
    for param in group.params:
        struct_names = static_to_struct.get(param.name)
        ref = base[struct_names[0]]                   # tie 权重共享一段，取第一个作基准
        shard = group.params_buffer.param_shard(param)
        if shard is None:                             # 该参数完全不在本 rank
            for n in struct_names: result.pop(n, None)
            continue
        begin, end, flattened_range = shard
        for n in struct_names:
            result[n] = self._create_sharded_weight(
                n, group.params_buffer.data_buffer, slice_info)
    self._shard_descs[params_buffer.data_buffer.name] = param_slice_info
return result
```

三个细节：

- **`shard is None` 时要从 state_dict 里删掉**，否则会存出一个空片段，
  加载端会认为该参数缺失或形状不符。
- **tie 权重**：多个结构化名指向同一 `param.name`，它们共享同一段描述；
  但只有第一个 key 会进入 `_shard_descs`（optimizer 状态只存一份）。
- **`_shard_descs`** 记录 `fused buffer 名 -> {结构化名: slice 信息}`，
  供 8.5 切分 optimizer 状态复用，避免重复计算。

### 8.4 EP 场景的 axis-0 折叠（`:914`）

```python
if ep_nranks > 1 and global_shape == local_shape:
    # 模型把 expert 描述成“replicated”，但各 EP rank 上其实是不同的专家。
    # 若不区分，多个 rank 会写到同一个 key，互相覆盖。
    global_shape = (local_shape[0] * ep_nranks, *local_shape[1:])
    global_offset = (ep_rank * local_shape[0], *(0 for _ in local_shape[1:]))
```

即把 EP 维度“折叠”进 axis 0，让每个 EP rank 的专家占据全局张量的不同区间，key 才能唯一。

### 8.5 优化器状态的名字反解析（`optimizer_sharded_state_dict`, `:1001`）

optimizer 的 `state_dict()` 的 key 形如
`fuse_params_0_adamw_fp32_master_0_moment1_3`，需要反解析成 `(buffer 名, 状态 tag)`：

```python
tags = sorted({getattr(cls, attr)                       # 收集所有 '_*_acc*_str' 声明
               for cls in type(inner).__mro__ for attr in vars(cls)
               if attr.endswith("_str") and "_acc" in attr
               and isinstance(getattr(cls, attr), str)},
              key=len, reverse=True)                    # 长的优先匹配，避免前缀歧义

def _split_optimizer_state_name(vname):
    for tag in tags:
        marker = "_" + tag + "_"
        idx = vname.rfind(marker)
        if idx < 0 or not vname[idx + len(marker):].isdigit(): continue
        base = vname[:idx]
        if opt_name and base.endswith("_" + opt_name):   # 去掉优化器名前缀
            base = base[: -len(opt_name) - 1]
        matched = _master_weight_suffix.match(base)      # 去掉 '_fp32_master_\d+'
        if matched: base = matched.group(1)
        return base, f"{tag.lstrip('_')}_0"
    return None, None
```

随后按 `_shard_descs` 把每个状态张量切给各结构化参数名：

```python
if int(tensor.numel()) == 1:                    # beta1_pow 这类标量 → 每个 key 复制一份
    sharded_state[key] = _replicated_scalar(key, tensor)
else:
    sharded_state[key] = self._create_sharded_weight(key, tensor, slice_info)
```

`master_weights` 单独处理，存为 `<struct_name>.w_0`。

**解析失败会抛 `NotImplementedError` 而不是跳过**：

```python
raise NotImplementedError(
    f"FSDP cannot save optimizer state '{vname}' ... the accumulator is not declared "
    "through a '_*_acc*_str' attribute, and skipping it would silently reset the state on load.")
```

理由写在错误信息里：静默跳过会导致恢复训练时该状态被悄悄重置（比如动量丢失），
这种错误极难发现，所以宁可让 save 失败。**自定义优化器接 FSDP 时，
务必用 `_xxx_acc_str` 类属性声明 accumulator 名。**

---

## 九、显存与性能定量分析

### 9.1 稳态显存

$$M \approx \underbrace{\frac{(2 + 4 + 4 + 4)\Psi}{N}}_{\text{param/master/m/v 分片}}
+ \underbrace{\frac{4\Psi}{N}}_{\text{fp32 梯度分片}}
+ \underbrace{C_{\text{unshard}}}_{\text{double buffer}}
+ \underbrace{C_{\text{grad tmp}}}_{\text{在飞的梯度}}
+ M_{\text{activation}}$$

其中两个常数项才是 FSDP 调参真正能动的部分：

$$C_{\text{unshard}} \approx \text{double\_buffer\_limit} \times \overline{S_{\text{buffer}}} \times s_{\text{param}}$$
$$C_{\text{grad tmp}} \approx (\text{queue\_limit} + 1) \times \overline{S_{\text{buffer}}} \times 4\ \text{bytes}$$

$\overline{S_{\text{buffer}}}$ 是单个 buffer 的元素数（约等于半层参数量）。
注意梯度 tmp 是 **fp32**，所以它的单位显存是参数侧的 2 倍 —— 显存吃紧时，
减小梯度队列长度（源码里 `_wait_for_grad_comm` 的默认 `queue_limit=2`）往往比动
`double_buffer_limit` 更立竿见影。

### 9.2 举例：7B 模型、$N=8$

假设 32 层、每层约 200M 参数、每层 2 个 buffer（$\overline{S_{\text{buffer}}} \approx 100$M）：

- 常驻分片：$18\Psi/N = 18 \times 7\text{e}9 / 8 \approx 15.8$ GB
- $C_{\text{unshard}} = 4 \times 100\text{M} \times 2\text{B} = 0.8$ GB
- $C_{\text{grad tmp}} = 3 \times 100\text{M} \times 4\text{B} = 1.2$ GB
- 合计约 17.8 GB + 激活值

与 DDP 的 $16\Psi = 112$ GB（单卡装不下）对比即可看出 FSDP 的价值。
上述是量级估算，实际还要加上 allocator 碎片、通信 buffer 与对齐 padding。

### 9.3 初始化期的峰值（容易被忽略）

`TensorFusionBuffer.__init__` 会先按**完整大小**分配 buffer，把参数拷进去，再切分。
此刻同时存在“完整 buffer + 参数原始数据”，峰值约 $2\Psi_{\text{group}}$。
逐 group 构造意味着这个峰值不会累积到整模型，但仍是 $\max_g 2\Psi_g$。
源码中的 `TODO(lizhenxing)` 计划让完整 buffer 建在 CPU 上以消掉这一项。
**若在 `fully_shard()` 调用期间 OOM，先看这里，而不是训练配置。**

### 9.4 通信时间能否被隐藏

单层参数 $\Psi_L$、bf16、带宽 $B$（有效值，NVLink 内约 200–400 GB/s，跨机 IB 低一个量级）：

$$T_{\text{comm}}^{L} = \frac{2 \cdot \frac{N-1}{N} \Psi_L \cdot 2\text{B}}{B}, \qquad
T_{\text{compute}}^{L} \approx \frac{6 \cdot \Psi_L \cdot B_{\text{tokens}}}{\text{FLOPS}_{\text{eff}}}$$

$T_{\text{compute}}$ 与 token 数成正比，而 $T_{\text{comm}}$ 与之无关，
所以**增大 micro-batch/序列长度是让 FSDP 通信被隐藏的最直接手段**。
反之，小 batch + 跨机低带宽的组合下，FSDP 会明显慢于 DDP —— 这时应考虑
`sharding_degree` 只开到机内（配合 dp 维度做跨机），即混合并行。

### 9.5 profiling 时该看什么

- **是否有大量同步 all_gather**：说明预取失效（排序错、`prefetch_units=0`、
  或层间计算太快来不及）。
- **`reduce_scatter` 是否堆在反向末尾**：说明 unit 划分过粗、或大量参数不属于任何 unit
  （`fsdp_unit_id is None`），只能在 `finish_grads_sync` 里同步兜底。
- **`empty_cache` / allocator 抖动**：`clear_tmp_buffer` 频繁分配释放同尺寸块通常能被
  allocator 缓存命中；若看到大量 cudaMalloc，检查是否 buffer 尺寸差异过大。

---

## 十、使用、调参与精度对齐

### 10.1 最小可运行示例

```python
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fsdp import fully_shard

# 1) 初始化：FSDP 的并行度来自 sharding_degree
dist.init_parallel_env()
strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {"sharding_degree": 2, "dp_degree": 1,
                           "mp_degree": 1, "pp_degree": 1}
fleet.init(is_collective=True, strategy=strategy)

# 2) 先 fully_shard，再构造 optimizer（顺序重要，见 10.2）
model = MyTransformer()
model = fully_shard(model, enable_tensor_fusion_and_overlap=True)   # 原地改造，返回同一个 module

opt = paddle.optimizer.AdamW(learning_rate=1e-3,
                             parameters=model.parameters(),
                             multi_precision=True)

for x in data:
    loss = model(x).mean()
    loss.backward()
    opt.step()
    opt.clear_grad()
```

`fully_shard` 的签名（`fully_shard.py:85`）：

```python
fully_shard(module, *, mesh=None, reshard_after_forward=None, shard_placement_fn=None,
            mp_policy=None, offload_policy=None, ignored_params=None,
            enable_tensor_fusion_and_overlap=True)
```

手动并行分支目前只真正使用 `mp_policy.reduce_dtype`（→ `main_grad_dtype`）与
`enable_tensor_fusion_and_overlap`；`mesh` / `reshard_after_forward` /
`shard_placement_fn` / `offload_policy` / `ignored_params` 尚未接线（见 13.1）。

### 10.2 使用顺序与常见坑

- **先 `fully_shard`，再包 `MixPrecisionLayer` / 构造 optimizer**。
  测试 `dygraph_group_fsdp.py` 的顺序即为参考：`fully_shard` → `MixPrecisionLayer`
  → `AdamW` → `MixPrecisionOptimizer`。原因是 `fully_shard` 会重建参数存储，
  之后再取 `model.parameters()` 才拿到正确的对象状态。
- **不要用 optimizer `param_groups`**（会抛 `NotImplementedError`）。
  需要分组超参时，考虑改用多个 optimizer，或等待该能力补齐。
- **豁免 weight decay 的参数要提前标记** `param.no_weight_decay = True`（见 7.3）。
- **自定义 Transformer 层想按层通信**，让类名进 `fsdp_unit_layers`：
  目前 `fully_shard()` 没有透出这个参数，需要直接构造
  `FullyShardFusion(model, fsdp_unit_layers=['MyDecoderLayer'])`。
  否则会退化为“每个直接持参的 layer 一组”，通信更碎。

### 10.3 调参清单

- `enable_tensor_fusion_and_overlap=False`：关闭预取与异步通信（全同步）。
  **排查精度问题的第一步**，因为它把时序简化到最容易推理的状态。
- `double_buffer_limit`（需直连 `FullyShardFusion`）：显存↓ / overlap↓。
- `mp_policy=MixedPrecisionPolicy(reduce_dtype=paddle.bfloat16)`：
  梯度以 bf16 累加与通信，通信量减半、显存更省，但多步累加有精度风险。
- `sharding_degree`：不必等于总卡数。跨机带宽差时，让 sharding 只覆盖机内 8 卡、
  机间走 dp，通信效率通常更好。

### 10.4 精度如何对齐（测试的做法）

`test/collective/fleet/dygraph_group_fsdp.py` 的对齐方法值得照抄：

```python
sharding_model, fsdp_model = Model(), Model()
sharding_model.set_state_dict(fsdp_model.state_dict())     # 保证初值完全一致
sharding_loss = train_mlp(sharding_model, use_fsdp=False, data=data)   # group_sharded os
fsdp_loss     = train_mlp(fsdp_model,     use_fsdp=True,  data=data)
assert fsdp_loss == sharding_loss                          # 要求逐 step 严格相等
```

要点：**同一份初始权重 + 同一批数据 + 与 ZeRO-1（`level="os"`）对比**。
之所以能要求严格相等，是因为两者的数学操作序列被设计成一致
（`scale_ + SUM` 的 reduce_scatter 等价于 all_reduce mean 后取本段）。
第二个用例则覆盖 fp32 + 关闭 overlap + tie_weights 的组合路径。

自查顺序建议：单卡（`sharding_degree=1`，只融合不分片）→ 2 卡关 overlap
→ 2 卡开 overlap。三步中哪一步开始不对，就能定位问题在融合、分片还是异步调度。

---

## 十一、FAQ 与排查手册

**Q1：loss 与非 FSDP 基线不一致，怎么定位？**

按 10.4 的三步收敛法。若“单卡融合”就不对 → 问题在 `TensorFusionBuffer` 的布局/对齐或
optimizer 替换；若“2 卡同步”不对 → 问题在分片边界（`param_shard`）或 reduce_scatter 语义；
若只有“开 overlap”不对 → 问题在时序（预取过早/释放过早/`_ensure_grads_writable` 未覆盖）。

**Q2：weight decay 好像没生效？**

见 7.3。检查 optimizer 是否设置了 `apply_decay_param_fun`，以及日志中是否出现过
`bind_decay_param_fun`。注意 `bind_decay_param_fun` 是在第一次 `step()` 时才绑定的。

**Q3：`fully_shard()` 阶段就 OOM。**

初始化期峰值约 $2\Psi_{\text{group}}$（9.3）。可行手段：减小最大 buffer（拆分 unit）、
或在 CPU 上初始化模型再逐层搬。

**Q4：训练中期 OOM，但稳态显存看起来够。**

优先怀疑两个常数项：`double_buffer_limit`（参数侧）与梯度队列（fp32，单位显存更大）。
另外 tie 权重与冻结参数组会长期停在 `READY`，也会吃住一块完整显存。

**Q5：多卡 hang 住。**

FSDP 的所有通信都是**集合通信**，要求组内所有 rank 以相同顺序发起相同次数的通信。
常见诱因：模型存在**数据相关的控制流**（某些 rank 走了不同分支 → 某层参数没被 gather）、
MoE 路由导致某 rank 的专家一次也没被用到（本实现用 `tmp_data_buffer is None` 守卫规避），
或 `if step % k == 0` 之类只在部分 rank 成立的条件里调用了模型的一部分。

**Q6：梯度累加（`accumulate_steps > 1`）需要额外做什么？**

不需要。`accumulate_reduced_grad` 用 `add_` 累加、清零由 `need_zero_grads` 懒触发，
只要**不在中间调用 `optimizer.step()`** 即可。注意每个 micro-batch 都会付一次
reduce_scatter（没有 `no_sync` 语义），通信量随累加步数线性增长。

**Q7：如何确认预取真的生效？**

打点 `comm_manager.buffer_cnt_in_using` 与 `_issue_async_gather` 的调用次数，
或直接看 profiler：`SYNCING` 路径的 all_gather 应与前一层的计算 kernel 重叠。
若 `prefetch_units == 0`（`double_buffer_limit <= 2`），预取被完全关闭。

**Q8：能和 MP / PP 一起用吗？**

`hybrid_configs` 里可以同时开 mp/pp，FSDP 只在 sharding 维度工作。但需注意 MP 参数
（`is_distributed` 类属性）与 PP 的分层结构会影响分组与执行顺序，
这条组合路径在 PR 中尚无覆盖测试，建议自行加对齐验证。

**Q9：为什么某些参数 `sharded_state_dict` 里没有？**

`param_shard` 返回 `None` 意味着该参数完全落在别的 rank，本 rank 的 state_dict 里
被有意删掉了（8.3）。这是正确行为，拼装由 flex_checkpoint 完成。

---

## 十二、与 PyTorch FSDP 的对比

- **PyTorch FSDP1** 用 `FlatParameter`：把一个 unit 的参数摊平成单个大参数，
  optimizer 直接看到扁平参数。Paddle 这套实现思路最接近 FSDP1 —— 都是 flatten + 分片 +
  optimizer 更新扁平张量，因此也共享同一类问题（按参数名工作的机制失效、
  param_groups 难支持）。
- **PyTorch FSDP2** 改为 per-parameter 的 `DTensor` 分片（按 dim-0 切），
  保留了参数的语义结构，`state_dict` 与 param_groups 更自然。
  Paddle 的对应物是 `FullyShardAuto`（自动并行分支，基于 placement/DTensor）。
- **本实现的取舍**：牺牲一部分“语义纯净”，换来 tensor fusion 带来的通信与 kernel 效率，
  并通过 `ShardedWeight(is_flattened=True)` 把 checkpoint 的并行度解耦补回来。

一句话：Paddle 手动并行 FSDP ≈ “FSDP1 的性能路径 + FSDP2 的 checkpoint 语义”。

---

## 十三、已知限制与源码索引

### 13.1 限制

- 不支持 optimizer `param_groups`（显式报错）。
- 同一 fused buffer 内不能混 weight decay 行为（显式报错，需 `no_weight_decay` 标注）。
- `fsdp_unit_layers` 默认只认三个内置层名，且未从 `fully_shard()` 透出。
- `offload_policy`（CPU offload）、`shard_placement_fn`、`reshard_after_forward`、
  `ignored_params`、`mesh` 在手动并行分支尚未接线。
- 自定义优化器必须用 `_*_acc*_str` 声明 accumulator，否则 save 会失败（这是有意为之）。
- MP / PP 组合路径缺少覆盖测试。

### 13.2 源码索引（`fully_shard_fusion.py`）

- `BufferState` — `:58`
- `BufferGroup` — `:74`
- `TensorFusionBuffer` — `:91`；`__init__` `:100`；`rebind_main_grad` `:225`；
  `get_padded_size` `:242`；`param_shard` `:254`；`get_tmp_buffer` `:268`；
  `do_all_gather` `:281`；`do_reduce_scatter` `:290`；`accumulate_reduced_grad` `:305`
- `FSDPBufferManager` — `:316`
- `FSDPCommManager` — `:445`；`_release_one_buffer_if_needed` `:477`；
  `_next_buffer_id` `:495`；`all_gather_params` `:529`；`shard_params` `:612`；
  `reset_params_buffer_status` `:636`；`_maybe_zero_grads` `:655`；
  `_ensure_grads_writable` `:663`；`reduce_scatter_grads` `:675`；
  `_reduce_group_grads` `:696`；`_flush_expert_grads_after_unit` `:714`；
  `_wait_for_grad_comm` `:726`；`finish_grads_sync` `:738`
- `FusionBackwardHook` `:754` / `FusionForwardHook` `:776`
- `FullyShardFusion` — `:796`；`_base_sharded_state_dict` `:832`；
  `sharded_state_dict` `:881`；`bind_decay_param_fun` `:947`；
  `owns_optimizer_params` `:980`；`init_optimizer_state` `:989`；
  `optimizer_sharded_state_dict` `:1001`；`comm_sync_and_reset_status` `:1102`；
  `_bind_expert_main_grads` `:1114`；`_register_grad_hook_once` `:1125`；
  `register_tensor_fusion_hooks` `:1152`；`_register_fusion_layer_hooks` `:1224`

### 13.3 和自动并行分支的关系

`fully_shard()`（`fully_shard.py:170`）按 `in_auto_parallel_mode()` 二选一：

- 自动并行 → `paddle.distributed.auto_parallel.fully_shard.FullyShardAuto`（基于 DTensor placement）
- 手动并行 → 本文的 `FullyShardFusion`（基于 hook + fused buffer）

两条路径共用 `_fsdp_context.py` 的全局注册表，因此 `Optimizer._apply_optimize` 只需一处
`get_fsdp_context()` 就能同时适配 —— 这正是 PR #78577 把 context 从 `auto_parallel`
模块中抽出来单独成文件的原因。
