# PyTorch FSDP 通信 Overlap / Prefetch / Buffer 管理实现分析

> 源码基线：`pytorch` HEAD `ed1fcb4b`（2.16.0a0 dev）
> FSDP2（`fully_shard`，主分析对象）：`torch/distributed/fsdp/_fully_shard/`
> - `_fsdp_state.py`：`FSDPState`，模块级 hook、root 逻辑、反向收尾
> - `_fsdp_param_group.py`：`FSDPParamGroup` / `FSDPCommContext`，流、unshard/reshard、预取、RS buffer 上限
> - `_fsdp_collectives.py`：`foreach_all_gather` / `foreach_all_gather_copy_out` / `foreach_reduce`，通信实现类
> - `_fsdp_param.py`：`FSDPParam`，单参数的分片/非分片表示、storage 分配与释放
> - `_fully_shard.py`：`fully_shard()` 与 `FSDPModule` 的用户 API
>
> FSDP1（`FullyShardedDataParallel`，FlatParameter）：`torch/distributed/fsdp/_runtime_utils.py`、`_flat_param.py`、`_limiter_utils.py`、`_exec_order_utils.py`
>
> 下文行号均指上述文件，未特别说明时描述的是 FSDP2。

## 0. 结论速览

| 问题 | FSDP2 答案 |
| --- | --- |
| 通信粒度 | 一次 `fully_shard(module)` 调用 = 一个 `FSDPParamGroup` = 一次 AG + 一次 RS。参数先拼成一块平坦的 staging buffer，再做单次 `all_gather_single` / `reduce_scatter_single`，不用 coalescing manager |
| 流 | 全局共享 4 条 side stream：`all_gather_copy_in_stream`、`all_gather_stream`、`reduce_scatter_stream`（三者优先级为 -1），以及 HSDP 用的 `all_reduce_stream`（默认优先级） |
| 前向预取量 | 默认是**隐式预取 1 层**：下一层的 AG 在它自己的 pre-forward 里发出，靠 CPU 领先 GPU 以及独立的 AG 流来和当前层计算重叠。显式预取用 `set_modules_to_forward_prefetch([...])`，列表有多长就预取多少层 |
| 反向预取量 | 默认**显式预取 1 层**：在当前层的 pre-backward 中，取本轮 `post_forward_order` 里的前一个 group 发起 AG。可以用 `set_modules_to_backward_prefetch([...])` 覆盖 |
| 预取顺序来源 | 前向使用用户显式列表或运行时自然顺序；反向使用**每轮运行时录制**的 post-forward 顺序（带重复次数），每轮反向结束后清空 |
| AG/RS 通信器 | 默认共用同一个 shard PG（同一个 NCCL communicator，线上串行）。`set_separate_reduce_scatter_group()` 可为 RS 单独建一个 PG，让 RS 与 AG 并发 |
| HSDP | RS 走 shard 组，AR 走 replicate 组，AR 在 `all_reduce_stream` 上，和 AG/RS 并发 |
| 内存安全 | **不用 `record_stream`**。跨流使用的张量由 Python 引用保活，并配一个 CUDA event：`AllGatherState` / `ReduceScatterState` / `AllReduceState`。等 event 之后再丢引用，caching allocator 才能回收 |
| AG staging buffer | 前向中，当前层的 AG 输出被保留到**下一层 copy-in 之后**才释放，这样 copy-in(i+1) 和 AG(i) 能重叠。反向中 copy-out 后立即释放 |
| 非分片参数 | 每个参数有自己的 unsharded tensor，用 `storage.resize_(size)` / `resize_(0)` 分配和释放，tensor 对象复用，data_ptr 不固定（除非 `set_keep_unsharded_storage(True)`） |
| RS 输入 buffer | 默认最多 **1** 个 RS 输入在途（`reduce_scatter_max_input_buffers=1`）。下一层 copy-in 之前，计算流要等上一个 RS 完成。可以调大，用显存换 RS 暴露时的计算流停顿 |
| reshard_after_forward | `True`：前向后释放；`False`：保留；`None`：root 为 False、非 root 为 True；`int`：前向后 reshard 到更小的组（如节点内），反向 AG 只在这个小组里做 |
| 迭代边界 | root pre-forward 让 AG 流等当前流（等优化器 step 完成），或者等用户给的 `post_optim_event`。最后一次反向的 final callback 让当前流等所有 post-reduce event |

| 问题 | FSDP1 答案（对照） |
| --- | --- |
| 流 | `_pre_unshard_stream`（混精 cast / H2D）、`_unshard_stream`（AG）、`_post_backward_stream`（RS），HSDP 另有 `_all_reduce_stream` |
| 预取 | `forward_prefetch=False`（默认关），开启后按**首轮**录制的静态前向顺序预取 1 个。`backward_prefetch=BACKWARD_PRE`（默认），按本轮 post-forward 逆序预取 1 个 |
| 限流 | `limit_all_gathers=True`：最多 2 个 AG 在途，超出时 CPU 端 `event.synchronize()` 阻塞 |
| 内存安全 | 大量使用 `record_stream`（unsharded flat param、mp shard、梯度等） |

## 1. FSDP2 的基本组织

### 1.1 四层对象

| 对象 | 粒度 | 作用 |
| --- | --- | --- |
| `FSDPState`（`_fsdp_state.py`） | 每次 `fully_shard()` 调用一个 | 注册 module forward pre/post hook，处理 root 逻辑、显式预取列表、反向 final callback |
| `FSDPParamGroup`（`_fsdp_param_group.py`） | 通常每个 state 一个；`shard_placement_fn` 给参数指定不同 mesh 时（如 MoE 专家参数）每个 mesh 一个 | **通信单位**：一个 group 的参数一起 AG、一起 RS、一起 reshard |
| `FSDPParam`（`_fsdp_param.py`） | 每个参数一个 | 管理 sharded DTensor、unsharded tensor、post-forward shard，以及 `resize_` 分配/释放 |
| `FSDPCommContext`（`_fsdp_param_group.py:78`） | 全模型共享一个（root lazy init 时下发给所有 state） | 持有 4 条流、`all_gather_state`、`reduce_scatter_states`、`post_forward_order`、`_last_post_reduce_events` |

`fully_shard` 要**自底向上**调用：每次调用把 `module.parameters()` 中还没被子模块认领的参数归入新 group（docstring 在 `_fully_shard.py:145-155`）。典型用法是每个 TransformerBlock 调一次，最后对整个模型调一次（root，包住 embedding/head 等剩余参数）。`fully_shard([norm, head])` 可以把多个模块并成一个 group（`_register_group_forward_hooks`）。

### 1.2 分片表示

- 参数以 DTensor 表示，默认 `Shard(0)` 放置。`shard_placement_fn` 可改成 `Shard(i)`，要求整除。
- 分片做了 padding：`padded_sharded_param_size`，每个 rank 的分片大小相同，AG/RS 因此可以用等长的 `*_single` 接口。
- mesh 为 1D 时是 FSDP；为 2D 时是 HSDP，放置为 `(Replicate(), Shard(0))`。
- `MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype, cast_forward_inputs)` 决定 AG 用的 dtype 和 RS 用的 dtype。

### 1.3 TrainingState

`_fsdp_common.py` 中定义了 `IDLE / FORWARD / PRE_BACKWARD / POST_BACKWARD`。group 的 `_training_state` 决定：

- AG 走哪条流（见 §2.2）
- 是否推迟释放 AG 结果（只在 FORWARD）
- 是否 reshard（重计算阶段的 PRE_BACKWARD 下 post-forward 不 reshard）

## 2. 流与同步规则

### 2.1 流的创建

`FSDPCommContext.lazy_init`（`_fsdp_param_group.py:84-122`）：

```python
high_priority = -1
self.all_gather_copy_in_stream = Stream(priority=high_priority)  # copy-in：把分片拷进 AG staging buffer
self.all_gather_stream         = Stream(priority=high_priority)  # 发 AG 通信
self.reduce_scatter_stream     = Stream(priority=high_priority)  # 梯度除法 + RS + 梯度累加
self.all_reduce_stream         = Stream()                        # HSDP 的 AR（默认优先级）
```

源码注释说明了高优先级的用途：避免 copy-in/copy-out 这类小 kernel 被计算 kernel 挤后，进而阻塞计算。这和 NCCL 的高优先级流是两回事。

所有 state 共享一个 `comm_ctx`（root `_lazy_init` → `_init_shared_state`，`_fsdp_state.py:179-234`），所以**整个模型只有这 4 条 side stream**。

### 2.2 哪段工作在哪条流上

| 工作 | 流 | 位置 |
| --- | --- | --- |
| AG copy-in（分片 → staging buffer，`_foreach_copy_`） | `all_gather_copy_in_stream` | `foreach_all_gather`，`_fsdp_collectives.py:362-387` |
| AG 通信（`all_gather_single`） | `all_gather_stream`（先 `wait_stream(copy_in)`） | `:388-396` |
| AG copy-out（staging → 每个参数的 unsharded tensor，`split_with_sizes_copy`） | **当前（计算）流**，先 `wait_event(AG event)` | `foreach_all_gather_copy_out`，`:510-545` |
| 前向/反向计算、reshard（`resize_(0)`） | 当前流 | — |
| RS copy-in（unsharded grad → RS 输入，`chunk_cat_mixed_dtype`） | **当前流** | `foreach_reduce`，`:624-638` |
| 预除、RS 通信、HSDP 累加 | `reduce_scatter_stream`（先 `wait_stream(current)`） | `:639-664` |
| HSDP all-reduce / 用户 all_reduce_hook | `all_reduce_stream` | `:682-719` |
| 后除、dtype cast、写回 `sharded_param.grad`（view）、CPU offload D2H | post-reduce 流（RS 流，或 HSDP 时的 AR 流） | `:722-794` |

`get_all_gather_streams`（`_fsdp_param_group.py:124-134`）决定 AG 用哪两条流：

```python
if not async_op and training_state in (FORWARD, PRE_BACKWARD):
    return self.all_gather_copy_in_stream, self.all_gather_stream
current_stream = self.device_handle.current_stream()
return current_stream, current_stream
```

正常前向/反向走 side stream。用户手动调 `module.unshard(async_op=True)`，或者在 IDLE 状态下 unshard 时，copy-in 和 AG 都在当前流上，重叠只能来自 NCCL 内部流和 `Work.wait()`。

### 2.3 全部同步点

| 同步 | 位置 | 目的 |
| --- | --- | --- |
| `copy_in_stream / ag_stream.wait_event(post_optim_event)`，没有该 event 时改为 `wait_stream(current)` | root `_root_pre_forward`，`_fsdp_state.py:145-177` | 每轮第一次 AG 必须读到优化器 step 之后的分片参数。**每轮只做一次**，后续层的 AG 不再等计算流 |
| `ag_stream.wait_stream(copy_in_stream)` | `foreach_all_gather:388` | AG 在 copy-in 完成后才开始 |
| `current.wait_event(all_gather_event)`，async 时还会 `work.wait()` | `_wait_all_gather:500-507` | copy-out 在 AG 完成后才开始 |
| 前向：保存 `AllGatherState(result, copy_out_event)`；由下一个 group 的 `wait_for_unshard` 调 `release_all_gather_state_for_comm_reuse`，让两条 AG 流 `wait_event(copy_out_event)` 后丢掉引用 | `wait_for_unshard:497-577`，`:146-156` | 推迟释放 staging buffer，让 copy-in(i+1) 与 AG(i) 重叠（§4.3） |
| 非前向：两条 AG 流立即 `wait_event(copy_out_event)` | `wait_for_unshard` | staging 内存回到 copy-in 流的 pool 之前，copy-out 必须读完 |
| AG 流 `wait_event(_reshard_after_forward_event)` | `unshard:450-494` | 使用 post-forward mesh 时，AG 的输入是前向末尾在计算流上 clone 出来的小分片，必须等它写完 |
| `rs_stream.wait_stream(current)` | `foreach_reduce:639` | RS 必须等梯度算完、copy-in 完成 |
| `ar_stream.wait_stream(rs_stream)` | `:686`, `:712` | AR / hook 必须在 RS 之后 |
| RS 输入上限：最老的 `ReduceScatterState` 出队，`allocation_stream.wait_event(event)`；当前流不同时也等 | `post_backward:721-749` | 限制在途 RS 输入 buffer 的个数（§6.3） |
| `current.wait_event(self._post_reduce_event)`，以及 `self._all_reduce_state.event` | `_wait_for_post_backward:1048-1057` | **同一 group** 上一次的 reduce（上一个 micro-batch）完成后，才能再往 `sharded_param.grad` 里累加 |
| `current.wait_event(e)`，`e` 取自 `_last_post_reduce_events` 的每一项 | `_end_backward_iteration:547-566` | 反向结束时一次性等所有 RS/AR 完成。同一条流上后记录的 event 覆盖前面的，所以每条流只需等最后一个 |
| `allocation_stream.wait_event(rs_state.event)`，然后清空 `reduce_scatter_states` | `_end_backward_iteration` | 释放剩余的 RS 输入 buffer |

`_fsdp_state.py:613-628` 有一个 TODO：反向开始时 AG 流没有 `wait_stream(current)`。所以「前向和反向之间改参数」在语义上不受支持（`fully_shard` docstring 中有说明）。

## 3. 通信器（ProcessGroup）管理

### 3.1 各集合通信用的 PG

| 集合通信 | PG | 属性 |
| --- | --- | --- |
| 前向/反向 AG | `post_forward_mesh_info` 的 shard 组（int reshard 时）或 `mesh_info.shard_process_group` | `_all_gather_process_group` |
| 梯度 RS | 默认是 `shard_process_group`；开启 separate 后为专用 PG | `_reduce_scatter_process_group` |
| HSDP AR | `replicate_process_group` | `_all_reduce_process_group` |

### 3.2 AG 与 RS 共用 communicator 的问题与 `set_separate_reduce_scatter_group`

默认 AG 和 RS 虽然在**不同 CUDA 流**上发出，但共用一个 ProcessGroup，也就是同一个 NCCL communicator。communicator 一次只处理一个集合通信，二者在线上**串行**。所以反向中 AG(i-1) 和 RS(i) 谁先发出很重要，这也是默认反向预取要在 RS 之前提前发出 AG 的原因（§5.2）。

`FSDPModule.set_separate_reduce_scatter_group(enable=True)`（`_fully_shard.py:1013-1049`）→ `_set_separate_reduce_scatter_group`（`_fsdp_param_group.py:1250-1287`）：

```python
dist.new_group(ranks, use_local_synchronization=True, group_desc="fsdp_reduce_scatter")
```

- 新组按 shard rank 元组缓存，相同 rank 集合的 mesh 共用一个新 communicator，通常全模型只有 1 个。
- 这是集合操作，所有 rank 必须一致地调用。
- 开启后 RS 和 AG 可以在网络上并发（前提是带宽能撑住）。这和 Megatron 的 `--create-all-gather-group` 思路相同，只是 Megatron 给 AG 单独建组，这里给 RS 单独建组。

### 3.3 可替换的通信实现

`_fsdp_collectives.py:47-236`。每个实现都包含 `allocate()`（分配 staging buffer）和 `__call__()`（发起通信）：

| 实现 | 分配方式 | 用法 |
| --- | --- | --- |
| `DefaultAllGather` / `DefaultReduceScatter` | `torch.empty`（caching allocator） | 默认 |
| `ProcessGroupAllocAllGather/ReduceScatter` | `backend.allocate_tensor()`（如 NCCL 注册内存，可用 SHARP/NVLS 零拷贝） | `set_allocate_memory_from_process_group_for_comm(True)` |
| `SymmMemAllGather/ReduceScatter` | symm_mem 的 MemPool（复用对称 buffer，避免反复 rendezvous），通信前 `symm_mem.rendezvous` | `set_symm_mem_for_comm("NCCL")`。单机可走 Copy-Engine AG（需 `NCCL_CTA_POLICY=2`，即 zero-CTA），多机走 symmetric kernel |
| 用户自定义 | 自定义 | `set_custom_all_gather` / `set_custom_reduce_scatter`（不支持多 param group） |

限制：

- SymmMem RS 只在 `force_sum_reduction_for_comms=True` 时使用，因为 NCCL 2.29.3 的对称 RS 只支持 SUM。
- SymmMem RS 与 `reduce_scatter_max_input_buffers>1` 互斥（`_validate_reduce_scatter_max_input_buffers`，`:1337-1356`）。

## 4. Hook 与前向 all-gather 流水线

### 4.1 Hook 一览

| Hook | 注册方式 | 做什么 |
| --- | --- | --- |
| forward pre-hook | `register_forward_pre_hook(prepend=True, with_kwargs=True)` | `FSDPState._pre_forward`（`_fsdp_state.py:309-351`）：root 逻辑 → 每个 group 的 `pre_forward`（unshard + wait）→ 显式前向预取 |
| forward hook | `register_forward_hook` | `_post_forward`（`:354-370`）：每个 group 的 `post_forward`（reshard + 记录 post_forward_order）→ 在 output 上注册 pre-backward hook；root 还会 flush AG state |
| pre-backward | `output.register_hook(self._pre_backward)`，只注册在 requires_grad 的输出上 | `_pre_backward`（`:407-419`）：注册 root final callback → 每个 group 的 `pre_backward(default_prefetch)` → 显式反向预取 |
| post-backward | `RegisterPostBackwardFunction`（autograd.Function，`_fsdp_param_group.py:1390-1407`），只在首次进入 `pre_forward` 时套在 forward 输入上，其 `backward` 调用 `post_backward()` | reshard + RS/AR |
| final callback | `Variable._execution_engine.queue_callback` | `_root_post_backward_final_callback`（`:422-460`）：补跑没触发的 post_backward，然后 `_end_backward_iteration` |

### 4.2 单个 group 的 unshard：两阶段

**阶段 1：`unshard(async_op)`，发通信**（`_fsdp_param_group.py:450-494`）

1. 如果已有 `_all_gather_result`（已被预取）就直接返回；已经是 unsharded 状态也返回；`unshard_in_backward=False` 且处于 PRE_BACKWARD 时也返回。
2. 两条 AG 流 `wait_event(_reshard_after_forward_event)`（仅 post-forward mesh 时存在）。
3. 调用 `foreach_all_gather(fsdp_params, pg, async_op, *get_all_gather_streams(...), device, comm)`，结果存入 `self._all_gather_result`。

`foreach_all_gather`（`_fsdp_collectives.py:350-404`）：

```python
with stream(copy_in_stream):
    param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
    # bf16/fp32 混精快速路径：一次 torch.empty + _foreach_copy_ 完成 cast
    all_gather_output = comm.allocate((input_numel * world_size,), dtype, device)
    all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(...)
    # input 是 output 中本 rank 那段的 narrow 视图 → in-place AG，无额外输入 buffer
ag_stream.wait_stream(copy_in_stream)
with stream(ag_stream):
    work  = comm(output_tensor=all_gather_output, input_tensor=all_gather_input, group, async_op)
    event = ag_stream.record_event()
```

- 多种 dtype 混合时（如 fp8 extension 加 bf16），整块 buffer 按 `uint8` 处理，按字节切分。
- 混精快速路径（`_get_param_all_gather_inputs:407-453`）：先在 copy-in 流上 `torch.empty` 一块 param_dtype 的平坦临时张量，再把 fp32 分片 `_foreach_copy_` 进去完成 cast，最后拷入 staging buffer。这个临时张量在 copy-in 结束后由 `del` 释放（同流，安全）。

**阶段 2：`wait_for_unshard()`，copy-out**（`:497-577`）

1. 处于 FORWARD 时，先释放**上一个 group** 保留的 `comm_ctx.all_gather_state`。
2. `foreach_all_gather_copy_out`：当前流 `wait_event(AG event)`，然后对每个参数执行 `init_all_gather_outputs`（首次 `torch.empty`）和 `alloc_all_gather_outputs`（`storage.resize_(size)`），再调 `split_with_sizes_copy` 按 rank-major 布局拆回各参数。
3. `init_unsharded_param()`：首次用 `as_strided` 构造 `nn.Parameter`，之后复用。`to_unsharded()` 把 module 上的参数换成 unsharded 版本。
4. 在当前流记录 copy-out event：
   - `not async_op and FORWARD and world_size>1`：`comm_ctx.all_gather_state = AllGatherState(all_gather_result, event)`，推迟释放。
   - 否则：两条 AG 流 `wait_event(event)`。
5. `self._all_gather_result = None`。如果没有被 AllGatherState 保留，staging buffer 的引用就在这里释放。

### 4.3 为什么推迟释放 AG staging buffer

源码 Note（`_fsdp_param_group.py:66-75`）：

> 隐式前向预取希望下一次 copy-in 与当前 AG 重叠，为此使用独立的 copy-in 流。AG 输入是输出的一个视图，所以下一次 copy-in 必须写入**另一块**内存。因此保留当前 AG 输出的引用，由下一个 group 在**它的 copy-in 之后**释放；最后由 root 的 post-forward flush 这个引用。

如果 copy-out 后立刻释放 staging(i)，staging(i+1) 就可能复用同一块内存。那时 copy-in 流必须等 copy-out(i)，而 copy-out(i) 在计算流上排在计算 (i-1) 后面，copy-in(i+1) 就会被拖到计算流上。推迟释放让 copy-in 流始终不依赖计算流。代价是前向同时存在 **2 块** staging buffer（当前层与下一层）。

## 5. 预取：预取多少、何时发出

### 5.1 前向

**隐式预取（默认）：等效于 1 层**

默认情况下没有显式的「预取下一层」调用，AG(i+1) 在 layer i+1 自己的 pre-forward 中发出。它仍然能和 layer i 的计算重叠，原因有三：

1. CPU 发射 kernel 通常领先 GPU 执行。layer i 的计算 kernel 都已入队、GPU 还在执行时，CPU 已经走到 pre_forward(i+1)。
2. AG(i+1) 在独立的 copy-in/AG 流上，与计算流之间**没有依赖**：每轮只有 root 那一次 `wait_stream`。
3. copy-in(i+1) 不必等 copy-out(i)，原因见 §4.3。

只有 copy-out(i+1) 排在计算流上 compute(i) 之后，此时才真正等 AG(i+1)。

```
CPU : pre_fwd(i)[AG i+1 已在途?]  launch compute(i)  post_fwd(i)  pre_fwd(i+1)[copy-in/AG(i+1)] [copy-out(i+1)] launch compute(i+1)
copy-in: ......................................................copy-in(i+1)
AG     : .......................................................AG(i+1)==========
compute: copy-out(i) | compute(i) ======================== | reshard(i) | wait AG | copy-out(i+1) | compute(i+1)
```

这种方式依赖 CPU 领先 GPU。如果 CPU 跑得慢（CPU-bound，小模型/小 batch），AG(i+1) 发得太晚就会暴露。

**显式预取：`set_modules_to_forward_prefetch(modules)`**（`_fully_shard.py:728-746`）

`_pre_forward` 在本 state 所有 group 的 `pre_forward`（unshard + copy-out）完成后，按顺序对每个目标执行 `_prefetch_unshard(target, "forward")`（`_fsdp_state.py:309-351`，`_fsdp_param_group.py:1097-1112`）：

```python
with use_training_state(target, FORWARD):
    target.unshard(target.unshard_async_op)   # 只发 AG，不做 copy-out
```

- 预取量等于列表长度。docstring 说明：只放「下一层」时 overlap 行为和默认相同，只是 CPU 更早发出；**长度 ≥2 才是更激进的预取**，会占用更多 reserved 显存（每多一层就多一块 staging buffer 在途）。
- 被预取的 group 只完成了 AG。它的 copy-out（以及 unsharded 参数的 `resize_` 分配）推迟到它自己的 pre-forward 才执行，所以预取阶段**只占 staging buffer**，不占 unsharded 参数内存。
- 显式前向预取在激活重计算（PRE_BACKWARD）期间不执行。

### 5.2 反向

**为什么反向需要显式预取**：layer i-1 的 pre-backward hook 要等 autograd 走到它输出的梯度时才触发。在 CPU 发射顺序上，post_backward(i) 的 RS(i) 已经先发出。AG 和 RS 默认共用一个 communicator，AG(i-1) 会**排在 RS(i) 后面**，计算流也就跟着空等。所以反向必须在 layer i 开始计算之前就把 AG(i-1) 发出去。

**默认反向预取：1 个 group**

`FSDPState._pre_backward`（`_fsdp_state.py:407-419`）：

```python
default_prefetch = len(self._states_to_backward_prefetch) == 0
for g in self._fsdp_param_groups:
    g.pre_backward(default_prefetch)        # unshard(若已预取则 no-op) → wait_for_unshard → _backward_prefetch()
if not default_prefetch:  # 显式
    for target_state in self._states_to_backward_prefetch:
        for target_group in reversed(target_state._fsdp_param_groups):
            FSDPParamGroup._prefetch_unshard(target_group, "backward")
```

`_backward_prefetch`（`_fsdp_param_group.py:1059-1094`）：

```python
curr_index = self._post_forward_indices.pop()       # 本 group 在 post_forward_order 中最近一次出现的位置
if self._num_param_groups > 1:
    if self._param_group_index != 1: return          # 多 group 的 state 只由一个 group 负责预取
    # 沿 post_forward_order 往回找，把「上一个 module」的所有 group 都预取
elif curr_index > 0:
    target = self.comm_ctx.post_forward_order[curr_index - 1]
    self._prefetch_unshard(target, "backward")
```

- `post_forward_order` 由 `post_forward → _record_post_forward` 在**每轮前向运行时**追加。同一模块被调用多次时会出现多次，`_post_forward_indices` 是一个栈，支持共享/复用模块。反向结束时在 `_end_backward_iteration` 中清空。所以顺序是**每轮动态录制**的，不像 FSDP1 的前向那样是静态的，也不像 Megatron 那样依赖 bucket id。
- 预取时机在 `wait_for_unshard`（copy-out）之后、本层反向计算发射之前，因此 CPU 上 AG(i-1) 先于 RS(i) 发出，二者在 communicator 上的顺序就是 AG 在前。
- 反向的 `wait_for_unshard` 不推迟 staging 的释放：copy-out 后两条 AG 流立刻 `wait_event`，staging 随即归还。

**显式反向预取：`set_modules_to_backward_prefetch(modules)`**（`_fully_shard.py:748-766`）会覆盖默认行为。列表长度 ≥2 时可以预取更多层（同样以 staging 显存为代价）。

### 5.3 其他影响预取的开关

| 开关 | 作用 |
| --- | --- |
| `unshard_async_op`（group 字段，默认 False） | 预取时 `target.unshard(async_op=True)`：AG 走当前流 + NCCL async Work，等待推迟到 `_wait_all_gather` 的 `work.wait()` |
| `FSDPModule.unshard(async_op=True)` | 用户手动提前 unshard，返回 `UnshardHandle`；FSDP 会在 pre-forward 中自动等待（`_fully_shard.py:396-424`） |
| `set_unshard_in_backward(False)` | 反向完全不 AG（如 embedding 这类反向不需要权重的层），pre_backward 中的 unshard 直接返回 |
| `reshard_after_forward=False` / root | 前向后不释放，反向 unshard 是 no-op，也不会触发 AG |
| 激活重计算 | `_pre_forward` 在 PRE_BACKWARD 下只做「需要时 unshard + wait」，不做显式预取；`_post_forward` 直接返回，不 reshard |

## 6. 反向梯度规约流水线

### 6.1 `post_backward` 主流程（`_fsdp_param_group.py:688-881`）

1. `reduce_grads=False`（`set_requires_gradient_sync(False)`，即梯度累积不通信）：标记 pending；按 `reshard_after_backward` 决定是否 reshard，然后返回。unsharded 梯度留在 unsharded 参数上继续累加（dtype 为 `reduce_dtype` 或 `grad_dtype`）。
2. `_take_unsharded_grads_to_reduce()`：取出 unsharded 梯度。如果 `reshard_after_backward`，执行 `reshard()`，unsharded 参数随即 `resize_(0)`。
3. **RS 输入上限**（`:721-749`，见 §6.3）。
4. `_wait_for_post_backward()`：当前流等**本 group 上一次** reduce 的 `_post_reduce_event` / AR event。
5. HSDP 且 `all_reduce_grads=False` 时，在 RS 流上执行 `_prepare_partial_reduce_output`。
6. `foreach_reduce(...)`，返回 `(rs_input, rs_event, post_reduce_stream, post_reduce_event, ar_input, ar_event, partial_reduce_output)`。
7. `comm_ctx._last_post_reduce_events[post_reduce_stream] = post_reduce_event`；`reduce_scatter_states.append(ReduceScatterState(rs_input, rs_event, current_stream, self))`；保存 `AllReduceState(ar_input, ar_event)`。

### 6.2 `foreach_reduce`（`_fsdp_collectives.py:570-807`）

```python
pre, post, rs_op, ar_op = _get_gradient_divide_factors(...)
copy_in = prepare_reduce_scatter_inputs(unsharded_grads, shard_dims, world_size)  # Shard(i>0) 先重排
rs_input = comm.allocate((rs_out_numel * world_size,), reduce_dtype)    # 当前流上分配
copy_in(rs_input)                     # chunk_cat_mixed_dtype：拼接 + pad + cast，当前流
unsharded_grads.clear()               # 同流，立即释放 unsharded 梯度
rs_stream.wait_stream(current)
with stream(rs_stream):
    rs_output = comm.allocate((rs_out_numel,), reduce_dtype)  # RS 流上分配
    _div_if_needed(rs_input, pre)
    comm(rs_output, rs_input, group=rs_pg, op=rs_op)
    rs_event = rs_stream.record_event()
    if HSDP:
        if not all_reduce_grads: partial += rs_output; return ...   # 只累加，不 AR
        all_reduce_stream.wait_stream(rs_stream)
        with stream(all_reduce_stream):
            dist.all_reduce(rs_output, group=replicate_pg, op=ar_op)
            ar_input, ar_event = rs_output, ar_stream.record_event()
if all_reduce_hook: 在 all_reduce_stream（或用户指定的 hook 流）上调用 hook(rs_output)
with stream(post_reduce_stream):
    _div_if_needed(rs_output, post)
    sharded_grads = _cast_and_view_sharded_grads(rs_output, ...)  # 按 dtype 分段 cast，其余为 as_strided 视图
    for p, g in zip(...): accumulate 到 p.sharded_param.grad（或直接赋值为 DTensor），可选 D2H offload
    post_reduce_event = post_reduce_stream.record_event()
```

几个设计要点：

- **RS copy-in 在计算流上，不在 RS 流上**。这样 unsharded 梯度可以在 copy-in 后立刻同流释放，不需要 `record_stream`。代价是 copy-in kernel 占用计算流时间，量级为一次 memcpy。
- **RS 输出就是 sharded grad 的存储**：sharded grad 是 rs_output 的视图。这块内存在 RS 流上分配，却在默认流（优化器）上使用。源码注释说明不需要额外同步，因为 sharded 参数的 `.grad` 一直持有引用，内存不会被别的 RS 复用。混合 dtype 时，若需要 cast 的部分比不 cast 的部分大，会把不 cast 的部分 clone 出来，避免整块 rs_output 一直被视图保活（`:836-848`）。
- **HSDP AR 的保活**：`AllReduceState` 持有 rs_output，直到下一次 `_wait_for_post_backward` 或反向结束。否则 cast 之后旧 buffer 会回到 caching allocator，下一层的 RS 可能在本层 AR 还在跑时复用同一块物理内存，造成跨层梯度别名（PR #140044 / #180900）。
- **除法策略**（`_get_gradient_divide_factors:910-967`）：
  - fp32/bf16 默认直接用 NCCL `ReduceOp.AVG`，没有额外的除法 kernel；world_size==1 时用 SUM。
  - 自定义 `gradient_divide_factor` 时：factor 等于 RS 组大小则用 AVG，否则用 `PREMUL_SUM(1/factor)`。
  - fp16 或 `force_sum_reduction_for_comms` 时：SUM，配合 pre/post 两次除法。fp16 的 pre_factor 约为 √N，取 2 的幂，用来防止上溢/下溢。
  - MTIA 强制使用 SUM。

### 6.3 RS 输入 buffer 上限（cap-K）

`comm_ctx.reduce_scatter_states` 是**全局共享**的列表，上限 K = `reduce_scatter_max_input_buffers`（默认 1；`_init_shared_state` 取所有 group 中的最大值）。每个 state 的最后一个 param group 在 append 之前执行：

```python
while len(states) >= K:
    oldest = states.pop(0)
    oldest.allocation_stream.wait_event(oldest.event)   # 分配流（计算流）等该 RS 完成
    if current != oldest.allocation_stream: current.wait_event(oldest.event)
    del oldest                                            # 丢掉引用，allocator 可复用
```

- K=1（默认）：layer i 的 RS 输入在 layer i-1 做 post_backward 时回收。计算流必须先等 RS(i) 完成，才能执行 copy-in(i-1)。**RS 比 layer i-1 的反向计算慢时，计算流会停顿。**
- K=2..：可以把下一块 copy-in 写进新的 buffer，消除停顿，代价是多 K-1 块 RS 输入的峰值显存。docstring 建议 K ≥ RS 流水线深度。这里没有额外的流，也没有 `record_stream`。
- 这个机制只在 RS 暴露时有收益。

## 7. Buffer 与显存管理汇总

### 7.1 Buffer 清单

| Buffer | 分配位置/流 | 释放时机 | 保活机制 |
| --- | --- | --- | --- |
| sharded 参数（DTensor 本地分片，含 padding） | 初始化 | 常驻 | — |
| 混精 cast 临时张量（`flat_foreach_copy_input`） | copy-in 流，`torch.empty` | copy-in 后 `del`（同流） | 无需 |
| AG staging（平坦 `all_gather_output`，in-place AG） | copy-in 流，`comm.allocate` | 前向：下一个 group 的 `wait_for_unshard` 中释放；root post-forward 时 flush。反向：copy-out 后立即释放 | `AllGatherState` + copy-out event；释放前两条 AG 流先 `wait_event` |
| unsharded 参数（每个参数一个 `all_gather_outputs[i]`） | 首次 `torch.empty`；之后在 copy-out 时于当前流 `storage.resize_(size)` | reshard 时 `storage.resize_(0)`（当前流） | 单流使用，无需。tensor 对象跨迭代复用，autograd 保存的仍是同一个对象 |
| post-forward 小分片（int reshard） | 当前流，从 AG 输出 `narrow(...).clone()` | 下一次 `to_unsharded()` 时置 None（copy-out 之后） | `_reshard_after_forward_event` 让 AG 流等 clone 完成 |
| unsharded 梯度（autograd 产生） | 计算流 | RS copy-in 后 `unsharded_grads.clear()` | 同流 |
| RS 输入（平坦，reduce_dtype） | 当前流，`comm.allocate` | 超过 cap-K 被挤出时，或反向结束时 | `ReduceScatterState` + rs_event |
| RS 输出 = sharded grad 存储 | RS 流 | 用户清梯度（`zero_grad(set_to_none)`） | `sharded_param.grad` 持有引用 |
| HSDP AR buffer / hook 输入 | 即 RS 输出 | 下一次 `_wait_for_post_backward` 或反向结束 | `AllReduceState` |
| HSDP partial reduce（不 AR 的 micro-batch） | RS 流 | AR 后 | `_partial_reduce_output` |
| CPU offload 梯度 | pinned CPU，D2H `non_blocking` | — | `grad_offload_event`，优化器之前在 CPU 上等待 |

### 7.2 不用 `record_stream` 的原因与做法

`record_stream` 会让 caching allocator 推迟复用，直到被记录流上「将来某个点」完成，回收时机不确定，容易导致显存碎片和峰值偏高（FSDP1 的经验教训）。FSDP2 改用以下方式：

1. 每块跨流内存都在**生产者流**上分配（staging 在 copy-in 流，RS 输出在 RS 流，RS 输入在计算流）。
2. 用 NamedTuple 保存 Python 引用和完成 event。
3. 先让**分配流**（必要时还有当前流）`wait_event`，再丢引用。caching allocator 的语义是「在分配流上释放的块，只给该流后续的分配复用」，而该流已经等过 event，所以复用是安全的。

`record_grad_output_stream`（`_fsdp_collectives.py:322-347`）默认是 no-op，只给「free 是设备侧操作」的后端预留扩展点。

### 7.3 `resize_` 管理 unsharded storage

```python
def alloc_storage(t): t.untyped_storage().resize_(t.numel() * t.itemsize)   # _fsdp_param.py:1657
def free_storage(t):  t.untyped_storage().resize_(0)                         # :1663
```

- unsharded `nn.Parameter` 对象只构造一次（`init_unsharded_param:915-971`）。每次 unshard 只重新给 storage 分配内存，所以 autograd 保存的张量和 module 上的引用都保持有效。
- `set_keep_unsharded_storage(True)`（`_fully_shard.py:373-394`）：reshard 时不释放 storage，data_ptr 不变（CUDA Graph 等场景需要），代价是常驻 unsharded 显存。
- 用了 `fsdp_pre_all_gather/fsdp_post_all_gather` extension（如 fp8 all-gather）时，内部张量 `_unsharded_inner_tensors` 也按同样方式 resize。如果 extension 不需要保留 AG 输出，会在 post-all-gather 后立刻释放（`_release_all_gather_outputs_if_needed`）。

### 7.4 显存峰值估算（默认配置，L 为单层 unsharded 参数大小）

- **前向**：unsharded 参数(i) 约 L，加上 staging(i) 约 L（保留到 i+1 copy-in 之后），加上 staging(i+1) 约 L 在途。layer i 的参数在 post_forward(i) 中先于 copy-out(i+1) 释放，所以 unsharded 参数同一时刻约为 1 层。显式预取 k 层时 staging 变为约 (k+1)·L。
- **反向**：unsharded 参数(i) L、unsharded 梯度(i)（按 reduce_dtype）、staging(i-1) L（预取在途）、RS 输入 K 块、累计的 sharded 梯度。
- 被保留和在途的 buffer 分属不同流的 pool，reserved 显存一般高于上述 allocated 估算。

## 8. `reshard_after_forward`、root 与 post-forward mesh

### 8.1 取值语义（`_fsdp_init.py:152-195`，`_fully_shard.py:262-285`）

内部统一转成 `post_forward_mesh_info`：

| 用户传入 | `post_forward_mesh_info` | 行为 |
| --- | --- | --- |
| `None`（默认，`auto_reshard_after_forward`） | 先按 True 处理；`_lazy_init` 中 root 的被置 None（`_fsdp_state.py:222-226`） | root 前向后不释放，非 root 释放 |
| `True` / `int == shard_mesh_size` | `mesh_info` 本身 | 前向后 reshard 回完整分片，反向重新在整个 shard 组 AG |
| `False` / `int == 1`（会告警） | `None` | 前向后保留 unsharded 参数，反向 unshard 是 no-op（ZeRO-2 语义） |
| `int = d`，1<d<shard_size，且整除 | 新建 `HSDPMeshInfo(mesh.view(-1, d))`，`shard_mesh_dim=1` | 前向后只保留 1/d 的分片；反向 AG 只在 d 个 rank 的小组内做（例如节点内 8 卡） |

- `_reshard_after_forward = post_forward_mesh_info is not None`；`_use_post_forward_mesh = reshard 且 post_forward_mesh_info != mesh_info`（`_fsdp_param_group.py:1207-1216`）。
- `set_reshard_after_forward(bool)` 运行时修改（只接受 bool），同时把 `_auto_reshard_after_forward` 置 False。因此对 root 调用 `set_reshard_after_forward(True)` 可以强制 root 前向后也释放（`_fully_shard.py:665-700`）。
- int 形式不支持 SPMD mesh（`dp_mesh_dims`）和 DTensor 参数（`to_sharded_post_forward` 直接抛 `NotImplementedError`）。

**root 不 reshard 的原因**：root 的 post_forward 之后紧接着就是 loss 计算和反向，反向第一个 pre_backward 就是 root 自己，释放后马上又要 AG 回来，纯属浪费。

### 8.2 post-forward mesh 的实现

`reshard()`（`_fsdp_param_group.py:580-590`）在 FORWARD 状态下：

```python
if not self._reshard_after_forward: return
if self._use_post_forward_mesh:
    self._to_sharded_post_forward()             # 每个 FSDPParam.to_sharded_post_forward()
    self._reshard_after_forward_event = Event(); event.record()   # 当前流
    return
self._to_sharded()
```

`FSDPParam.to_sharded_post_forward`（`_fsdp_param.py:1046-1085`）：

```python
shard_world_size = post_forward_mesh_info.shard_mesh_size
sharded_numel = all_gather_outputs[0].numel() // shard_world_size
self._sharded_post_forward_param_data = all_gather_outputs[0].narrow(
    0, sharded_numel * shard_rank, sharded_numel).clone()      # clone 后才能释放 AG 输出
self._sharded_post_forward_param = nn.Parameter(to_sharded_post_forward_dtensor(...))
self._setattr_on_modules(...)
if not self.keep_unsharded_storage: self.free_unsharded_param()
self.sharded_state = SHARDED_POST_FORWARD
```

- clone 在**当前（计算）流**上执行，结果却要被 AG 流读取。所以记一个 `_reshard_after_forward_event`，下一次 `unshard()` 开头让两条 AG 流 `wait_event` 它（`_fsdp_param_group.py:460-466`）。这是 FSDP2 中唯一一处「AG 流等计算流」的逐层同步。
- 反向 AG 的 PG 由 `_all_gather_process_group` 决定：`is_sharded_post_forward` 时使用 post-forward mesh 的 shard 组（`:1222-1234`）。AG 输入是 1/d 的小分片，输出仍是完整参数。
- 只支持单个 `all_gather_outputs`（没有 fsdp_pre_all_gather extension 多输出的情况）。
- 显存：常驻 sharded 参数 + 1/d 的 post-forward 分片。用小组内（如 NVLink）AG 代替跨节点 AG，用显存换反向 AG 带宽。

## 9. 激活重计算（AC）

FSDP2 通过 `TrainingState.PRE_BACKWARD` 识别重计算：

- `_pre_forward`（`_fsdp_state.py:319-330`）：状态为 PRE_BACKWARD 时，对还没 unshard 的 group 执行 `unshard() + wait_for_unshard()`（嵌套 FSDP、多次前向的情况下，参数可能已被前一个 post_backward reshard），然后**直接返回**：不做 root 逻辑、不做显式前向预取、不注册 post_backward hook。
- `_post_forward`（`:358-359`）：PRE_BACKWARD 下只做 output dtype cast，**不 reshard**，也不追加 `post_forward_order`。参数保留到本层 post_backward 时才 reshard。
- `FSDPParamGroup.post_forward` 中用 `is_bw()` 判断。`AC(fully_shard(model))` 这种 AC 包在外层的写法下，重计算时也不会 reshard，也不会改动 post_forward_order（`_fsdp_param_group.py:657-661`）。
- `pre_forward` 中 `if torch.is_grad_enabled() and not is_bw()` 才设置 unsharded grad dtype，重计算沿用原前向的选择（`:641-644`）。

所以 AC 不会带来额外的 AG：重计算用的是 pre_backward 已经 unshard 好的参数。反向预取照常进行。

## 10. 迭代边界与梯度累积

### 10.1 前向开头：等优化器（`_root_pre_forward`，`_fsdp_state.py:145-177`）

每轮只在 root 第一次 pre-forward 时执行一次（`iter_forward_root` 去重）：

```python
if (event := state_ctx.post_optim_event) is not None:
    all_gather_copy_in_stream.wait_event(event); all_gather_stream.wait_event(event)
    state_ctx.post_optim_event = None
else:
    all_gather_copy_in_stream.wait_stream(current_stream)
    all_gather_stream.wait_stream(current_stream)
```

- 默认等整个当前流（包括优化器 step 之后用户在当前流上排的所有工作）。
- `set_post_optim_event(event)`（`_fully_shard.py:837-854`）：用户在优化器 step 后记一个 event，AG 流只等这个 event。这样 step 之后、下一次前向之前的其他计算（例如 logging、EMA）就能和第一层 AG 重叠。
- **反向开头没有对应的同步**（`_fsdp_state.py:621-625` 的 TODO）：如果在前向和反向之间修改 sharded 参数，AG 流不会等这个修改。源码注释说，想支持的话要在 `_register_root_post_backward_final_callback` 中加 `wait_stream`，代价是第一层反向 AG 无法和前向尾部重叠。

### 10.2 反向结束：final callback（`_fsdp_state.py:422-460`）

由第一个 `_pre_backward` 通过 `Variable._execution_engine.queue_callback` 注册，每次 backward 一次：

1. 清 `iter_forward_root`、各 state 的 `_modules_to_run_forward`。
2. 按逆序遍历所有 group：反向中没跑过 post_backward 的（输入不需要梯度、或 group 没被用到），补一次 `post_backward()`。手动 finalization 且 `reduce_grads` 时只标记 `_post_backward_pending`。
3. `is_last_backward and not manual` 时调用 `wait_for_gradient_reduction()` → `_end_backward_iteration`（`:547-566`）：
   - 当前流 `wait_event` 所有 `_last_post_reduce_events`（每个 post-reduce 流各 1 个，即 RS 流和 HSDP 的 AR 流）；
   - 每个 group 执行 `finalize_backward()`（`_fsdp_param_group.py:883-895`）：清 `_post_reduce_event` / `AllReduceState`；CPU offload 时 `grad_offload_event.synchronize()`；有误预取（AG 发了但没人等）时 `_wait_all_gather` 并丢弃；
   - `release_all_gather_state_on_current_stream()`；
   - 清空 `post_forward_order`；
   - 对所有残留的 `ReduceScatterState` 让其分配流 `wait_event`，然后清空。
4. 否则（非最后一次 backward）：只调用 `release_all_gather_state_on_current_stream()`，避免梯度累积期间一直保留 AG buffer。

final callback 只让**当前流**等待，不阻塞 CPU（offload 除外）。优化器 step 排在当前流上，自然在 RS/AR 和梯度累加之后执行。

### 10.3 梯度累积相关 API

| API | 效果 |
| --- | --- |
| `set_requires_gradient_sync(False)` | `reduce_grads = all_reduce_grads = False`（`_fully_shard.py:612-647`）。post_backward 不做 RS，unsharded 梯度累加在 unsharded 参数上（`reduce_dtype` 或 `grad_dtype`）。等价于 FSDP1 的 `no_sync`。如果同时 `reshard_after_backward=True`，unsharded 参数的 storage 会释放，但梯度张量仍然持有，**显存占用 = 完整 unsharded 梯度** |
| `set_requires_all_reduce(False)` | 仅 HSDP：每个 micro-batch 做 RS，不做 AR，在 RS 流上把结果累加进 `_partial_reduce_output`；最后一个 micro-batch 再 AR。省掉跨节点的 AR 次数 |
| `set_reshard_after_backward(False)` | 反向后不释放 unsharded 参数，下一个 micro-batch 的前向不再需要 AG（显存换通信）。必须在最后一个 micro-batch 之前改回来 |
| `set_is_last_backward(False)` | 非最后一次 backward 不等 RS，也不清 `post_forward_order` 和 RS 状态 |
| `set_manual_backward_finalization(True)` + `finalize_backward(async_op=...)` | 由用户决定何时收尾（`_fully_shard.py:470-560`，`_fsdp_state.py:462-545`）：先对 pending group 补 `post_backward()`，或者对 `reshard_after_backward` 的 group 做 reshard，再等待。`async_op=True` 返回 `GradientReductionHandle`，等待推迟到 `handle.wait()`。期间不能开始下一次前向（`_pre_forward` 会检查 `gradient_reduction_pending`）。不支持 HSDP partial reduce；HSDP 下必须开启 all-reduce |
| `reset_iter_state()` | 前向/反向抛异常后在 root 上调用（`_fully_shard.py:427-458`，`_fsdp_state.py:630+`）：等所有在途 AG/RS/AR event，reshard 所有 group，清空迭代状态，丢弃本轮梯度 |

## 11. 一个 step 的完整时间线（默认配置）

模型：root（embedding + head）+ L0..L2 各自 `fully_shard`，非 HSDP，K=1，无显式预取。CI = copy-in，CO = copy-out，C = 计算。

**前向**

```
             root pre_fwd          L0 pre_fwd            L1 pre_fwd            L2 pre_fwd
CPU      : [wait_stream][CI/AG r] [CO r][C r..] [CI/AG 0][CO 0][C 0][rs 0] [CI/AG 1][CO 1]...
copy-in  :  CI(r)                    CI(0)                  CI(1)                  CI(2)
AG       :     AG(r)=====               AG(0)=====             AG(1)=====             AG(2)====
compute  :  (opt step) | wait | CO(r) C(r_embed) | wait AG0 | CO(0) C(0) reshard(0) | CO(1) C(1) ...
staging  :            [r --------------- 释放于 L0 CO 时] [0 ------- 释放于 L1 CO 时] ...
```

- 每层的 AG 在上一层计算 kernel 还在 GPU 上执行时就已发出（CPU 领先），计算流只在 CO 前 `wait_event(ag_event)`。
- `reshard(i)` 在 post_forward(i) 中执行，即 `resize_(0)` 排在计算流上 C(i) 之后。
- root post_forward 时 flush 最后一个 `AllGatherState`。root 不 reshard。

**反向**（autograd 逆序；Lk 的 pre_backward 在其输出梯度就绪时触发）

```
CPU     : root pre_bwd: CO? no-op, prefetch AG(2) | C_bwd(head) ... | L2 pre_bwd: wait AG2, CO(2), prefetch AG(1) | C_bwd(2) | L2 post_bwd: reshard(2), RS(2)
          | L1 pre_bwd: wait AG1, CO(1), prefetch AG(0) | C_bwd(1) | L1 post_bwd: cap-K 等 RS(2), RS(1) | ... | final callback
AG/RS PG: AG(2) | AG(1) | RS(2) | AG(0) | RS(1) | RS(0) | RS(r)        <- 同一 communicator，按 CPU 发射顺序串行
compute : C_bwd(head) | CO(2) C_bwd(2) RScopyin(2) | CO(1) C_bwd(1) [wait RS(2)] RScopyin(1) | ... | wait post_reduce events | opt step
RS 流   :                               RS(2)+div+accum         RS(1)+...                 RS(0) ...
```

要点：

- AG(i-1) 在 C_bwd(i) **之前**发出，所以在 communicator 上排在 RS(i) 前面，反向 AG 一般不暴露。
- RS(i) 和 C_bwd(i-1) 重叠。K=1 时，copy-in(i-1) 前计算流要等 RS(i) 完成。RS 比一层反向计算慢时，这里会有停顿，可以用 K≥2 或 `set_separate_reduce_scatter_group` 缓解。
- 最后一层（root）的 RS 暴露在反向末尾。final callback 只让计算流等待，优化器 step 排在后面。
- 下一轮 root pre_forward 让 AG 流等计算流（优化器 step 完成），第一层 AG 暴露。可以用 `post_optim_event` 部分缓解。

## 12. FSDP1（`FullyShardedDataParallel`）实现对照

FSDP1 把每个 wrap 单元内的所有参数拼成一个 `FlatParameter`，由 `FlatParamHandle` 管理。源码在 `_runtime_utils.py`、`_flat_param.py`、`_exec_order_utils.py`、`_limiter_utils.py`。

### 12.1 流（`_runtime_utils.py:237-273` `_init_streams`）

```python
high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
state._default_stream       = current_stream()
state._unshard_stream       = Stream(priority=high_priority)   # AG
state._post_backward_stream = Stream(priority=high_priority)   # RS
state._pre_unshard_stream   = Stream(priority=high_priority)   # 混精 cast / CPU offload H2D
state._all_reduce_stream    = Stream() if HSDP else _default_stream
```

- 只在 root 上创建，`_share_state_and_init_handle_attrs` 把这些流、`_exec_order_data`、`_free_event_queue` 分享给所有子 state（`:217-223`）。
- 与 FSDP2 的区别：混精 cast 在独立的 pre_unshard 流上；AG 输出直接写进 `_full_param_padded`，没有单独的 copy-in/copy-out；优先级只在 `limit_all_gathers + HSDP` 时为 -1。

| 流 | 工作 |
| --- | --- |
| pre_unshard | `pre_unshard()`：cast 到 `_mp_shard`；offload 时 H2D（`_flat_param.py:1307-1357`） |
| unshard | `_full_param_padded` resize 分配 + `all_gather_single` + 释放 `_mp_shard`（`_runtime_utils.py:304-306`） |
| default | 计算、reshard（free）、限流 event 记录 |
| post_backward | cast 到 reduce_dtype、pad、预除、`reduce_scatter_single`、后除、累加到 `_saved_grad_shard`、offload |
| all_reduce（HSDP） | 跨节点 `all_reduce` + 后除 + 累加 |

同步点：

- `unshard_stream.wait_stream(pre_unshard_stream)`（`:296`）
- pre-forward / pre-backward 中计算流 `wait_stream(unshard_stream)`（`:425-430`、`:685`）。等待的是整条 unshard 流，但下一个 handle 的预取 AG 在这次 wait **之后**才发出（`:430→:434`、`:685→:693`），因此计算流不会提前等到预取的 AG
- `post_backward_stream.wait_stream(current)`（`:751-753`）；HSDP 下 `all_reduce_stream.wait_stream(post_backward_stream)`（`:866`）
- 每轮 root pre-forward：`_wait_for_computation_stream`，unshard 和 pre_unshard 流等计算流（`:1556-1560`，`:589-593`）
- final callback：计算流等 post_backward 流（以及 HSDP 的 AR 流）；offload 时 `current_stream().synchronize()`（`:1104-1111`）

### 12.2 前向预取：`forward_prefetch`（默认 False）

- 顺序：`_ExecOrderData.record_pre_forward`（`_exec_order_utils.py:150-170`）只在**第一轮**（`_iter == 0`）、每个 handle 首次使用时追加到 `handles_pre_forward_order`，之后不再更新，是**静态顺序**。
- 目标：`get_handle_to_forward_prefetch`（`:109-128`）返回 `handles_pre_forward_order[cur + 1]`。第一轮前向时下一个还没录制，所以不预取。
- 数量：**1**。`fully_sharded_data_parallel.py:476-477` 把 `forward_prefetch_limit` 和 `backward_prefetch_limit` 都写死为 1。
- 时机：`_pre_forward_unshard`（`_runtime_utils.py:411-434`）：unshard 当前 handle（若未预取）→ 计算流 wait → `_prefetch_handle(next, FORWARD)`，发出下一个 handle 的 pre_unshard + 限流 + AG。这些都在本模块前向计算发射之前。
- docstring 定位：适合 CPU-bound、静态图的场景。默认不开时，前向完全依赖 CPU 领先，和 FSDP2 默认的隐式预取类似。

### 12.3 反向预取：`backward_prefetch`（默认 `BACKWARD_PRE`）

`api.py:73-109` 定义了两种模式。注释（`:97-101`）明确说明：同一个 NCCL PG 上的集合通信争用同一条 NCCL 流，所以**发出顺序很重要**。

| 模式 | 触发点 | CPU 发出顺序 | 峰值显存 |
| --- | --- | --- | --- |
| `BACKWARD_PRE`（默认） | `_pre_backward_hook`（`:693`）：当前 handle unshard + wait 之后，梯度计算发射之前 | AG(i-1) → C_bwd(i) → RS(i) | 当前参数 + 下一层参数 + 当前梯度，最高 |
| `BACKWARD_POST` | `_post_backward_reshard`（`:793-807`）：释放当前参数之后、发 RS 之前 | C_bwd(i) → free(i) → AG(i-1) → RS(i) | 较低；但 AG(i-1) 要等 C_bwd(i) 结束才发，C_bwd(i-1) 一开始就可能等待 |
| `None` | 不预取 | AG(i-1) 在 i-1 的 pre_backward 才发，排在 RS(i) 之后 | 最低，反向 AG 暴露 |

- 目标：`get_handle_to_backward_prefetch`（`:88-107`）返回 `handles_post_forward_order[cur._post_forward_index - 1]`。`record_post_forward` **每轮都录制**，`next_iter` 时清空，是本轮运行时顺序的逆序。
- 数量：**1**（写死）。
- 防重复：pre-backward unshard 后把 `_needs_pre_backward_unshard` 置 False；`_prefetched` 标志防止重复预取。SHARD_GRAD_OP 下前向没有释放，`needs_unshard()` 为 False，只切换视图，不发 AG。
- 冻结参数也会触发：`_post_backward_reshard_only_hook` 通过 `register_multi_grad_hook` 挂在输入上（`:777-790`）。

### 12.4 限流器：`limit_all_gathers`（默认 True）

FSDP1 的 AG 输出（`_full_param_padded`）在 **unshard 流**上分配，在计算流上释放（带 `record_stream`）。CPU 跑得太快时，unshard 流会不停分配新块，而旧块因为 `record_stream` 迟迟不能回收，显存暴涨。限流器就是为了解决这个问题（`_limiter_utils.py:6-32`）：

```python
class _FreeEventQueue:
    _max_num_inflight_all_gathers = 2   # empirically chosen
    def dequeue_if_needed(self):
        if len(self._queue) >= 2: return self._queue.popleft()
```

- **入队**：`_reshard`（`_runtime_utils.py:320-326`）在真正释放 unsharded 参数时，在计算流上 `Event().record()` 并入队。没有释放（root post-forward、SHARD_GRAD_OP 前向后、NO_SHARD）时不入队。
- **出队**：`_unshard`（`:297-303`）在 pre_unshard 之后、发 AG 之前，如果有 ≥2 个 pending event，就弹出最老的一个并调用 **`event.synchronize()`，阻塞 CPU**，直到计算流执行完那次 free。profiler 中显示为 `FullyShardedDataParallel.rate_limiter`。
- 效果：同一时刻最多约 2 个 handle 的 unsharded 参数存活（当前 + 预取）。代价是 CPU 被阻塞，失去 run-ahead，可能拖慢 kernel 发射。
- FSDP2 不需要它：staging 由 `AllGatherState` 精确控制，unsharded 参数在计算流上分配和释放，内存复用是确定的。

### 12.5 FlatParameter 内存

| 字段 | 内容 | 生命周期 |
| --- | --- | --- |
| `_local_shard` | 本 rank 的 padded 分片（offload 时为 pinned CPU） | 常驻 |
| `_full_param_padded` | `numel * world_size` 的 unsharded 参数（param_dtype） | unshard 时 `_typed_storage()._resize_(n)`（在 unshard 流上）；reshard 时 `record_stream(current)` 后 `_resize_(0)` |
| `_full_prec_full_param_padded` | 强制全精度时（summon_full_params / eval）的 AG 目标 | 同上 |
| `_mp_shard` | 低精度分片 | pre_unshard 中分配；AG 后在 `post_unshard` 中 `record_stream` + 释放 |
| `_cpu_grad` | pinned CPU 梯度 | 常驻 |
| `_saved_grad_shard` | sharded 梯度累加器 | 每步创建、删除 |

- `_alloc_storage` / `_free_storage` 在 `torch/distributed/utils.py:180-215`。
- 和 FSDP2 一样，FlatParameter 对象复用，只 resize storage。但**分配流是 unshard 流、释放流是计算流**，所以必须 `record_stream`，这是碎片和限流器问题的根源。
- `record_stream` 调用点：`_flat_param.py:1486/1543/1829/1858`，`_runtime_utils.py:772/871/991/1054`（offload H2D 副本、`_mp_shard`、unsharded flat param、autograd 产生的 unsharded 梯度、HSDP sharded 梯度、offload 梯度、低精度梯度）。

### 12.6 梯度 RS 流程

`_post_backward_hook`（`_runtime_utils.py:699-774`）挂在 FlatParameter 的 `AccumulateGrad` 节点上，首轮前向时注册一次：

1. 状态设为 BACKWARD_POST（重入式 AC 下允许重复触发）。
2. `_post_backward_reshard`：释放 unsharded 参数；BACKWARD_POST 模式下在这里预取。
3. `no_sync`（`_sync_gradients=False`）时直接返回，unsharded 梯度留在 `.grad`。
4. `post_backward_stream.wait_stream(current)`，在 post_backward 流上执行：
   - cast 到 `reduce_dtype`；
   - `_reduce_grad`（`:831-886`）：取出 `flat_param.grad` 并置 None → `chunk` 成 world_size 份，需要 padding 时用 `F.pad` **新分配**一块 → 预除 → `reduce_scatter_single(new_sharded_grad, padded, pg)` → HSDP 时 AR 流等 RS 流，在 `_inter_node_pg` 上 `all_reduce` → 后除 → 累加到 `_saved_grad_shard`；
   - 最后对原 autograd 梯度 `record_stream(post_backward_stream)`。
5. 除法因子（`_init_utils.py:128-138`，`default_hooks.py:38-42`）：pre = 不超过 √N 的最大 2 的幂，post = N / pre，**无论 dtype 都分两次除**（FSDP2 在 fp32/bf16 下改用 `ReduceOp.AVG`）。
6. final callback `_post_backward_final_callback`（`:1082-1126`）：计算流等 post_backward 流（以及 HSDP 的 AR 流）→ `next_iter()` → `_catch_all_reshard`（补 reshard 没触发 post_backward 的 handle）→ `_finalize_params`（把 `_saved_grad_shard` 挂到 `.grad`）→ 重置各 handle 标志。

与 FSDP2 的差异：

- FSDP1 的 RS **输入就是 autograd 产生的 unsharded 梯度**（不需要 padding 时没有拷贝），靠 `record_stream` 保活。FSDP2 在计算流上 copy-in 到一块新 buffer，用 cap-K 控制在途数量。
- FSDP1 没有在途 RS 数量上限。RS 输入的释放时机完全由 `record_stream` 和 allocator 决定。
- 每个 FlatParameter 是一块连续内存，RS 天然是单次调用；FSDP2 要为每次 RS 拼一次。

### 12.7 进程组、策略与 AC

- **PG**：AG（`handle.process_group`）和 RS（`state.process_group`）是**同一个 PG 对象**（`_init_utils.py:661`，`_flat_param.py:570`），只是在不同的 CUDA 流上发出；没有拆分 RS 组的选项。HSDP 的 AR 使用独立的 `_inter_node_pg`。`FSDP_USE_FAKE_REDUCE=1` / `FSDP_USE_FAKE_ALL_GATHER=1` 可以换成 `FakeProcessGroup` 用于 profiling。
- **策略**：

| ShardingStrategy | 前向后释放（非 root） | 反向后释放 | 通信 |
| --- | --- | --- | --- |
| FULL_SHARD | 是 | 是（`no_sync` 下也释放） | 前向 AG + 反向 AG + RS |
| SHARD_GRAD_OP | 否 | 只在同步梯度时 | 前向 AG + RS（反向 unshard 是 no-op） |
| HYBRID_SHARD | 是 | 是 | 节点内 AG/RS + 节点间 AR |
| _HYBRID_SHARD_ZERO2 | 否 | 只在同步梯度时 | 同上，反向不 AG |
| NO_SHARD | — | — | 梯度 `all_reduce` |

  root 永远不在前向后释放（`_runtime_utils.py:496-499`）；world_size=1 时强制 NO_SHARD。
- **AC**：handle 处于 BACKWARD_PRE 时，`_pre_forward` / `_post_forward` 直接返回（`:375-380`、`:470-471`），重计算不 unshard、不 reshard、不录制顺序，用的是 pre_backward 已 unshard 的参数。
- `torch/distributed/fsdp/experimental/__init__.py` 只服务 FSDP2，提供 `all_gather_output_fn_with_native_copy` 和 `reduce_scatter_input_fn_with_native_copy`（用 `torch.ops.fsdp` 原生拷贝 kernel 替换 copy-out / RS copy-in）。

## 13. FSDP2 / FSDP1 / Megatron-FSDP 对比

| 维度 | PyTorch FSDP2 | PyTorch FSDP1 | Megatron-FSDP（v1，见 `megatron_fsdp_overlap_prefetch.md`） |
| --- | --- | --- | --- |
| 通信单元 | `fully_shard` 调用 → FSDPParamGroup，每次通信时拼成平坦 buffer | FlatParameter（常驻连续） | bucket；同 unit 的 bucket 用 `_coalescing_manager` 合并 |
| 参数表示 | 每参数 DTensor `Shard(0)` | FlatParameter + 视图 | 每 dtype 一块大 buffer + 视图 |
| 前向预取 | 默认隐式 1 层（靠 CPU 领先）；显式列表任意长 | 默认关；开启后按首轮静态顺序预取 1 个 | 静态 bucket 顺序 `bucket_id+1`，按字节量 ≈ 1 个 unit |
| 反向预取 | 默认 1 个，按本轮运行时 post-forward 逆序；显式列表可覆盖 | 默认 BACKWARD_PRE，1 个，按本轮 post-forward 逆序 | 静态 `bucket_id-1`，按字节量 |
| 流 | copy-in / AG / RS（-1 优先级）+ AR | pre_unshard / unshard / post_backward + AR | AG side stream + RS/拷贝 side stream（HFSDP 再加 2 条） |
| AG/RS 通信器 | 默认同一个 shard PG；`set_separate_reduce_scatter_group` 可拆分 | 同一个 PG，不可拆 | 可用 `--create-all-gather-group` 给 AG 单独建 PG |
| 跨流内存安全 | 无 `record_stream`：Python 引用 + event | 大量 `record_stream` | 无 `record_stream`：`wait_stream` + `event.wait()` 后再 free |
| AG 在途上限 | 由预取列表长度决定；staging 前向保留到下一层 copy-in | `limit_all_gathers`：2 个，CPU 阻塞 | `fsdp_double_buffer` 时最多 2 个 unit |
| RS 在途上限 | cap-K（默认 1 个 RS 输入），计算流等待 | 无显式上限 | `suggested_RS_queue_capacity` ≈ 2 个 unit |
| 非分片参数存储 | 每参数 `storage.resize_` | FlatParameter `storage._resize_` | 临时 bucket，4 种 allocator（含双 buffer / 固定池） |
| 梯度除法 | fp32/bf16 用 AVG / PREMUL_SUM；fp16 用 pre/post | 统一 pre/post 两次除 | 在 RS 流上预缩放：`average_in_collective` 用 AVG，`gradient_reduce_div_fusion`（非 bf16）用 premul_sum，否则 `mul_` + SUM |
| 梯度累加 | sharded grad 是 RS 输出的视图，跨 micro-batch 用 `+=` | `_saved_grad_shard +=` | RS 输出写临时 shard，再 `+=` 到持久 main_grad buffer |
| ZeRO-2 / 节点内 reshard | `reshard_after_forward=False` / `int` | SHARD_GRAD_OP / _HYBRID_SHARD_ZERO2 | `optim_grads` 策略 |

主要设计取向：

- **预取顺序**：FSDP2 / FSDP1 的反向按运行时录制的顺序，能适应动态控制流和模块复用；Megatron 用静态 bucket id，要求执行顺序与注册顺序一致。
- **预取量控制**：FSDP2 按「模块个数」，Megatron 按「字节量」（`suggested_communication_unit_size`），FSDP1 写死 1 个。
- **显存安全**：FSDP2 和 Megatron 都放弃了 `record_stream`，改为显式 event 加引用；FSDP1 因为 `record_stream` 需要用 CPU 阻塞的限流器兜底。
- **通信器争用**：三者默认都让 AG 和 RS 走同一个 DP 通信器。FSDP2 和 Megatron 都提供了拆分选项（FSDP2 拆 RS，Megatron 拆 AG），FSDP1 只能靠调整发出顺序（BACKWARD_PRE）来缓解。

## 14. 调优开关与注意事项

### 14.1 调优开关一览（FSDP2）

| 开关 | 默认 | 作用 / 代价 |
| --- | --- | --- |
| `reshard_after_forward` | None（root False，其余 True） | False：省反向 AG，多占一层 unsharded 参数；int：反向 AG 限于节点内 |
| `set_modules_to_forward_prefetch` | 无（隐式 1 层） | CPU-bound 时提前发 AG；长度 ≥2 时每多一层多一块 staging |
| `set_modules_to_backward_prefetch` | 无（默认 1 层） | 覆盖默认的反向预取目标，可预取多层 |
| `set_separate_reduce_scatter_group` | 关 | RS 用独立 communicator，与 AG 并发；多一个 NCCL comm 的资源 |
| `set_reduce_scatter_max_input_buffers(K)` | 1 | RS 暴露时消除计算流停顿；多 K-1 块 RS 输入显存 |
| `set_post_optim_event` | 无 | 第一层 AG 只等优化器，不等整个当前流 |
| `unshard_async_op` / `unshard(async_op=True)` | False | AG 改走当前流 + async work；手动控制 unshard 时机 |
| `set_unshard_in_backward(False)` | True | 反向不需要权重的模块（如 embedding）跳过反向 AG |
| `set_requires_gradient_sync` / `set_requires_all_reduce` / `set_reshard_after_backward` / `set_is_last_backward` | True | 梯度累积：在通信量与显存之间权衡 |
| `set_keep_unsharded_storage(True)` | False | data_ptr 固定（CUDA Graph 等），常驻 unsharded 显存 |
| 自定义 AG/RS 实现（ProcessGroupAlloc / SymmMem / 用户 Comm） | Default | 注册 buffer / 对称内存，减少 NCCL 拷贝；Copy-Engine AG 需要 `NCCL_CTA_POLICY=2` |
| `MixedPrecisionPolicy(param_dtype, reduce_dtype)` | 无 | 降低 AG/RS 字节量；reduce_dtype 也决定累积梯度的 dtype |

### 14.2 局限与注意事项

1. **隐式前向预取依赖 CPU 领先**。CPU-bound（小模型、小 batch、Python 开销大）时 AG 会暴露，需要显式预取或 `torch.compile`。
2. **AG 与 RS 默认共用 communicator**，线上串行。反向时 RS(i) 会延后 AG(i-2) 等后续通信；RS 很长时，可以拆分 RS 组或调大 K。
3. **cap-K=1 的计算流停顿**：RS 比单层反向计算慢时，copy-in 前计算流要等上一个 RS 完成。
4. **前向 staging 双 buffer**：为了让 copy-in(i+1) 和 AG(i) 重叠，前向会多保留一块 staging（约 1 层 unsharded 参数大小）。
5. **反向开头 AG 流不等计算流**（`_fsdp_state.py:621-625` TODO）：在前向与反向之间原地修改 sharded 参数是不安全的。
6. **int `reshard_after_forward`** 不支持 DTensor 参数和 SPMD mesh，也不支持多输出的 all-gather extension。
7. 一个 state 有多个 param group 时（例如 `shard_placement_fn` 把一个 block 拆成 dense + moe 两组），默认反向预取只由 `_param_group_index == 1` 的 group 负责，它会沿 `post_forward_order` 往回找，预取上一个模块的全部 group（`_fsdp_param_group.py:1065-1091`）。
8. **手动 finalization** 不支持 HSDP partial reduce；`finalize_backward(async_op=True)` 之后、`handle.wait()` 之前不能开始下一次前向。
9. 若 `fully_shard` 输出视图张量后被原地修改，pre-backward hook 会丢失，反向不会 AG（`_register_pre_backward_hook` 会给出告警）。

对 FSDP1：`limit_all_gathers` 的 `event.synchronize()` 会阻塞 CPU；`forward_prefetch` 使用首轮静态顺序，动态图下可能预取错误的 handle（错误预取会被 `_prefetched` / `_needs_*_unshard` 标志兜住，但会浪费一次 AG 和显存）；预取数量固定为 1。
