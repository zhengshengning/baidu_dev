# Paddle FSDP 通信流（side stream）改造小白教程

> 背景：`Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py` 的流改造把训练吞吐从
> 14000 tokens/s/card 提升到 18087（只改流），配合 `prefetch_units=2` 再提升到 21087（+17%）。
> 本文从零讲清"流（stream）"是什么、旧实现为什么慢、新实现改了什么、为什么快。

---

## 目录

1. [预备知识：GPU 是怎么干活的](#一预备知识gpu-是怎么干活的)
2. [FSDP 里要通信什么](#二fsdp-里要通信什么)
3. [旧代码：为什么慢](#三旧代码为什么慢)
4. [新代码：改成了什么](#四新代码改成了什么)
5. [新旧时间线对比（一图看懂）](#五新旧时间线对比一图看懂)
6. [常见疑问](#六常见疑问)
7. [一句话总结](#七一句话总结)

---

## 一、预备知识：GPU 是怎么干活的？

### 1.1 GPU 是"排队干活"的

CPU（主机）和 GPU（设备）分工明确：

- CPU 决定"接下来干什么"，GPU 负责"把活干完"。
- CPU 干活的方式是**把任务（kernel）提交给 GPU**，提交完 CPU 可以继续干别的，GPU 在后台按顺序执行。
- 就像你把衣服放进洗衣机（提交任务），然后你去做饭（CPU 干别的），洗衣机洗完了（GPU 干完了）你再去晾（用结果）。

**一个流（stream）就是一条"任务队列"**。默认情况下只有一条队列（默认流/计算流），所有任务都在里面排队，一个接一个执行。

```
默认流:  [算A] → [算B] → [算C] → [算D]
                     ↑ GPU 按顺序一个接一个执行
```

### 1.2 多流的威力

GPU 里其实有**好几套硬件**可以同时干活：

| 硬件 | 干什么 |
|---|---|
| SM 计算核心 | 算矩阵乘法、注意力等 |
| 拷贝引擎（DMA） | 内存搬运 |
| 通信引擎（NIC/IB） | 网络收发（all_gather / reduce_scatter 就是它干的） |

如果所有任务都挤在一条流里，那"通信"和"计算"就**不可能同时进行**——通信时 SM 闲着，计算时网卡闲着。

**多流 = 多条队列 = 可以同时执行**：

```
流A(计算):  [算A]      [算B]      [算C]
流B(通信):       [通信1]      [通信2]
              ↑ 通信1 和 算A 同时进行！
```

FSDP 优化的大方向就是：**让通信这条流和计算这条流并行起来，互相不拖后腿**。

### 1.3 跨流同步：事件（Event）

两条流并行没问题，但**有依赖关系**时必须同步。比如"算A"的结果要给"通信1"用，那通信1 必须等算A 干完。

同步的手段是**事件（event）**——一面"旗子"：

```
1. 在流A 的末尾插旗子：   event.record(流A)       # 旗子会在流A 执行到此刻时"立起来"
2. 让流B 等旗子：         stream.wait_event(event) # 流B 执行到这就停，等旗子立起来再继续
```

旗子立起来 = "流A 已经干到的位置"，流B 等到了旗子 = "流A 的东西我可以用啦"。

---

## 二、FSDP 里要通信什么？

FSDP（Fully Sharded Data Parallel）把模型参数切分到多张卡上，因此：

| 操作 | 干什么 | 走谁 |
|---|---|---|
| **all_gather**（前向） | 把几张卡上的参数分片**拼成完整参数**，才能做计算 | 通信引擎 |
| **reduce_scatter**（反向） | 把各卡的梯度**求和 + 切分**，只留自己那份 | 通信引擎 |

一个训练 step 里，模型有几十层，每层都要一次 AG + 一次 RS，所以**通信是极度频繁的**。通信一条、计算一条，两条流如果能完全并行，步时间就能大幅缩短。

---

## 三、旧代码：为什么慢？

这是修改前的版本（`git HEAD` 的 `do_all_gather`）：

```python
def do_all_gather(self):
    return paddle.distributed.all_gather(
        self.get_tmp_buffer(),      # 输出：完整参数（通信结果写到这里）
        self.data_buffer,           # 输入：本卡的参数分片
        group=self.fsdp_group,
        sync_op=False,              # 异步：提交完立即返回，不等通信完成
    )
```

`sync_op=False` 只是"主机不等待"，但通信任务被发到哪条流上呢？——发到 **ProcessGroup 自带的 comm 流**（一条专门的通信流）。而 C++ 底层（`process_group_flagcx.cc`）每次调用还要做一套"双流握手"：

```
发射时：
  SyncCalcStream()  → 1. 在"计算流"当前末尾插旗子
                     2. 让 comm 流等这面旗子   ← comm 流必须等计算流"已入队的活全干完"才能开始通信
  （把通信任务发到 comm 流上执行）
完成时：
  UpdateWaitChain() → 3. 在 comm 流末尾插旗子
消费时：
  task.wait()       → 4. 计算流等 comm 流的旗子（等通信真正完成）
```

慢在哪里？

### 问题 1：所有通信挤在一条 comm 流上，互相排队

同一张卡的 **AG 和 RS 全部共用一个 comm 流**。前一个 unit 的 RS 还没发完，后面 unit 的 AG 就得等着——通信引擎一直是"串行处理"，利用率低。

### 问题 2：发射时要等"计算流已入队的全部活"

`SyncCalcStream` 的语义是"comm 流等我提交那一刻，计算流上**已经排进去的所有任务**"。而 backward 阶段触发 RS 时，计算流上已经排满了大量 backward 计算——于是**通信总是被之前的计算拖着，无法提前开始**。

### 问题 3：同步点太多，开销大

每次通信要"插 2 次旗子 + 等 2 次旗子 + 一堆簿记（任务表查找/清理）"。几十层 × 每步 = 几十次通信，这些开销全部落在关键路径上。

**一句话：旧实现"通信和计算"确实分了两条流，但同步毛刺多、通信内部还互相排队，GPU 的通信引擎和计算引擎没有真正充分并行起来。**

---

## 四、新代码：改成了什么？

现在看修改后的核心（`fully_shard_fusion.py` L89–109）：

```python
_comm_streams = {}  # group id -> 每 group 自己的一条 side 流

class _StreamTask:
    """一次异步通信的"凭证"：拿着它可以等通信完成"""
    def __init__(self, stream):
        self._event = stream.record_event()   # 在 side 流末尾插旗子

    def wait(self):
        # 让"当前流"（主计算流）等这面旗子
        paddle.device.current_stream().wait_event(self._event)

def _async_collective(group, tensors, collective):
    stream = _comm_streams.get(group.id)      # 每个 group 只建一次
    if stream is None:
        stream = _comm_streams[group.id] = paddle.device.Stream()  # ① 独立 side 流

    stream.wait_stream(paddle.device.current_stream())  # ② side 流等主流当前进度

    with paddle.device.stream_guard(stream):            # ③ 把"当前流"切到 side 流！
        # ④ 内存安全：告诉分配器"这块内存还在 side 流上用，先别复用"
        for t in tensors:
            t._record_stream()
        collective()      # ⑤ 通信任务发到 side 流上执行
    return _StreamTask(stream)   # ⑥ 返回旗子，谁用结果谁等
```

逐行拆解：

### ① 每 group 一条独立流水线

```python
stream = _comm_streams[group.id] = paddle.device.Stream()
```

这是**全新的、独立的一条流**，不是 PG 自带的 comm 流。用 `group.id` 做 key，**每个 FSDP group 一条流**——不同 group 的 AG/RS 互不干扰，可以真正并行。

### ② 数据就绪：side 流等主流

```python
stream.wait_stream(paddle.device.current_stream())
```

通信要读的参数分片，是主流（计算流）上算出来的，所以 **side 流要先等主流"目前排到的位置"**——保证主流把分片写完了，side 流才开始通信。

注意：**这是单向依赖**——只让 side 等主流，**没有**反过来让主流等 side。主流可以继续往前跑，这就是并行的来源。

### ③ 切换"当前流"——本改动的核心魔法

```python
with paddle.device.stream_guard(stream):
```

Paddle 的 `stream_guard` 会把"当前流"（`paddle.device.current_stream()` 返回的那个）**切成我们这条 side 流**。而 Paddle 的通信接口有一个参数：

```python
paddle.distributed.stream.all_gather(..., sync_op=True, use_calc_stream=True)
```

`use_calc_stream=True` 的含义是：**"通信任务用当前流来执行"**（API 文档原话：*whether the communication is done on calculation stream*）。

所以组合起来的效果是：

```
stream_guard 切了当前流 → use_calc_stream 读取"当前流" → 通信任务落在 side 流上
```

也就是说：**通信任务真的被发到了我们这条独立的 side 流上**，而不是旧代码的 comm 流。

> 底层原理：`paddle.device.set_current_stream()` 会直接改写设备上下文池（`DeviceContextPool`）里当前设备上下文的流（见 `paddle/fluid/pybind/cuda_streams_py.cc` 的 `set_current_stream`），而 `use_calc_stream=True` 的集体通信正是从该上下文取流执行的，所以 guard 内发起的通信就落到了 side 流上。

### ④ `_record_stream()`：保护内存不被"抢走"

```python
for t in tensors:
    t._record_stream()
```

这是本改动**必不可少**的一步，讲一下为什么：

- GPU 显存是**池化复用**的：一个 tensor 用完了，它的显存会归还给池子，之后新建的 tensor 可能**拿到同一块地址**。
- 通信是异步的：side 流还在读写 `tmp_buffer` / `data_buffer`，但主流这边可能已经觉得"这块 buffer 的活我交出去了"，转而把这块内存分配给了别的变量——**数据就被覆盖了**，训练数值就错了。
- `t._record_stream()` 的作用就是登记："**这块内存现在被 side 流用着，还没完**。" 显存分配器看到这个登记，就不会把它复用于其他分配。
- 什么时候能真正复用？等 `_StreamTask.wait()` 执行完——主流已经等到了通信完成的旗子，side 流不再碰这块内存了。

> 旧代码为什么不显式写这个？因为旧代码走的是 comm 流 + 事件完整握手，C++ 内部有一套 holder 机制（`allocation_stream_pairs_` / `EraseTensorHolders`）在维护这个关系。新代码把通信挪到了 **Python 可见的 side 流**上，所以要在 Python 侧显式登记，语义一样。

### ⑤ 异步发射，主机不阻塞

```python
collective()   # 就是 all_gather / reduce_scatter 的实际调用
```

`sync_op=True` 但配合 `use_calc_stream=True` 时：通信已经排在 side 流上了，顺序天然正确，C++ 的 `wait()` 变成一个空操作——**主机立刻返回**，CPU 继续干别的。

### ⑥ 返回"旗子"，消费时等待

```python
return _StreamTask(stream)
```

之后在 `all_gather_params` 的消费点（`fully_shard_fusion.py` L747 附近）：

```python
params_buffer.comm_task.wait()   # 参数要开始算了 → 主流等 side 流通信完成
```

主计算流现在才真正等通信——**只等一次**。

---

## 五、新旧时间线对比（一图看懂）

### 旧代码

```
主计算流:  [算Unit1] ────────────→ [算Unit2] ────────────→ [算Unit3]
comm流:   [等Unit1计算完][AG1][RS0] [等Unit2计算完][AG2][RS1]
              ↑                  ↑                ↑
           AG1要等Unit1算完    AG2要等Unit2算完    AG/RS挤一条流，互相排队
           （保守同步，通信被计算拖着）
```

### 新代码

```
主计算流:  [算Unit1] ──────────────────────→ [算Unit2] ────────────────→ [算Unit3]
side流:      [AG1]          [AG2]  [RS1]     [AG3]  [RS2]
              ↑ 与算Unit1 同 ↑ 与算Unit2 同    ↑ 与算Unit3 同时进行
              时进行(真正并行) 时进行            （每 group 一条流，互不排队）
```

### 区别总结

| | 旧 | 新 |
|---|---|---|
| 通信在哪条流 | PG 自带 comm 流（AG/RS 共用） | 每 group 独立 side 流 |
| 同步点 | 发射端 2 次 + 消费端 1 次（4 个事件操作） | 发射端 1 次 + 消费端 1 次 |
| 通信能否并行 | 只能与"之后才入队"的计算并行，且 AG/RS 互相排队 | 与主计算流完全并行，多 group 通信也互相并行 |
| 内存保护 | C++ holder 机制隐式维护 | Python 侧 `_record_stream()` 显式登记 |

---

## 六、常见疑问

**Q1：为什么不直接用 `sync_op=False` 老写法，改一个参数就行？**

`sync_op=False` 只是"主机不等"，但通信还是发在 comm 流上，还是要走 `SyncCalcStream` + `UpdateWaitChain` 那套双流握手。本改动是把通信**整体搬到了一条新的、更轻量的流上**，同步方式整个换掉了——所以是"只改流就能 +29%"。

**Q2：为什么用一个全局字典 `_comm_streams`，而不是每次新建？**

流是有创建成本的。而且同一个 group 的通信需要**同一条流**才知道先后顺序（如果每次新流，两次 AG 之间没有依赖，反而可能乱序）。所以按 `group.id` 缓存，一个 group 固定一条流。

**Q3：`_record_stream` 什么时候能省略？**

不能省略。只要通信是异步的（主机不等通信完成就继续跑），这块内存就可能被提前复用。省略了就会出现"偶发数值错误/训练崩"，是隐蔽 bug。

**Q4：`prefetch_units=2` 和这个流改动是一回事吗？**

不是同一层。流改动解决的是"**通信和计算怎么并行**"；`prefetch_units` 解决的是"**提前多少步发起通信**"（把下一个/下两个 unit 的参数镜像提前发起 all_gather，让通信的"尾巴"被计算藏住）。两者叠加：流的改动让通信真正并行（+29%），加深预取让并行窗口更大（再 +17%）。

**Q5：为什么 `sync_op=True` 反而"异步"？**

旧代码 `sync_op=False` 是"主机不等待"的异步；新代码 `sync_op=True` + `use_calc_stream=True` 时，C++ 侧完成事件从未被记录（wait-chain 被跳过），`task->Wait()` 变成 no-op——通信已经排在 side 流上，主机不用等它，等效于异步发射。真正负责"等"的，是 Python 侧 `_StreamTask.wait()` 那面旗子。

---

## 七、一句话总结

> **旧实现**：通信任务挤在 PG 共用的一条 comm 流上，每次发射要和计算流做两次事件握手（保守同步），AG 和 RS 还互相排队——通信引擎没有被充分用起来。
>
> **新实现**：每个 FSDP group 建一条独立 side 流，通过 `stream_guard` + `use_calc_stream=True` 把通信任务直接发到这条流上，用 `_record_stream` 保护显存、用 `_StreamTask` 的旗子做唯一一次消费端同步——**通信与计算真正并行、通信之间也互不排队**，所以只改流就拿到了 +29% 的吞吐提升；再叠加 `prefetch_units=2` 加深流水，进一步拿到 +17%。

---

*相关文件：`Paddle/python/paddle/distributed/fsdp/fully_shard_fusion.py`（改动），
`Paddle/paddle/fluid/distributed/collective/process_group_flagcx.cc`、`Paddle/paddle/fluid/pybind/cuda_streams_py.cc`（底层机制）。*