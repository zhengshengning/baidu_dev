# inbuild H5：按样本跨源采样 + 混行装箱（mix_sample_packing）

## 1. 目标与口径

现状（行模式）：`_plan_rows` 把每个 H5 内**连续**样本顺序贪心装成行，sampler 的配比、shuffle、
`data_status` 都以「行」为单位；一行只来自一个文件、一个源、一种模态。

目标（混行模式，开关 `mix_sample_packing=1`）：

- sampler 以**样本**为单位按权重从各 part 抽（和 Megatron energon 同口径：权重乘样本数，不再≈乘 token）；
- 抽出的样本在**主进程 sampler 里**确定性装箱成行，一行可混不同源、不同模态（文本 / 图 / 音频 / 视频）；
- loss、seg_lens、文档 mask、MTP mask 的计算口径不变；
- 按源 loss（`record_data_source_loss`）与按模态 loss（`log_modality_loss`）改为**按段**归属；
- 开关关闭时行为与现在逐位一致（行模式代码路径不动）。

续训（含扩缩卡）要求不重训、不漏训任何样本，见 §4.5。只需支持混行自己的新格式：
从行模式 ckpt 切过来时按 `saved_data_status` 复原已训样本并跳过，见 §4.7。

## 2. 组网侧已支持混行（不用改）

- `src/utils/hyperbody_batch.py:_run_modalities` 优先用逐段 `context_modality`（0 image / 1 audio），
  图像块与 mel 段各自按出现顺序消费；行级 `task_id` 只在缺 `context_modality` 时兜底。
- `src/utils/data_utils.py:273` collator 已把 `image_resolutions / seg_lens / audio_lengths / context_modality`
  沿段轴拼、`images / audios` 沿像素 / 帧轴拼。
- 因此只要 reader 产出的一行里这几样**按样本顺序拼接、彼此顺序一致**，组网就能吃。
  `_check_segments` 与模态校验在不一致时会报错，不会静默训错。

## 3. 开关与约束

`PreTrainingArguments`（`src/trainers/pretraining_trainer.py`，挨着 `deterministic_blend` 加）：

```python
mix_sample_packing: bool = field(default=False, metadata={"help": "inbuild: 按样本跨源采样并在 sampler 里混行装箱"})
mix_pack_open_rows: int = field(default=1, metadata={"help": "混行装箱同时开着的行数；1 = 顺序贪心，>1 = 窗口内 first-fit"})
mix_chunk_samples: int = field(default=100_000, metadata={"help": "混行 sampler 每次生成的样本占位数（流式，不物化整 epoch）"})
mix_reset_sample_pos: bool = field(default=False, metadata={"help": "混行续训时丢弃 ckpt 里的 per-part 样本位点，从 0 开始（§4.5）"})
```

开关只影响 **train sampler**；reader 根据 index 的形状分流（见 §5），不需要 reader 侧开关，eval 仍走行模式。

`PreTrainingArguments._validate_mix_sample_packing` 与 sampler `_init_mix` 里各断言一遍
（另外要求 `use_inbuild_data`，且不支持 `pp_need_data_degree` / `pp_shuffle_dataset_degree`：消费记账要求每个 dp 组只有一个读数卡）：

| 条件 | 原因 |
|---|---|
| 数据集是 `PretrainInbuildTask` 的 `ExampleSet` | 只有它有逐样本读接口 |
| `combine_batch == 1` | collator 的 combine 分支断言一组只有一个 `src_id`（data_utils.py:242） |
| `shuffle_consecutive == False` | 二次 shuffle 打乱的是样本占位，不是行 |
| `dp_invariant_num_partitions == 0` | inbuild 本来就不支持 |
| `online_generate_data == False` | 该分支往 index 末尾追加 global_step，与行 index 冲突 |
| `num_consecutive == 1` | 行模式下它是「同 part 连续取几行」（按块打乱、换 AFS 读连续性），`sample_sampler` 不实现它；不断言的话 >1 会被静默忽略。混行要换 IO 连续性走 §10(b) 的 `mix_sample_block` |
| `use_train_part_sharding=1` | 各 rank 只装自己 part 的样本，`roundup_and_shard` 直接跳过；§4.5 的位点要求每个 part 同一时刻只属一个 dp rank |

## 4. Sampler（`WeightedDistributedSampler`，pretraining_trainer.py:2454）

### 4.1 part 侧：逐样本序列（`ExampleSetSingleDataSource`，pretrain_inbuild_task.py:529）

新增两个方法，行模式的 `__len__ / sampler / example_id` 不动：

```python
def sample_lens(self, fp_idx):
    """该文件逐样本长度（int32），按 ds16_offset 差分；> seqlen 的样本记为 -1（丢弃）。首次用到时算，缓存。
    与 _row_start 一样直接 h5py 读 offsets，不依赖 fps 是否 IPC。"""

def sample_sampler(self, start=0, skip=()):
    """无穷生成 (epoch, sample_idx, length, k)。每个 epoch: fp_idx = epoch % len(self.path)，
    取 sample_lens(fp_idx) >= 0 的下标，shuffle 时用 default_rng(SeedSequence([seed, part, epoch])).permutation
    打乱（不用 seed + epoch + part：part0/epoch1 与 part1/epoch0 会撞种子）。全被丢弃时同 _plan_rows raise。
    k = 该 part 跨 epoch 的样本序号（续训位点，§4.5）；从序号 start 开始出，序号在 skip 里的跳过。"""
```

- 第 k 个样本只由 `(seed, part, k)` 决定，与 dp 数、part 归哪个 rank 无关——这是扩缩卡续训不重复的前提。

- 保留 `epoch % len(self.fps)` 选文件的语义，和行模式一致。
- 长度缓存只在主进程（sampler 所在进程）里建；worker 不需要。内存 ≈ 4 B × 本 rank 样本数
  （文本 500B 约 785M 样本 / dp16 ≈ 200 MB/rank），可接受；若吃紧改成懒加载 + LRU 只留当前 epoch 文件。

**没有 ds16_offset 的纯文本 H5（pretrain content 这类连续 token 流）**：`sample_lens` 按行模式同口径切滑窗
（窗口 `seqlen`、步长 `stride = max_seq_length`，相邻窗口重叠 1 个 token），一个窗口当一个「样本」，
`_read_mix_row` 用 `window_span` 取 `[i·stride, i·stride+seqlen)`。

- 满长窗口恰好占满一行，不和别的样本拼；只有文件末尾的短窗口会被拼进别的行。
  单个满窗口行的 input_ids / labels / mask 与行模式那一行逐位相同（单测覆盖）；
  差别只有 MTP lookahead：混行补 pad 不计损，行模式读下一窗口的开头。
- 配比按样本数计，一个窗口约 seqlen 个 token，而带 offset 的文本平均约 600 token。
  这类源的 data_weights 要按窗口长度折算（大致除以 `seqlen / 其它文本平均样本长度`），
  否则它的 token 占比会被放大十几倍。每个 part 首次读到时打一条 warning 提醒。
- 非文本（图 / 音频 / 视频）H5 没有 offset 直接 raise：媒体按样本区间取，没有样本边界就对不上。

### 4.2 样本占位：按 chunk 流式生成

`__iter__` 里 `mix_sample_packing` 分支，不走 `gen_data_seq_weighted`、不 pickle data_seq：

占位流按 chunk 切开、每块独立生成，任意位置都能直接定位（续训跳转用，§4.6）：

```python
def _mix_chunk_sizes(self, c):                # 第 c 个 chunk 里各 part 的占位数，只由 (stream_seed, c) 决定
    if deterministic_blend:                   # 确定性：闭式累计配额
        return [floor(w*C*(c+1)) - floor(w*C*c) for w in weights]
    rng = default_rng(md5(f"{stream_seed}-c{c}-multinomial"))
    return rng.multinomial(C, weights)        # 默认：每个占位独立按权重抽 part

def _mix_chunk_order(self, c):
    sizes = [(ex.part, n) for ex, n in zip(exs, _mix_chunk_sizes(c)) if n > 0]
    rng = random.Random(f"{stream_seed}-c{c}")
    order = deterministic_blend ? grouped_blend_order(...) : shuffle(展开)
```

- **默认（deterministic_blend=False）用多项分布抽样**（2026-10-31 起）：每块恰好 `C` 个占位，
  等价于逐个占位按权重独立抽 part。前 c 块的累计抽取数没有闭式，`_mix_cum_counts` 逐块累加
  （游标单调前进，缓存最近一次，300 块约 0.3s）。改动原因：按 part 配额 `floor(w·C)` 取整时，
  `w·C < 1` 的 part 要攒好几块才出一条；一个源摊到上万个 part 时整个源在前几块一条都不出
  （线上 tusou：1% 权重摊到约 2500 part/rank、每 part `w·C≈0.39`，前 150 步 0 条），之后又集中出现。
  非确定性模式的游标指纹带 `|multinomial`，旧 ckpt 的行流游标不能逐位接续，退回 per-part 位点。
- deterministic_blend 仍用闭式 `floor(w·C·(c+1)) - floor(w·C·c)`，不用 `ceil`：行模式的 `ceil(w × num_examples)`
  每 epoch 只取整一次、基数大，误差可忽略；混行每个 chunk 都取整，`ceil` 的误差同向累加，
  `w × C` 越接近 1 越严重（2.3 → 每轮给 3，多 30%；`C` 调小到 1 万时 0.56 → 给 1，多近一倍）。
  闭式下每个 part 在任意 chunk 边界处的累计计数恰好是 `floor(w·C·c)`，与理论值相差 < 1；
  chunk 总长在 `C ± part 数` 内浮动。（最初的「逐 chunk 累积余数」实现等价但不可跳转，已替换。）
- 权重口径：part 权重 `source_weight / num_parts` 不变，只是现在乘的是**样本数**。
  长样本多的源 token 占比会比行模式高，和 Megatron 一致；和行模式实验不可直接比。
- 新行流的 `stream_seed = f"{seed}-{global_shuffle_seed}-{global_step}-{dp_rank}"`：首次训练与扩缩卡续训
  都起新流（带 global_step，不会重放旧流的开头）；同 dp 续训沿用 ckpt 里存的 stream_seed，见 §4.6。

### 4.3 装箱

```python
def _mix_rows(self, start):                   # start = 行流状态（新流 / ckpt 游标 / 进程内上次消费处）
    gens = {p: ex.sample_sampler(start.next_k[p], start.skip[p])}
    open_rows = start.open_rows               # [[used_len, [(part, epoch, sidx, k), ...]], ...]
    c, j = start.c, start.j
    while True:
        order = self._mix_chunk_order(c)
        while j < len(order):
            item = (part, *next(gens[part]))  # part = order[j]
            first-fit 放进 open_rows；放不下时：
                满了先关最老的一行，cursor = (c, j, 关完之后的 open_rows)   # j 处的样本还没放进去
                self._emitted.append((row, cursor)); yield ("mix", row)
                open_rows.append([length, [item]])
            j += 1
        c, j = c + 1, 0
```

- `mix_pack_open_rows=1` 就是顺序贪心（与 `_plan_rows` 同策略，只是跨源）；4~16 能显著降 padding，
  代价是行内样本顺序与抽样顺序轻微错开，仍是确定性的。
- 每个 rank 用自己的 part（`use_train_part_sharding=1`）独立装，rank 间不需要同步。
- 外层的 batch 组装照抄 `ret()`：攒满 `batch_size` 个行 index 就 `yield buf`，同样维护 `global_step`；
  `consumed_samples` / `data_status` 在混行模式下不用（`set_epoch`、`load_data_status` 直接返回），
  续训位点见 §4.5。
- 打点：每 N 行 log 一次「平均每行样本数、填充率 Σlen/seqlen、行内源数、行内模态数」，冒烟时看这几个数。

### 4.4 index 形状与 DataLoader

行 index 是 `("mix", ((part, epoch, sidx, k), ...))`，变长 tuple，**不能**再 `np.array(indices)`。
paddle `DataLoader(batch_sampler=...)` 会把 batch_sampler 产出的 list 原样经队列发给 worker 再逐个
`dataset[idx]`，tuple 可以过 pickle；冒烟时在 worker 里 log 一次 `type(idx)` 确认没被转成 ndarray。

### 4.5 续训与扩缩卡：per-part 样本位点

目标：任意次重启、任意改 dp 数之后，同一个 `(part, k)` 不会被训两次，也不会被跳过
（part 读完一轮进下一 epoch 是正常多轮，不算重复）。原理同现在行模式的 `data_status`：
part 内顺序与 dp 无关（§4.1），只记「每个 part 消费到哪」，混合流丢掉重起。

**为什么不能只记一个计数**：`mix_pack_open_rows > 1` 时 first-fit 会让同 part 后抽的样本先关行
（A、B 依次抽出，A 还在开着的行里，B 的行已经训完）。只存计数，续训会漏 A、重训 B。
所以每个 part 存两样：

- `W[p]`：水位线，`k < W[p]` 的样本全部已消费；
- `E[p]`：`k >= W[p]` 但已消费的序号集合。规模 ≤ 所有 rank 开着的行与 prefetch 里的样本数（千级）；
  `open_rows=1` 时恒为空，退化成单计数。

**记账（训练侧，不是 sampler 侧）**：sampler 产出的行连同游标同时进 `self._emitted`（deque）。
`DataTraceCallback.on_load_data_end` 混行分支：本 micro-batch 有几行就从 `_emitted` 头部弹几行，
逐行断言 `data_id == row[0][2]`（对不上直接 raise，说明 FIFO 假设被破坏），再把行内每个
`(part, k)` 加进该 part 的已消费集合并推进 `W`。prefetch 出来但没训的行、open_rows 里的样本都不记账，
续训后由新 owner 重新抽到。按 micro-batch 弹，与 acc / Progressive-Batching 无关。

**存**：`DataTraceCallback.on_step_end` 在 `should_save` 时（所有 rank 都要调，集合通信）：

1. `W` 打成 `[max_part_id+1]` int64，全局 `all_reduce(MAX)`。非 need_data 卡、非 owner 卡贡献 0 或
   续训时载入的旧值；owner 的值一定 ≥ 旧值，所以 MAX 正确。
2. `E` 用 `all_gather_object` 收齐后求并，丢掉 `k < W[p]` 的项。
3. 写 `state.trial_params["mix_sample_pos"] = {"W": [...], "E": {p: [...]}, "meta": {...}}`，
   随 `trainer_state.json` 落盘。`meta` 存 `seed`、`seqlen`、filelist 的 md5、`shuffle`。

**载入**：`init_data_state` 里混行分支，所有 rank 都载入**全量** `W/E`（不只自己的 part），
owner 以它为初值继续记账。`sampler.load_mix_pos(W, E)` 之后，新行流的起点是
`next_k[p] = W[p]`、`skip[p] = E[p]`，喂给 `sample_sampler(start, skip)`（同 dp 时优先走 §4.6 的游标）。`consumed_samples` / `data_status` 在混行下不用。

**扩缩卡**：part 重新 `equal_shard` 后，新 owner 直接从全量位点里取自己 part 的 `W/E` 接着读，
与旧 owner 是谁无关。rank 内权重按新 part 集合重新归一，与行模式扩缩卡的行为一致。

**冒烟查重**：设环境变量 `MIX_CONSUMED_LOG_DIR` 时，每行消费的 `[part, k, epoch, sidx]` 追加写到
`$MIX_CONSUMED_LOG_DIR/mix_consumed_dp{r}.jsonl`（带 global_step），正式训练不要开。

**校验**：`meta` 任一项与当前配置不同就 raise。`seqlen` 变了会改变哪些样本被丢弃（长度 > seqlen 记 -1），
`seed` / `shuffle` 变了会改变 part 内顺序，两者都会让同一个 k 指向别的样本；filelist 变了 part id 会错位。
需要强行重开时显式设 `mix_reset_sample_pos=1`（全部位点归 0 并打 warning）。

**`k -> (epoch, 文件内位置)`**：k 是跨 epoch 的累计序号，`sample_sampler(start=W)` 需要按
`epoch % len(path)` 逐 epoch 减去该文件的有效样本数定位起点，会读到 part 内多个文件的
`ds16_offset`（走 `sample_lens` 缓存，只在主进程）。

### 4.7 从行模式 ckpt 切到混行：已训数据去重

行模式 ckpt 的 `trial_params["saved_data_status"][p]` 是 part p 累计被训练消费的行数（训练侧计数，prefetch 不算）。
part 内行顺序只由 `(seed, part, epoch, num_consecutive)` 和行计划（offset + 当时的 seqlen）决定，无论中途
续训几次、`same_data` 真假，消费的都是这个序列的前缀，所以「训过哪些样本」能直接算出来：

- 切换时（ckpt 有 `saved_data_status`、没有 `mix_sample_pos`），从旧 ckpt 的 `training_args.bin`
  （`resume_from_checkpoint` 或 `resume_train_state_from_checkpoint`，优先带 trainer_state 的那个）读
  seed / max_seq_length / num_consecutive / no_part_shuffle，连同每个 part 的行数存成
  `trial_params["mix_row_base"]`。之后每次混行续训都沿用这份（`saved_data_status` 在混行下不再维护）。
- 每个 part（`_row_base_consumed`）：按旧参数复刻 `sampler()` + `example_id` 走一遍行序列，得到
  整轮训完的 part-epoch 数 `e_full` 与第 `e_full` 轮里训过的样本（offset 文件：行内样本，去掉当时
  超 seqlen 被丢的；无 offset 文件：窗口下标）。混行的样本序列（`_epoch_items`）跳过前 `e_full` 轮，
  第 `e_full` 轮去掉训过的样本，之后正常。k 在这个去重后的序列上编号，W/E 与游标照常用；
  row base 进游标指纹。
- **旧 ckpt 一「行」的口径**（`mix_row_base_layout`，默认 `auto`）：
  - `plan`：现在的 inbuild reader（`use_inbuild_data=True`），`_plan_rows` 顺序贪心装行、超 seqlen 的样本丢弃（2026-10-13 起）；
  - `plan_keep_oversize`：2026-09-17 ~ 10-13 的 inbuild reader，贪心装行、超长样本独占一行（截断训过）；
  - `entry`：`use_inbuild_data=False` 时默认的 PretrainOnlineTask（PretrainTask 同口径）：一个 offset 条目就是一行，
    条目可以是装好的整行，超长的截断训过。
  - `auto`：旧 args 里 `use_inbuild_data` 为真取 `plan`，否则 `entry`（该字段 2026-09-17 才加，更早的 ckpt 都是
    PretrainOnlineTask）。`plan` 下若文件有超旧 seqlen 的样本会 warning，提示更早的代码要选 `plan_keep_oversize`。
  - 每个 offset 条目都 ≤ 旧 seqlen、相邻两条装不进一行时（离线装好行的数据），三种口径切出的行完全相同。
  - `online_generate_data=1`（在线 tokenize）的旧 ckpt 行不对应 H5 里的 token，直接 raise。
- 混行可以换 max_seq_length（offset 文件按旧 seqlen 复刻行，新 seqlen 只影响哪些样本有效）；
  无 offset 文件的窗口随 seqlen 变，换了直接 raise。
- 旧配置无法精确复原时 raise：`combine_batch != 1`、`use_train_part_sharding=0`（rank 内切片 +
  padding 会重复取行）、`dp_invariant_dataflow`、`data_sample_ratio != 1`。找不到 `training_args.bin` 也 raise。
  确认放弃去重时设 `mix_reset_sample_pos=1`（同时丢掉 row base）。
- 无法检测：行模式期间开过 `ignore_data_skip`（会把累计计数清零）、或中途换过 filelist / seed /
  max_seq_length。这些情况下复原的是最后一段的口径，需要用户自己确认。
- 代价：启动时每个 part 读一次 offset、按旧 seqlen 算一次行计划（sampler 初始化本来就要做）。

### 4.6 同 dp 续训：行流游标，逐位接续

§4.5 只保证不重复不漏，混合顺序续训时重起。同 dp、同 part 分配时可以做到行流与不中断时逐位一致，
而且是直接跳转、不空跑：

- **游标**：每产出一行，带上「这一行之后」的 `(c, j, open_rows)`。训练每消费一行就记下它的游标
  （prefetch 出来没训的行天然不算）。
- **存**：`DataTraceCallback` 存 ckpt 时，每个读数卡把最后消费那行的游标展开成自包含状态
  `{stream_seed, c, j, open_rows, next_k, skip, fp}`，`all_gather_object` 后按 dp_rank 放进
  `trial_params["mix_sample_pos"]["cursors"]`（trainer_state 只有 rank0 落盘）。
  `next_k[p]` = 流起点的序号 + 位置 (c, j) 处的累计抽取数（前 c 块累计 `_mix_cum_counts(c)` + 当前 chunk 前 j 个里的个数）
  - 起点处的累计抽取数，再按起点的跳过集合（来自 W/E 起的新流）修正；不用回放。
- **载入**：`init_data_state` 载完 W/E 后调 `sampler.load_mix_cursor(cursors[dp_rank])`。
  指纹 `fp` = (dp_rank, dp_size, 本 rank 的 part 列表, 权重, seqlen, open_rows, chunk, deterministic_blend, shuffle)。
  一致：从游标起流（同一位置重新抽到同一个 k，first-fit 结果不变）；不一致（扩缩卡等）：打 warning，
  退回 §4.5。还校验游标里开着的样本都没被消费过、`next_k >= W`，不一致直接 raise。
- **bs / acc 可以变**：游标记的是「消费到第几行」，和每步几行无关；变了只是行到 step 的划分变。
- 同一进程里重建迭代器（dataloader 重建）也从最近一次消费的行之后接着出。
- 存储：每个 rank 一份 next_k（本 rank 的 part 数个整数）+ 至多 open_rows 行的样本列表，KB 级。
- loss 能否逐位相同还取决于数值确定性（通信规约顺序、非确定 kernel），与混行无关，行模式也一样。

## 5. Reader（pretrain_inbuild_task.py）

### 5.1 把 `__getitem__` 拆成「读区间」+「收尾」

现在 `ExampleSetSingleDataSource.__getitem__`（:710-1016）一口气做完：取 [a, b) → ids / lossmask →
图 / 音频 / 视频 → MTP lookahead → mask。拆成两段，行模式调用顺序不变，逐位一致：

```python
def _read_span(self, fp, a, b, s, e, seg_lens, idx, is_row_tail=True, mask_starts=False):
    """读 ds16[s:e]（样本区间 [a, b)）的 ids / lossmask / 三种媒体，返回 dict（ids, lossmask, task_id,
    images, image_resolutions, audios, audio_lengths, context_modality, *_data_error_count）。
    行区间的推算（_row_start / _trim_oversize_tail）和 MTP lookahead 读取留在行模式 __getitem__ 里。"""

def _finish_row(self, sp, seg_lens, ids_mtp, lossmask_mtp, src=None, extra=None):
    """MTP 补齐 + lookahead 拼接 + cu_seqlens / startend / MTP mask -> Example。
    extra = 混行字段 dict，放进 Example 新增的 `mix` 字段（行模式为 None）。"""
```

行模式：`__getitem__((epoch, row))` = `_row_start` 求 [a, b) → `_read_span(..., is_row_tail=True,
mask_starts=...)` → 读 lookahead → `_finish_row`，和改动前逐位一致（tests 外的 golden 对比：
五种 H5 × mask_sample_start_label × MTP 共 360 行 + 行模式 sampler 前 60 个 batch，逐字段 array_equal）。

**`is_row_tail` 坑**：图像分支数占位符用的是 `input_view = ids[:-1]`（:807），理由是行末 token 只作
label、不进解码器。混行时只有**行内最后一个样本**该砍这一位，中间样本必须用完整 ids，否则样本恰好以
CONTEXT 结尾时会少数一个占位、误报 SpanError。`_read_audio_row` / `_read_video_row` 里有同类
`[:-1]` 的地方一并改成看 `is_row_tail`（改前 grep 一下 `[:-1]`）。

### 5.2 混行读：`ExampleSet.__getitem__`（:1411）按 index 形状分流

```python
if isinstance(idx, (tuple, list)) and idx and idx[0] == "mix":
    ret = self._read_mix_row(idx[1])
    eidx = int(idx[1][0][2])   # data_id = 首样本 sidx，§4.5 记账时用来核对 FIFO
else:
    ...  # 现有 int / 三元组分支不动
```

`_read_mix_row(items)`：

1. 对每个 `(part, epoch, sidx, k)`（k 只给 §4.5 记账用，reader 不看）：`ex = self.partid2ex[part]`，`fp_idx = epoch % len(ex.fps)`，
   `sp = ex._read_span(fp_idx, sidx, sidx + 1, is_row_tail=(k == last))`。
2. 按顺序拼：`ids`、`lossmask`（某段 H5 没有 lossmask 时补全 1，与 `_mask_sample_starts` 同口径）、
   `seg_lens`（每段一个）、`images`（axis=1）、`image_resolutions`、`audios`（axis=1）、`audio_lengths`、
   `context_modality`（axis=0；文本样本、读失败的样本都不贡献条目）。
3. 逐段字段：`seg_src = [part]*段数`、`seg_task = [sp.task_id]*段数`（int64，长度 == len(seg_lens)）。
4. 计数：`image_data_error_count = Σ sp.err['img']`，音频 / 视频同；另发
   `image_sample_count / audio_sample_count / video_sample_count` = 行内对应模态的样本数（失败率分母，§7）。
5. `mask_sample_start_label`：各段 `ex.mask_sample_start_label` 一致时对整行调一次 `_mask_sample_starts`；
   不一致直接 raise（不同词表的 H5 混在一起本身就是配错了）。
6. MTP：lookahead 不再读文件，`ids_mtp = [pad_token_id]*K`、`lossmask_mtp = [0]*K`；然后 `_finish_row`。
   `build_mtp_masks_from_cu_seqlens` 按 seg_lens 算，跨段本来就屏蔽，不受影响。
7. `Example(src=items[0].part, part=items[0].part, task_id=第一段的 task_id, ...)`：行级字段只给老代码兜底
   （collator、DataTrace 的 data_status 索引）；混行行有媒体时一定带 `context_modality`，组网不会走 task_id 兜底。
8. 断言 `sum(seg_lens) == len(ids_before_mtp) <= seqlen`，以及
   `len(context_modality) == len(image_resolutions or []) + len(audio_lengths or [])`。

读失败仍是 all-or-nothing，但粒度从「整行」变成「单个样本」：失败样本的 ids 置 1、lossmask 置 0、
不贡献媒体与 `context_modality` 条目，同一行其它样本照常训练。

### 5.3 `example_to_feature`（:1599）

features 里加 `seg_src`、`seg_task`（`np.int64`）与三个 `*_sample_count`（int），来自 `Example.mix`。混行模式每行都发，
行模式都不发：collator 用 `keys = list(batch[0].keys())`（data_utils.py:235），同一个 run 内 key 集合必须固定
（PP 要求每个 micro-batch 字段一致）。

## 6. Collator 与 DistDataLoader

- `src/utils/data_utils.py:273`：把 `seg_src`、`seg_task` 加进沿段轴拼接的列表
  （`["image_resolutions", "seg_lens", "audio_lengths", "context_modality", "seg_src", "seg_task"]`）。
  三个 `*_sample_count` 是 int，走开头的 `np.stack` 分支，不用改。
- `src/datasets/dist_data_loader.py`：
  - :379-600 是逐字段显式解包 / 广播 / 回填的，按 `context_modality` 的写法把 `seg_src`、`seg_task`、
    三个 `*_sample_count` 各加一份（取值、None 占位、broadcast 列表、`to_return`）。
  - :63 `_PER_SAMPLE_MD5_SKIP` 加 `seg_src`、`seg_task`（首维是段数，不是样本数）。
  - 新参数 `broadcast_mix_seg`（= 混行且开了按源 / 按模态 loss）：把 `seg_src`、`seg_task` 与段长 / 段数
    广播给 PP 末级。段长 / 段数用单独的 key `loss_seg_lens / loss_seg_count` 发：末级 stage 收到 `seg_lens`
    会改变组网侧行为。TP 组内不广播这些字段（只有 tp0 采集统计，TP 冗余卡不需要）。

## 7. 按源 / 按模态 loss 与失败率

loss 层不动：`_eval_token_loss_hook` 已经把主 head 的逐 token CE（mask、归约之前）回调给 callback。

### 7.1 把段信息带进 step_inputs

`pretraining_trainer.py:4056 _src_loss_inputs` 在 pipeline pop 字段之前取引用，加上：

```python
{"labels": ..., "src_id": ..., "task_id": ...,
 "seg_lens": item.get("seg_lens"), "seg_count": item.get("seg_count"),
 "seg_src": item.get("seg_src"), "seg_task": item.get("seg_task")}
```

### 7.2 `DataSourceLossCallback`（data_source_loss_callback.py）

- `record`（:361）：record 时拿不到 `seg_src`，所以在 `setup(args)` 里记下 `args.mix_sample_packing`，
  打开时多存一份逐 token 的 `masked CE` 与 `mask`（形状 `[n_row, S]`，`detach()`），放 records 第 5 个槽位；
  关闭时不存，行模式不受影响。显存：mbs2 × 8k × 2 × 8B ≈ 256 KB / micro-batch，可忽略。
- `collect_step`（:486）/ `_expand_rows`（:432）：按 labels 指纹配对 micro-batch 的逻辑不变。
  配上之后若该 item 有 `seg_src`，改走按段聚合，一次 device→host 后在 numpy 里做：

```python
# 一行: seg_lens 是 ids 口径（含行首、长度和 <= S+1）; labels[j] 预测 ids[j+1]
seg_of_ids = np.repeat(np.arange(n_seg), seg_lens)
label_seg = seg_of_ids[1:S + 1]                      # 长度 < S 的部分是 padding，mask 本来就是 0
ce_seg  = np.bincount(label_seg, weights=ce_row[:len(label_seg)],  minlength=n_seg)
tok_seg = np.bincount(label_seg, weights=mask_row[:len(label_seg)], minlength=n_seg)
for k in range(n_seg):  stat[source_of(seg_src[k])] += (ce_seg[k], tok_seg[k], 1)   # rows -> 段数
```

  多行 micro-batch 用 `seg_count` 把拼平的 `seg_lens / seg_src` 切回各行。
  「段 k 的 token」= label 落在段 k 的 ids 上；也就是说上一段末位预测下一段 BOS 那一格归下一段
  （`mask_sample_start_label=1` 时这格本来就不计损）。
- jsonl 里的 `rows` 在混行模式下含义是样本数；`meta.json` 加 `"unit": "sample"` 标注。
- 校验：每个 step `Σ_k tok_seg == mask.sum()`，不等就记 unmatched 并 warn，不猜。

### 7.3 `ModalityLossCallback`（同文件 :997 起）

同样存逐 token CE、按段聚合，桶取 `seg_task[k]`（映射表 :981 不变）。

### 7.4 媒体失败率（pretraining_trainer.py:7225-7250）

分子仍是 `*_data_error_count.sum()`；分母在 batch 有 `*_sample_count` 时改用它的和，
否则沿用「task_id 属于 IMAGE/AUDIO/VIDEO_TASK_IDS 的行数」。混行下按行数做分母会把失败率算错。

### 7.5 不改的

- eval 侧 `_EvalSourceLossTracker` / eval_ppl：eval 不走混行 sampler，仍是行模式，不动。
- `sep_parallel_size > 1` 时 callback 本来就关，不变。
- `data_trace_callback.py:93` 用行级 `src_id` 累 `data_status`：混行下只记第一段、且续训不读它，不改；
  混行的消费记账与位点存取是在同一个 callback 里新加的分支（§4.5）。

## 8. 单测（`ernie5/tests/`，参照现有 `test_hyperbody_batch.py` / `test_inbuild_lossmask.py` 的造 H5 方式）

| 用例 | 断言 |
|---|---|
| 开关关闭 | 同一组 H5、同 seed，行模式产出的前 N 个 batch 与改动前逐字段 `np.array_equal`（改前先 dump 一份 golden） |
| 装箱 | 每行 `Σlen <= seqlen`；超长样本不出现；`open_rows=1` 时等价顺序贪心；同 seed 两次生成完全相同 |
| 配比 | 取一个 `w × C` 非整数的小 part（如 `C=1000`、`w×C=2.3`）：连续 100 个 chunk 的每个 chunk 边界处，累计计数与 `w × 已出 chunk 数 × C` 相差 < 1；`w×C<1` 的 part 不会每个 chunk 都出现、长期频率正确；全体 part 在 10 万占位里偏差 < 1%（deterministic_blend 下应是常数级偏差） |
| 混模态拼接 | 一行 = 文本 + 图（含切图多段）+ 音频（多 chunk）+ 视频；`build_hyperbody_packed_batch` 不报错，`image_list / audio_list` 的条目数与顺序和逐样本单独读拼起来一致 |
| 行尾 CONTEXT | 中间样本以 CONTEXT 结尾时不误报；只有行末样本按 `ids[:-1]` 数 |
| 单样本读失败 | mock BOS 让一张图失败：只有该段 ids=1 / lossmask=0、无 `context_modality` 条目，同行其它段正常，`image_data_error_count==1` |
| MTP | K=1 时 lookahead 为 pad 且不计损，`mtp_*` mask 与 seg_lens 一致 |
| 按段 loss | 构造已知 CE 的 `[2, S]` 张量 + 两行各 3 段不同源：按段聚合结果与手算一致；`Σtok_seg == mask.sum()` |
| 模态 loss | 同上，按 `seg_task` 归桶 |
| 位点记账 | `open_rows=8`，按产出顺序消费若干行、留 10 行在「prefetch」里不消费：`W/E` 与手算一致；`open_rows=1` 时 `E` 恒空 |
| FIFO 核对 | 故意错位喂一个 micro-batch：`data_id` 核对 raise |
| 扩缩卡不重复 | 纯 CPU 模拟 sampler + 记账 + 存载：dp4 跑 N 步存 → dp8 续 → dp2 续，每段中途都留 prefetch 未消费；全程消费的 `(part, k)` 无重复，且每个 part 消费集合 = `[0, W) ∪ E`（无空洞） |
| 多 epoch | 小 part 被读穿进入下一 epoch：k 连续递增，`k -> (epoch, sidx)` 与不中断时一致 |
| 配置变更 | 续训时改 `max_seq_length` / `seed` / filelist：raise；`mix_reset_sample_pos=1` 时位点归 0 并 warning |

## 9. 两机冒烟

### 9.1 数据：四模态小批拼一份 list（每模态 25%）

```bash
D=/root/paddlejob/pfsl3-qnt-pretrain-public/songyi05/hyperbody_h5_1010
OUT=/root/paddlejob/gpfsspace/hyperbody_h5/mix_dsv4_4mod_smoke; mkdir -p $OUT
: > $OUT/data.list.afs; : > $OUT/data_weights.txt
for m in dsv4_text_small_pretrain dsv4_tiling_small_pretrain dsv4_audio_1020_small_pretrain dsv4_video_small_pretrain; do
  cat $D/$m/data.list.afs >> $OUT/data.list.afs
  # data_weights 的源前缀要和 list 里的 AFS 子目录名对上（video 小批的 AFS 名是 dsv4_video_1021_small_pretrain）
  pre=$(head -1 $D/$m/data.list.afs | awk -F/ '{print $(NF-2)}')
  awk -v p="$pre" '{s+=$2; n[NR]=$1; w[NR]=$2} END{for(i=1;i<=NR;i++) printf "%s/%s %.12g 1\n", p, n[i], w[i]/s*0.25}' \
      $D/$m/data_weights.txt >> $OUT/data_weights.txt
done
```

跑之前对一下 `data_weights.txt` 的源名格式和 `mix_dsv4_tiling_audio_smoke/data_weights.txt` 一致
（形如 `dsv4_tiling_1010_small/baike_000/ 0.00035 1`），`parse_data_weight` 是按前缀匹配 list 路径的。

### 9.2 配置

在 `hyperbody_dsv4_mix-pretrain-16k-hzz1-2node-mbs2.yaml` 基础上新建 `hyperbody_dsv4_mix4_mixpack-pretrain-8k-hzz1-2node.yaml`：

```yaml
output_dir: ./output/hyperbody_dsv4_mix4_mixpack_2node
data_filelist: /root/paddlejob/gpfsspace/hyperbody_h5/mix_dsv4_4mod_smoke/data.list.afs
data_weights: /root/paddlejob/gpfsspace/hyperbody_h5/mix_dsv4_4mod_smoke/data_weights.txt
mix_sample_packing: true
mix_pack_open_rows: 8
use_train_part_sharding: 1
deterministic_blend: true
shuffle_consecutive: false
record_data_source_loss: true
log_modality_loss: true
max_steps: 50
```

```bash
mpirun -x MODEL_TYPE bash script/train.sh --configs \
    conf/experiment/HyperBody/hyperbody_real_image_all.yaml \
    conf/experiment/HyperBody/hyperbody_dsv4_mix-pretrain-16k-hzz1-2node-mbs2.yaml \
    conf/experiment/HyperBody/hyperbody_dsv4_mix4_mixpack-pretrain-8k-hzz1-2node.yaml
```

对照组：同一份数据、`mix_sample_packing: false` 跑 50 步。

续训组（验 §4.5）：两机 `save_steps: 20` 跑到 25 步后 kill，改成**单机**（dp 减半）从 checkpoint-20 续到 50。
冒烟期间临时打开 debug 日志，把每个 micro-batch 消费的 `(part, k)` 逐行写到
`output_dir/mix_consumed_rank{r}.jsonl`（每条带 global_step），两段合起来离线查。

### 9.3 看什么

1. sampler 打点：平均每行样本数、填充率（应 > 行模式）、行内源数 / 模态数（应明显 > 1）。
2. 前几步无 `SpanError` / `context_modality has N entries but ...` / 段长和超 seq_len。
3. `data_source_loss/loss.jsonl`：四个模态的源都有 tokens，`Σ源 tokens` == step 有效 token 数，`unmatched=0`。
4. 模态 loss 四个桶都有值；`[IMG_CHECK]/[AUD_CHECK]/[VID_CHECK]` 失败率与行模式量级一致。
5. 吞吐 `tokens_per_sec_per_card` 与对照组比：掉得多说明 reader 逐样本读成了瓶颈（见 §10）。
6. loss 曲线前 50 步和对照组同量级（不要求一致：配比口径变了）。
7. 续训组：只取两段里 step ≤ 20 和 step > 20 的记录（21~25 步被回滚、属于重训，排除），
   `(part, k)` 无重复；续训起点日志里各 part 的 `start` 等于 checkpoint-20 里的 `W[p]`；
   `trainer_state.json` 里 `mix_sample_pos.E` 的总条数在千级以内。

## 10. 风险与备选

- **逐样本读的 IO 次数**：文本平均 ~630 token/样本，64k 一行 ≈ 100 次 h5py 小切片。AFS fuse 上慢的话：
  （a）`_read_mix_row` 里把同一 `(part, fp_idx)` 的相邻 `sidx` 合并成一次切片再按 offset 拆；
  （b）`sample_sampler` 改成按 K 个连续样本为块打乱（加 `mix_sample_block`，不复用 `num_consecutive`），
  换随机性换 IO；k 仍按块内实际出样顺序编号，§4.5 的位点口径不变；
  （c）加 dataloader worker 数。先测再选。
- **主进程长度缓存**：大 list（mm4 full）启动时要读本 rank 所有文件的 `ds16_offset`；
  慢的话改成首次被抽到时再读（`sample_sampler` 是惰性的，天然支持）。
- **配比口径**：按样本配比后长样本源 token 占比上升，若要和行模式 token 比例对齐，需按
  `平均样本长度` 重算 data_weights，不在本次改动范围。
- **行级 `src_id / task_id`**：混行行只代表第一段，任何还在按行用它们做统计的地方都会偏；
  改完 grep 一遍 `["src_id"]`、`["task_id"]` 的读点确认只剩 collator / data_trace。

## 11. 实现与冒烟记录（2026-10-29）

代码：`pretrain_inbuild_task.py`（_read_span / _finish_row / sample_sampler / _read_mix_row）、
`pretraining_trainer.py`（参数、`WeightedDistributedSampler` 混行分支、失败率分母、`_src_loss_inputs`）、
`data_trace_callback.py`（消费记账 + 位点存取）、`data_utils.py`（collator、`_init_mix_pos`）、
`dist_data_loader.py`（混行字段 + `broadcast_mix_seg`）、`data_source_loss_callback.py`（按段归属）。
顺手修了一个旧 bug：`_blend_source` 写成 `getattr(ex, "source", ex.src)`，默认值先求值，
inbuild + `deterministic_blend=true` 的行模式之前会直接 AttributeError。

单测：`ernie5/tests/test_mix_sample_packing.py`（19 个，fixture 在 `mix_pack_fixtures.py`），
现有 hyperbody / inbuild / loss 相关测试全过。行模式 golden（改动前后逐字段 array_equal）0 差异。

冒烟数据：§9.1 的四模态 list，去掉了 177 个「全部样本 > 8193」的 H5（行模式 / 混行对这种文件都会 raise）。
两机 16 卡、mbs2 × 8k、gbs32、EP8、`mix_pack_open_rows=8`、`deterministic_blend=true`：

| 项 | 混行 | 行模式对照 |
|---|---|---|
| 50 步跑通 | 是，无 SpanError / 段长报错 | 是 |
| loss（step 50） | 9.39 | 9.76 |
| tokens_per_sec_per_card（11-50 步均值） | 4088 | 5063 |
| max_mem_allocated_gb | 41.2 | 36.9 |
| 每行样本数 / 填充率（CPU 实读） | 7.3 / 0.98-0.99 | — |
| 有效 token 占比 text / img / aud / vid | 60 / 16 / 3 / 22 % | 75 / 8 / 7 / 9 % |
| `loss.jsonl` unmatched / `Σ源 tokens == 模态 tokens` | 0 / 是（41-50 步都是 824897） | — |
| img / aud / vid 失败率 | 0 / 0 / 0 | 0 / 0 / 0 |

- 吞吐低约 19%，原因没拆开：两边 token 构成不同（混行的图像 / 视频 token 占比约为行模式的 2 倍，encoder 更重），逐样本读（每行约 7 次 h5py 切片 + 媒体）也可能有份；要定位需在同配比下对比或打 dataloader 等待时间。§10 的 IO 备选还没做。
- 四源按样本各 25%，但实测样本数 video 约 17%、其余约 27%：每个 rank 每步固定出行数、各自在本 rank 的
  part 上归一权重；分到 video 大 part 的 rank 每行只装 2-3 个长样本，消耗的样本数少。这是「rank 内归一 +
  固定行数」的口径，不是混行 bug；行模式同样存在（按行数）。

续训冒烟（§4.5）：两机 dp16 跑到 25 步、checkpoint-20 存在 /dev/shm；单机 dp8 用
`resume_train_state_from_checkpoint` 接 checkpoint-20 的 trainer_state 续到 40 步。
`MIX_CONSUMED_LOG_DIR` 记的消费：A 段 1-20 步 4739 个样本、B 段 21-40 步 2241 个，合起来 6980 个
`(part, k)` 无重复、每个 part 无空洞；checkpoint-20 里的 `W/E` 与 A 段 1-20 步的消费逐 part 一致
（Σ W = 4733，E 6 条）。B 段 `loss.jsonl` unmatched = 0。
（B 段只接了训练状态没接权重，loss 不可比；接权重的完整续训受 GPFS 剩余 40G 限制没做。）

按源 / 按模态 / 按 task loss 的核对（混行冒烟）：
- **token 归属（独立复算）**：用 `MIX_CONSUMED_LOG_DIR` 记的 `(part, k)` 按 `sample_sampler` 同口径还原样本，
  直接读 H5 算每个样本该计损的 token（lossmask=1、非 CONTEXT、去掉样本首 token），按源 / 按模态汇总。
  50 步的四个模态 token 逐步一致；`loss.jsonl` 3882 个（窗口, 源）的 tokens 与样本数全部一致。
- **按 task（`log_data_source_loss`，`lm_loss/<task>`）**：data_weights 第 4 列填模态名
  （`hyperbody_h5/mix_dsv4_4mod_smoke/data_weights.task.txt`，前 3 列不变、采样不变）另跑 12 步，
  `dump_steps=1`。逐步 `lm_loss/<task>` 与按模态 tracker 的 `*_loss` 完全相等（两条独立路径：
  part→source→name 与 seg_task→模态），与 `loss.jsonl` 按 name 汇总的差 < 1e-6（jsonl 保留 6 位小数），
  unmatched 全 0。
- 没法独立验的：loss 数值本身（需要模型 forward）。按段把 CE 切开的那一步由单测按手算 CE 覆盖。

同 dp 逐位续训（§4.6，行流游标）冒烟：
- A：两机 dp16 不中断跑 30 步，第 20 步存 ckpt（/dev/shm）。
- B：从 A 的 checkpoint-20 完整续训（权重 + 优化器 + trainer_state，`ignore_load_lr_and_optim=false`）到 30 步。
  各 rank 日志都是 `[mix] start mix sampler (exact resume): stream_seed=23-1-0-<r> chunk=0 pos=...`。
- 行流：21-30 步 16 个 rank × 10 步 = 160 组 (dp_rank, step)、2325 个样本，A 与 B 每行的 `(part, k)` 逐位一致；
  各步 lr、各模态 token 数一致。
- loss：step 21 完全相同，22-30 步差 ≤ 5e-5。另跑了一次与 A 同配置的不中断运行 A2 作对照：A 与 A2 从
  step 3 起就差到 2e-5 ~ 8e-5（MoE / 通信的非确定数值），所以续训差异在不中断重跑的波动以内，
  数据侧是逐位接上的。
- 单测补了：同 dp 续训（含中途换 bs）、扩缩卡新流之后再同 dp 续训、进程内重建迭代器、指纹不一致退回 W/E。

从行模式 ckpt 切到混行（§4.7）冒烟：
- 行模式两机 dp16 跑 12 步，第 10 步存 ckpt；混行从 checkpoint-10 完整续训（权重 + 优化器 + trainer_state）到 20 步。
  日志：`[mix] 从行模式 ckpt 切到混行: 累计已训 320 行，按 .../checkpoint-10/training_args.bin 的参数 ...`。
- 核对（`output/mix_consumed_row2mix/check_row2mix.py`）：行模式前 10 步各 rank 的 data_seq 前 20 行 → 行计划 → 4293 个样本；
  混行 11-20 步消费 2486 个样本，其中 1518 个来自行模式训到一半的 part-epoch。交集 0；
  不去重时按各 part-epoch 已训比例估计应重复约 218 个。
- 多机下 training_args.bin 只在 rank0 节点：`get_train_dataloader` 里先用 `distributed_file` 同步到各节点
  （第一次冒烟没做这一步，node2 直接报找不到文件）。

同 token 配比下的吞吐（回答 §11 里「吞吐低 19%」的原因）：
- 按混行冒烟与行模式对照实测的 token 占比，把四个模态的权重各乘一个系数（text 1.27 / image 0.53 / audio 2.15 /
  video 0.43，`hyperbody_h5/mix_dsv4_4mod_smoke/data_weights.tokenmatch.txt`），其余配置不变，混行再跑 50 步。

| 11-50 步 | 行模式 | 混行（原权重） | 混行（token 配比对齐） |
|---|---|---|---|
| token 占比 text / img / aud / vid | 75.4 / 8.3 / 7.1 / 9.3 | 59.6 / 15.6 / 3.3 / 21.5 | 73.0 / 8.1 / 8.3 / 10.6 |
| 每步耗时 | 3.27 s | 4.05 s | 3.40 s |
| `tokens_per_sec_per_card` | 5063 | 4088 | 4856 |
| 每步计损 token 数 | 87226 | 83480 | 108491 |
| 计损 token / 秒 / 卡 | 1667 | — | 1996 |

- 原来的 19% 主要来自配比：图像 / 视频 token 占比翻倍，encoder 更重。对齐后每步只慢 4%（逐样本读等开销）。
- `tokens_per_sec_per_card` 按 `gbs × max_seq_length` 算（含 padding），不反映装箱率。混行每行装得更满，
  同样 32 行每步多训 24% 的计损 token，按计损 token 算的有效吞吐比行模式高约 20%。

