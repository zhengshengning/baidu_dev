#!/usr/bin/env python3
"""
从训练日志中提取指标，绘制多条曲线对比图。

使用方式：直接修改下方 CONFIG 字典，然后 python plot_loss_curve.py
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
#  配置区 —— 直接修改这里即可
# ============================================================
CONFIG = {
    # 图表标题，None 则自动生成
    "title": None,

    # 输出图片路径
    "output": "loss_curve.png",

    # X 轴是否用日志中的 global_step（False 则用顺序编号）
    "use_global_step": True,

    # Y 轴范围，None 则自动，如 [5.0, 8.0]
    "ylim": None,

    # loss diff 子图的 Y 轴范围，None 则自动
    "loss_diff_ylim": None,

    # grad_norm diff 子图的 Y 轴范围，None 则自动
    "grad_norm_diff_ylim": None,

    # global_grad_norm 子图的 Y 轴范围，None 则自动
    "grad_norm_ylim": None,

    # diff 对列表：每项 [a_idx, b_idx] 表示画 curve[b] - curve[a] 的 diff 曲线
    # 索引基于上方 curves 列表（0-based），可自由组合
    # None 或空列表则不画 diff 子图
    "diff_pairs": [
        [0, 1],
        [0, 2],
        [0, 3]
    ],

    # 曲线列表：每条曲线独立指定 日志路径、提取指标、图例标签
    # 同一个日志文件可以出现多次，提取不同指标画不同曲线
    "curves": [
    
        #{
        #    "path": "/root/paddlejob/pfsl3space/chenruibiao/ernie_v2_0430/baidu/ernie/erniebot/output/paddle_distributed_logs_ec_baseline_0501/rank_0/workerlog.0",
        #    "key": "pure_text_loss",
        #    "label": "ec_baseline",
        #},
        
        {
            # "path": "/root/paddlejob/paddlefleet-public/ningzhengsheng/erniebot/outputs/loss_ernie5_5_flash_ladder_v90_T3_h1024_L16_17.56BA0.710B-pretrain-8k-hzz1/rank_0/workerlog.0",
            "path": "/root/paddlejob/paddlefleet-public/ningzhengsheng/FSDP/baidu/ernie/erniebot/outputs/ernie5_5_flash_ladder_v90_T3_h1024_L16_17.56BA0.710B-pretrain-8k-hzz1/rank_0/workerlog.0",
            "key": "loss",
            "label": "pp+sharding1",
        },
        {
            # "path": "/root/paddlejob/paddlefleet-public/ningzhengsheng/erniebot/outputs/loss_ernie5_5_flash_ladder_v90_T3_h1024_L16_17.56BA0.710B-pretrain-8k-hzz1-fsdp/rank_0/workerlog.0",
            "path": "/root/paddlejob/paddlefleet-public/ningzhengsheng/FSDP/baidu/ernie/erniebot/outputs/ernie5_5_flash_ladder_v90_T3_h1024_L16_17.56BA0.710B-pretrain-8k-hzz1-fsdp/rank_1/workerlog.0",
            "key": "loss",
            "label": "fsdp",
        },
        #         {
        #     "path": "/root/paddlejob/share-storage/gpfs/system-public/sundong/code/v2_508_1/erniebot/backup/output_gate_init/paddle_distributed_logs_0/workerlog.0",
        #     "key": "loss",
        #     "label": "fleet_gate_normal_init",
        # },
        # {
        #     "path": "/root/paddlejob/share-storage/gpfs/system-public/sundong/code/v2_508_1/erniebot/backup/output_xairver_init/paddle_distributed_logs_0/workerlog.0",
        #     "key": "loss",
        #     "label": "fleet_gate_Xavier_init",
        # },

        #{
        #     "path": "/root/paddlejob/share-storage/gpfs/system-public/zhuxinming/zhuxinming/ablation_attn/erniebot/output/paddle_distributed_logs/rank_0/workerlog.0",
        #     "key": "loss",
        #     "label": "fleet_resume_align_rope",
        #},

        #{
        #     "path": "/root/paddlejob/share-storage/gpfs/system-public/wangxiangzhe/test_0430_new/baidu/ernie/erniebot/output/paddle_distributed_logs_fleet_resume_0501/rank_0/workerlog.0",
        #     "key": "loss",
        #     "label": "fleet_resume_0501",
        #},

        # =============NOPP===========================================
        # {
        #      "path": "/root/paddlejob/share-storage/gpfs/system-public/huangjiyi/ernie_v2_a3b_nopp_latest/erniebot/output/v2_nopp_ec_baseline/paddle_distributed_logs_0/workerlog.0",
        #      "key": "pure_text_loss",
        #      "label": "ec_baseline_nopp",
        # },
        # {
        #      "path": "/root/paddlejob/share-storage/gpfs/system-public/huangjiyi/ernie_v2_a3b_nopp_latest/erniebot/output/v2_nopp_pf_xaiver_gate/paddle_distributed_logs_0/workerlog.0",
        #      "key": "loss",
        #      "label": "fleet_gate_xaiver_nopp",
        # },
        #{
        #     "path": "/root/paddlejob/share-storage/gpfs/system-public/huangjiyi/ernie_v2_a3b_nopp_latest/erniebot/output/v2_nopp_pf_resume_ec/paddle_distributed_logs_0/workerlog.0",
        #     "key": "loss",
        #     "label": "fleet_resume_nopp",
        #},




    ],

    # 移动平均窗口，0 表示不平滑
    "smooth": 0,

    # 每隔多少步采样一个点，1 表示全量
    "step_interval": 1,

    # 只绘制指定步数范围，None 则全部，如 [500, 2000] 表示只画 step 500~2000
    "step_range": None,
}

# ============================================================
#  以下为逻辑代码，一般不需要修改
# ============================================================

LOG_LINE_RE = re.compile(r"\[logging_callback\.py:\s+\d+\]")
FIELD_RE = re.compile(r"(\w+):\s*([\d.eE+\-]+)")


def extract_metrics(path: str, key: str, use_global_step: bool = False):
    steps, values = [], []
    seen = set()
    seq = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not LOG_LINE_RE.search(line):
                continue
            fields = dict(FIELD_RE.findall(line))
            if key not in fields:
                continue
            # 保存 ckpt 等日志行没有 global_step（只有 global_save_step），按 step 画图时跳过
            if use_global_step and "global_step" not in fields:
                continue
            gs = int(fields.get("global_step", seq + 1))
            if gs in seen:
                continue
            seen.add(gs)
            seq += 1
            steps.append(gs if use_global_step else seq)
            values.append(float(fields[key]))
    return steps[0:3000], values[0:3000]


def detect_interval(steps):
    """日志的 logging_steps：相邻 global_step 差值的众数"""
    gaps = [b - a for a, b in zip(steps, steps[1:]) if b > a]
    return max(set(gaps), key=gaps.count) if gaps else 1


def window_mean(raw, step, fine, coarse):
    """把 logging_steps=fine 的日志在 step 处按 coarse 窗口取平均（与 trainer 打印 loss 的口径一致）"""
    ks = range(step - coarse + fine, step + 1, fine)
    if all(k in raw for k in ks):
        return sum(raw[k] for k in ks) / len(ks)
    return None


def smooth(values, window):
    if window <= 1 or len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        s = max(0, i - window + 1)
        out.append(sum(values[s:i + 1]) / (i - s + 1))
    return out


def _process_curve(steps, vals, cfg):
    """对单条曲线做步数过滤、采样、平滑，返回 (steps, processed_vals)"""
    sr = cfg.get("step_range")
    if sr:
        pairs = [(s, v) for s, v in zip(steps, vals) if sr[0] <= s <= sr[1]]
        if not pairs:
            return None, None
        steps, vals = list(zip(*pairs))

    si = cfg.get("step_interval", 1)
    if si > 1:
        idxs = list(range(0, len(steps), si))
        steps = [steps[i] for i in idxs]
        vals = [vals[i] for i in idxs]

    w = cfg.get("smooth", 0)
    pv = smooth(vals, w) if w > 0 else vals
    return steps, pv


def _plot_curves(curves, key, colors, cfg):
    """提取并画曲线，返回 plotted 列表 [(steps_dict, label, color), ...]
    key=None 时使用每条 curve 自身的 key，否则统一使用指定 key
    """
    plotted = []
    for idx, c in enumerate(curves):
        path, label = c["path"], c["label"]
        use_key = c["key"] if key is None else key
        color = colors[idx % len(colors)]

        try:
            steps, vals = extract_metrics(path, use_key, cfg.get("use_global_step", False))
        except FileNotFoundError:
            print(f"[WARN] 文件不存在: {path}")
            continue

        if not vals:
            print(f"[WARN] {path} 中未找到指标 '{use_key}'")
            continue

        raw = dict(zip(steps, vals))
        interval = detect_interval(steps)
        steps, pv = _process_curve(steps, vals, cfg)
        if steps is None:
            continue

        plt.plot(steps, pv, color=color, linewidth=1.2)
        min_step, max_step = min(steps), max(steps)
        legend_label = '{} (Step {}-{})'.format(label, min_step, max_step)
        plotted.append((dict(zip(steps, pv)), legend_label, color, raw, interval))
        print(f"  {use_key} {label}: {len(vals)} steps, logging_steps={interval}, "
              f"[{min(vals):.6f}, {max(vals):.6f}]")

    return plotted


def _aligned_diff(a, b, window_avg):
    """返回 (common_steps, diff_vals)。

    两条曲线 logging_steps 不同且 window_avg=True（loss 是窗口均值）时，
    把间隔小的一条按大间隔窗口取平均，再在大间隔曲线的 step 上做差（使用原始值，不做平滑）。
    window_avg=False（grad_norm 是当前步的值）时直接取共同 step。
    """
    a_data, _, _, a_raw, a_int = a
    b_data, _, _, b_raw, b_int = b
    if not window_avg or a_int == b_int:
        common = sorted(set(a_data) & set(b_data))
        return common, [b_data[s] - a_data[s] for s in common]

    coarse, fine = max(a_int, b_int), min(a_int, b_int)
    if coarse % fine:
        print(f"[WARN] logging_steps {a_int} 与 {b_int} 不成整数倍，按共同 step 直接做差")
        common = sorted(set(a_data) & set(b_data))
        return common, [b_data[s] - a_data[s] for s in common]
    a_coarse = a_int == coarse
    coarse_steps = a_data if a_coarse else b_data  # 已按 step_range / step_interval 过滤
    common, diffs = [], []
    for s in sorted(coarse_steps):
        if a_coarse:
            av, bv = a_raw[s], window_mean(b_raw, s, fine, coarse)
        else:
            av, bv = window_mean(a_raw, s, fine, coarse), b_raw[s]
        if av is None or bv is None:
            continue
        common.append(s)
        diffs.append(bv - av)
    print(f"  [align] logging_steps {a_int} vs {b_int}: 间隔 {fine} 的曲线按 {coarse} 步窗口取平均后做差")
    return common, diffs


def _plot_diff(plotted, diff_pairs, ylabel_suffix, window_avg=False):
    """画 diff 子图"""
    diff_legends = []
    for pair_idx, (a_idx, b_idx) in enumerate(diff_pairs):
        if a_idx >= len(plotted) or b_idx >= len(plotted):
            print(f"[WARN] diff_pairs[{pair_idx}] 索引 [{a_idx}, {b_idx}] 超出范围（共 {len(plotted)} 条曲线），跳过")
            continue

        a_label = plotted[a_idx][1]
        b_label, b_color = plotted[b_idx][1], plotted[b_idx][2]

        common_steps, diff_vals = _aligned_diff(plotted[a_idx], plotted[b_idx], window_avg)
        if not common_steps:
            print(f"[WARN] {a_label} 与 {b_label} 无重叠 step，跳过 diff")
            continue

        plt.plot(common_steps, diff_vals, color=b_color)
        a_short = a_label.split(' (Step')[0]
        b_short = b_label.split(' (Step')[0]
        diff_legends.append('{} - {}'.format(b_short, a_short))
        mean_diff = sum(diff_vals) / len(diff_vals)
        print(f"  diff ({b_short} - {a_short}): "
              f"mean={mean_diff:.6f}, [{min(diff_vals):.6f}, {max(diff_vals):.6f}]")

    plt.axhline(y=0, color='gray', linewidth=1.0, linestyle='--')
    if diff_legends:
        plt.legend(diff_legends)


def main():
    cfg = CONFIG
    curves = cfg["curves"]
    if not curves:
        print("[ERROR] CONFIG['curves'] 为空，请至少添加一条曲线配置")
        return

    colors = ['r', 'g', 'b', 'y', 'pink', 'orange', 'black', 'm', 'c']
    diff_pairs = cfg.get("diff_pairs") or []
    title = cfg.get("title") or "Training Curve Comparison"

    # 2 行 2 列：上行 loss，下行 grad_norm
    ncols = 2 if diff_pairs else 1
    fig_h = 20 * 2  # 两行
    fig_w = 20 * ncols

    plt.figure(figsize=(fig_w, fig_h))
    plt.rcParams.update({'font.size': 28})

    # ========== (1,1) loss 曲线 ==========
    plt.subplot(2, ncols, 1)
    loss_plotted = _plot_curves(curves, None, colors, cfg)  # 用 curves 各自的 key

    if not loss_plotted:
        print("[ERROR] 所有曲线均未提取到数据，请检查配置")
        return

    plt.legend([p[1] for p in loss_plotted])
    plt.xlabel('step')
    plt.ylabel('loss')
    ylim = cfg.get("ylim")
    if ylim:
        plt.xlim(ylim)
    sr = cfg.get("step_range")
    if sr:
        plt.xlim(sr)
    plt.title(title)
    plt.grid(True)

    # ========== (1,2) loss diff ==========
    if diff_pairs:
        plt.subplot(2, ncols, 2)
        _plot_diff(loss_plotted, diff_pairs, "loss", window_avg=True)
        plt.xlabel('step')
        plt.ylabel('loss diff')
        loss_diff_ylim = cfg.get("loss_diff_ylim")
        if loss_diff_ylim:
            plt.ylim(loss_diff_ylim)
        sr = cfg.get("step_range")
        if sr:
            plt.xlim(sr)
        plt.title(f"{title} - loss diff")
        plt.grid(True)

    # ========== (2,1) grad_norm 曲线 ==========
    plt.subplot(2, ncols, ncols + 1)
    grad_plotted = _plot_curves(curves, "global_grad_norm", colors, cfg)

    if grad_plotted:
        plt.legend([p[1] for p in grad_plotted])
    plt.xlabel('step')
    plt.ylabel('global_grad_norm')
    grad_norm_ylim = cfg.get("grad_norm_ylim")
    if grad_norm_ylim:
        plt.ylim(grad_norm_ylim)
    sr = cfg.get("step_range")
    if sr:
        plt.xlim(sr)
    plt.title("global_grad_norm")
    plt.grid(True)

    # ========== (2,2) grad_norm diff ==========
    if diff_pairs:
        plt.subplot(2, ncols, ncols + 2)
        _plot_diff(grad_plotted, diff_pairs, "grad_norm")
        plt.xlabel('step')
        plt.ylabel('grad_norm diff')
        grad_norm_diff_ylim = cfg.get("grad_norm_diff_ylim")
        if grad_norm_diff_ylim:
            plt.ylim(grad_norm_diff_ylim)
        sr = cfg.get("step_range")
        if sr:
            plt.xlim(sr)
        plt.title("global_grad_norm diff")
        plt.grid(True)

    plt.tight_layout()
    out_path = cfg.get("output", "loss_curve.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n图片已保存到: {out_path}")


if __name__ == "__main__":
    main()