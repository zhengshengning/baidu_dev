#!/usr/bin/env python3
"""比较两个训练日志中每一步 (global_step) 的 loss 是否相同。

日志中每步的记录形如：
    - loss: 12.5338459, learning_rate: 2.639e-05, global_step: 1, ...
脚本按 global_step 对齐两份日志的 loss 并逐步比较。
"""
import argparse
import re
import sys
import unicodedata

# 去除 ANSI 颜色转义码（如 \x1b[32m ... \x1b[0m）
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# 从一行中提取 loss 与 global_step
LOSS_RE = re.compile(r"(?:^|[\s,])loss:\s*([-+0-9.eE]+)")
STEP_RE = re.compile(r"global_step:\s*(\d+)")
# 提取每卡吞吐（不匹配 tokens_per_sec_per_card_average）
TPS_RE = re.compile(r"tokens_per_sec_per_card:\s*([-+0-9.eE]+)")


def parse_log(path):
    """返回 ({global_step: loss}, {global_step: tokens_per_sec_per_card})。
    同一 step 出现多次时保留首次。"""
    steps = {}
    tps = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = ANSI_RE.sub("", line)
            step_m = STEP_RE.search(line)
            loss_m = LOSS_RE.search(line)
            if not step_m or not loss_m:
                continue
            step = int(step_m.group(1))
            try:
                loss = float(loss_m.group(1))
            except ValueError:
                continue
            if step not in steps:
                steps[step] = loss
                tps_m = TPS_RE.search(line)
                if tps_m:
                    try:
                        tps[step] = float(tps_m.group(1))
                    except ValueError:
                        pass
    return steps, tps


def log_names(path1, path2):
    """根据文件路径是否包含 fsdp 判断列名；无法区分时回退为 log1/log2。"""
    n1 = "FSDP" if "fsdp" in path1.lower() else "非FSDP"
    n2 = "FSDP" if "fsdp" in path2.lower() else "非FSDP"
    if n1 == n2:
        return "log1", "log2"
    return n1, n2


def rjust(s, width):
    """按显示宽度右对齐（中文字符占 2 列）。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return " " * max(width - w, 0) + s


def compare_tps(t1, t2, skip, diff_only, n1, n2):
    """逐步对比 tokens_per_sec_per_card，并汇总均值（跳过前 skip 步预热）。"""
    common = sorted(set(t1) & set(t2))
    if not common:
        print("\n未在日志中解析到共同步的 tokens_per_sec_per_card，跳过速度对比")
        return
    print(f"\n===== tokens_per_sec_per_card 对比（共同步数: {len(common)}）=====")
    print(f"{'step':>6} {rjust(n1, 14)} {rjust(n2, 14)} {'diff(2-1)':>14} {'ratio(2/1)':>11}")
    print("-" * 64)
    for step in common:
        a, b = t1[step], t2[step]
        ratio = b / a if a else float("nan")
        if not diff_only or a != b:
            print(f"{step:>6} {a:>14.2f} {b:>14.2f} {b - a:>14.2f} {ratio:>11.4f}")
    print("-" * 64)
    stable = [s for s in common if s > skip]
    if not stable:
        print(f"共同步均 <= 预热步数 {skip}，不统计均值")
        return
    avg1 = sum(t1[s] for s in stable) / len(stable)
    avg2 = sum(t2[s] for s in stable) / len(stable)
    ratio = avg2 / avg1 if avg1 else float("nan")
    print(f"平均 tokens_per_sec_per_card（跳过前 {skip} 步，统计 {len(stable)} 步）:")
    print(f"  {n1}: {avg1:.2f}")
    print(f"  {n2}: {avg2:.2f}")
    print(f"  {n2}/{n1}: {ratio:.4f} ({(ratio - 1) * 100:+.2f}%)")


def main():
    ap = argparse.ArgumentParser(description="比较两个日志每步 loss 是否相同")
    ap.add_argument("log1")
    ap.add_argument("log2")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="绝对误差容忍度，默认 0（要求完全相同）")
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="相对误差容忍度，默认 0")
    ap.add_argument("--diff-only", action="store_true",
                    help="只打印 loss 不同的步（默认打印每一步）")
    ap.add_argument("--no-tps", action="store_true",
                    help="不对比 tokens_per_sec_per_card")
    ap.add_argument("--tps-skip", type=int, default=1,
                    help="统计平均 tokens_per_sec_per_card 时跳过的前 N 步（预热），默认 1")
    args = ap.parse_args()

    s1, t1 = parse_log(args.log1)
    s2, t2 = parse_log(args.log2)

    n1, n2 = log_names(args.log1, args.log2)

    print(f"log1({n1}): {args.log1}  共 {len(s1)} 步")
    print(f"log2({n2}): {args.log2}  共 {len(s2)} 步")

    common = sorted(set(s1) & set(s2))
    only1 = sorted(set(s1) - set(s2))
    only2 = sorted(set(s2) - set(s1))

    if only1:
        print(f"\n仅在 log1 出现的步: {only1}")
    if only2:
        print(f"仅在 log2 出现的步: {only2}")

    print(f"\n共同步数: {len(common)}")
    print(f"{'step':>6} {rjust(n1, 18)} {rjust(n2, 18)} {'diff':>14} {'状态'}")
    print("-" * 70)

    n_diff = 0
    max_diff = 0.0
    max_diff_step = None
    for step in common:
        l1, l2 = s1[step], s2[step]
        diff = abs(l1 - l2)
        same = diff <= max(args.atol, args.rtol * max(abs(l1), abs(l2)))
        if diff > max_diff:
            max_diff = diff
            max_diff_step = step
        if not same:
            n_diff += 1
        if not args.diff_only or not same:
            status = "相同" if same else "不同 <<<"
            print(f"{step:>6} {l1:>18.8g} {l2:>18.8g} {diff:>14.6g} {status}")

    print("-" * 70)
    if n_diff == 0:
        print(f"结果: 所有 {len(common)} 个共同步的 loss 均相同"
              f"（atol={args.atol}, rtol={args.rtol}）")
    else:
        print(f"结果: {n_diff}/{len(common)} 个步的 loss 不同")
    if max_diff_step is not None:
        print(f"最大绝对差异: {max_diff:.6g} @ global_step {max_diff_step}")

    # 速度对比仅作参考，不影响退出码
    if not args.no_tps:
        compare_tps(t1, t2, args.tps_skip, args.diff_only, n1, n2)

    sys.exit(1 if (n_diff or only1 or only2) else 0)


if __name__ == "__main__":
    main()
