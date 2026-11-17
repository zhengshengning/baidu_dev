#!/usr/bin/env python3
"""统计 workerlog 中第 m 到第 n 步（含两端）的 tokens_per_sec_per_card 与 max_mem_reserved_gb 平均值。

用法:
    python avg_step_metrics.py <workerlog> <m> <n>
"""
import argparse
import re

STEP_RE = re.compile(r"\bglobal_step: (\d+)")
METRIC_RES = {
    "tokens_per_sec_per_card": re.compile(r"\btokens_per_sec_per_card: ([\d.eE+-]+)"),
    "max_mem_reserved_gb": re.compile(r"\bmax_mem_reserved_gb: ([\d.eE+-]+)"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("m", type=int, help="起始步（含）")
    parser.add_argument("n", type=int, help="结束步（含）")
    args = parser.parse_args()

    # 同一步多次出现时（如续训）以最后一次为准
    records = {}
    with open(args.log, errors="ignore") as f:
        for line in f:
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            if not args.m <= step <= args.n:
                continue
            values = {}
            for name, pattern in METRIC_RES.items():
                metric_match = pattern.search(line)
                if metric_match:
                    values[name] = float(metric_match.group(1))
            if len(values) == len(METRIC_RES):
                records[step] = values

    if not records:
        print(f"第 {args.m}~{args.n} 步没有找到数据")
        return

    missing = [s for s in range(args.m, args.n + 1) if s not in records]
    print(f"步数范围: {args.m}~{args.n}, 有效步数: {len(records)}")
    if missing:
        print(f"缺失步: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    for name in METRIC_RES:
        avg = sum(r[name] for r in records.values()) / len(records)
        print(f"{name} 平均值: {avg:.4f}")


if __name__ == "__main__":
    main()
