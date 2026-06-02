import csv
import os
import numpy as np
import torch
import paddle
import random

paddle.set_device("gpu")

precision = 15
paddle.set_printoptions(precision=precision)
torch.set_printoptions(
    precision=precision,
    sci_mode=False,
    linewidth=200,
    threshold=10_000,
)


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    paddle.seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_deterministic(SEED)


# Mapping between paddle / torch / numpy dtypes for the dtypes we want to test.
DTYPE_MAP = {
    "float32": (torch.float32, np.float32),
    "float64": (torch.float64, np.float64),
    "float16": (torch.float16, np.float16),
    "bfloat16": (torch.bfloat16, None),  # numpy has no native bf16
}


def paddle_device_to_torch_device(device: str) -> str:
    if device in ("gpu", "gpu:0"):
        return "cuda"
    return device


def torch_to_np(torch_data: torch.Tensor):
    return torch_data.detach().cpu().numpy()


def paddle_to_np(paddle_data: paddle.Tensor):
    return paddle_data.detach().cpu().numpy()


def paddle_xavier_normal_init(
    shape,
    fan_in=None,
    fan_out=None,
    gain=1.0,
    dtype="float32",
    device="gpu",
):
    paddle.set_device(device)
    initializer = paddle.nn.initializer.XavierNormal(
        fan_in=fan_in, fan_out=fan_out, gain=gain
    )
    param = paddle.create_parameter(
        shape=shape,
        dtype=dtype,
        default_initializer=initializer,
    )
    return paddle_to_np(param)


def torch_xavier_normal_init(
    shape,
    gain=1.0,
    dtype=torch.float32,
    device="cuda",
):
    param = torch.empty(
        shape, dtype=dtype, device=paddle_device_to_torch_device(device)
    )
    torch.nn.init.xavier_normal_(param, gain=gain)
    return torch_to_np(param)


def run_one_case(shape, gain, dtype_str, device="gpu"):
    torch_dtype, _ = DTYPE_MAP[dtype_str]

    set_deterministic(SEED)
    pd_out = paddle_xavier_normal_init(
        shape, gain=gain, dtype=dtype_str, device=device
    )

    set_deterministic(SEED)
    th_out = torch_xavier_normal_init(
        shape, gain=gain, dtype=torch_dtype, device=device
    )

    diff = np.abs(pd_out - th_out)
    return {
        "shape": tuple(shape),
        "dtype": dtype_str,
        "gain": gain,
        "numel": int(np.prod(shape)),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "pd_mean": float(pd_out.mean()),
        "th_mean": float(th_out.mean()),
    }


if __name__ == "__main__":
    device = "gpu"

    # XavierNormal needs at least 2D (fan_in, fan_out)
    SHAPES = [
        [32, 32],
        [102, 105],
        [128, 256],
        [512, 512],
        [4, 8, 16],
        [2, 3, 4, 5],
        [8, 16, 32, 64],
        [1, 1],         # edge: trivial 2D
        [1, 1024],      # edge: very asymmetric
        [1024, 1],      # edge: very asymmetric (transposed)
    ]
    DTYPES = ["float32", "float64", "float16", "bfloat16"]
    GAINS = [1.0, 2.0, 0.5]

    results = []
    for dtype_str in DTYPES:
        for shape in SHAPES:
            for gain in GAINS:
                try:
                    r = run_one_case(shape, gain, dtype_str, device)
                    r["status"] = "ok"
                except Exception as e:
                    r = {
                        "shape": tuple(shape),
                        "dtype": dtype_str,
                        "gain": gain,
                        "numel": int(np.prod(shape)),
                        "status": f"error: {type(e).__name__}: {e}",
                    }
                results.append(r)

    # ------------ Per-case table ------------
    header = (
        f"{'dtype':<10}{'shape':<22}{'gain':<10}"
        f"{'numel':>8}{'max_abs':>14}{'mean_abs':>14}{'status':>10}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        gain_str = f"{r['gain']}"
        if r["status"] == "ok":
            line = (
                f"{r['dtype']:<10}"
                f"{str(r['shape']):<22}"
                f"{gain_str:<10}"
                f"{r['numel']:>8}"
                f"{r['max_abs_diff']:>14.6g}"
                f"{r['mean_abs_diff']:>14.6g}"
                f"{'ok':>10}"
            )
        else:
            line = (
                f"{r['dtype']:<10}"
                f"{str(r['shape']):<22}"
                f"{gain_str:<10}"
                f"{r['numel']:>8}"
                f"{'-':>14}{'-':>14}"
                f"{r['status']:>10}"
            )
        print(line)

    # ------------ Aggregated stats ------------
    print()
    print("=" * 80)
    print("Aggregated statistics")
    print("=" * 80)

    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]
    bit_exact = [r for r in ok if r["max_abs_diff"] == 0.0]
    nonzero = [r for r in ok if r["max_abs_diff"] != 0.0]

    print(f"total cases     : {len(results)}")
    print(f"  ok            : {len(ok)}")
    print(f"  bit-identical : {len(bit_exact)}  ({100*len(bit_exact)/max(1,len(ok)):.1f}% of ok)")
    print(f"  with diff     : {len(nonzero)}")
    print(f"  errored       : {len(err)}")

    if nonzero:
        max_overall = max(r["max_abs_diff"] for r in nonzero)
        mean_overall = np.mean([r["mean_abs_diff"] for r in nonzero])
        worst = max(nonzero, key=lambda r: r["max_abs_diff"])
        print(f"\nNon-zero diff summary:")
        print(f"  worst max_abs_diff   : {max_overall:.6g}")
        print(f"  avg of mean_abs_diff : {mean_overall:.6g}")
        print(
            f"  worst case           : "
            f"dtype={worst['dtype']} shape={worst['shape']} "
            f"gain={worst['gain']} -> max={worst['max_abs_diff']:.6g}"
        )

    # Per-dtype breakdown
    print("\nPer-dtype breakdown:")
    for dt in DTYPES:
        sub = [r for r in ok if r["dtype"] == dt]
        if not sub:
            continue
        be = sum(1 for r in sub if r["max_abs_diff"] == 0.0)
        worst = max(sub, key=lambda r: r["max_abs_diff"])
        print(
            f"  {dt:<10} cases={len(sub):>3}  bit-exact={be:>3}/{len(sub)}  "
            f"worst max_abs={worst['max_abs_diff']:.6g}"
        )

    if err:
        print("\nErrors:")
        for r in err:
            print(f"  dtype={r['dtype']} shape={r['shape']} gain={r['gain']} -> {r['status']}")

    # ------------ Save results to CSV ------------
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_xavier_normal_old.csv")
    fields = [
        "dtype", "shape", "gain", "numel",
        "max_abs_diff", "mean_abs_diff",
        "pd_mean", "th_mean",
        "status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                "dtype": r["dtype"],
                "shape": "x".join(str(s) for s in r["shape"]),
                "gain": r["gain"],
                "numel": r["numel"],
                "max_abs_diff": r.get("max_abs_diff", ""),
                "mean_abs_diff": r.get("mean_abs_diff", ""),
                "pd_mean": r.get("pd_mean", ""),
                "th_mean": r.get("th_mean", ""),
                "status": r["status"],
            }
            writer.writerow(row)
    print(f"\nResults saved to: {csv_path}")
