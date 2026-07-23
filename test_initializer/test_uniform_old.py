import numpy as np
import torch
import paddle
import random

paddle.set_device("gpu")

# 设置 Paddle 和 PyTorch 打印张量时的显示格式
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


def paddle_device_to_torch_device(device: str) -> str:
    if device in ("gpu", "gpu:0"):
        return "cuda"
    return device


def compare_diff(np_pd, np_torch, name="", max_show=10):
    diff = np_pd - np_torch
    abs_diff = np.abs(diff)
    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)

    print(f"{name}, max abs diff:", max_abs)
    print(f"{name}, mean abs diff:", mean_abs)

    # Print positions and values where diff is non-zero
    nonzero_idx = np.argwhere(abs_diff > 0)
    n_diff = len(nonzero_idx)
    if n_diff == 0:
        print(f"{name}, all elements bit-identical")
        return

    print(f"{name}, {n_diff}/{abs_diff.size} elements differ "
          f"({100.0 * n_diff / abs_diff.size:.4f}%)")

    # Sort by abs diff descending; show top-K
    flat_abs = abs_diff.ravel()
    flat_order = np.argsort(-flat_abs)
    show = min(max_show, n_diff)
    print(f"{name}, top {show} positions by abs diff:")
    for i in range(show):
        flat_i = flat_order[i]
        idx = np.unravel_index(flat_i, abs_diff.shape)
        print(
            f"  index={tuple(int(x) for x in idx)}  "
            f"paddle={np_pd[idx]!r}  torch={np_torch[idx]!r}  "
            f"abs_diff={abs_diff[idx]!r}"
        )


def torch_to_np(torch_data: torch.Tensor):
    return torch_data.detach().cpu().numpy()


def paddle_to_np(paddle_data: paddle.Tensor):
    return paddle_data.detach().cpu().numpy()


def paddle_uniform_init(
    shape,
    low=-1.0,
    high=1.0,
    dtype="float32",
    device="gpu",
    seed=0,
):
    """
    Use paddle.nn.initializer.Uniform to initialize a tensor.
    Returns the initialized tensor as numpy array.
    """
    paddle.set_device(device)
    initializer = paddle.nn.initializer.Uniform(low=low, high=high)
    param = paddle.create_parameter(
        shape=shape,
        dtype=dtype,
        default_initializer=initializer,
    )
    return paddle_to_np(param)


def torch_uniform_init(
    shape,
    low=-1.0,
    high=1.0,
    dtype=torch.float32,
    device="cuda",
    seed=0,
):
    """
    Use torch.nn.init.uniform_ to initialize a tensor.
    Returns the initialized tensor as numpy array.
    """
    param = torch.empty(
        shape, dtype=dtype, device=paddle_device_to_torch_device(device)
    )
    torch.nn.init.uniform_(param, a=low, b=high)
    return torch_to_np(param)


if __name__ == "__main__":
    device = "gpu"

    # --- Common Uniform hyperparameters ---
    low = -5.0
    high = 5.0
    shape = [102, 105]

    # ======== Test 1: Uniform initialization, default range ========
    print("=" * 60)
    print("Test 1: Uniform init with low=-1.0, high=1.0")
    print("=" * 60)

    set_deterministic(SEED)
    pd_out = paddle_uniform_init(
        shape, low=low, high=high, dtype="float16", device=device, seed=SEED
    )

    set_deterministic(SEED)
    th_out = torch_uniform_init(
        shape, low=low, high=high, dtype=torch.float16, device=device, seed=SEED
    )

    compare_diff(pd_out, th_out, "uniform_out (low=-1, high=1)")

    # ======== Test 2: Uniform initialization, custom range ========
    print("=" * 60)
    print("Test 2: Uniform init with low=0.0, high=10.0")
    print("=" * 60)

    low2, high2 = 0.0, 10.0

    set_deterministic(SEED)
    pd_out2 = paddle_uniform_init(
        shape, low=low2, high=high2, dtype="float16", device=device, seed=SEED
    )

    set_deterministic(SEED)
    th_out2 = torch_uniform_init(
        shape, low=low2, high=high2, dtype=torch.float16, device=device, seed=SEED
    )

    compare_diff(pd_out2, th_out2, "uniform_out (low=0, high=10)")
    
