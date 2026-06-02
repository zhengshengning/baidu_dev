import numpy as np
import torch
import paddle
import random
from paddle import _C_ops
from paddle.framework import _current_expected_place

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


def compare_diff(np_pd, np_torch, name=""):
    diff = np_pd - np_torch
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))

    print(f"{name}, max abs diff:", max_abs)
    print(f"{name}, mean abs diff:", mean_abs)


def torch_to_np(torch_data: torch.Tensor):
    return torch_data.float().detach().cpu().numpy()


def paddle_to_np(paddle_data: paddle.Tensor):
    return paddle_data.astype("float32").detach().cpu().numpy()


def paddle_uniform_init(
    shape,
    low=-1.0,
    high=1.0,
    dtype="float32",
    device="gpu",
    seed=0,
):
    """
    Directly call paddle._C_ops.uniform, which is the underlying op used by
    paddle.nn.initializer.Uniform in dynamic graph mode.
    """
    paddle.set_device(device)
    out_dtype = paddle.framework.convert_np_dtype_to_dtype_(np.dtype(dtype))
    place = _current_expected_place()
    param = _C_ops.uniform(
        list(shape),
        out_dtype,
        float(low),
        float(high),
        0,
        place,
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
