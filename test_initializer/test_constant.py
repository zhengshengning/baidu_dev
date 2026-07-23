import numpy as np
import torch
import paddle
import random
from paddle import _C_ops
from paddle.framework import _current_expected_place

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


def paddle_constant_init(
    shape,
    value=0.0,
    dtype="float32",
    device="gpu",
):
    """
    Directly call paddle._C_ops.full, which is the underlying op used by
    paddle.nn.initializer.Constant in dynamic graph mode.
    Signature: full(shape, value, dtype, place)
    """
    paddle.set_device(device)
    out_dtype = paddle.framework.convert_np_dtype_to_dtype_(np.dtype(dtype))
    place = _current_expected_place()
    param = _C_ops.full(
        list(shape),
        float(value),
        out_dtype,
        place,
    )
    return paddle_to_np(param)


def torch_constant_init(
    shape,
    value=0.0,
    dtype=torch.float32,
    device="cuda",
):
    param = torch.empty(
        shape, dtype=dtype, device=paddle_device_to_torch_device(device)
    )
    torch.nn.init.constant_(param, val=value)
    return torch_to_np(param)


if __name__ == "__main__":
    device = "gpu"
    shape = [102, 105]

    # ======== Test 1: Constant 0.0 ========
    print("=" * 60)
    print("Test 1: Constant init with value=0.0")
    print("=" * 60)

    set_deterministic(SEED)
    pd_out = paddle_constant_init(shape, value=0.0, device=device)

    set_deterministic(SEED)
    th_out = torch_constant_init(shape, value=0.0, device=device)

    compare_diff(pd_out, th_out, "constant_out (value=0.0)")

    # ======== Test 2: Constant 3.14 ========
    print("=" * 60)
    print("Test 2: Constant init with value=3.14")
    print("=" * 60)

    set_deterministic(SEED)
    pd_out2 = paddle_constant_init(shape, value=3.14, device=device)

    set_deterministic(SEED)
    th_out2 = torch_constant_init(shape, value=3.14, device=device)

    compare_diff(pd_out2, th_out2, "constant_out (value=3.14)")
