import math
import numpy as np
import torch
import paddle
import random
from paddle import _C_ops
from paddle.framework import _current_expected_place
from paddle.nn.initializer.kaiming import calculate_gain

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


def paddle_kaiming_uniform_init(
    shape,
    fan_in=None,
    negative_slope=0.0,
    nonlinearity="leaky_relu",
    dtype="float32",
    device="gpu",
):
    """
    Directly call paddle._C_ops.uniform with Kaiming-uniform bound.
    Equivalent to paddle.nn.initializer.KaimingUniform in dynamic graph mode.

    KaimingUniform (mode='fan_in') computes:
        gain  = calculate_gain(nonlinearity, negative_slope)
        limit = gain * sqrt(3 / fan_in)

    NOTE on fan_in convention for 2D weight [D0, D1]:
      - paddle._compute_fans treats it as [in, out] -> fan_in = D0
      - torch._calculate_fan_in_and_fan_out treats it as [out, in] -> fan_in = D1
    To bit-align with torch.nn.init.kaiming_uniform_(mode='fan_in'), this
    helper defaults fan_in to shape[1] (the torch convention). Pass fan_in
    explicitly to override.
    """
    paddle.set_device(device)
    out_dtype = paddle.framework.convert_np_dtype_to_dtype_(np.dtype(dtype))
    place = _current_expected_place()

    if fan_in is None:
        fan_in = shape[1] if len(shape) > 1 else shape[0]

    gain = calculate_gain(nonlinearity, negative_slope)
    limit = gain * math.sqrt(3.0 / float(fan_in))

    param = _C_ops.uniform(
        list(shape),
        out_dtype,
        float(-limit),
        float(limit),
        0,  # seed=0 -> default generator
        place,
    )
    return paddle_to_np(param)


def torch_kaiming_uniform_init(
    shape,
    a=0.0,
    mode="fan_in",
    nonlinearity="leaky_relu",
    dtype=torch.float32,
    device="cuda",
):
    param = torch.empty(
        shape, dtype=dtype, device=paddle_device_to_torch_device(device)
    )
    torch.nn.init.kaiming_uniform_(
        param, a=a, mode=mode, nonlinearity=nonlinearity
    )
    return torch_to_np(param)


if __name__ == "__main__":
    device = "gpu"
    shape = [102, 105]

    # ======== Test 1: KaimingUniform default (leaky_relu, slope=0) ========
    print("=" * 60)
    print("Test 1: KaimingUniform init with negative_slope=0.0, nonlinearity=leaky_relu")
    print("=" * 60)

    set_deterministic(SEED)
    pd_out = paddle_kaiming_uniform_init(
        shape, negative_slope=0.0, nonlinearity="leaky_relu", device=device
    )

    set_deterministic(SEED)
    th_out = torch_kaiming_uniform_init(
        shape, a=0.0, mode="fan_in", nonlinearity="leaky_relu", device=device
    )

    compare_diff(pd_out, th_out, "kaiming_uniform_out (slope=0.0, leaky_relu)")

    # ======== Test 2: KaimingUniform with relu nonlinearity ========
    print("=" * 60)
    print("Test 2: KaimingUniform init with nonlinearity=relu")
    print("=" * 60)

    set_deterministic(SEED)
    pd_out2 = paddle_kaiming_uniform_init(
        shape, negative_slope=0.0, nonlinearity="relu", device=device
    )

    set_deterministic(SEED)
    th_out2 = torch_kaiming_uniform_init(
        shape, a=0.0, mode="fan_in", nonlinearity="relu", device=device
    )

    compare_diff(pd_out2, th_out2, "kaiming_uniform_out (relu)")

    # ======== Test 3: KaimingUniform with negative_slope=math.sqrt(5) ========
    # This matches the default initialization in torch.nn.Linear.
    print("=" * 60)
    print("Test 3: KaimingUniform init with negative_slope=sqrt(5), leaky_relu")
    print("=" * 60)

    a = math.sqrt(5)

    set_deterministic(SEED)
    pd_out3 = paddle_kaiming_uniform_init(
        shape, negative_slope=a, nonlinearity="leaky_relu", device=device
    )

    set_deterministic(SEED)
    th_out3 = torch_kaiming_uniform_init(
        shape, a=a, mode="fan_in", nonlinearity="leaky_relu", device=device
    )

    compare_diff(pd_out3, th_out3, "kaiming_uniform_out (slope=sqrt(5))")
