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
    if torch_data.dtype in (torch.bfloat16, torch.float16):
        torch_data = torch_data.float()
    return torch_data.detach().cpu().numpy()


def paddle_to_np(paddle_data: paddle.Tensor):
    if paddle_data.dtype in (paddle.bfloat16, paddle.float16):
        paddle_data = paddle_data.astype("float32")
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


if __name__ == "__main__":
    device = "gpu"
    shape = [102, 105]

    # ======== Test 1: XavierNormal default gain=1.0 ========
    print("=" * 60)
    print("Test 1: XavierNormal init with gain=1.0")
    print("=" * 60)

    set_deterministic(SEED)
    pd_out = paddle_xavier_normal_init(shape, gain=1.0, dtype="float64", device=device)

    set_deterministic(SEED)
    th_out = torch_xavier_normal_init(shape, gain=1.0, dtype=torch.float64, device=device)

    compare_diff(pd_out, th_out, "xavier_normal_out (gain=1.0)")

    # ======== Test 2: XavierNormal gain=2.0 ========
    print("=" * 60)
    print("Test 2: XavierNormal init with gain=2.0")
    print("=" * 60)

    set_deterministic(SEED)
    pd_out2 = paddle_xavier_normal_init(shape, gain=2.0, dtype="float64", device=device)

    set_deterministic(SEED)
    th_out2 = torch_xavier_normal_init(shape, gain=2.0, dtype=torch.float64, device=device)

    compare_diff(pd_out2, th_out2, "xavier_normal_out (gain=2.0)")
