import numpy as np
import torch
import paddle
import random
from torch.optim.adamw import adamw

# import os
# os.environ["PADDLE_ADAMW_TORCH_COMPAT"] = "1"

paddle.set_device("gpu")

# 设置了 Paddle 和 PyTorch 打印张量时的显示格式
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

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_deterministic(SEED)


def paddle_device_to_torch_device(device: str) -> str:
    """Convert Paddle device name to PyTorch device name."""
    if device in ("gpu", "gpu:0"):
        return "cuda"
    return device


def compare_diff(np_pd, np_torch, name=""):
    diff = np_pd - np_torch
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))

    print(f"{name}, max abs diff:", max_abs)
    print(f"{name}, mean abs diff:", mean_abs)


def np_to_torch(np_data, dtype=torch.float, device="cuda") -> torch.Tensor:
    return torch.from_numpy(np_data).to(dtype).to(paddle_device_to_torch_device(device))


def torch_to_np(torch_data: torch.Tensor):
    return torch_data.float().detach().cpu().numpy()


def np_to_paddle(np_data, dtype="float32", device="gpu") -> paddle.Tensor:
    return paddle.to_tensor(np_data, dtype=dtype).to(device)


def paddle_to_np(paddle_data: paddle.Tensor):
    return paddle_data.astype("float32").detach().cpu().numpy()


def paddle_adamw_step(
    param_np,
    grad_np,
    moment1_np,
    moment2_np,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    beta1_pow_val=None,
    beta2_pow_val=None,
    device="gpu",
    amsgrad=False,
):
    """
    Use paddle._C_ops.adamw_ to perform one AdamW update step.
    Returns updated param, moment1, moment2 as numpy arrays.
    """
    param = np_to_paddle(param_np, "float32", device)
    grad = np_to_paddle(grad_np, "float32", device)
    moment1 = np_to_paddle(moment1_np, "float32", device)
    moment2 = np_to_paddle(moment2_np, "float32", device)
    moment2_max = paddle.zeros_like(param)

    lr = np_to_paddle(np.array([learning_rate]).astype("float64"), "float64", device)

    if beta1_pow_val is None:
        beta1_pow_val = beta1
    if beta2_pow_val is None:
        beta2_pow_val = beta2
    beta1_pow = np_to_paddle(np.array([beta1_pow_val]).astype("float32"), "float32", device)
    beta2_pow = np_to_paddle(np.array([beta2_pow_val]).astype("float32"), "float32", device)

    master_weight = None
    find_inf = None
    lazy_mode = False
    lr_ratio = 1.0
    # lr_ratio = learning_rate
    with_decay = True

    res = paddle._C_ops.adamw_(
        param,
        grad,
        lr,
        moment1,
        moment2,
        moment2_max,
        beta1_pow,
        beta2_pow,
        master_weight,
        find_inf,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        weight_decay,
        with_decay,
        lazy_mode,
        1000, # min_row_size_to_use_multithread
        False, # multi_precision
        False, # use_global_beta_pow
        amsgrad,
    )

    print("len(paddle._C_ops.adamw_):", len(res))

    return paddle_to_np(param), paddle_to_np(moment1), paddle_to_np(moment2)


def torch_adamw_step(
    param_np,
    grad_np,
    moment1_np,
    moment2_np,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    step=1,
    device="cuda",
):
    """
    Use adamw() functional API to perform one AdamW update step.
    Returns updated param, moment1, moment2 as numpy arrays.
    """
    param = np_to_torch(param_np, torch.float, device=device)
    grad = np_to_torch(grad_np, torch.float, device=device)
    exp_avg = np_to_torch(moment1_np, torch.float, device=device)
    exp_avg_sq = np_to_torch(moment2_np, torch.float, device=device)
    state_step = torch.tensor(
        float(step), device=paddle_device_to_torch_device(device)
    )

    adamw(
        params=[param],
        grads=[grad],
        exp_avgs=[exp_avg],
        exp_avg_sqs=[exp_avg_sq],
        max_exp_avg_sqs=[],
        state_steps=[state_step],

        fused=True,

        amsgrad=False,
        beta1=beta1,
        beta2=beta2,
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=epsilon,
        maximize=False,
    )

    param_out = torch_to_np(param)
    moment1_out = torch_to_np(exp_avg)
    moment2_out = torch_to_np(exp_avg_sq)

    return param_out, moment1_out, moment2_out


if __name__ == "__main__":
    device = "gpu"

    # --- Common AdamW hyperparameters ---
    learning_rate = 1e-5
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-8
    weight_decay = 0.01
    step = 1  # simulate the first step

    # beta1_pow and beta2_pow correspond to beta^step for paddle._C_ops.adamw_
    beta1_pow_val = beta1**step
    beta2_pow_val = beta2**step

    # --- Generate shared random inputs ---
    shape = (102, 105)
    param_np = np.random.uniform(-1, 1, shape).astype("float32")
    grad_np = np.random.uniform(-1, 1, shape).astype("float32")
    moment1_np = np.zeros(shape).astype("float32")
    moment2_np = np.zeros(shape).astype("float32")

    # ======== Test 1: Zero-initialized moments, single step update ========
    print("=" * 60)
    print("Test 1: AdamW step with zero-initialized moments (step=1)")
    print("=" * 60)

    # --- Paddle _C_ops.adamw_ ---
    pd_param, pd_m1, pd_m2 = paddle_adamw_step(
        param_np,
        grad_np,
        moment1_np,
        moment2_np,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        beta1_pow_val=beta1_pow_val,
        beta2_pow_val=beta2_pow_val,
        device=device,
    )

    # --- Torch AdamW ---
    # NOTE: torch's _fused_adam does torch._foreach_add_(state_steps, 1) before
    # the kernel call, so we pass step-1 so the kernel sees the same step as Paddle.
    th_param, th_m1, th_m2 = torch_adamw_step(
        param_np,
        grad_np,
        moment1_np,
        moment2_np,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        step=step - 1,
        device=device,
    )

    # --- Compare ---
    compare_diff(pd_param, th_param, "param_out (step=1)")
    compare_diff(pd_m1, th_m1, "moment1_out (step=1)")
    compare_diff(pd_m2, th_m2, "moment2_out (step=1)")
