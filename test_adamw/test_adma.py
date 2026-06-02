import numpy as np
import torch
import paddle
import random
from typing import cast, Optional, Union

from torch import Tensor
from torch.optim.adam import adam
from torch.optim.optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict,
    DeviceDtypeDict,
    Optimizer,
    ParamsT,
)

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


def compare_diff(np_pd, np_torch, name=""):
    diff = np_pd - np_torch
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))

    print(f"{name}, max abs diff:", max_abs)
    print(f"{name}, mean abs diff:", mean_abs)

def np_to_torch(np_data, dtype=torch.float, device="cuda") -> torch.Tensor:
    return torch.from_numpy(np_data).to(dtype).to(device)

def torch_to_np(torch_data: torch.Tensor):
    return torch_data.float().detach().cpu().numpy()

def np_to_paddle(np_data, dtype="float32", device="cuda") -> paddle.Tensor:
    return paddle.to_tensor(np_data, dtype=dtype).to(device)

def paddle_to_np(paddle_data: paddle.Tensor):
    return paddle_data.astype("float32").detach().cpu().numpy()



#####################################################################################################
#####################################################################################################


#####################################################################################################
#####################################################################################################


def paddle_adam_step(param_np, grad_np, moment1_np, moment2_np,
                     learning_rate=0.001, beta1=0.9, beta2=0.999,
                     epsilon=1e-8, beta1_pow_val=None, beta2_pow_val=None,
                     device="cuda"):
    """
    Use paddle._C_ops.adam_ to perform one Adam update step.
    Returns updated param, moment1, moment2 as numpy arrays.
    """
    param = np_to_paddle(param_np, "float32", device)
    grad = np_to_paddle(grad_np, "float32", device)
    moment1 = np_to_paddle(moment1_np, "float32", device)
    moment2 = np_to_paddle(moment2_np, "float32", device)
    moment2_max = paddle.zeros_like(param)

    lr = np_to_paddle(np.array([learning_rate]).astype("float32"), "float32", device)

    if beta1_pow_val is None:
        beta1_pow_val = beta1
    if beta2_pow_val is None:
        beta2_pow_val = beta2
    beta1_pow = np_to_paddle(np.array([beta1_pow_val]).astype("float32"), "float32", device)
    beta2_pow = np_to_paddle(np.array([beta2_pow_val]).astype("float32"), "float32", device)

    master_weight = None
    find_inf = None
    lazy_mode = False
    amsgrad = False

    res = paddle._C_ops.adam_(
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
        lazy_mode,
        1000,
        False,
        False,
        amsgrad,
    )

    print("len(paddle._C_ops.adam_):", len(res))

    return paddle_to_np(param), paddle_to_np(moment1), paddle_to_np(moment2)


def torch_adam_step(param_np, grad_np, moment1_np, moment2_np,
                    learning_rate=0.001, beta1=0.9, beta2=0.999,
                    epsilon=1e-8, step=1, device="cuda"):
    """
    Use adam() functional API to perform one Adam update step.
    Returns updated param, moment1, moment2 as numpy arrays.
    """
    param = np_to_torch(param_np, device=device)
    grad = np_to_torch(grad_np, device=device)
    exp_avg = np_to_torch(moment1_np, device=device)
    exp_avg_sq = np_to_torch(moment2_np, device=device)
    state_step = torch.tensor(float(step), device=device)

    adam(
        params=[param],
        grads=[grad],
        exp_avgs=[exp_avg],
        exp_avg_sqs=[exp_avg_sq],
        max_exp_avg_sqs=[],
        state_steps=[state_step],
        amsgrad=False,
        beta1=beta1,
        beta2=beta2,
        lr=learning_rate,
        weight_decay=0.0,
        eps=epsilon,
        maximize=False,
    )

    param_out = torch_to_np(param)
    moment1_out = torch_to_np(exp_avg)
    moment2_out = torch_to_np(exp_avg_sq)

    return param_out, moment1_out, moment2_out


if __name__ == "__main__":
    device = "cuda"

    # --- Common Adam hyperparameters ---
    learning_rate = 1e-05
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-08
    step = 1  # simulate the first step

    # beta1_pow and beta2_pow correspond to beta^step for paddle._C_ops.adam_
    beta1_pow_val = beta1 ** step
    beta2_pow_val = beta2 ** step

    # --- Generate shared random inputs ---
    shape = (102, 105)
    param_np = np.random.uniform(-1, 1, shape).astype("float32")
    grad_np = np.random.uniform(-1, 1, shape).astype("float32")
    moment1_np = np.zeros(shape).astype("float32")
    moment2_np = np.zeros(shape).astype("float32")


    # ======== Test 1 │ 零初始化 moments，单步更新，对比 param/moment1/moment2 ====
    print("=" * 60)
    print("Test 1: Adam step with zero-initialized moments (step=1)")
    print("=" * 60)

    # --- Paddle _C_ops.adam_ ---
    pd_param, pd_m1, pd_m2 = paddle_adam_step(
        param_np, grad_np, moment1_np, moment2_np,
        learning_rate=learning_rate, beta1=beta1, beta2=beta2,
        epsilon=epsilon, beta1_pow_val=beta1_pow_val,
        beta2_pow_val=beta2_pow_val, device=device,
    )

    # --- Torch Adam ---
    th_param, th_m1, th_m2 = torch_adam_step(
        param_np, grad_np, moment1_np, moment2_np,
        learning_rate=learning_rate, beta1=beta1, beta2=beta2,
        epsilon=epsilon, step=step, device=device,
    )

    # --- Compare ---
    compare_diff(pd_param, th_param, "param_out (step=1)")
    compare_diff(pd_m1, th_m1, "moment1_out (step=1)")
    compare_diff(pd_m2, th_m2, "moment2_out (step=1)")
