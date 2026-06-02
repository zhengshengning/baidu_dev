import numpy as np
import torch
import paddle
import random

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


def compare_diff(np_pd, np_torch, name=""):
    diff = np_pd - np_torch
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))

    print(f"{name}, max abs diff:", max_abs)
    print(f"{name}, mean abs diff:", mean_abs)


if __name__ == "__main__":
    device = "gpu"

    # --- Common AdamW hyperparameters ---
    learning_rate = 1e-5
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-8
    weight_decay = 0.01

    total_steps = 50

    # --- Generate shared random inputs ---
    shape = (102, 105)
    param_np = np.random.uniform(-1, 1, shape).astype("float32")

    # Pre-generate gradients for all steps (use different random grads each step)
    np.random.seed(SEED + 1000)
    grad_all = [
        np.random.uniform(-1, 1, shape).astype("float32")
        for _ in range(total_steps)
    ]

    # ======== Setup Paddle optimizer ========
    pd_param = paddle.to_tensor(param_np.copy(), dtype="float32", place=paddle.CUDAPlace(0))
    pd_param.stop_gradient = False

    pd_optimizer = paddle.optimizer.AdamW(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        parameters=[pd_param],
    )

    # ======== Setup PyTorch optimizer ========
    th_param = torch.tensor(param_np.copy(), dtype=torch.float32, device="cuda", requires_grad=True)

    th_optimizer = torch.optim.AdamW(
        [th_param],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
        fused=True,
    )

    # ======== Multi-step AdamW comparison (step 1 ~ 50) ========
    print("=" * 80)
    print(f"AdamW precision comparison: Paddle vs PyTorch over {total_steps} steps")
    print(f"  Interface: paddle.optimizer.AdamW vs torch.optim.AdamW(fused=True)")
    print(f"  Shape: {shape}, lr={learning_rate}, beta1={beta1}, beta2={beta2}")
    print("=" * 80)
    print(f"{'step':>4}  {'param max_abs':>14}  {'param mean_abs':>15}  "
          f"{'m1 max_abs':>12}  {'m2 max_abs':>12}")
    print("-" * 80)

    for step in range(1, total_steps + 1):
        grad_np = grad_all[step - 1]

        # --- Paddle step ---
        pd_param.grad = paddle.to_tensor(
            grad_np, dtype="float32", place=paddle.CUDAPlace(0)
        )
        pd_optimizer.step()
        pd_optimizer.clear_grad()

        # --- PyTorch step ---
        th_param.grad = torch.tensor(grad_np, dtype=torch.float32, device="cuda")
        th_optimizer.step()
        th_optimizer.zero_grad()

        # --- Extract numpy for comparison ---
        pd_param_np = pd_param.numpy().astype(np.float32)
        th_param_np = th_param.detach().cpu().numpy().astype(np.float32)

        # Extract moments from optimizer state for comparison
        # Paddle: optimizer state is stored internally
        pd_state = pd_optimizer._accumulators
        pd_m1_np = pd_state['moment1'][pd_param.name].numpy().astype(np.float32)
        pd_m2_np = pd_state['moment2'][pd_param.name].numpy().astype(np.float32)

        # PyTorch: optimizer state dict
        th_state = th_optimizer.state[th_param]
        th_m1_np = th_state['exp_avg'].detach().cpu().numpy().astype(np.float32)
        th_m2_np = th_state['exp_avg_sq'].detach().cpu().numpy().astype(np.float32)

        # --- Compare ---
        param_diff = np.abs(pd_param_np - th_param_np)
        m1_diff = np.abs(pd_m1_np - th_m1_np)
        m2_diff = np.abs(pd_m2_np - th_m2_np)

        param_max = np.max(param_diff)
        param_mean = np.mean(param_diff)
        m1_max = np.max(m1_diff)
        m2_max = np.max(m2_diff)

        print(f"{step:>4}  {param_max:>14.6e}  {param_mean:>15.6e}  "
              f"{m1_max:>12.6e}  {m2_max:>12.6e}")

    # --- Final summary ---
    print("=" * 80)
    print(f"Final state after step {total_steps}:")
    compare_diff(pd_param_np, th_param_np, "param_out")
    compare_diff(pd_m1_np, th_m1_np, "moment1_out")
    compare_diff(pd_m2_np, th_m2_np, "moment2_out")
