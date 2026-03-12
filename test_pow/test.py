import paddle
import torch
import numpy as np
import os

paddle.set_device("cpu")


def test_pow_diff(base_scalar, shape):
    # 1. 构造数据
    exponent_np = np.random.random(shape).astype(np.float32)

    # 2. PyTorch 基准 (Torch 通常两者一致)
    t_exp = torch.tensor(exponent_np)
    t_out_scalar = base_scalar ** t_exp

    # t_base_tensor = torch.tensor(base_scalar, dtype=t_exp.dtype)
    # t_out_tensor = t_base_tensor ** t_exp



    # 3. Paddle 测试
    p_exp = paddle.to_tensor(exponent_np)
    # 方式 A: Scalar ** Tensor (网络当前写法)
    p_out_scalar = base_scalar ** p_exp
    # 方式 B: Tensor ** Tensor (单测写法 / 推荐修复写法)
    # p_base_tensor = paddle.to_tensor(base_scalar, dtype=p_exp.dtype)
    # p_out_tensor = p_base_tensor ** p_exp


    # 5. 与 Torch 对比
    print("元素总个数: ", t_out_scalar.numel())
    # for idx in range(t_out_scalar.numel()):
    #     if ((p_out_scalar[idx].item() != t_out_scalar[idx].item())):
    #         print(f"Error Found at idx: {idx}, p_exp[{idx}] = {np.float32(p_exp[idx].item()).tolist()}")
    #         print(f"  p_out_scalar[{idx}] = {np.float32(p_out_scalar[idx].item()).tolist()} != t_out_scalar[{idx}] = {np.float32(t_out_scalar[idx].item()).tolist()}")
    #         break
    diff_scalar_vs_torch = np.abs(p_out_scalar.numpy() - t_out_scalar.numpy()).max()

    # print("p_out_scalar.device", p_out_scalar.device)
    # print("t_out_scalar.device", t_out_scalar.device)
    # # print("p_out_tensor.device", p_out_tensor.device)
    # # print("t_out_tensor.device", t_out_tensor.device)
    
    print("-" * 60)
    print(f"Scalar(Paddle) vs Torch Diff: {diff_scalar_vs_torch} <--- 现在的 Diff")
    # print(f"Tensor(Paddle) vs Torch Diff: {diff_tensor_vs_torch} <--- 修复后的 Diff")

    if(diff_scalar_vs_torch != 0):
        print("=" * 60)
        print("[❌] 存在 Diff！请查看以上输出分析原因")
        print("=" * 60)
        exit(1)
    

if __name__ == "__main__":
    # for i in range(1, 100):
    #     for j in range(1, 100):
    #         test_pow_diff(i * j * 1.0, (i, j))
    test_pow_diff(1000000, (1, 32))