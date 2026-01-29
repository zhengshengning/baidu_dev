import paddle
import torch
import numpy as np
import os

paddle.set_device("cpu")

def test_pow_diff():
    print("=" * 60)
    print("[Test] 复现 Scalar ** Tensor 与 Tensor ** Tensor 的差异")
    print("=" * 60)

    # 1. 构造数据
    base_scalar = 1000000
    # exponent_np = np.array([0.015625, 0.5, 0.984375], dtype=np.float32)
    file_path = "./exponent.npy"
    print(f"Loading exponent from: {file_path}")
    if not os.path.exists(file_path):
        print("❌ 错误：找不到文件，请确认路径是否正确。")
        return

    exponent_np = np.load(file_path)
    print(f"Data Loaded. Shape: {exponent_np.shape}")

    # 2. PyTorch 基准 (Torch 通常两者一致)
    t_exp = torch.tensor(exponent_np)
    t_out_scalar = base_scalar ** t_exp
    t_out_tensor = torch.tensor(base_scalar) ** t_exp
    
    # print(f"[Torch] Scalar ** Tensor: {t_out_scalar.numpy()}")
    # print(f"[Torch] Tensor ** Tensor: {t_out_tensor.numpy()}")
    # print(f"[Torch] Diff: {(t_out_scalar - t_out_tensor).abs().max().item()}\n")

    # 3. Paddle 测试
    
    p_exp = paddle.to_tensor(exponent_np)
    
    # 方式 A: Scalar ** Tensor (网络当前写法)
    p_out_scalar = base_scalar ** p_exp
    
    # 方式 B: Tensor ** Tensor (单测写法 / 推荐修复写法)
    p_base_tensor = paddle.to_tensor(base_scalar, dtype=p_exp.dtype)
    p_out_tensor = p_base_tensor ** p_exp
    
    
    # 4. 计算 Paddle 内部差异
    diff_internal = (p_out_scalar - p_out_tensor).abs().max().item()
    print(f"[Paddle] Internal Diff: {diff_internal}")

    # 5. 与 Torch 对比
    diff_scalar_vs_torch = np.abs(p_out_scalar.numpy() - t_out_scalar.numpy()).max()
    diff_tensor_vs_torch = np.abs(p_out_tensor.numpy() - t_out_tensor.numpy()).max()

    print("-" * 60)
    print(f"Scalar(Paddle) vs Torch Diff: {diff_scalar_vs_torch} <--- 现在的 Diff")
    print(f"Tensor(Paddle) vs Torch Diff: {diff_tensor_vs_torch} <--- 修复后的 Diff")
    
    # if diff_tensor_vs_torch < diff_scalar_vs_torch:
    #     print("\n✅ 结论: 将 base 转为 Tensor 计算可以显著降低与 Torch 的 Diff！")
    # else:
    #     print("\n❌ 结论: 转换 Tensor 并没有帮助，可能不是这个原因。")


if __name__ == "__main__":
    test_pow_diff()
    # test_matmul_diff()