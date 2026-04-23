"""
对比 paddle.tril 和 torch.tril 在 0-size 场景下的计算结果
"""

import numpy as np
import paddle
import torch

def compare_tril_0size():
    """测试 0-size 场景下两个框架的 tril 行为"""
    
    print("=" * 80)
    print("paddle.tril vs torch.tril 0-size 场景对比")
    print("=" * 80)
    
    # 设置随机种子
    np.random.seed(42)
    paddle.seed(42)
    torch.manual_seed(42)
    
    test_cases = [
        # (shape, diagonal, description)
        # Case 1: 2D 输入，行数为 0
        ((0, 5), 0, "2D input: shape=(0, 5), diagonal=0"),
        ((0, 5), 1, "2D input: shape=(0, 5), diagonal=1"),
        ((0, 5), -1, "2D input: shape=(0, 5), diagonal=-1"),
        
        # Case 2: 2D 输入，列数为 0
        ((4, 0), 0, "2D input: shape=(4, 0), diagonal=0"),
        ((4, 0), 1, "2D input: shape=(4, 0), diagonal=1"),
        ((4, 0), -1, "2D input: shape=(4, 0), diagonal=-1"),
        
        # Case 3: 2D 输入，两个维度都为 0
        ((0, 0), 0, "2D input: shape=(0, 0), diagonal=0"),
        
        # Case 4: 3D 输入，batch 为 0
        ((0, 3, 3), 0, "3D input: shape=(0, 3, 3), diagonal=0"),
        ((0, 3, 3), 1, "3D input: shape=(0, 3, 3), diagonal=1"),
        
        # Case 5: 3D 输入，中间维度为 0
        ((2, 0, 4), 0, "3D input: shape=(2, 0, 4), diagonal=0"),
        
        # Case 6: 3D 输入，最后维度为 0
        ((2, 3, 0), 0, "3D input: shape=(2, 3, 0), diagonal=0"),
    ]
    
    for i, (shape, diagonal, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {desc}")
        print("=" * 70)
        
        # 准备数据
        if 0 not in shape:
            input_np = np.random.randn(*shape).astype(np.float32)
        else:
            input_np = np.empty(shape, dtype=np.float32)
        
        print(f"Input shape: {input_np.shape}, diagonal: {diagonal}")
        
        # ============ PyTorch 测试 ============
        print("\n--- PyTorch ---")
        try:
            torch_input = torch.tensor(input_np, dtype=torch.float32)
            torch_output = torch.tril(torch_input, diagonal=diagonal)
            print(f"  Output shape: {torch_output.shape}")
            print(f"  Output dtype: {torch_output.dtype}")
            if torch_output.numel() > 0 and torch_output.numel() <= 20:
                print(f"  Output:\n{torch_output}")
            else:
                print(f"  Output: (numel={torch_output.numel()})")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")
        
        # ============ PaddlePaddle 测试 ============
        print("\n--- PaddlePaddle ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype='float32')
            paddle_output = paddle.tril(paddle_input, diagonal=diagonal)
            print(f"  Output shape: {paddle_output.shape}")
            print(f"  Output dtype: {paddle_output.dtype}")
            if paddle_output.numel() > 0 and paddle_output.numel() <= 20:
                print(f"  Output:\n{paddle_output}")
            else:
                print(f"  Output: (numel={paddle_output.numel()})")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")
        
        # 比较结果
        print("\n--- 对比结果 ---")
        if torch_output.shape == paddle_output.shape:
            print(f"  Shape 一致: {list(torch_output.shape)}")
            if torch_output.numel() > 0:
                torch_np = torch_output.numpy()
                paddle_np = paddle_output.numpy()
                if np.allclose(torch_np, paddle_np):
                    print("  数值一致")
                else:
                    print("  数值不一致")
            else:
                print("  空张量，无需比较数值")
        else:
            print(f"  Shape 不一致: PyTorch={list(torch_output.shape)}, Paddle={list(paddle_output.shape)}")

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    compare_tril_0size()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()