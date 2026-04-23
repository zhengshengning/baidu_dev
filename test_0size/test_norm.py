"""
对比 paddle.nn.functional.normalize 和 torch.nn.functional.normalize 在 0-size 场景下的计算结果
"""

import numpy as np
import paddle
import paddle.nn.functional as paddle_F

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn.functional as torch_F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch 未安装，将仅测试 PaddlePaddle")


def compare_normalize_0size():
    """测试 0-size 场景下两个框架的 normalize 行为"""
    
    print("=" * 80)
    print("paddle.nn.functional.normalize vs torch.nn.functional.normalize 0-size 场景对比")
    print("=" * 80)
    
    # 设置随机种子
    np.random.seed(42)
    paddle.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)
    
    test_cases = [
        # (shape, axis, p, description)
        # Case 1: 第一维度为 0
        ((0, 10), -1, 2, "shape=(0, 10), axis=-1, p=2"),
        ((0, 10), 0, 2, "shape=(0, 10), axis=0, p=2"),
        ((0, 10), 1, 2, "shape=(0, 10), axis=1, p=2"),
        
        # Case 2: 第二维度为 0
        ((4, 0), -1, 2, "shape=(4, 0), axis=-1, p=2"),
        ((4, 0), 0, 2, "shape=(4, 0), axis=0, p=2"),
        ((4, 0), 1, 2, "shape=(4, 0), axis=1, p=2"),
        
        # Case 3: 两个维度都为 0
        ((0, 0), -1, 2, "shape=(0, 0), axis=-1, p=2"),
        ((0, 0), 0, 2, "shape=(0, 0), axis=0, p=2"),
        
        # Case 4: 3D 输入，第一维为 0
        ((0, 5, 10), -1, 2, "shape=(0, 5, 10), axis=-1, p=2"),
        ((0, 5, 10), 1, 2, "shape=(0, 5, 10), axis=1, p=2"),
        
        # Case 5: 3D 输入，中间维度为 0
        ((4, 0, 10), -1, 2, "shape=(4, 0, 10), axis=-1, p=2"),
        ((4, 0, 10), 1, 2, "shape=(4, 0, 10), axis=1, p=2"),
        
        # Case 6: 3D 输入，最后维度为 0
        ((4, 5, 0), -1, 2, "shape=(4, 5, 0), axis=-1, p=2"),
        ((4, 5, 0), 2, 2, "shape=(4, 5, 0), axis=2, p=2"),
        
        # Case 7: 不同的 p 值
        ((0, 10), -1, 1, "shape=(0, 10), axis=-1, p=1 (L1 norm)"),
        ((0, 10), -1, float('inf'), "shape=(0, 10), axis=-1, p=inf"),
        ((4, 0), -1, 1, "shape=(4, 0), axis=-1, p=1 (L1 norm)"),
        
        # Case 8: 1D 输入
        ((0,), 0, 2, "shape=(0,), axis=0, p=2"),
    ]
    
    results = []
    
    for i, (shape, axis, p, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {desc}")
        print("=" * 70)
        
        # 准备数据
        if 0 not in shape:
            input_np = np.random.randn(*shape).astype(np.float32)
        else:
            input_np = np.empty(shape, dtype=np.float32)
        
        print(f"Input shape: {input_np.shape}")
        
        torch_result = None
        paddle_result = None
        torch_error = None
        paddle_error = None
        
        # ============ PyTorch 测试 ============
        print("\n--- PyTorch ---")
        if HAS_TORCH:
            try:
                torch_input = torch.tensor(input_np, dtype=torch.float32)
                # PyTorch normalize: dim 参数
                torch_output = torch_F.normalize(torch_input, p=p, dim=axis)
                print(f"  Output shape: {torch_output.shape}")
                print(f"  Output dtype: {torch_output.dtype}")
                if torch_output.numel() > 0:
                    print(f"  Output: {torch_output}")
                else:
                    print(f"  Output: (empty tensor)")
                torch_result = torch_output
            except Exception as e:
                torch_error = f"{type(e).__name__}: {e}"
                print(f"  Error: {torch_error}")
        else:
            print("  [PyTorch 未安装]")
        
        # ============ PaddlePaddle 测试 ============
        print("\n--- PaddlePaddle ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype='float32')
            # Paddle normalize: axis 参数
            paddle_output = paddle_F.normalize(paddle_input, p=p, axis=axis)
            print(f"  Output shape: {paddle_output.shape}")
            print(f"  Output dtype: {paddle_output.dtype}")
            if paddle_output.numel() > 0:
                print(f"  Output: {paddle_output}")
            else:
                print(f"  Output: (empty tensor)")
            paddle_result = paddle_output
        except Exception as e:
            paddle_error = f"{type(e).__name__}: {e}"
            print(f"  Error: {paddle_error}")
        
        # 比较结果
        print("\n--- 对比结果 ---")
        if torch_error and paddle_error:
            print(f"  两者都报错")
            match = "Both Error"
        elif torch_error:
            print(f"  PyTorch 报错，Paddle 成功")
            match = "Diff (Torch Error)"
        elif paddle_error:
            print(f"  Paddle 报错，PyTorch 成功")
            match = "Diff (Paddle Error)"
        elif torch_result is not None and paddle_result is not None:
            if list(torch_result.shape) == list(paddle_result.shape):
                print(f"  Shape 一致: {list(torch_result.shape)}")
                match = "Match"
            else:
                print(f"  Shape 不一致: PyTorch={list(torch_result.shape)}, Paddle={list(paddle_result.shape)}")
                match = "Shape Mismatch"
        else:
            match = "N/A"
        
        results.append({
            'case': desc,
            'shape': shape,
            'axis': axis,
            'p': p,
            'torch_error': torch_error,
            'paddle_error': paddle_error,
            'match': match
        })
    
    return results


def compare_normalize_with_eps():
    """测试带 eps 参数的 0-size 场景"""
    
    print("\n\n" + "=" * 80)
    print("带 eps 参数的 0-size 场景对比测试")
    print("=" * 80)
    
    test_cases = [
        # (shape, axis, p, eps, description)
        ((0, 10), -1, 2, 1e-12, "shape=(0, 10), eps=1e-12"),
        ((0, 10), -1, 2, 1e-6, "shape=(0, 10), eps=1e-6"),
        ((4, 0), -1, 2, 1e-12, "shape=(4, 0), eps=1e-12"),
    ]
    
    for i, (shape, axis, p, eps, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"EPS Test Case {i}: {desc}")
        print("=" * 60)
        
        input_np = np.empty(shape, dtype=np.float32)
        
        # PyTorch
        print("\n--- PyTorch ---")
        if HAS_TORCH:
            try:
                torch_input = torch.tensor(input_np, dtype=torch.float32)
                torch_output = torch_F.normalize(torch_input, p=p, dim=axis, eps=eps)
                print(f"  Output shape: {torch_output.shape}, dtype: {torch_output.dtype}")
            except Exception as e:
                print(f"  Error: {type(e).__name__}: {e}")
        else:
            print("  [PyTorch 未安装]")
        
        # PaddlePaddle
        print("\n--- PaddlePaddle ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype='float32')
            paddle_output = paddle_F.normalize(paddle_input, p=p, axis=axis, epsilon=eps)
            print(f"  Output shape: {paddle_output.shape}, dtype: {paddle_output.dtype}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


def compare_normalize_backward():
    """测试反向传播在 0-size 场景下的行为"""
    
    print("\n\n" + "=" * 80)
    print("反向传播在 0-size 场景下的对比测试")
    print("=" * 80)
    
    test_cases = [
        ((0, 10), -1, "shape=(0, 10), axis=-1"),
        ((4, 0), -1, "shape=(4, 0), axis=-1"),
        ((0, 0), -1, "shape=(0, 0), axis=-1"),
    ]
    
    for i, (shape, axis, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Backward Test Case {i}: {desc}")
        print("=" * 60)
        
        input_np = np.empty(shape, dtype=np.float32)
        
        # PyTorch
        print("\n--- PyTorch ---")
        if HAS_TORCH:
            try:
                torch_input = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)
                torch_output = torch_F.normalize(torch_input, p=2, dim=axis)
                
                # 尝试反向传播
                if torch_output.numel() > 0:
                    loss = torch_output.sum()
                    loss.backward()
                    print(f"  Forward output shape: {torch_output.shape}")
                    print(f"  Gradient shape: {torch_input.grad.shape if torch_input.grad is not None else 'None'}")
                else:
                    print(f"  Forward output shape: {torch_output.shape} (empty, skip backward)")
                    # 对空张量尝试 sum 和 backward
                    try:
                        loss = torch_output.sum()
                        loss.backward()
                        print(f"  Backward on empty: success, grad={torch_input.grad}")
                    except Exception as e:
                        print(f"  Backward on empty: {type(e).__name__}: {e}")
            except Exception as e:
                print(f"  Error: {type(e).__name__}: {e}")
        else:
            print("  [PyTorch 未安装]")
        
        # PaddlePaddle
        print("\n--- PaddlePaddle ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype='float32', stop_gradient=False)
            paddle_output = paddle_F.normalize(paddle_input, p=2, axis=axis)
            
            if paddle_output.numel() > 0:
                loss = paddle_output.sum()
                loss.backward()
                print(f"  Forward output shape: {paddle_output.shape}")
                print(f"  Gradient shape: {paddle_input.grad.shape if paddle_input.grad is not None else 'None'}")
            else:
                print(f"  Forward output shape: {paddle_output.shape} (empty, skip backward)")
                try:
                    loss = paddle_output.sum()
                    loss.backward()
                    print(f"  Backward on empty: success, grad={paddle_input.grad}")
                except Exception as e:
                    print(f"  Backward on empty: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


def print_summary(results):
    """打印测试结果总结"""
    
    print("\n\n" + "=" * 80)
    print("                         测 试 结 果 总 结")
    print("=" * 80)
    
    print("\n基础 normalize 测试结果:")
    print("-" * 80)
    print(f"{'Case':<50} {'Match Status':<20}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['case']:<50} {r['match']:<20}")
    
    # 统计
    match_count = sum(1 for r in results if r['match'] == 'Match')
    diff_count = sum(1 for r in results if 'Diff' in r['match'] or 'Mismatch' in r['match'])
    error_count = sum(1 for r in results if 'Error' in r['match'])
    
    print("-" * 80)
    print(f"\n统计:")
    print(f"  - 完全一致: {match_count}/{len(results)}")
    print(f"  - 存在差异: {diff_count}/{len(results)}")
    print(f"  - 双方报错: {error_count}/{len(results)}")
    
    print("""
关键发现总结:
1. 当 batch 维度为 0 时，两个框架通常都能正确返回空张量
2. 当 normalize 的目标 axis 维度为 0 时，行为可能存在差异
3. 不同的 p 值（L1, L2, Linf）在 0-size 场景下的行为需要特别关注
4. 反向传播在 0-size 场景下的行为也是测试重点

建议: 查看详细输出以了解具体差异
""")


def main():
    if HAS_TORCH:
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("PyTorch: 未安装")
    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # 基础 0-size 对比测试
    results = compare_normalize_0size()
    
    # eps 参数测试
    compare_normalize_with_eps()
    
    # 反向传播测试
    compare_normalize_backward()
    
    # 打印总结
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()