"""
对比 paddle.nn.functional.one_hot 和 torch.nn.functional.one_hot 在 0-size 场景下的计算结果
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


def compare_one_hot_0size():
    """测试 0-size 场景下两个框架的 one_hot 行为"""
    
    print("=" * 80)
    print("paddle.nn.functional.one_hot vs torch.nn.functional.one_hot 0-size 场景对比")
    print("=" * 80)
    
    # 设置随机种子
    np.random.seed(42)
    paddle.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)
    
    test_cases = [
        # (input_shape, num_classes, description)
        # Case 1: 1D 输入，元素个数为 0
        ((0,), 10, "1D input: shape=(0,), num_classes=10"),
        ((0,), 5, "1D input: shape=(0,), num_classes=5"),
        ((0,), 0, "1D input: shape=(0,), num_classes=0"),
        
        # Case 2: 2D 输入，第一维为 0
        ((0, 5), 10, "2D input: shape=(0, 5), num_classes=10"),
        
        # Case 3: 2D 输入，第二维为 0
        ((4, 0), 10, "2D input: shape=(4, 0), num_classes=10"),
        
        # Case 4: 2D 输入，两个维度都为 0
        ((0, 0), 10, "2D input: shape=(0, 0), num_classes=10"),
        
        # Case 5: 3D 输入
        ((0, 3, 4), 10, "3D input: shape=(0, 3, 4), num_classes=10"),
        ((2, 0, 4), 10, "3D input: shape=(2, 0, 4), num_classes=10"),
        ((2, 3, 0), 10, "3D input: shape=(2, 3, 0), num_classes=10"),
        
        # Case 6: num_classes = 0
        ((5,), 0, "1D input: shape=(5,), num_classes=0"),
        ((3, 4), 0, "2D input: shape=(3, 4), num_classes=0"),
        
        # Case 7: 单元素但 num_classes=0
        ((1,), 0, "1D input: shape=(1,), num_classes=0"),
    ]
    
    results = []
    
    for i, (shape, num_classes, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {desc}")
        print("=" * 70)
        
        # 准备数据
        if 0 not in shape and num_classes > 0:
            # 有效数据：值在 [0, num_classes) 范围内
            input_np = np.random.randint(0, num_classes, size=shape).astype(np.int64)
        elif 0 not in shape:
            # shape 非空但 num_classes=0，创建随机数据（会导致越界）
            input_np = np.zeros(shape, dtype=np.int64)
        else:
            # shape 中有 0，创建空数组
            input_np = np.empty(shape, dtype=np.int64)
        
        print(f"Input shape: {input_np.shape}, num_classes: {num_classes}")
        
        torch_result = None
        paddle_result = None
        torch_error = None
        paddle_error = None
        
        # ============ PyTorch 测试 ============
        print("\n--- PyTorch ---")
        if HAS_TORCH:
            try:
                torch_input = torch.tensor(input_np, dtype=torch.int64)
                torch_output = torch_F.one_hot(torch_input, num_classes=num_classes)
                print(f"  Output shape: {torch_output.shape}")
                print(f"  Output dtype: {torch_output.dtype}")
                if torch_output.numel() > 0 and torch_output.numel() <= 20:
                    print(f"  Output:\n{torch_output}")
                elif torch_output.numel() > 20:
                    print(f"  Output: (too large to display, numel={torch_output.numel()})")
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
            paddle_input = paddle.to_tensor(input_np, dtype='int64')
            paddle_output = paddle_F.one_hot(paddle_input, num_classes=num_classes)
            print(f"  Output shape: {paddle_output.shape}")
            print(f"  Output dtype: {paddle_output.dtype}")
            if paddle_output.numel() > 0 and paddle_output.numel() <= 20:
                print(f"  Output:\n{paddle_output}")
            elif paddle_output.numel() > 20:
                print(f"  Output: (too large to display, numel={paddle_output.numel()})")
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
            torch_shape = list(torch_result.shape)
            paddle_shape = list(paddle_result.shape)
            if torch_shape == paddle_shape:
                print(f"  Shape 一致: {torch_shape}")
                # 检查值是否一致
                if torch_result.numel() > 0:
                    torch_np = torch_result.numpy()
                    paddle_np = paddle_result.numpy()
                    if np.allclose(torch_np, paddle_np):
                        print(f"  Value 一致")
                        match = "Match"
                    else:
                        print(f"  Value 不一致")
                        match = "Value Mismatch"
                else:
                    match = "Match (empty)"
            else:
                print(f"  Shape 不一致: PyTorch={torch_shape}, Paddle={paddle_shape}")
                match = "Shape Mismatch"
        else:
            match = "N/A"
        
        results.append({
            'case': desc,
            'shape': shape,
            'num_classes': num_classes,
            'torch_error': torch_error,
            'paddle_error': paddle_error,
            'match': match
        })
    
    return results


def compare_one_hot_dtype():
    """测试不同输入类型在 0-size 场景下的行为"""
    
    print("\n\n" + "=" * 80)
    print("不同输入数据类型的 0-size 场景对比测试")
    print("=" * 80)
    
    test_cases = [
        # (dtype_name, torch_dtype, paddle_dtype)
        ("int32", torch.int32 if HAS_TORCH else None, 'int32'),
        ("int64", torch.int64 if HAS_TORCH else None, 'int64'),
    ]
    
    shape = (0,)
    num_classes = 10
    
    for dtype_name, torch_dtype, paddle_dtype in test_cases:
        print(f"\n{'='*60}")
        print(f"Dtype Test: {dtype_name}, shape={shape}, num_classes={num_classes}")
        print("=" * 60)
        
        input_np = np.empty(shape, dtype=np.int64)
        
        # PyTorch
        print("\n--- PyTorch ---")
        if HAS_TORCH and torch_dtype is not None:
            try:
                torch_input = torch.tensor(input_np).to(torch_dtype)
                torch_output = torch_F.one_hot(torch_input, num_classes=num_classes)
                print(f"  Input dtype: {torch_input.dtype}")
                print(f"  Output shape: {torch_output.shape}, dtype: {torch_output.dtype}")
            except Exception as e:
                print(f"  Error: {type(e).__name__}: {e}")
        else:
            print("  [PyTorch 未安装或不支持该类型]")
        
        # PaddlePaddle
        print("\n--- PaddlePaddle ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype=paddle_dtype)
            paddle_output = paddle_F.one_hot(paddle_input, num_classes=num_classes)
            print(f"  Input dtype: {paddle_input.dtype}")
            print(f"  Output shape: {paddle_output.shape}, dtype: {paddle_output.dtype}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


def compare_one_hot_negative_num_classes():
    """测试负数 num_classes 的行为（边界情况）"""
    
    print("\n\n" + "=" * 80)
    print("负数 num_classes 的边界情况测试")
    print("=" * 80)
    
    test_cases = [
        ((0,), -1, "shape=(0,), num_classes=-1"),
        ((5,), -1, "shape=(5,), num_classes=-1 (PyTorch 推断模式)"),
    ]
    
    for shape, num_classes, desc in test_cases:
        print(f"\n{'='*60}")
        print(f"Negative num_classes Test: {desc}")
        print("=" * 60)
        
        if 0 not in shape:
            input_np = np.array([0, 1, 2, 3, 4], dtype=np.int64)[:np.prod(shape)].reshape(shape)
        else:
            input_np = np.empty(shape, dtype=np.int64)
        
        # PyTorch (num_classes=-1 表示自动推断)
        print("\n--- PyTorch ---")
        if HAS_TORCH:
            try:
                torch_input = torch.tensor(input_np, dtype=torch.int64)
                # PyTorch: num_classes=-1 表示自动根据输入最大值推断
                torch_output = torch_F.one_hot(torch_input, num_classes=num_classes)
                print(f"  Output shape: {torch_output.shape}, dtype: {torch_output.dtype}")
                if torch_output.numel() <= 20:
                    print(f"  Output:\n{torch_output}")
            except Exception as e:
                print(f"  Error: {type(e).__name__}: {e}")
        else:
            print("  [PyTorch 未安装]")
        
        # PaddlePaddle
        print("\n--- PaddlePaddle ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype='int64')
            paddle_output = paddle_F.one_hot(paddle_input, num_classes=num_classes)
            print(f"  Output shape: {paddle_output.shape}, dtype: {paddle_output.dtype}")
            if paddle_output.numel() <= 20:
                print(f"  Output:\n{paddle_output}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


def print_summary(results):
    """打印测试结果总结"""
    
    print("\n\n" + "=" * 80)
    print("                         测 试 结 果 总 结")
    print("=" * 80)
    
    print("\n基础 one_hot 测试结果:")
    print("-" * 90)
    print(f"{'Case':<55} {'Match Status':<25}")
    print("-" * 90)
    
    for r in results:
        case_str = r['case'][:52] + "..." if len(r['case']) > 55 else r['case']
        print(f"{case_str:<55} {r['match']:<25}")
    
    # 统计
    match_count = sum(1 for r in results if 'Match' in r['match'] and 'Mismatch' not in r['match'])
    diff_count = sum(1 for r in results if 'Diff' in r['match'] or 'Mismatch' in r['match'])
    error_count = sum(1 for r in results if r['match'] == 'Both Error')
    
    print("-" * 90)
    print(f"\n统计:")
    print(f"  - 完全一致: {match_count}/{len(results)}")
    print(f"  - 存在差异: {diff_count}/{len(results)}")
    print(f"  - 双方报错: {error_count}/{len(results)}")
    
    # 分析差异
    diff_cases = [r for r in results if 'Diff' in r['match'] or 'Mismatch' in r['match']]
    if diff_cases:
        print(f"\n存在差异的场景:")
        for r in diff_cases:
            print(f"  - {r['case']}: {r['match']}")
            if r['torch_error']:
                print(f"    PyTorch Error: {r['torch_error'][:80]}...")
            if r['paddle_error']:
                print(f"    Paddle Error: {r['paddle_error'][:80]}...")
    
    print("""
关键发现总结:
1. 当输入 shape 包含 0 时，输出 shape 会相应包含 0
2. 当 num_classes=0 时，两框架行为可能存在差异
3. PyTorch 支持 num_classes=-1 自动推断，Paddle 可能不支持
4. 输出 dtype 可能因框架而异（torch.int64 vs paddle.int64/float32）

建议: 重点关注 num_classes=0 和边界情况的处理差异
""")


def main():
    if HAS_TORCH:
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("PyTorch: 未安装")
    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # 基础 0-size 对比测试
    results = compare_one_hot_0size()
    
    # 不同数据类型测试
    compare_one_hot_dtype()
    
    # 负数 num_classes 测试
    compare_one_hot_negative_num_classes()
    
    # 打印总结
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()