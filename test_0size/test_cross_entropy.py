"""
对比 paddle.cross_entropy 和 torch.nn.functional.cross_entropy 在 0-size 场景下的计算结果
"""

import numpy as np
import paddle

# 尝试导入 PyTorch（可选）
import torch
import torch.nn.functional as F
HAS_TORCH = True


def compare_cross_entropy_0size():
    """测试 0-size 场景下两个框架的 cross_entropy 行为"""
    
    print("=" * 80)
    print("paddle.cross_entropy vs torch.cross_entropy 0-size 场景对比测试")
    print("=" * 80)
    
    # 设置随机种子
    np.random.seed(42)
    paddle.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)
    
    test_cases = [
        # (input_shape, label_shape, description)
        # Case 1: batch_size = 0
        ((0, 10), (0,), "batch_size=0, num_classes=10"),
        # Case 2: num_classes = 0
        ((4, 0), (4,), "batch_size=4, num_classes=0"),
        # Case 3: 两个维度都为 0
        ((0, 0), (0,), "batch_size=0, num_classes=0"),
        # Case 4: 3D输入，batch_size = 0 (axis=1, label shape 应为 (N, d1, d2, ...))
        # PyTorch: input (N, C, d1, d2, ...), label (N, d1, d2, ...)
        # Paddle: 默认 axis=-1, 所以 input (N, d1, C), label (N, d1) 或使用 axis=1
        ((0, 10, 5), (0, 5), "3D input: batch=0, classes=10, seq_len=5 (axis=1)"),
        # Case 5: 3D输入，seq_len = 0
        ((4, 10, 0), (4, 0), "3D input: batch=4, classes=10, seq_len=0 (axis=1)"),
    ]
    
    for i, (input_shape, label_shape, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {desc}")
        print(f"Input shape: {input_shape}, Label shape: {label_shape}")
        print("=" * 60)
        
        # 准备数据
        if 0 not in input_shape:
            input_np = np.random.randn(*input_shape).astype(np.float32)
        else:
            input_np = np.empty(input_shape, dtype=np.float32)
        
        # 创建标签 (需要在有效范围内)
        num_classes = input_shape[1] if len(input_shape) > 1 else 1
        if 0 not in label_shape and num_classes > 0:
            label_np = np.random.randint(0, num_classes, size=label_shape).astype(np.int64)
        else:
            label_np = np.empty(label_shape, dtype=np.int64)
        
        print(f"Input data shape: {input_np.shape}")
        print(f"Label data shape: {label_np.shape}")
        
        # ============ PyTorch 测试 ============
        print("\n--- PyTorch ---")
        if HAS_TORCH:
            try:
                torch_input = torch.tensor(input_np, dtype=torch.float32)
                torch_label = torch.tensor(label_np, dtype=torch.int64)
                
                # 测试不同 reduction 模式
                for reduction in ['mean', 'sum', 'none']:
                    try:
                        torch_output = F.cross_entropy(torch_input, torch_label, reduction=reduction)
                        print(f"  reduction='{reduction}': shape={torch_output.shape}, value={torch_output}")
                    except Exception as e:
                        print(f"  reduction='{reduction}': Error - {type(e).__name__}: {e}")
            except Exception as e:
                print(f"  PyTorch Error: {type(e).__name__}: {e}")
        else:
            # PyTorch 预期行为（基于 PyTorch 文档和测试）
            print("  [PyTorch 预期行为 - 未安装，基于文档]")
            if input_shape[0] == 0:  # batch_size = 0
                print("  reduction='mean': shape=torch.Size([]), value=nan")
                print("  reduction='sum': shape=torch.Size([]), value=0.0")
                print(f"  reduction='none': shape=torch.Size([{label_shape}]), value=tensor([])")
            elif len(input_shape) > 1 and input_shape[1] == 0:  # num_classes = 0
                print("  Error - IndexError: 访问空张量")
        
        # ============ PaddlePaddle 测试 ============
        print("\n--- PaddlePaddle ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype='float32')
            paddle_label = paddle.to_tensor(label_np, dtype='int64')
            
            # 测试不同 reduction 模式
            # 对于 3D+ 输入，需要指定 axis=1 以匹配 PyTorch 行为
            axis = 1 if len(input_shape) > 2 else -1
            for reduction in ['mean', 'sum', 'none']:
                try:
                    paddle_output = paddle.nn.functional.cross_entropy(
                        paddle_input, paddle_label, reduction=reduction, axis=axis
                    )
                    print(f"  reduction='{reduction}': shape={paddle_output.shape}, value={paddle_output}")
                except Exception as e:
                    print(f"  reduction='{reduction}': Error - {type(e).__name__}: {e}")
        except Exception as e:
            print(f"  PaddlePaddle Error: {type(e).__name__}: {e}")


def compare_with_soft_label_0size():
    """测试 soft_label 模式下的 0-size 场景"""
    
    print("\n\n" + "=" * 80)
    print("Soft Label 模式下的 0-size 场景对比测试")
    print("=" * 80)
    
    test_cases = [
        # (input_shape, description)
        ((0, 10), "batch_size=0, num_classes=10"),
        ((4, 0), "batch_size=4, num_classes=0"),
        ((0, 0), "batch_size=0, num_classes=0"),
    ]
    
    for i, (input_shape, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Soft Label Test Case {i}: {desc}")
        print(f"Input/Label shape: {input_shape}")
        print("=" * 60)
        
        # 准备数据
        if 0 not in input_shape:
            input_np = np.random.randn(*input_shape).astype(np.float32)
            # soft label 需要是概率分布
            soft_label_np = np.random.rand(*input_shape).astype(np.float32)
            soft_label_np = soft_label_np / soft_label_np.sum(axis=-1, keepdims=True)
        else:
            input_np = np.empty(input_shape, dtype=np.float32)
            soft_label_np = np.empty(input_shape, dtype=np.float32)
        
        # ============ PyTorch 测试 (使用 kl_div 或手动实现) ============
        print("\n--- PyTorch (手动实现 soft cross entropy) ---")
        if HAS_TORCH:
            try:
                torch_input = torch.tensor(input_np, dtype=torch.float32)
                torch_soft_label = torch.tensor(soft_label_np, dtype=torch.float32)
                
                log_softmax = torch.nn.functional.log_softmax(torch_input, dim=-1)
                torch_output = -(torch_soft_label * log_softmax).sum(dim=-1).mean()
                print(f"  Output: shape={torch_output.shape}, value={torch_output}")
            except Exception as e:
                print(f"  PyTorch Error: {type(e).__name__}: {e}")
        else:
            print("  [PyTorch 预期行为 - 未安装]")
            if 0 in input_shape:
                print("  Output: shape=torch.Size([]), value=nan (空张量 mean)")
        
        # ============ PaddlePaddle 测试 ============
        print("\n--- PaddlePaddle (soft_label=True) ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype='float32')
            paddle_soft_label = paddle.to_tensor(soft_label_np, dtype='float32')
            
            paddle_output = paddle.nn.functional.cross_entropy(
                paddle_input, paddle_soft_label, soft_label=True
            )
            print(f"  Output: shape={paddle_output.shape}, value={paddle_output}")
        except Exception as e:
            print(f"  PaddlePaddle Error: {type(e).__name__}: {e}")


def compare_with_weight_0size():
    """测试带 weight 参数的 0-size 场景"""
    
    print("\n\n" + "=" * 80)
    print("带 Weight 参数的 0-size 场景对比测试")
    print("=" * 80)
    
    test_cases = [
        # (input_shape, label_shape, weight_shape, description)
        ((0, 10), (0,), (10,), "batch=0, classes=10"),
        ((4, 0), (4,), (0,), "batch=4, classes=0"),
    ]
    
    for i, (input_shape, label_shape, weight_shape, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Weight Test Case {i}: {desc}")
        print(f"Input: {input_shape}, Label: {label_shape}, Weight: {weight_shape}")
        print("=" * 60)
        
        # 准备数据
        input_np = np.empty(input_shape, dtype=np.float32) if 0 in input_shape else \
                   np.random.randn(*input_shape).astype(np.float32)
        
        num_classes = input_shape[1] if len(input_shape) > 1 else 1
        label_np = np.empty(label_shape, dtype=np.int64) if 0 in label_shape or num_classes == 0 else \
                   np.random.randint(0, num_classes, size=label_shape).astype(np.int64)
        
        weight_np = np.empty(weight_shape, dtype=np.float32) if 0 in weight_shape else \
                    np.random.rand(*weight_shape).astype(np.float32)
        
        # ============ PyTorch 测试 ============
        print("\n--- PyTorch ---")
        if HAS_TORCH:
            try:
                torch_input = torch.tensor(input_np, dtype=torch.float32)
                torch_label = torch.tensor(label_np, dtype=torch.int64)
                torch_weight = torch.tensor(weight_np, dtype=torch.float32)
                
                torch_output = F.cross_entropy(torch_input, torch_label, weight=torch_weight)
                print(f"  Output: shape={torch_output.shape}, value={torch_output}")
            except Exception as e:
                print(f"  PyTorch Error: {type(e).__name__}: {e}")
        else:
            print("  [PyTorch 预期行为 - 未安装]")
            if input_shape[0] == 0:
                print("  Output: shape=torch.Size([]), value=nan")
        
        # ============ PaddlePaddle 测试 ============
        print("\n--- PaddlePaddle ---")
        try:
            paddle_input = paddle.to_tensor(input_np, dtype='float32')
            paddle_label = paddle.to_tensor(label_np, dtype='int64')
            paddle_weight = paddle.to_tensor(weight_np, dtype='float32')
            
            paddle_output = paddle.nn.functional.cross_entropy(
                paddle_input, paddle_label, weight=paddle_weight
            )
            print(f"  Output: shape={paddle_output.shape}, value={paddle_output}")
        except Exception as e:
            print(f"  PaddlePaddle Error: {type(e).__name__}: {e}")


def main():
    if HAS_TORCH:
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("PyTorch: 未安装")
    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # 基础 0-size 对比测试
    compare_cross_entropy_0size()
    
    # Soft label 模式测试
    compare_with_soft_label_0size()
    
    # Weight 参数测试
    compare_with_weight_0size()
    

if __name__ == "__main__":
    main()