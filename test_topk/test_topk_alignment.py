"""
Test script to verify that Paddle's topk indices align with PyTorch's behavior.

When there are duplicate values, both frameworks should return the same indices
(preferring smaller original indices for equal values).

This version tests the KeMatrixTopK path (used when CUDA version conditions 
for RadixTopK are not met, or as fallback).
"""

import numpy as np

def test_topk_with_paddle():
    """Test Paddle's topk implementation."""
    import paddle
    paddle.set_device('gpu')
    
    print("=" * 60)
    print("Testing Paddle TopK with duplicate values")
    print("=" * 60)
    
    # Test case 1: 1D tensor with duplicates
    print("\n[Test 1] 1D tensor with duplicates:")
    data = np.array([1.0, 3.0, 2.0, 3.0, 3.0, 1.0, 2.0], dtype=np.float32)
    x = paddle.to_tensor(data)
    values, indices = paddle.topk(x, k=5, largest=True, sorted=True)
    print(f"  Input:   {data}")
    print(f"  Values:  {values.numpy()}")
    print(f"  Indices: {indices.numpy()}")
    # Expected: values=[3,3,3,2,2], indices=[1,3,4,2,6] (smaller indices first for equal values)
    
    # Test case 2: 2D tensor, topk on last axis
    print("\n[Test 2] 2D tensor, topk on axis=-1:")
    data2d = np.array([
        [5.0, 3.0, 5.0, 1.0, 5.0],
        [2.0, 2.0, 2.0, 2.0, 1.0]
    ], dtype=np.float32)
    x2d = paddle.to_tensor(data2d)
    values2d, indices2d = paddle.topk(x2d, k=3, axis=-1, largest=True, sorted=True)
    print(f"  Input:\n{data2d}")
    print(f"  Values:\n{values2d.numpy()}")
    print(f"  Indices:\n{indices2d.numpy()}")
    # Row 0 expected: values=[5,5,5], indices=[0,2,4] (smaller indices first)
    # Row 1 expected: values=[2,2,2], indices=[0,1,2] (smaller indices first)
    
    # Test case 3: 2D tensor, topk on axis=0
    print("\n[Test 3] 2D tensor, topk on axis=0:")
    data_axis0 = np.array([
        [3.0, 1.0, 2.0],
        [3.0, 2.0, 2.0],
        [1.0, 2.0, 3.0]
    ], dtype=np.float32)
    x_axis0 = paddle.to_tensor(data_axis0)
    values_axis0, indices_axis0 = paddle.topk(x_axis0, k=2, axis=0, largest=True, sorted=True)
    print(f"  Input:\n{data_axis0}")
    print(f"  Values:\n{values_axis0.numpy()}")
    print(f"  Indices:\n{indices_axis0.numpy()}")
    
    # Test case 4: Small tensor to test KeMatrixTopK path
    print("\n[Test 4] Small 1D tensor (testing KeMatrixTopK path):")
    np.random.seed(42)
    small_data = np.random.randint(0, 10, size=50).astype(np.float32)
    x_small = paddle.to_tensor(small_data)
    values_small, indices_small = paddle.topk(x_small, k=10, largest=True, sorted=True)
    print(f"  Input shape: {small_data.shape}")
    print(f"  Top 10 values:  {values_small.numpy()}")
    print(f"  Top 10 indices: {indices_small.numpy()}")
    
    # Verify stability: for equal values, indices should be in ascending order
    vals = values_small.numpy()
    inds = indices_small.numpy()
    is_stable = True
    for i in range(len(vals) - 1):
        if vals[i] == vals[i+1] and inds[i] > inds[i+1]:
            is_stable = False
            print(f"  WARNING: Unstable at position {i}: val={vals[i]}, indices {inds[i]} > {inds[i+1]}")
    print(f"  Stability check: {'PASSED' if is_stable else 'FAILED'}")
    
    # Test case 5: smallest=True (largest=False)
    print("\n[Test 5] 1D tensor with smallest values (largest=False):")
    values_smallest, indices_smallest = paddle.topk(x, k=3, largest=False, sorted=True)
    print(f"  Input:   {data}")
    print(f"  Values:  {values_smallest.numpy()}")
    print(f"  Indices: {indices_smallest.numpy()}")
    
    print("\n" + "=" * 60)
    print("Paddle TopK tests completed!")
    print("=" * 60)


def test_topk_with_torch():
    """Test PyTorch's topk implementation for comparison."""
    try:
        import torch
        torch.set_default_device('cuda')
        
        print("\n" + "=" * 60)
        print("Testing PyTorch TopK for comparison")
        print("=" * 60)
        
        # Test case 1: 1D tensor with duplicates
        print("\n[Test 1] 1D tensor with duplicates:")
        data = np.array([1.0, 3.0, 2.0, 3.0, 3.0, 1.0, 2.0], dtype=np.float32)
        x = torch.tensor(data)
        values, indices = torch.topk(x, k=5, largest=True, sorted=True)
        print(f"  Input:   {data}")
        print(f"  Values:  {values.cpu().numpy()}")
        print(f"  Indices: {indices.cpu().numpy()}")
        
        # Test case 2: 2D tensor, topk on last axis
        print("\n[Test 2] 2D tensor, topk on dim=-1:")
        data2d = np.array([
            [5.0, 3.0, 5.0, 1.0, 5.0],
            [2.0, 2.0, 2.0, 2.0, 1.0]
        ], dtype=np.float32)
        x2d = torch.tensor(data2d)
        values2d, indices2d = torch.topk(x2d, k=3, dim=-1, largest=True, sorted=True)
        print(f"  Input:\n{data2d}")
        print(f"  Values:\n{values2d.cpu().numpy()}")
        print(f"  Indices:\n{indices2d.cpu().numpy()}")
        
        # Test case 4: Small tensor
        print("\n[Test 4] Small 1D tensor:")
        np.random.seed(42)
        small_data = np.random.randint(0, 10, size=50).astype(np.float32)
        x_small = torch.tensor(small_data)
        values_small, indices_small = torch.topk(x_small, k=10, largest=True, sorted=True)
        print(f"  Input shape: {small_data.shape}")
        print(f"  Top 10 values:  {values_small.cpu().numpy()}")
        print(f"  Top 10 indices: {indices_small.cpu().numpy()}")
        
        print("\n" + "=" * 60)
        print("PyTorch TopK tests completed!")
        print("=" * 60)
        
    except ImportError:
        print("\nPyTorch not available, skipping comparison tests.")


def compare_paddle_torch():
    """Direct comparison between Paddle and PyTorch topk results."""
    try:
        import paddle
        import torch
        
        paddle.set_device('gpu')
        torch.set_default_device('cuda')
        
        print("\n" + "=" * 60)
        print("Direct Comparison: Paddle vs PyTorch TopK")
        print("=" * 60)
        
        # Test with duplicates
        np.random.seed(123)
        test_cases = [
            ("Small 1D with duplicates", np.array([1, 3, 2, 3, 3, 1, 2], dtype=np.float32), 5),
            ("Medium 1D (KeMatrixTopK)", np.random.randint(0, 5, size=100).astype(np.float32), 10),
            ("2D tensor axis=-1", np.array([[5, 3, 5, 1, 5], [2, 2, 2, 2, 1]], dtype=np.float32), 3),
        ]
        
        all_passed = True
        for name, data, k in test_cases:
            print(f"\n[{name}]")
            
            # Paddle
            x_paddle = paddle.to_tensor(data)
            if len(data.shape) == 1:
                vals_paddle, inds_paddle = paddle.topk(x_paddle, k=k, largest=True, sorted=True)
            else:
                vals_paddle, inds_paddle = paddle.topk(x_paddle, k=k, axis=-1, largest=True, sorted=True)
            vals_paddle = vals_paddle.numpy()
            inds_paddle = inds_paddle.numpy()
            
            # PyTorch
            x_torch = torch.tensor(data, device='cuda')
            if len(data.shape) == 1:
                vals_torch, inds_torch = torch.topk(x_torch, k=k, largest=True, sorted=True)
            else:
                vals_torch, inds_torch = torch.topk(x_torch, k=k, dim=-1, largest=True, sorted=True)
            vals_torch = vals_torch.cpu().numpy()
            inds_torch = inds_torch.cpu().numpy()
            
            values_match = np.allclose(vals_paddle, vals_torch)
            indices_match = np.array_equal(inds_paddle, inds_torch)
            
            print(f"  Values match:  {values_match}")
            print(f"  Indices match: {indices_match}")
            
            if not indices_match:
                all_passed = False
                print(f"  Paddle indices:  {inds_paddle}")
                print(f"  PyTorch indices: {inds_torch}")
        
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL TESTS PASSED! Paddle and PyTorch topk indices are aligned.")
        else:
            print("SOME TESTS FAILED! Indices are not fully aligned.")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\nCannot run comparison: {e}")


if __name__ == "__main__":
    test_topk_with_paddle()
    test_topk_with_torch()
    compare_paddle_torch()