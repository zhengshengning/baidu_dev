import numpy as np
import paddle
import torch
import time
import csv

results = []

def run_benchmark(shape, k, dtype_str, warmup_iters=100, test_iters=1000):
    """
    测试 paddle.topk 与 torch.topk 的性能对比
    
    Args:
        shape: 输入张量的形状，如 (M, N)
        k: topk 的 k 值
        dtype_str: 数据类型字符串，如 'float32', 'float16', 'bfloat16'
        warmup_iters: 预热迭代次数
        test_iters: 测试迭代次数
    """
    # 设置数据类型映射
    paddle_dtype_map = {
        'float32': 'float32',
        'float16': 'float16',
        'bfloat16': 'bfloat16',
    }
    torch_dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    
    paddle_dtype = paddle_dtype_map[dtype_str]
    torch_dtype = torch_dtype_map[dtype_str]
    
    # 生成随机数据
    np_data = np.random.randn(*shape).astype(np.float32)
    
    # 创建 Paddle tensor
    paddle_data = paddle.to_tensor(np_data, dtype=paddle_dtype)
    
    # 创建 PyTorch tensor
    torch_data = torch.from_numpy(np_data).to(torch_dtype).cuda()
    
    print(f"\n{'='*60}")
    print(f"Shape: {shape}, K: {k}, Dtype: {dtype_str}")
    print(f"{'='*60}")
    
    # ============== Paddle Benchmark ==============
    # Warmup
    for _ in range(warmup_iters):
        paddle.topk(paddle_data, k, axis=-1, sorted=True)
    paddle.device.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(test_iters):
        pd_values, pd_indices = paddle.topk(paddle_data, k, axis=-1, sorted=True)
    paddle.device.synchronize()
    end_time = time.time()
    
    paddle_total_time = (end_time - start_time) * 1000.0  # ms
    paddle_avg_time = paddle_total_time / test_iters
    
    print(f"[Paddle topk]")
    print(f"  Total time ({test_iters} runs): {paddle_total_time:.3f} ms")
    print(f"  Average time per run: {paddle_avg_time:.4f} ms")
    
    # ============== PyTorch Benchmark ==============
    # Warmup
    for _ in range(warmup_iters):
        torch.topk(torch_data, k, dim=-1, sorted=True)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(test_iters):
        torch_values, torch_indices = torch.topk(torch_data, k, dim=-1, sorted=True)
    torch.cuda.synchronize()
    end_time = time.time()
    
    torch_total_time = (end_time - start_time) * 1000.0  # ms
    torch_avg_time = torch_total_time / test_iters
    
    print(f"[Torch topk]")
    print(f"  Total time ({test_iters} runs): {torch_total_time:.3f} ms")
    print(f"  Average time per run: {torch_avg_time:.4f} ms")
    
    # ============== 性能对比 ==============
    speedup = torch_avg_time / paddle_avg_time if paddle_avg_time > 0 else 0
    print(f"\n[Performance Comparison]")
    print(f"  Speedup (Torch/Paddle): {speedup:.2f}x")
    if speedup > 1:
        print(f"  -> Paddle is {speedup:.2f}x faster than Torch")
    else:
        print(f"  -> Torch is {1/speedup:.2f}x faster than Paddle")
    
    # ============== 正确性验证 ==============
    pd_values_np = pd_values.astype("float32").numpy()
    pd_indices_np = pd_indices.numpy()
    torch_values_np = torch_values.float().cpu().numpy()
    torch_indices_np = torch_indices.cpu().numpy()
    
    values_diff = np.max(np.abs(pd_values_np - torch_values_np))
    indices_match = np.array_equal(pd_indices_np, torch_indices_np)
    
    print(f"\n[Correctness Check]")
    print(f"  Max abs diff (values): {values_diff:.6f}")
    print(f"  Indices match: {indices_match}")
    
    # 保存结果
    results.append({
        'Shape': str(shape),
        'K': k,
        'Dtype': dtype_str,
        'Paddle_Avg(ms)': f"{paddle_avg_time:.4f}",
        'Torch_Avg(ms)': f"{torch_avg_time:.4f}",
        'Speedup': f"{speedup:.2f}",
        'Values_Diff': f"{values_diff:.6f}",
        'Indices_Match': indices_match,
    })


def save_results_to_csv(filename='topk_benchmark_results.csv'):
    """保存结果到 CSV 文件"""
    if results:
        fieldnames = [
            'Shape', 'K', 'Dtype', 'Paddle_Avg(ms)', 'Torch_Avg(ms)', 
            'Speedup', 'Values_Diff', 'Indices_Match'
        ]
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n结果已保存至: {filename}")


if __name__ == "__main__":
    # 检查CUDA是否可用
    if paddle.device.is_compiled_with_cuda() and torch.cuda.is_available():
        print("CUDA可用，使用GPU进行测试")
        paddle.set_device('gpu:2')
        torch.cuda.set_device(2)
    else:
        print("警告: CUDA不可用或PyTorch未检测到CUDA")
        if not paddle.device.is_compiled_with_cuda():
            print("  - Paddle CUDA不可用")
        if not torch.cuda.is_available():
            print("  - PyTorch CUDA不可用")
        exit(1)
    
    # ============== 测试配置 ==============
    # 不同的 shape 配置
    shapes = [
        (1024, 128),      # 小规模
        (2048, 256),      # 中等规模
        (4096, 512),      # 较大规模
        (8192, 1024),     # 大规模
        (16384, 2048),    # 超大规模
        (32768, 64),      # 高 batch，小 dim
        (16384, 512),     # 中 batch，大 dim
        (512, 8192),      # 低 batch，大 dim
        (1, 369303),    # 超高 batch，中等 dim
    ]
    
    # 不同的 k 值
    ks = [1, 8, 16, 32, 64, 128]
    
    # 不同的数据类型
    dtypes = ['float32', 'float16', 'bfloat16']
    
    # 运行测试
    for shape in shapes:
        for k in ks:
            # 确保 k 不超过最后一个维度
            if k > shape[-1]:
                continue
            for dtype in dtypes:
                try:
                    run_benchmark(shape, k, dtype, warmup_iters=50, test_iters=500)
                except Exception as e:
                    print(f"测试失败: shape={shape}, k={k}, dtype={dtype}")
                    print(f"错误信息: {e}")
    
    # 保存结果
    save_results_to_csv()
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("性能测试汇总")
    print("="*80)
    print(f"{'Shape':<20} {'K':<6} {'Dtype':<10} {'Paddle(ms)':<12} {'Torch(ms)':<12} {'Speedup':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['Shape']:<20} {r['K']:<6} {r['Dtype']:<10} {r['Paddle_Avg(ms)']:<12} {r['Torch_Avg(ms)']:<12} {r['Speedup']:<10}")