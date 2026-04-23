import numpy as np
import paddle
import torch
import time
import csv

# paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

results = []

def run_benchmark(shape, normalized_shape, dtype_str, warmup_iters=100, test_iters=1000):
    """
    测试 paddle.nn.functional.layer_norm 与 torch.nn.functional.layer_norm 的性能对比

    Args:
        shape: 输入张量的形状，如 (B, S, D)
        normalized_shape: layer_norm 归一化的维度，如 [D] 或 [S, D]
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
    np_weight = np.random.randn(*normalized_shape).astype(np.float32)
    np_bias = np.random.randn(*normalized_shape).astype(np.float32)

    # 创建 Paddle tensor
    paddle_data = paddle.to_tensor(np_data, dtype=paddle_dtype)
    paddle_weight = paddle.to_tensor(np_weight, dtype=paddle_dtype)
    paddle_bias = paddle.to_tensor(np_bias, dtype=paddle_dtype)

    # 创建 PyTorch tensor
    torch_data = torch.from_numpy(np_data).to(torch_dtype).cuda()
    torch_weight = torch.from_numpy(np_weight).to(torch_dtype).cuda()
    torch_bias = torch.from_numpy(np_bias).to(torch_dtype).cuda()

    print(f"\n{'='*60}")
    print(f"Shape: {shape}, Normalized Shape: {normalized_shape}, Dtype: {dtype_str}")
    print(f"{'='*60}")

    # ============== Paddle Benchmark ==============
    # Warmup
    for _ in range(warmup_iters):
        paddle.nn.functional.layer_norm(paddle_data, normalized_shape, paddle_weight, paddle_bias)
    paddle.device.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(test_iters):
        pd_out = paddle.nn.functional.layer_norm(paddle_data, normalized_shape, paddle_weight, paddle_bias)
    paddle.device.synchronize()
    end_time = time.time()

    paddle_total_time = (end_time - start_time) * 1000.0  # ms
    paddle_avg_time = paddle_total_time / test_iters

    print(f"[Paddle layer_norm]")
    print(f"  Total time ({test_iters} runs): {paddle_total_time:.3f} ms")
    print(f"  Average time per run: {paddle_avg_time:.4f} ms")

    # ============== PyTorch Benchmark ==============
    # Warmup
    for _ in range(warmup_iters):
        torch.nn.functional.layer_norm(torch_data, normalized_shape, torch_weight, torch_bias)
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(test_iters):
        torch_out = torch.nn.functional.layer_norm(torch_data, normalized_shape, torch_weight, torch_bias)
    torch.cuda.synchronize()
    end_time = time.time()

    torch_total_time = (end_time - start_time) * 1000.0  # ms
    torch_avg_time = torch_total_time / test_iters

    print(f"[Torch layer_norm]")
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
    pd_out_np = pd_out.astype("float32").numpy()
    torch_out_np = torch_out.float().cpu().numpy()

    max_diff = np.max(np.abs(pd_out_np - torch_out_np))
    mean_diff = np.mean(np.abs(pd_out_np - torch_out_np))

    print(f"\n[Correctness Check]")
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")

    # 保存结果
    results.append({
        'Shape': str(shape),
        'Normalized_Shape': str(normalized_shape),
        'Dtype': dtype_str,
        'Paddle_Avg(ms)': f"{paddle_avg_time:.4f}",
        'Torch_Avg(ms)': f"{torch_avg_time:.4f}",
        'Speedup': f"{speedup:.2f}",
        'Max_Diff': f"{max_diff:.6f}",
        'Mean_Diff': f"{mean_diff:.6f}",
    })


def save_results_to_csv(filename='layer_norm_benchmark_results.csv'):
    """保存结果到 CSV 文件"""
    if results:
        fieldnames = [
            'Shape', 'Normalized_Shape', 'Dtype', 'Paddle_Avg(ms)', 'Torch_Avg(ms)',
            'Speedup', 'Max_Diff', 'Mean_Diff'
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
    # (input_shape, normalized_shape) 组合
    test_configs = [
        # --- 典型 Transformer 场景: (batch, seq_len, hidden_dim), norm on [hidden_dim] ---
        # ((8, 128, 768),    [768]),       # BERT-base
        # ((8, 128, 1024),   [1024]),      # BERT-large
        # ((8, 512, 768),    [768]),       # 长序列 BERT-base
        # ((8, 512, 1024),   [1024]),      # 长序列 BERT-large
        # ((4, 2048, 4096),  [4096]),      # LLaMA-7B 级别
        # ((4, 2048, 5120),  [5120]),      # LLaMA-13B 级别
        # ((2, 4096, 8192),  [8192]),      # LLaMA-65B 级别
        # ((1, 8192, 4096),  [4096]),      # 超长序列

        # # --- 不同 batch size ---
        # ((1, 512, 768),    [768]),       # batch=1
        # ((16, 512, 768),   [768]),       # batch=16
        # ((32, 128, 768),   [768]),       # batch=32
        # ((64, 128, 768),   [768]),       # batch=64

        # --- 2D 输入: (batch, features) ---
        # ((4096, 768),      [768]),       # 2D 小维度
        # ((4096, 4096),     [4096]),      # 2D 大维度
        # ((16384, 1024),    [1024]),      # 2D 大 batch

        # 非对齐
        ((6996, 1152),    [1152]),  # norm on [seq_len, hidden_dim]
        ((8856, 1152),    [1152]),  # norm on [seq_len, hidden_dim]
        ((9052, 1152),    [1152]),  # norm on [seq_len, hidden_dim]

        # --- 多维 normalized_shape ---
        # ((8, 128, 768),    [128, 768]),  # norm on [seq_len, hidden_dim]
        # ((4, 64, 32, 64),  [32, 64]),    # 4D 输入，norm on 后两维
    ]

    # 不同的数据类型
    dtypes = ['float32', 'float16', 'bfloat16']

    # 运行测试
    for shape, normalized_shape in test_configs:
        for dtype in dtypes:
            try:
                run_benchmark(shape, normalized_shape, dtype, warmup_iters=50, test_iters=500)
            except Exception as e:
                print(f"测试失败: shape={shape}, normalized_shape={normalized_shape}, dtype={dtype}")
                print(f"错误信息: {e}")

    # 保存结果
    save_results_to_csv()

    # 打印汇总表格
    print("\n" + "="*100)
    print("性能测试汇总")
    print("="*100)
    print(f"{'Shape':<22} {'Norm_Shape':<14} {'Dtype':<10} {'Paddle(ms)':<12} {'Torch(ms)':<12} {'Speedup':<10} {'Max_Diff':<12}")
    print("-"*100)
    for r in results:
        print(f"{r['Shape']:<22} {r['Normalized_Shape']:<14} {r['Dtype']:<10} {r['Paddle_Avg(ms)']:<12} {r['Torch_Avg(ms)']:<12} {r['Speedup']:<10} {r['Max_Diff']:<12}")
