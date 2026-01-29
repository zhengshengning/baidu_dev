import csv
import time
import torch
import paddle
import numpy as np
import paddle.nn.functional as F

def str2dtype_torch(dtype_str):
    """转换字符串到PyTorch dtype"""
    dtype_str = dtype_str.lower()
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype string: {dtype_str}")

def str2dtype_paddle(dtype_str):
    """转换字符串到Paddle dtype"""
    dtype_str = dtype_str.lower()
    if dtype_str == "float32":
        return "float32"
    elif dtype_str == "float64":
        return "float64"
    elif dtype_str == "float16":
        return "float16"
    elif dtype_str == "bfloat16":
        return "bfloat16"
    else:
        raise ValueError(f"Unknown dtype string: {dtype_str}")

def benchmark_paddle_vs_torch_rms_norm_backward():
    """对比Paddle和PyTorch的rms_norm反向传播性能"""
    
    # 设置随机种子
    paddle.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    epsilon = 1e-5
    
    # 测试参数
    batch_sizes = [1, 8, 32, 128, 200, 500, 1000]
    hidden_sizes1 = [768, 1024, 1536, 2304, 2048, 3072, 4096, 8192, 10240]
    # batch_sizes = [32]
    # hidden_sizes1 = [32]
    hidden_sizes2 = [2]
    seq_lens = [32]
    dtypes = ['bfloat16', 'float16', 'float32', 'float64']
    
    print("=" * 80)
    print("Paddle LayerNorm vs PyTorch LayerNorm 反向传播性能对比测试")
    print("=" * 80)
    results = []
    
    for hidden_size1 in hidden_sizes1:
        for hidden_size2 in hidden_sizes2:
            for batch_size in batch_sizes:
                for seq_len in seq_lens:
                    print(f"\n测试配置: batch_size={batch_size}, hidden_size1={hidden_size1}, hidden_size2={hidden_size2}, seq_len={seq_len}")
                    
                    # 生成基准数据 (float32)
                    x_np = np.random.randn(batch_size, seq_len, hidden_size1, hidden_size2).astype('float32')
                    weight_np = np.ones((hidden_size1, hidden_size2)).astype('float32')
                    grad_output_np = np.random.randn(batch_size, seq_len, hidden_size1, hidden_size2).astype('float32')
                    
                    for dtype in dtypes:
                        print(f"  [Dtype: {dtype}, x.shape: {x_np.shape}, weight.shape: {weight_np.shape}]")
                        try:
                            # === Paddle 准备 ===
                            x_paddle = paddle.to_tensor(x_np, dtype=str2dtype_paddle(dtype), stop_gradient=False)
                            weight_paddle = paddle.to_tensor(weight_np, dtype=str2dtype_paddle(dtype), stop_gradient=False)
                            grad_output_paddle = paddle.to_tensor(grad_output_np, dtype=str2dtype_paddle(dtype))
                            
                            # === PyTorch 准备 ===
                            torch_dtype = str2dtype_torch(dtype)
                            x_torch = torch.from_numpy(x_np).to(dtype=torch_dtype).cuda().requires_grad_(True)
                            weight_torch = torch.from_numpy(weight_np).to(dtype=torch_dtype).cuda().requires_grad_(True)
                            grad_output_torch = torch.from_numpy(grad_output_np).to(dtype=torch_dtype).cuda()
                            
                            # # === 预热 GPU (包含前向和反向) ===
                            # for _ in range(500):
                            #     # Paddle预热
                            #     y_paddle = F.rms_norm(x_paddle, (hidden_size1, hidden_size2), weight_paddle, eps=epsilon)
                            #     y_paddle.backward(grad_output_paddle, retain_graph=True)
                            #     x_paddle.clear_gradient()
                            #     weight_paddle.clear_gradient()
                                
                            #     # PyTorch预热
                            #     y_torch = torch.nn.functional.rms_norm(x_torch, (hidden_size1, hidden_size2), weight_torch, eps=epsilon)
                            #     y_torch.backward(grad_output_torch, retain_graph=True)
                            #     x_torch.grad = None
                            #     weight_torch.grad = None
                                
                            paddle.device.cuda.synchronize()
                            torch.cuda.synchronize()

                            # === 测试 Paddle LayerNorm 反向性能 ===
                            # 前向传播只执行一次（不计入时间）
                            y_paddle = F.rms_norm(x_paddle, (hidden_size1, hidden_size2), weight_paddle, eps=epsilon)
                            if isinstance(y_paddle, tuple):
                                y_paddle = y_paddle[0]
                            
                            paddle_times = []
                            paddle.device.cuda.synchronize()
                            start = time.time()
                            # 只重复执行反向传播
                            for _ in range(1):
                                y_paddle.backward(grad_output_paddle, retain_graph=True)
                                x_paddle.clear_gradient()
                                weight_paddle.clear_gradient()
                            paddle.device.cuda.synchronize()
                            end = time.time()
                            paddle_times.append((end - start) * 1000)
                            
                            # 保存Paddle梯度用于精度对比（重新执行一次反向）
                            # 注意：上面的循环中最后一次 y_paddle 已经被 backward 且清除了 graph（除非 retain_graph=True），
                            # 但这里为了清晰起见，我们通常不再重用上面的 y_paddle，而是重新前向一次。
                            # 不过原始代码是直接调用的，这意味着它依赖于上面的循环中 retain_graph=True
                            # 但上面的循环已经结束，y_paddle 对应的 graph 可能还在（如果 loop range > 0）
                            # 无论如何，为了保险，重新前向一次
                            y_paddle = F.rms_norm(x_paddle, (hidden_size1, hidden_size2), weight_paddle, eps=epsilon)
                            if isinstance(y_paddle, tuple):
                                y_paddle = y_paddle[0]
                            y_paddle.backward(grad_output_paddle)
                            grad_x_paddle = x_paddle.grad.astype('float32').numpy() if x_paddle.grad is not None else None
                            grad_weight_paddle = weight_paddle.grad.astype('float32').numpy() if weight_paddle.grad is not None else None
                            
                            # === 测试 PyTorch LayerNorm 反向性能 ===
                            # 前向传播只执行一次（不计入时间）
                            y_torch = torch.nn.functional.rms_norm(x_torch, (hidden_size1, hidden_size2), weight_torch, eps=epsilon)
                            
                            torch_times = []
                            torch.cuda.synchronize()
                            start = time.time()
                            # 只重复执行反向传播
                            for _ in range(1):
                                y_torch.backward(grad_output_torch, retain_graph=True)
                                x_torch.grad = None
                                weight_torch.grad = None
                            torch.cuda.synchronize()
                            end = time.time()
                            torch_times.append((end - start) * 1000)
                            
                            # 保存PyTorch梯度用于精度对比（重新执行一次反向）
                            y_torch.backward(grad_output_torch)
                            grad_x_torch = x_torch.grad.cpu().to(dtype=torch.float32).numpy() if x_torch.grad is not None else None
                            grad_weight_torch = weight_torch.grad.cpu().to(dtype=torch.float32).numpy() if weight_torch.grad is not None else None
                            
                            # === 计算统计信息 ===
                            paddle_avg = np.mean(paddle_times)
                            paddle_min = np.min(paddle_times)
                            paddle_max = np.max(paddle_times)
                            
                            torch_avg = np.mean(torch_times)
                            torch_min = np.min(torch_times)
                            torch_max = np.max(torch_times)
                            
                            # 性能对比
                            speedup = (torch_avg - paddle_avg) / torch_avg * 100 if torch_avg > 0 else 0
                            
                            # === 梯度精度对比 ===
                            grad_x_mse = np.mean((grad_x_paddle - grad_x_torch) ** 2) if grad_x_paddle is not None and grad_x_torch is not None else 0
                            grad_x_max_diff = np.max(np.abs(grad_x_paddle - grad_x_torch)) if grad_x_paddle is not None and grad_x_torch is not None else 0
                            
                            grad_weight_mse = np.mean((grad_weight_paddle - grad_weight_torch) ** 2) if grad_weight_paddle is not None and grad_weight_torch is not None else 0
                            grad_weight_max_diff = np.max(np.abs(grad_weight_paddle - grad_weight_torch)) if grad_weight_paddle is not None and grad_weight_torch is not None else 0
                            
                            print("\n" + "=" * 80)
                            # print(f"    Paddle 反向时间: {paddle_avg:.3f}ms (min:{paddle_min:.3f}, max:{paddle_max:.3f})")
                            # print(f"    PyTorch 反向时间: {torch_avg:.3f}ms (min:{torch_min:.3f}, max:{torch_max:.3f})")
                            # print(f"    Paddle 性能提升: {speedup:+.1f}% (相对于PyTorch)")
                            print(f"    梯度精度对比:")
                            print(f"      grad_x: MSE={grad_x_mse:.2e}, MaxDiff={grad_x_max_diff:.2e}")
                            print(f"      grad_weight: MSE={grad_weight_mse:.2e}, MaxDiff={grad_weight_max_diff:.2e}")
                            print("=" * 80)
                            
                            results.append({
                                'Batch Size': batch_size,
                                'Hidden Size': (hidden_size1,hidden_size2),
                                'Seq Len': seq_len,
                                'Dtype': dtype,
                                'Paddle Backward Time(ms)': f"{paddle_avg:.3f}",
                                'PyTorch Backward Time(ms)': f"{torch_avg:.3f}",
                                'Paddle Speedup(%)': f"{speedup:.1f}",
                                'grad_x MSE': f"{grad_x_mse:.2e}",
                                'grad_x MaxDiff': f"{grad_x_max_diff:.2e}",
                                'grad_weight MSE': f"{grad_weight_mse:.2e}",
                                'grad_weight MaxDiff': f"{grad_weight_max_diff:.2e}",
                            })
                        
                        except Exception as e:
                            print(f"    测试失败: {e}")
                            import traceback
                            traceback.print_exc()
    
    # 保存结果到CSV
    if results:
        csv_file = 'paddle_vs_torch_ln_backward_benchmark.csv'
        fieldnames = [
            'Batch Size', 'Hidden Size', 'Seq Len', 'Dtype',
            'Paddle Backward Time(ms)', 'PyTorch Backward Time(ms)', 'Paddle Speedup(%)',
            'grad_x MSE', 'grad_x MaxDiff',
            'grad_weight MSE', 'grad_weight MaxDiff'
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n结果已保存至: {csv_file}")


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
    
    # 运行测试
    benchmark_paddle_vs_torch_rms_norm_backward()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)