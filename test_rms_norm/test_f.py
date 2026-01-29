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

def benchmark_paddle_vs_torch_layer_norm():
    """对比Paddle和PyTorch的layer_norm性能"""
    
    # 设置随机种子
    paddle.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    epsilon = 1e-5
    
    # 测试参数
    # batch_sizes = [32, 128, 200, 500, 800]
    # hidden_sizes1 = [766,1534,2302, 3070,4094,5122,8190,10242,12802,14334,16382,18430,20482]
    # batch_sizes = [32]
    # hidden_sizes1 = [32]
    hidden_sizes2 = [2]
    seq_lens = [32]
    dtypes = ['bfloat16', 'float16', 'float32', 'float64']
    epsilon = 1e-5
    
    print("=" * 80)
    print("Paddle LayerNorm vs PyTorch LayerNorm 性能对比测试")
    print("=" * 80)
    results = []
    
    for hidden_size1 in hidden_sizes1: # H1
      for hidden_size2 in hidden_sizes2: # H2
            for batch_size in batch_sizes: # B
                for seq_len in seq_lens: # C
                    print(f"\n测试配置: batch_size={batch_size}, hidden_size1={hidden_size1}, hidden_size2={hidden_size2}, seq_len={seq_len}")
                    
                    # 生成基准数据 (float32)
                    x_np = np.random.randn(batch_size, seq_len, hidden_size1, hidden_size2).astype('float32')
                    scale_np = np.ones((hidden_size1, hidden_size2)).astype('float32')
                    
                    for dtype in dtypes:
                        print(f"  [Dtype: {dtype}, x.shape: {x_np.shape}, weight.shape: {scale_np.shape}]")
                        try:
                            # === Paddle 准备 ===
                            x_paddle = paddle.to_tensor(x_np, dtype=str2dtype_paddle(dtype))
                            weight_paddle = paddle.to_tensor(scale_np, dtype=str2dtype_paddle(dtype))
                            
                            # === PyTorch 准备 ===
                            x_torch = torch.from_numpy(x_np).to(dtype=str2dtype_torch(dtype)).cuda()
                            weight_torch = torch.from_numpy(scale_np).to(dtype=str2dtype_torch(dtype)).cuda()
                            
                            # # === 预热 GPU ===
                            # for _ in range(500):
                            #     _ = paddle.incubate.nn.functional.fused_rms_norm_ext(x_paddle, weight_paddle, epsilon=epsilon)
                            #     # _ = paddle.incubate.nn.functional.fast_rms_norm(x_paddle, weight_paddle, epsilon=epsilon)
                            #     # _ = F.rms_norm_nzs(x_paddle, weight_paddle, epsilon=epsilon)

                            #     _ = torch.nn.functional.rms_norm(x_torch, (hidden_size1,hidden_size2), weight_torch, eps=epsilon)
                                
                            # paddle.device.cuda.synchronize()
                            # torch.cuda.synchronize()

                            # === 测试 Paddle LayerNorm 性能 ===
                            paddle_times = []
                            paddle.device.cuda.synchronize()
                            start = time.time()
                            for _ in range(1):
                                y_paddle = paddle.nn.functional.rms_norm(x_paddle, (hidden_size1,hidden_size2), weight_paddle, eps=epsilon)
                                # y_paddle = paddle.incubate.nn.functional.fused_rms_norm_ext(x_paddle, weight_paddle, epsilon=epsilon)

                                # y_paddle = paddle.incubate.nn.functional.fast_rms_norm(x_paddle, weight_paddle, epsilon=epsilon)
                                # y_paddle = F.rms_norm_nzs(x_paddle, weight_paddle, epsilon=epsilon)
                            paddle.device.cuda.synchronize()
                            end = time.time()
                            paddle_times.append((end - start) * 1000)
                            
                            # === 测试 PyTorch LayerNorm 性能 ===
                            torch_times = []
                            torch.cuda.synchronize()
                            start = time.time()
                            for _ in range(1):
                                y_torch = torch.nn.functional.rms_norm(x_torch, (hidden_size1,hidden_size2), weight_torch, eps=epsilon)
                            torch.cuda.synchronize()
                            end = time.time()
                            torch_times.append((end - start) * 1000)
                            
                            # === 计算统计信息 ===
                            paddle_avg = np.mean(paddle_times)
                            paddle_min = np.min(paddle_times)
                            paddle_max = np.max(paddle_times)
                            
                            torch_avg = np.mean(torch_times)
                            torch_min = np.min(torch_times)
                            torch_max = np.max(torch_times)
                            
                            # 性能对比
                            speedup = (torch_avg - paddle_avg) / torch_avg * 100 if torch_avg > 0 else 0

                            # === 精度对比 ===
                            y_paddle_np = y_paddle[0].astype('float32').numpy()
                            y_torch_np = y_torch.cpu().to(dtype=torch.float32).numpy()
                            
                            mse = np.mean((y_paddle_np - y_torch_np) ** 2)
                            max_diff = np.max(np.abs(y_paddle_np - y_torch_np))
                            mean_abs_diff = np.mean(np.abs(y_paddle_np - y_torch_np))
                            
                            print("\n" + "=" * 80)
                            # print(f"    Paddle LayerNorm: {paddle_avg:.3f}ms (min:{paddle_min:.3f}, max:{paddle_max:.3f})")
                            # print(f"    PyTorch LayerNorm: {torch_avg:.3f}ms (min:{torch_min:.3f}, max:{torch_max:.3f})")
                            # print(f"    Paddle 性能提升: {speedup:+.1f}% (相对于PyTorch)")
                            print(f"    精度对比:")
                            print(f"      MSE={mse:.2e}, MaxDiff={max_diff:.2e}, MeanAbsDiff={mean_abs_diff:.2e}")
                            print("=" * 80)

                            results.append({
                                'Batch Size': batch_size,
                                'Hidden Size': hidden_size1,
                                'Seq Len': seq_len,
                                'Dtype': dtype,
                                'Paddle Time(ms)': f"{paddle_avg:.3f}",
                                'PyTorch Time(ms)': f"{torch_avg:.3f}",
                                'Paddle Speedup(%)': f"{speedup:.1f}",
                                'MSE': f"{mse:.2e}",
                                'MaxDiff': f"{max_diff:.2e}",
                                'MeanAbsDiff': f"{mean_abs_diff:.2e}",
                            })
                        
                        except Exception as e:
                            print(f"    测试失败: {e}")
                            import traceback
                            traceback.print_exc()
    
    # 保存结果到CSV
    if results:
        csv_file = 'paddle_vs_torch_ln_benchmark.csv'
        fieldnames = [
            'Batch Size', 'Hidden Size', 'Seq Len', 'Dtype',
            'Paddle Time(ms)', 'PyTorch Time(ms)', 'Paddle Speedup(%)',
            'MSE', 'MaxDiff', 'MeanAbsDiff'
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
        paddle.set_device('gpu:1')
        torch.cuda.set_device(1)
    else:
        print("警告: CUDA不可用或PyTorch未检测到CUDA")
        if not paddle.device.is_compiled_with_cuda():
            print("  - Paddle CUDA不可用")
        if not torch.cuda.is_available():
            print("  - PyTorch CUDA不可用")
        exit(1)
    
    # 运行测试
    benchmark_paddle_vs_torch_layer_norm()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)