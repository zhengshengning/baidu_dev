#!/usr/bin/env python3
"""
Performance test for Paddle topk operation.
Compares different scenarios and measures execution time.
"""

import paddle
import time
import numpy as np

def warmup(device='gpu'):
    """Warmup GPU"""
    if device == 'gpu':
        x = paddle.randn([1000, 1000])
        for _ in range(10):
            paddle.topk(x, 10)
        paddle.device.cuda.synchronize()

def benchmark_topk(shape, k, num_runs=100, largest=True, device='gpu'):
    """Benchmark topk operation"""
    x = paddle.randn(shape)
    
    # Warmup
    for _ in range(5):
        paddle.topk(x, k, largest=largest)
    paddle.device.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        paddle.topk(x, k, largest=largest)
    paddle.device.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time

def main():
    paddle.set_device('gpu')
    warmup()
    
    print("=" * 80)
    print("Paddle TopK Performance Benchmark")
    print("=" * 80)
    
    # Test scenarios
    test_cases = [
        # (shape, k, description)
        # Basic cases
        ([32, 1000], 10, "Small batch, small k"),
        ([32, 1000], 100, "Small batch, medium k"),
        ([32, 10000], 10, "Small batch, large slice"),
        ([32, 10000], 100, "Small batch, large slice, medium k"),
        ([32, 100000], 10, "Small batch, very large slice"),
        
        # NLP scenarios
        ([16, 50257], 50, "GPT vocab topk (beam search)"),
        ([32, 50257], 10, "GPT vocab topk (top-p sampling)"),
        
        # MoE scenarios
        ([4096, 64], 8, "MoE router (tokens x experts)"),
        ([8192, 64], 8, "MoE router large batch"),
        
        # Large batch
        ([1024, 1000], 10, "Large batch"),
        ([4096, 1000], 10, "Very large batch"),
        
        # Recommendation
        ([256, 1000000], 100, "Recommendation (large candidate pool)"),
    ]
    
    print(f"\n{'Shape':<25} {'k':<8} {'Time(ms)':<12} Description")
    print("-" * 80)
    
    for shape, k, desc in test_cases:
        try:
            avg_time = benchmark_topk(shape, k)
            print(f"{str(shape):<25} {k:<8} {avg_time:<12.4f} {desc}")
        except Exception as e:
            print(f"{str(shape):<25} {k:<8} {'ERROR':<12} {desc} - {e}")
    
    print("\n" + "=" * 80)
    print("K-value impact analysis (shape=[32, 10000])")
    print("=" * 80)
    
    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    print(f"\n{'k':<10} {'Time(ms)':<12}")
    print("-" * 30)
    
    for k in k_values:
        try:
            avg_time = benchmark_topk([32, 10000], k)
            print(f"{k:<10} {avg_time:<12.4f}")
        except Exception as e:
            print(f"{k:<10} {'ERROR':<12}")
    
    print("\n" + "=" * 80)
    print("Slice size impact analysis (batch=32, k=10)")
    print("=" * 80)
    
    slice_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    print(f"\n{'Slice Size':<15} {'Time(ms)':<12}")
    print("-" * 30)
    
    for size in slice_sizes:
        try:
            avg_time = benchmark_topk([32, size], 10)
            print(f"{size:<15} {avg_time:<12.4f}")
        except Exception as e:
            print(f"{size:<15} {'ERROR':<12}")

if __name__ == "__main__":
    main()