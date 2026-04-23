import numpy as np
import paddle
import torch
import csv

results = []


def test_correctness(shape, k, dtype_str):
    """
    测试 paddle.topk 与 torch.topk 的正确性对比

    Args:
        shape: 输入张量的形状，如 (M, N)
        k: topk 的 k 值
        dtype_str: 数据类型字符串，如 'float32', 'float16', 'bfloat16'
    """
    torch_dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'int32': torch.int32,
    }
    torch_dtype = torch_dtype_map[dtype_str]

    # 生成随机数据
    numel = 1
    for s in shape:
        numel *= s
    if dtype_str == 'int32':
        high = max(numel // (2 * k), 2)
        np_data = np.random.randint(1, high, size=shape).astype(np.int32)
    else:
        np_data = np.random.randn(*shape).astype(np.float32)

<<<<<<< HEAD:test_topk/testing_diff.py
    # Paddle topk
    paddle_data = paddle.to_tensor(np_data, dtype=dtype_str)
    pd_values, pd_indices = paddle.topk(paddle_data, k, axis=-1, sorted=True)
    pd_values_np = pd_values.astype("float32").numpy()
    pd_indices_np = pd_indices.numpy()

    # Torch topk
    torch_data = torch.from_numpy(np_data).to(torch_dtype).cuda()
    torch_values, torch_indices = torch.topk(torch_data, k, dim=-1, sorted=True)
    torch_values_np = torch_values.float().cpu().numpy()
    torch_indices_np = torch_indices.cpu().numpy()

    # 比较 values
    values_diff = np.max(np.abs(pd_values_np - torch_values_np))
    values_mean_diff = np.mean(np.abs(pd_values_np - torch_values_np))

    # 比较 indices
    indices_match = np.array_equal(pd_indices_np, torch_indices_np)
    indices_diff_count = np.sum(pd_indices_np != torch_indices_np)
    indices_total = pd_indices_np.size

    # 判断是否通过
    passed = indices_match and values_diff == 0
    status = "PASS" if passed else "FAIL"

    print(f"[{status}] shape={str(shape):<20} k={k:<4} dtype={dtype_str:<10} "
          f"val_max_diff={values_diff:.6f}  val_mean_diff={values_mean_diff:.6f}  "
          f"idx_mismatch={indices_diff_count}/{indices_total}")

    # 如果有差异，打印详细信息
    if not passed:
        diff_positions = np.argwhere(pd_indices_np != torch_indices_np)
        if len(diff_positions) > 0:
            print(f"  Indices diff positions (first 5):")
            for pos in diff_positions[:5]:
                pos_tuple = tuple(pos)
                print(f"    pos={pos_tuple}, paddle={pd_indices_np[pos_tuple]}, torch={torch_indices_np[pos_tuple]}")

    results.append({
        'Shape': str(shape),
        'K': k,
        'Dtype': dtype_str,
        'Values_Max_Diff': f"{values_diff:.6f}",
        'Values_Mean_Diff': f"{values_mean_diff:.6f}",
        'Indices_Match': indices_match,
        'Indices_Mismatch': f"{indices_diff_count}/{indices_total}",
        'Status': status,
    })


def save_results_to_csv(filename='topk_correctness_results.csv'):
    """保存结果到 CSV 文件"""
    if results:
        fieldnames = [
            'Shape', 'K', 'Dtype', 'Values_Max_Diff', 'Values_Mean_Diff',
            'Indices_Match', 'Indices_Mismatch', 'Status'
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
    shapes = [
        # # 1D
        (10,),
        (128,),
        (1024,),
        (8192,),
        (65536,),
        (369303,),
        # 2D: 小规模
        (1, 16),
        (1, 20),
        (1, 32),
        (1, 64),
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 4096),
        (1, 8192),
        (1, 65536),
        (1, 369303),
        # 2D: 中等 batch
        (4, 32),
        (4, 128),
        (4, 512),
        (4, 2048),
        (8, 64),
        (8, 128),
        (8, 256),
        (8, 1024),
        (16, 32),
        (16, 64),
        (16, 128),
        (16, 512),
        (16, 2048),
        (32, 64),
        (32, 128),
        (32, 256),
        (32, 1024),
        (64, 64),
        (64, 128),
        (64, 512),
        (128, 64),
        (128, 128),
        (128, 256),
        (256, 64),
        (256, 128),
        (256, 512),
        # 2D: 大 batch
        (512, 128),
        (512, 512),
        (512, 8192),
        (1024, 128),
        (1024, 1024),
        (2048, 256),
        (4096, 512),
        (8192, 128),
        (8192, 1024),
        (16384, 512),
        (16384, 2048),
        (32768, 64),
        (32768, 128),
        (65536, 64),
        # 2D: 极端比例
        (1, 1000000),
        (1000000, 2),
        (2, 131072),
        (131072, 2),
        # 2D: 非2的幂次（边界值）
        (7, 13),
        (13, 37),
        (33, 65),
        (100, 100),
        (127, 127),
        (255, 257),
        (1000, 1000),
        (1023, 1025),
        (4095, 129),
        (8191, 63),
        # 3D
        (2, 16, 64),
        (4, 32, 128),
        (8, 64, 256),
        (16, 128, 512),
        (2, 1024, 128),
        (4, 512, 1024),
        # 4D
        (2, 4, 8, 64),
        (2, 4, 16, 128),
    ]

    ks = [1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 24, 32, 50, 64, 128, 256]

    dtypes = ['float32', 'float16', 'bfloat16', 'int32']

    # 运行测试
    print("=" * 100)
    print("Paddle vs Torch topk 正确性测试")
    print("=" * 100)

    for shape in shapes:
        for k in ks:
            if k > shape[-1]:
                continue
            for dtype in dtypes:
                try:
                    test_correctness(shape, k, dtype)
                except Exception as e:
                    print(f"[ERROR] shape={shape}, k={k}, dtype={dtype}: {e}")

    # 保存结果
    save_results_to_csv()

    # 打印汇总
    total = len(results)
    passed = sum(1 for r in results if r['Status'] == 'PASS')
    failed = total - passed

    print("\n" + "=" * 100)
    print(f"测试汇总: 总计 {total} 项, 通过 {passed} 项, 失败 {failed} 项")
    print("=" * 100)

    if failed > 0:
        print("\n失败项:")
        print(f"{'Shape':<20} {'K':<6} {'Dtype':<10} {'Val_Max_Diff':<15} {'Idx_Mismatch':<15}")
        print("-" * 70)
        for r in results:
            if r['Status'] == 'FAIL':
                print(f"{r['Shape']:<20} {r['K']:<6} {r['Dtype']:<10} "
                      f"{r['Values_Max_Diff']:<15} {r['Indices_Mismatch']:<15}")
