import ast
import csv
from pathlib import Path

import numpy as np
import paddle
import torch

paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

results = []


def calculate_diff_metrics(lhs, rhs):
    diff = np.abs(lhs - rhs)
    return np.max(diff), np.mean(diff)


def load_test_configs(shape_file):
    test_configs = []
    with open(shape_file, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                shape = ast.literal_eval(text)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"shape.txt 第 {line_no} 行格式错误: {text}") from error

            if not isinstance(shape, (list, tuple)) or not shape:
                raise ValueError(f"shape.txt 第 {line_no} 行不是合法 shape: {text}")

            shape = tuple(int(dim) for dim in shape)
            normalized_shape = [shape[-1]]
            test_configs.append((shape, normalized_shape))
    return test_configs


def run_accuracy_check(shape, normalized_shape, dtype_str):
    """
    对比 paddle.nn.functional.layer_norm 与 torch.nn.functional.layer_norm 的前向和反向精度

    Args:
        shape: 输入张量的形状，如 (B, S, D)
        normalized_shape: layer_norm 归一化的维度，如 [D] 或 [S, D]
        dtype_str: 数据类型字符串，如 'float32', 'float16', 'bfloat16'
    """
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

    np_data = np.random.randn(*shape).astype(np.float32)
    np_weight = np.random.randn(*normalized_shape).astype(np.float32)
    np_bias = np.random.randn(*normalized_shape).astype(np.float32)
    np_out_grad = np.random.randn(*shape).astype(np.float32)

    paddle_data = paddle.to_tensor(np_data, dtype=paddle_dtype, stop_gradient=False)
    paddle_weight = paddle.to_tensor(np_weight, dtype=paddle_dtype, stop_gradient=False)
    paddle_bias = paddle.to_tensor(np_bias, dtype=paddle_dtype, stop_gradient=False)
    paddle_out_grad = paddle.to_tensor(np_out_grad, dtype=paddle_dtype)

    torch_data = torch.tensor(np_data, dtype=torch_dtype, device='cuda', requires_grad=True)
    torch_weight = torch.tensor(np_weight, dtype=torch_dtype, device='cuda', requires_grad=True)
    torch_bias = torch.tensor(np_bias, dtype=torch_dtype, device='cuda', requires_grad=True)
    torch_out_grad = torch.tensor(np_out_grad, dtype=torch_dtype, device='cuda')

    print(f"\n{'='*60}")
    print(f"Shape: {shape}, Normalized Shape: {normalized_shape}, Dtype: {dtype_str}")
    print(f"{'='*60}")

    pd_out = paddle.nn.functional.layer_norm(
        paddle_data, normalized_shape, paddle_weight, paddle_bias
    )
    torch_out = torch.nn.functional.layer_norm(
        torch_data, normalized_shape, torch_weight, torch_bias
    )

    pd_out_np = pd_out.astype('float32').detach().cpu().numpy()
    torch_out_np = torch_out.float().detach().cpu().numpy()
    forward_max_diff, forward_mean_diff = calculate_diff_metrics(pd_out_np, torch_out_np)

    pd_out.backward(paddle_out_grad)
    torch_out.backward(torch_out_grad)

    pd_input_grad = paddle_data.grad.astype('float32').cpu().numpy()
    torch_input_grad = torch_data.grad.float().detach().cpu().numpy()
    input_grad_max_diff, input_grad_mean_diff = calculate_diff_metrics(pd_input_grad, torch_input_grad)

    pd_weight_grad = paddle_weight.grad.astype('float32').cpu().numpy()
    torch_weight_grad = torch_weight.grad.float().detach().cpu().numpy()
    weight_grad_max_diff, weight_grad_mean_diff = calculate_diff_metrics(pd_weight_grad, torch_weight_grad)

    pd_bias_grad = paddle_bias.grad.astype('float32').cpu().numpy()
    torch_bias_grad = torch_bias.grad.float().detach().cpu().numpy()
    bias_grad_max_diff, bias_grad_mean_diff = calculate_diff_metrics(pd_bias_grad, torch_bias_grad)

    print("\n[Forward Accuracy]")
    print(f"  Max abs diff: {forward_max_diff:.6f}")
    print(f"  Mean abs diff: {forward_mean_diff:.6f}")

    print("\n[Backward Accuracy - Input Grad]")
    print(f"  Max abs diff: {input_grad_max_diff:.6f}")
    print(f"  Mean abs diff: {input_grad_mean_diff:.6f}")

    print("\n[Backward Accuracy - Weight Grad]")
    print(f"  Max abs diff: {weight_grad_max_diff:.6f}")
    print(f"  Mean abs diff: {weight_grad_mean_diff:.6f}")

    print("\n[Backward Accuracy - Bias Grad]")
    print(f"  Max abs diff: {bias_grad_max_diff:.6f}")
    print(f"  Mean abs diff: {bias_grad_mean_diff:.6f}")

    results.append({
        'Shape': str(shape),
        'Normalized_Shape': str(normalized_shape),
        'Dtype': dtype_str,
        'Forward_Max_Diff': f"{forward_max_diff:.6f}",
        'Forward_Mean_Diff': f"{forward_mean_diff:.6f}",
        'Input_Grad_Max_Diff': f"{input_grad_max_diff:.6f}",
        'Input_Grad_Mean_Diff': f"{input_grad_mean_diff:.6f}",
        'Weight_Grad_Max_Diff': f"{weight_grad_max_diff:.6f}",
        'Weight_Grad_Mean_Diff': f"{weight_grad_mean_diff:.6f}",
        'Bias_Grad_Max_Diff': f"{bias_grad_max_diff:.6f}",
        'Bias_Grad_Mean_Diff': f"{bias_grad_mean_diff:.6f}",
    })


def save_results_to_csv(filename='layer_norm_accuracy_results.csv'):
    """保存前向和反向精度测试结果到 CSV 文件"""
    if results:
        fieldnames = [
            'Shape',
            'Normalized_Shape',
            'Dtype',
            'Forward_Max_Diff',
            'Forward_Mean_Diff',
            'Input_Grad_Max_Diff',
            'Input_Grad_Mean_Diff',
            'Weight_Grad_Max_Diff',
            'Weight_Grad_Mean_Diff',
            'Bias_Grad_Max_Diff',
            'Bias_Grad_Mean_Diff',
        ]
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n精度结果已保存至: {filename}")

if __name__ == "__main__":
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
        raise SystemExit(1)

    script_dir = Path(__file__).resolve().parent
    shape_file = script_dir / "shape.txt"
    test_configs = load_test_configs(shape_file)
    dtypes = ['float32', 'float16', 'bfloat16']
    # dtypes = ['bfloat16']
    
    # run_accuracy_check((17340, 1280), (1280,), "bfloat16")
    # run_accuracy_check((16422, 1280), (1280,), "bfloat16")
    # run_accuracy_check((18070, 1280), (1280,), "bfloat16")

    # run_accuracy_check((18768, 1280), (1280,), "bfloat16")
    # run_accuracy_check((16744, 1280), (1280,), "bfloat16")

    # run_accuracy_check((17940, 1280), (1280,), "bfloat16")
    # run_accuracy_check((5382, 1280), (1280,), "bfloat16")


    for shape, normalized_shape in test_configs:
        for dtype in dtypes:
            try:
                run_accuracy_check(shape, normalized_shape, dtype)
            except Exception as error:
                print(f"测试失败: shape={shape}, normalized_shape={normalized_shape}, dtype={dtype}")
                print(f"错误信息: {error}")

    save_results_to_csv()

    print("\n" + "="*180)
    print("精度测试汇总")
    print("="*180)
    header_columns = [
        ("Shape", 22),
        ("Norm_Shape", 14),
        ("Dtype", 10),
        ("Fwd_Max_Diff", 14),
        ("Fwd_Mean_Diff", 14),
        ("InGrad_Max_Diff", 16),
        ("InGrad_Mean_Diff", 16),
        ("WGrad_Max_Diff", 17),
        ("WGrad_Mean_Diff", 17),
        ("BGrad_Max_Diff", 15),
        ("BGrad_Mean_Diff", 15),
    ]
    header = " ".join(f"{name:<{width}}" for name, width in header_columns)
    print(header)

    print("-"*180)
    for result in results:
        print(
            f"{result['Shape']:<22} {result['Normalized_Shape']:<14} {result['Dtype']:<10} "
            f"{result['Forward_Max_Diff']:<14} {result['Forward_Mean_Diff']:<14} "
            f"{result['Input_Grad_Max_Diff']:<16} {result['Input_Grad_Mean_Diff']:<16} "
            f"{result['Weight_Grad_Max_Diff']:<17} {result['Weight_Grad_Mean_Diff']:<17} "
            f"{result['Bias_Grad_Max_Diff']:<15} {result['Bias_Grad_Mean_Diff']:<15}"
        )

