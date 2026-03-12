import numpy as np
import paddle
import torch

paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

def test_softmax_precision():
    x_np = np.random.randn(1, 24, 221, 221).astype(np.float32)

    x_pd = paddle.to_tensor(x_np)
    result_pd = paddle.nn.functional.softmax(x_pd, dim=-1)

    x_pt = torch.from_numpy(x_np).to("cuda")
    result_pt = torch.nn.functional.softmax(x_pt, dim=-1)

    result_pd_np = result_pd.cpu().numpy()
    result_pt_np = result_pt.cpu().numpy()

    diff = np.abs(result_pd_np - result_pt_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print("Softmax Precision Test:")
    print(f"Test 1: Default dim=-1")
    print(f"  Input shape: {x_np.shape}")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance 1e-6"
    print("  PASSED!\n")


if __name__ == "__main__":
    test_softmax_precision()
