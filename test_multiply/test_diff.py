import numpy as np
import paddle
import torch

def test_complex64_multiply():
    real_a = np.random.randn(1, 144, 24, 64).astype(np.float32)
    imag_a = np.random.randn(144, 1, 64).astype(np.float32)
    real_b = np.random.randn(1, 144, 24, 64).astype(np.float32)
    imag_b = np.random.randn(144, 1, 64).astype(np.float32)

    # a_np = (real_a + 1j * imag_a).astype(np.complex64)
    # b_np = (real_b + 1j * imag_b).astype(np.complex64)
    a_np = (real_a + 0j * imag_a).astype(np.complex64)
    b_np = (real_b + 0j * imag_b).astype(np.complex64)


    # a_np = np.random.random((1, 144, 24, 64)).astype(np.complex64)
    # b_np = np.random.random((1, 144, 24, 64)).astype(np.complex64)

    a_pd = paddle.to_tensor(a_np)
    b_pd = paddle.to_tensor(b_np)
    result_pd = a_pd * b_pd

    a_pt = torch.from_numpy(a_np).to("cuda")
    b_pt = torch.from_numpy(b_np).to("cuda")
    result_pt = a_pt * b_pt

    result_pd_np = result_pd.cpu().numpy()
    result_pt_np = result_pt.cpu().numpy()

    diff = np.abs(result_pd_np - result_pt_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print("Complex64 Multiply Precision Test:")
    print(f"  a shape: {a_np.shape}, b shape: {b_np.shape}")
    print(f"  Result shape: {result_pd_np.shape}")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    assert max_diff < 1e-6, f"Max diff {max_diff} exceeds tolerance 1e-6"
    print("  PASSED!\n")


if __name__ == "__main__":
    test_complex64_multiply()


