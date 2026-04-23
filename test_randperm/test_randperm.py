"""
Test for shuffle consistency between torch and paddle implementations.
"""
import paddle
import torch

paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})
paddle.set_device('cpu')


def test_randperm_consistency():
    """Test that paddle and torch produce same shuffle with same seed."""
    # Set same seed
    paddle.seed(42)
    torch.manual_seed(42)

    # Generate random permutations
    paddle_idx = paddle.randperm(200).tolist()
    torch_idx = torch.randperm(200, device='cpu').tolist()

    print("*" * 80)
    assert paddle_idx == torch_idx, f"Mismatch: \n paddle={paddle_idx}, \n\n torch={torch_idx}"
    print("PASS: paddle and torch randperm produce identical results with seed=42")


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different shuffles."""
    paddle.seed(42)
    idx1 = paddle.randperm(200).tolist()

    paddle.seed(123)
    idx2 = paddle.randperm(200).tolist()

    assert idx1 != idx2, "Different seeds should produce different shuffles"
    print("PASS: different seeds produce different shuffles")


def test_randperm_range():
    """Test that randperm produces all values in range [0, n)."""
    paddle.seed(42)
    idx = paddle.randperm(200).tolist()

    assert sorted(idx) == list(range(200)), "randperm should produce all values in range"
    print("PASS: randperm produces all values in range [0, n)")


def test_torch_two_consecutive_randperm():
    """Test that two consecutive torch randperm calls with same seed produce different results."""
    torch.manual_seed(123123143)
    torch_idx1 = torch.randperm(10, device='cpu').tolist()
    torch_idx2 = torch.randperm(10, device='cpu').tolist()

    print("*" * 80)
    print(f"torch seed=123123143, randperm(10) call 1: {torch_idx1}")
    print(f"torch seed=123123143, randperm(10) call 2: {torch_idx2}")
    assert torch_idx1 != torch_idx2, "Two consecutive torch randperm calls should produce different results"
    print("PASS: two consecutive torch randperm calls produce different results")


if __name__ == "__main__":
    test_randperm_consistency()
    test_different_seeds_produce_different_results()
    test_randperm_range()
    test_torch_two_consecutive_randperm()
    print("\nAll tests passed!")
