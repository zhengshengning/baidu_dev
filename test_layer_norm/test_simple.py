import numpy as np
import paddle
import torch
import os

os.environ['FLAGS_use_accuracy_compatible_kernel'] = '1'

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def test_layer_norm_aligned(embed=1280, batch=1):
    # Create simple test data
    np_x = np.random.randn(batch, embed).astype('float32')
    np_weight = np.random.randn(embed).astype('float32')
    np_bias = np.random.randn(embed).astype('float32')

    # PyTorch
    torch_ln = torch.nn.LayerNorm(embed).cuda()
    torch_ln.weight.data = torch.from_numpy(np_weight).cuda()
    torch_ln.bias.data = torch.from_numpy(np_bias).cuda()
    torch_x = torch.from_numpy(np_x).cuda()
    torch_out = torch_ln(torch_x)
    np_torch_out = torch_out.float().detach().cpu().numpy()

    # Paddle
    paddle_ln = paddle.nn.LayerNorm(embed).cuda()
    paddle_ln.weight.data = paddle.to_tensor(np_weight).cuda()
    paddle_ln.bias.data = paddle.to_tensor(np_bias).cuda()
    paddle_x = paddle.to_tensor(np_x).cuda()
    paddle_out = paddle_ln(paddle_x)
    np_paddle_out = paddle_out.astype('float32').detach().cpu().numpy()

    # Compare
    diff = np_torch_out - np_paddle_out
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    max_rel = np.max(np.abs(diff / (np.abs(np_torch_out) + 1e-7)))

    print(f"Embed: {embed}, Batch: {batch}")
    print(f"Max abs diff: {max_abs}")
    print(f"Mean abs diff: {mean_abs}")
    print(f"Max rel diff: {max_rel}")
    print(f"Max value magnitude: {np.max(np.abs(np_torch_out))}")
    print(f"Torch stats: mean={np.mean(np_torch_out):.6f}, std={np.std(np_torch_out):.6f}")
    print(f"Paddle stats: mean={np.mean(np_paddle_out):.6f}, std={np.std(np_paddle_out):.6f}")
    print()

    # Check if bit-for-bit match
    if max_abs < 1e-6:
        print("PASS: Close enough")
    else:
        print(f"FAIL: diff {max_abs} > 1e-6")

    return max_abs, mean_abs, max_rel

if __name__ == "__main__":
    # Test various configurations
    test_layer_norm_aligned(embed=1280, batch=1)
    test_layer_norm_aligned(embed=1024, batch=1)
    test_layer_norm_aligned(embed=512, batch=2)
