import paddle
import torch
import numpy as np
import os

def test_pow_diff(shape):
    paddle.set_device("cpu")

    # theta=10000.0
    theta = np.random.rand() * 10
    dim=40.0
    # input_np = np.arange(0, dim, 2, dtype=np.float32)
    input_np = np.random.random(shape).astype(np.float32)
    print("shape：", shape)

    # [torch]
    inv_freq_torch = torch.tensor(input_np) / dim
    # inv_freq_torch = torch.tensor(input_np).cuda() / torch.tensor(dim).cuda()
    print("inv_freq_torch.device: ", inv_freq_torch.device)

    # [Paddle]
    # inv_freq_paddle = paddle.to_tensor(input_np) / dim
    # inv_freq_paddle = paddle.Tensor.__div__(paddle.to_tensor(input_np), dim)

    # inv_freq_paddle = paddle.to_tensor(input_np) / paddle.to_tensor(dim)
    inv_freq_paddle = paddle.divide(paddle.to_tensor(input_np), paddle.to_tensor(dim))
    print("inv_freq_paddle.device: ", inv_freq_paddle.device)

    # [验证精度]
    inv_freq_torch_np = inv_freq_torch.cpu().numpy()
    inv_freq_paddle_np = inv_freq_paddle.cpu().numpy()
    diff = np.abs(inv_freq_torch.cpu().numpy() - inv_freq_paddle.cpu().numpy()).max()
    print(f"Paddle vs Torch Diff: {diff} <--- 修复后的 Diff")

    if diff != 0:
        exit(1)
    # for i in range(len(inv_freq_torch_np)):
    #     if(inv_freq_torch_np[i] != inv_freq_paddle_np[i]):
    #         print(i, inv_freq_torch_np[i], inv_freq_paddle_np[i])
    #         exit(1)


if __name__ == "__main__":
    # for i in range(900, 1000):
    #     for j in range(900, 1000):
    #         test_pow_diff((i, j))
    test_pow_diff((1, 30))
   