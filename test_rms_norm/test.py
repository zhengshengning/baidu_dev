

import paddle
import torch
import numpy as np
import time

# batch_sizes = 200
# hidden_size = 2048
# seq_lens = 32

# np_x = np.random.randn(batch_sizes, seq_lens, hidden_size).astype("float32")

# x_paddle = paddle.to_tensor(np_x, dtype="float16")
# y_paddle = paddle.nn.functional.rms_norm(x_paddle, (hidden_size,), None, eps = 1e-5)
# # print(y_paddle)


# x_torch = torch.tensor(np_x, dtype=torch.float16).cuda()
# y_torch = torch.nn.functional.rms_norm(x_torch, (hidden_size,), None ,eps = 1e-5)
# # print(y_torch)

# # === 精度对比 ===
# y_paddle_np = y_paddle[0].astype('float32').numpy()
# y_torch_np = y_torch.cpu().to(dtype=torch.float32).numpy()
# np.testing.assert_allclose(y_paddle_np, y_torch_np, rtol=0, atol=0)
# print("精度对账通过！")

x_paddle = paddle.ones([5, 5], dtype="float32")
y_paddle = paddle.nn.functional.rms_norm(x_paddle, [5], None, eps = 1e-5)
print(y_paddle[0])

x_torch = torch.ones([5, 5], dtype=torch.float32).cuda()
y_torch = torch.nn.functional.rms_norm(x_torch, [5], None, eps = 1e-5)
print(y_torch)

# === 精度对比 ===
y_paddle_np = y_paddle[0].numpy()
y_torch_np = y_torch.cpu().numpy()
np.testing.assert_allclose(y_paddle_np, y_torch_np, rtol=0, atol=0)
print("精度对账通过！")