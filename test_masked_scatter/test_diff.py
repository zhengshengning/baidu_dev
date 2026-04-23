
import paddle
import numpy as np
import torch


x_shape = (120,)
mask_shape = (300, 120)
dtype = "float32"
value_shape = (300, 300)

x_np = np.random.rand(*x_shape).astype(dtype)
mask_np = np.random.rand(*mask_shape).astype(dtype)
value_np = np.random.rand(*value_shape).astype(dtype)


# paddle
paddle.set_device('cpu')
paddle_x = paddle.to_tensor(x_np, dtype=dtype)
paddle_x.stop_gradient = False
paddle_mask = paddle.to_tensor(mask_np).astype('bool')
paddle_value = paddle.to_tensor(value_np, dtype=dtype)
paddle_value.stop_gradient = False
paddle_result = paddle.masked_scatter(paddle_x, paddle_mask, paddle_value)
paddle_loss = paddle.sum(paddle_result)
paddle_loss.backward()

print("paddle x.grad device:", paddle_x.grad.place)
print("paddle value.grad device:", paddle_value.grad.place)
print("paddle_result.shape", paddle_result.shape)
print("paddle_x.grad.shape", paddle_x.grad.shape)
print("paddle_value.grad.shape", paddle_value.grad.shape)

# torch
torch_x = torch.tensor(x_np, dtype=torch.float32, device='cuda', requires_grad=True)
torch_mask = torch.tensor(mask_np, dtype=torch.bool, device='cuda')
torch_value = torch.tensor(value_np, dtype=torch.float32, device='cuda', requires_grad=True)
torch_result = torch.masked_scatter(torch_x, torch_mask, torch_value)
torch_loss = torch.sum(torch_result)
torch_loss.backward()

print("torch x.grad device:", torch_x.grad.device)
print("torch value.grad device:", torch_value.grad.device)
print("torch_result.shape", torch_result.shape)
print("torch_x.grad.shape", torch_x.grad.shape)
print("torch_value.grad.shape", torch_value.grad.shape)