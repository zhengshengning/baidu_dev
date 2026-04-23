import numpy as np

import torch
import paddle
import random

import os
os.environ['FLAGS_use_accuracy_compatible_kernel'] = '1'
os.environ['FLAGS_embedding_deterministic'] = '1'
os.environ['FLAGS_cudnn_deterministic'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

precision=15

paddle.set_printoptions(precision=precision)

torch.set_printoptions(
    precision=precision,        # 小数位数
    sci_mode=False,      # 关闭科学计数法
    linewidth=200,       # 每行字符数，防止被截断
    threshold=10_000,    # 超过这个元素数才省略
)

def set_deterministic(seed=42):                                                                                                 
    random.seed(seed)                                                                                                           
    np.random.seed(seed)                                                                                                        
    torch.manual_seed(seed)                                                                                                     
    torch.cuda.manual_seed(seed)                                                                                                
    torch.cuda.manual_seed_all(seed)                                                                                            
                                                                                                                                
    torch.use_deterministic_algorithms(True)                                                                                    
    torch.backends.cudnn.deterministic = True                                                                                   
    torch.backends.cudnn.benchmark = False                                                                                      


SEED = 42
set_deterministic(SEED)

def compare_diff(np_pd, np_torch, name = ""):
    diff = np_pd - np_torch
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))

    print(f"{name}, max abs diff:", max_abs)
    print(f"{name}, mean abs diff:", mean_abs)


def np_to_torch(np_data, dtype=torch.float, device="cpu") -> torch.Tensor:
    return torch.from_numpy(np_data).to(dtype).to(device)


def torch_to_np(torch_data: torch.Tensor):
    return torch_data.float().detach().cpu().numpy()


def np_to_paddle(np_data, dtype="float32", device="cpu") -> paddle.Tensor:
    return paddle.to_tensor(np_data, dtype=dtype).to(device)


def paddle_to_np(paddle_data: paddle.Tensor):
    return paddle_data.astype("float32").detach().cpu().numpy()
    
    
if __name__ == "__main__":
    channels = 1280
    length = 15000
    max_timescale = 10000

    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)

    device = "cpu"

    # compare exp & sin & cos: diff!
    torch_inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float().to(device))
    print(torch_inv_timescales.device)
    np_torch_inv_timescales = torch_to_np(torch_inv_timescales)

    paddle_inv_timescales = paddle.exp(-log_timescale_increment * paddle.arange(channels // 2).float().to(device))
    print(paddle_inv_timescales.device)
    np_paddle_inv_timescales = paddle_to_np(paddle_inv_timescales)

    compare_diff(np_torch_inv_timescales, np_paddle_inv_timescales, "inv_timescales")
    print("len(np_torch_inv_timescales)", len(np_torch_inv_timescales))

