import numpy as np
import paddle
import torch
import random

import os
paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})
os.environ['FLAGS_embedding_deterministic'] = '1'
os.environ['FLAGS_cudnn_deterministic'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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


def np_to_torch(np_data, dtype=torch.float) -> torch.Tensor:
    return torch.from_numpy(np_data).to(dtype).to("cuda")


def torch_to_np(torch_data: torch.Tensor):
    return torch_data.float().detach().cpu().numpy()


def np_to_paddle(np_data, dtype="float32") -> paddle.Tensor:
    return paddle.to_tensor(np_data, dtype=dtype).cuda()


def paddle_to_np(paddle_data: paddle.Tensor):
    return paddle_data.astype("float32").detach().cpu().numpy()
    
    
if __name__ == "__main__":
    hidden_states_file = "hidden_states_before_layernorm.npy"
    weight_file = "layernorm_weight.npy"
    bias_file = "layernorm_bias.npy"

    np_hidden_states = np.load(hidden_states_file)
    np_weight = np.load(weight_file)
    np_bias = np.load(bias_file)

    embed = 1280

    torch_layernorm = torch.nn.LayerNorm(embed).cuda()
    torch_layernorm.weight.data = np_to_torch(np_weight)
    torch_layernorm.bias.data = np_to_torch(np_bias)
    torch_hidden_states = np_to_torch(np_hidden_states)
    torch_layernorm_output = torch_layernorm(torch_hidden_states)
    np_torch_layernorm_output = torch_to_np(torch_layernorm_output)

    paddle_layernorm = paddle.nn.LayerNorm(embed).cuda()
    paddle_layernorm.weight.data = np_to_paddle(np_weight)
    paddle_layernorm.bias.data = np_to_paddle(np_bias)
    paddle_hidden_states = np_to_paddle(np_hidden_states)
    paddle_layernorm_output = paddle_layernorm(paddle_hidden_states)
    np_paddle_layernorm_output = paddle_to_np(paddle_layernorm_output)

    compare_diff(np_torch_layernorm_output, np_paddle_layernorm_output, "hidden_states_after_layernorm")

