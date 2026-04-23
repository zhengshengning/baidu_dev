import numpy as np
import paddle
import torch

def compare_log_float32():
    """对比 paddle.log 和 torch.log 在 float32 类型下的逐位差异"""
    
    # 设置平台为 GPU
    paddle.set_device('gpu')
    torch_device = torch.device('cuda')
    
    # 设置随机种子保证可复现
    np.random.seed(42)
    
    # 创建测试数据 (正数，避免 log 的负数问题)
    shape = (1000,)
    data = np.random.rand(*shape).astype(np.float32) * 10 + 0.01  # 范围 [0.01, 10.01]
    
    # 转换为 paddle 和 torch tensor (GPU)
    paddle_input = paddle.to_tensor(data, dtype=paddle.float32, place=paddle.CUDAPlace(0))
    torch_input = torch.tensor(data, dtype=torch.float32, device=torch_device)
    
    # 计算 log
    paddle_result = paddle.log(paddle_input)
    torch_result = torch.log(torch_input)
    
    # 转换为 numpy 进行对比 (GPU tensor 需先转到 CPU)
    paddle_np = paddle_result.cpu().numpy()
    torch_np = torch_result.cpu().numpy()
    
    # 逐位对比 (使用 view 转换为 uint32 进行二进制比较)
    paddle_bits = paddle_np.view(np.uint32)
    torch_bits = torch_np.view(np.uint32)
    
    # 找出不相同的位置
    diff_mask = paddle_bits != torch_bits
    diff_indices = np.where(diff_mask)[0]
    
    print(f"总元素数: {len(data)}")
    print(f"不相同的元素数: {len(diff_indices)}")
    print(f"差异比例: {len(diff_indices) / len(data) * 100:.2f}%")
    print("-" * 80)
    
    if len(diff_indices) > 0:
        print(f"{'Index':<8} {'Input':<15} {'Paddle':<20} {'Torch':<20} {'Paddle(hex)':<15} {'Torch(hex)':<15} {'ULP diff'}")
        print("-" * 120)
        
        # 只显示前 20 个差异
        for idx in diff_indices[:20]:
            input_val = data[idx]
            paddle_val = paddle_np[idx]
            torch_val = torch_np[idx]
            paddle_hex = format(paddle_bits[idx], '08x')
            torch_hex = format(torch_bits[idx], '08x')
            
            # 计算 ULP (Units in Last Place) 差异
            ulp_diff = abs(int(paddle_bits[idx]) - int(torch_bits[idx]))
            
            print(f"{idx:<8} {input_val:<15.8f} {paddle_val:<20.15f} {torch_val:<20.15f} {paddle_hex:<15} {torch_hex:<15} {ulp_diff}")
        
        if len(diff_indices) > 20:
            print(f"... 还有 {len(diff_indices) - 20} 个差异未显示")
    else:
        print("所有元素完全相同！")
    
    # 统计数值差异
    print("\n" + "=" * 80)
    print("数值差异统计:")
    abs_diff = np.abs(paddle_np - torch_np)
    rel_diff = abs_diff / (np.abs(torch_np) + 1e-10)
    print(f"  最大绝对差: {np.max(abs_diff):.2e}")
    print(f"  平均绝对差: {np.mean(abs_diff):.2e}")
    print(f"  最大相对差: {np.max(rel_diff):.2e}")
    print(f"  平均相对差: {np.mean(rel_diff):.2e}")

if __name__ == "__main__":
    compare_log_float32()