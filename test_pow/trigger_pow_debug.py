import torch

def run_test():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This test requires a GPU.")
        return

    print("准备开始测试 (Ensure you are using the modified PyTorch)...")
    
    # 创建一个简单的 CUDA 张量
    x = torch.tensor([2.0, 3.0], device='cuda')
    y = torch.tensor([2.0, 3.0], device='cuda')
    x_cpu = torch.tensor([2.0, 3.0])
    y_cpu = torch.tensor([2.0, 3.0])
    
    print("\n=== Case 1: Scalar ** Tensor (应该触发 pow_scalar_tensor_impl) ===")
    # 2.0 是标量, x 是 Tensor
    res1 = 2.0 ** x
    # 强制同步以确保 printf 输出（如果 printf 在 kernel 内部，需要同步；如果是在 host 端，则会直接打印）
    torch.cuda.synchronize()
    print("Case 1 完成")

    # print("\n=== Case 2: Tensor ** Scalar (应该触发 pow_tensor_scalar_kernel) ===")
    # # x 是 Tensor, 3.0 是标量
    # res2 = x ** 3.0
    # torch.cuda.synchronize()
    # print("Case 2 完成")

    # print("\n=== Case 3: Tensor ** Tensor (应该触发 pow_ generic kernel) ===")
    # # x 和 y 都是 Tensor
    # res3 = x ** y
    # torch.cuda.synchronize()
    # print("Case 3 完成")

    # print("\n=== Case 4: In-place Tensor.pow_(Scalar) ===")
    # # In-place 操作通常路径不同
    # x_clone = x.clone()
    # x_clone.pow_(2.0)
    # torch.cuda.synchronize()
    # print("Case 4 完成")

    # print("\n=== Case 5: Tensor ** 0-dim CPU Tensor (Mixed Device) ===")
    # # 尝试直接传 0-dim CPU Tensor，看是否能保留 Scalar 属性
    # s_cpu = torch.tensor(3.0, device='cpu')
    # try:
    #     # 某些版本允许混合设备运算，如果优化得当，可能会触发
    #     res5 = torch.pow(s_cpu, x)
    #     torch.cuda.synchronize()
    #     print("Case 5 完成")
    # except Exception as e:
    #     print(f"Case 5 Skipped: {e}")

    # print("\n=== Case 6: In-place Tensor.pow_(0-dim CPU Tensor) ===")
    # try:
    #     x_clone2 = x.clone()
    #     s_cpu = torch.tensor(2.0, device='cpu')
    #     # In-place 混合设备有时有特殊处理
    #     x_clone2.pow_(s_cpu)
    #     torch.cuda.synchronize()
    #     print("Case 6 完成")
    # except Exception as e:
    #     print(f"Case 6 Skipped: {e}")
        
if __name__ == "__main__":
    run_test()