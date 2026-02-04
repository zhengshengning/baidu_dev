import torch

def test_pow_backward():
    x = torch.tensor([2.,3.,4.], device="cuda").requires_grad_()
    p = torch.tensor([3.,2.,1.], device="cuda").requires_grad_()

    print("========================前向========================")
    y = torch.pow(x, p)      # y = x^p


    print("\n\n========================反向========================")
    loss = y.sum()           # 简化反向
    loss.backward()

    # print("x.grad:", x.grad)
    # print("p.grad:", p.grad)

if __name__ == "__main__":
    test_pow_backward()
