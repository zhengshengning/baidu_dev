import numpy as np
import paddle
import torch

def test_topk(top_k, np_router_logits):
    paddle_router_logits = paddle.to_tensor(np_router_logits, dtype="bfloat16").cuda()
    pd_router_top_value, pd_router_indices = paddle.topk(paddle_router_logits, top_k, dim=1, sorted=True)
    np_pd_router_top_value = pd_router_top_value.astype("float32").detach().cpu().numpy()
    np_pd_router_indices = pd_router_indices.astype("float32").detach().cpu().numpy()

    torch_router_logits = torch.from_numpy(np_router_logits).to(torch.bfloat16).to("cuda")
    torch_router_top_value, torch_router_indices = torch.topk(torch_router_logits, top_k, dim=1, sorted=True)
    np_torch_router_top_value = torch_router_top_value.float().detach().cpu().numpy()
    np_torch_router_indices = torch_router_indices.float().detach().cpu().numpy()


    # print("【top_indices】")
    print("paddle: ", type(pd_router_indices), pd_router_indices.shape, "\n", pd_router_indices)
    print("torch: ", type(torch_router_indices), torch_router_indices.shape, "\n", torch_router_indices)
    # h, w = np_pd_router_indices.shape
    # for i in range(h):
    #     for j in range(w):
    #         if np_pd_router_indices[i][j] != np_torch_router_indices[i][j]:
    #             print("top_indices error at [", i, ",", j, "]: paddle = ", np_pd_router_indices[i][j], " torch = ", np_torch_router_indices[i][j])

    # 输出diff
    diff = np_pd_router_indices - np_torch_router_indices
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    print("max abs diff:", max_abs)
    print("mean abs diff:", mean_abs)



if __name__ == "__main__":
    # router_logits_file = "router_logits.npy"
    # np_router_logits = np.load(router_logits_file)
    # test_topk(8, np_router_logits)
    # print(type(np_router_logits), np_router_logits.dtype, np_router_logits.shape)
    # print("np_router_logits[0][43]: ", np_router_logits[0][43], "    np_router_logits[0][47]: ", np_router_logits[0][47])

    # for i in range(1, 33):
    #     print("top_k = ", i)
    #     test_topk(i, np_router_logits)

    # np_router_logits = (np.random.randint(-10, 10, size=(20, 126))).astype("int32")
    np_router_logits = (np.ones((64, 64, 64)).astype("int32"))
    test_topk(8, np_router_logits)
