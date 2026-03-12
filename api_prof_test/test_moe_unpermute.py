# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import csv
import itertools
import unittest

import numpy as np

import paddle
from paddle.nn.functional import moe_permute, moe_unpermute

paddle.seed(42)

def fabricate_dispatch_result(
    seqlen,
    token_length,
    topk,
    num_experts,
    data_type="bfloat16",
    broadcast_ratio=0.5,
):
    """Helper function to generate test data."""
    hidden_states = paddle.randn([seqlen, token_length]).astype(data_type)

    scale = paddle.empty([0])
    if data_type == "float8_e4m3fn":
        scale_cols = (token_length + 127) // 128
        scale = paddle.randn([seqlen, scale_cols], dtype="float32")

    # Calculate expert counts with normal distribution
    expected_experts = max(1, min(broadcast_ratio * num_experts, topk))
    std_dev = max(1, expected_experts / 6)
    experts_count = paddle.normal(expected_experts, std_dev, [seqlen])
    experts_count = paddle.clip(
        paddle.round(experts_count), 1, min(topk, num_experts)
    )
    experts_count = paddle.cast(experts_count, "int32")

    # Preallocate results
    expert_routemap_topk = paddle.full([seqlen, topk], -1, dtype="int32")
    expert_prob_topk = paddle.zeros([seqlen, topk], dtype="float32")

    # Batch generate expert indices and probabilities
    for i in range(seqlen):
        count = experts_count[i].item()
        indices = paddle.randperm(num_experts)[:count]
        expert_routemap_topk[i, :count] = indices
        prob_value = 1.0 / count
        expert_prob_topk[i, :count] = paddle.full(
            [count], prob_value, dtype=data_type
        )

    # Calculate expert token counts
    valid_indices = expert_routemap_topk.reshape([-1])
    valid_mask = valid_indices >= 0
    valid_experts = valid_indices[valid_mask]
    tokens_per_expert = paddle.histogram(
        valid_experts, bins=num_experts, min=0, max=num_experts - 1
    )
    tokens_per_expert = paddle.cast(tokens_per_expert, "int32")
    # tokens_per_expert = list(tokens_per_expert)
    tokens_per_expert = tokens_per_expert.numpy()

    return (
        hidden_states,
        scale,
        expert_routemap_topk,
        expert_prob_topk,
        tokens_per_expert,
    )


def tensor_max_abs_rel_err(a, b, eps=1e-8):
    """Calculate max absolute and relative error between two tensors."""
    max_abs_err = paddle.max(paddle.abs(a - b))
    denom = paddle.maximum(paddle.abs(a), paddle.abs(b))
    denom = paddle.maximum(denom, paddle.to_tensor(eps, dtype=denom.dtype))
    max_rel_err = paddle.max(paddle.abs(a - b) / denom)
    return max_abs_err, max_rel_err


results = []

class TestFusedMoePermuteUnpermute(unittest.TestCase):
    """Test cases for moe_permute and moe_unpermute."""
    # 抓取的真实配置
    # SEQLENS = [7399, 7828, 7843, 8127, 9371, 9425, 9519, 9761, 9909, 10539, 10556, 10564, 10906, 11023, 11057, 11217, 11262, 11634, 11788, 11868, 11930, 12240, 12304, 12364, 12397, 12535, 12794, 12854, 13021, 13046, 13062, 13066, 13305, 13332, 13403, 13514, 13569, 13584, 13641, 13703, 13789, 14048, 14186, 14343, 14381, 14457, 14626, 14634, 14655, 14756, 14981, 15102, 15124, 15245, 15299, 15382, 15412, 15466, 15528, 15600, 15632, 15650, 15677, 15737, 15767, 15769, 15804, 15910, 15916, 15995, 16030, 16031, 16121, 16158, 16254, 16280, 16338, 16355, 16385, 16394, 16438, 16689, 16722, 16734, 16777, 16780, 16804, 16835, 16867, 16958, 16979, 16985, 17104, 17140, 17207, 17293, 17339, 17359, 17448, 17451, 17505, 17531, 17562, 17566, 17611, 17619, 17632, 17641, 17642, 17668, 17720, 17796, 17836, 17935, 17945, 17959, 18071, 18087, 18150, 18198, 18201, 18206, 18212, 18216, 18222, 18252, 18283, 18288, 18475, 18510, 18536, 18626, 18636, 18811, 18814, 18830, 18874, 18908, 18990, 19015, 19079, 19127, 19158, 19183, 19272, 19289, 19364, 19447, 19462, 19541, 19575, 19632, 19643, 19646, 19654, 19711, 19748, 19773, 19805, 19886, 19898, 19904, 19933, 19963, 19972, 20103, 20111, 20135, 20168, 20179, 20196, 20222, 20298, 20320, 20430, 20492, 20496, 20610, 20632, 20658, 20689, 20731, 20762, 20771, 20825, 20943, 20956, 20982, 21062, 21122, 21170, 21251, 21273, 21325, 21331, 21365, 21404, 21441, 21485, 21506, 21517, 21522, 21552, 21577, 21606, 21624, 21654, 21656, 21658, 21667, 21722, 21735, 21753, 21763, 21836, 21848, 21879, 21904, 21910, 21928, 21971, 21979, 21984, 22008, 22010, 22012, 22040, 22077, 22089, 22099, 22105, 22114, 22122, 22134, 22160, 22215, 22216, 22266, 22285, 22290, 22326, 22344, 22362, 22370, 22379, 22464, 22529, 22578, 22592, 22628, 22650, 22657, 22661, 22742, 22744, 22762, 22791, 22803, 22867, 22872, 22883, 22891, 22901, 23054, 23067, 23078, 23140, 23170, 23190, 23198, 23236, 23239, 23251, 23255, 23340, 23390, 23409, 23455, 23457, 23501, 23516, 23545, 23634, 23647, 23679, 23680, 23702, 23743, 23749, 23815, 23822, 23835, 23837, 23841, 23843, 23845, 23849, 23871, 23883, 23898, 23905, 23916, 23945, 23976, 23990, 24029, 24065, 24080, 24092, 24099, 24109, 24115, 24145, 24175, 24190, 24214, 24215, 24220, 24255, 24298, 24311, 24356, 24414, 24477, 24587, 24592, 24594, 24611, 24625, 24626, 24647, 24689, 24701, 24712, 24720, 24737, 24751, 24761, 24772, 24778, 24794, 24811, 24832, 24852, 24879, 24884, 24921, 24942, 24951, 24965, 25017, 25029, 25039, 25041, 25074, 25093, 25145, 25217, 25228, 25284, 25291, 25335, 25351, 25366, 25375, 25425, 25467, 25496, 25506, 25586, 25604, 25623, 25630, 25668, 25700, 25703, 25729, 25730, 25755, 25791, 25856, 25865, 25879, 25883, 25889, 25894, 25914, 25915, 25922, 25955, 25975, 26042, 26069, 26075, 26077, 26171, 26172, 26176, 26177, 26180, 26196, 26217, 26230, 26244, 26255, 26262, 26276, 26281, 26315, 26325, 26384, 26390, 26393, 26411, 26453, 26478, 26480, 26532, 26533, 26605, 26614, 26634, 26645, 26667, 26710, 26715, 26716, 26723, 26758, 26763, 26768, 26769, 26797, 26807, 26826, 26841, 26852, 26853, 26873, 26883, 26918, 26948, 26986, 27049, 27063, 27079, 27084, 27086, 27113, 27133, 27153, 27164, 27168, 27171, 27184, 27187, 27194, 27198, 27200, 27222, 27232, 27234, 27281, 27308, 27343, 27347, 27362, 27489, 27500, 27501, 27512, 27540, 27545, 27546, 27565, 27567, 27598, 27618, 27632, 27650, 27663, 27701, 27703, 27726, 27728, 27738, 27824, 27834, 27841, 27863, 27877, 27898, 27905, 27911, 27974, 27999, 28013, 28019, 28046, 28047, 28060, 28063, 28064, 28069, 28071, 28076, 28078, 28122, 28129, 28148, 28190, 28217, 28226, 28278, 28321, 28340, 28350, 28363, 28414, 28438, 28455, 28487, 28505, 28515, 28542, 28569, 28580, 28589, 28600, 28647, 28688, 28717, 28735, 28771, 28810, 28816, 28834, 28865, 28950, 28979, 28991, 29006, 29043, 29053, 29106, 29114, 29122, 29131, 29133, 29153, 29159, 29173, 29181, 29263, 29272, 29281, 29347, 29363, 29369, 29370, 29383, 29385, 29388, 29390, 29444, 29454, 29457, 29460, 29465, 29469, 29487, 29489, 29496, 29509, 29541, 29548, 29556, 29571, 29579, 29602, 29603, 29608, 29617, 29636, 29662, 29667, 29678, 29694, 29737, 29766, 29770, 29790, 29799, 29817, 29823, 29827, 29830, 29832, 29834, 29850, 29854, 29856, 29884, 29895, 29910, 29936, 29937, 29952, 29978, 29980, 29990, 30012, 30030, 30043, 30108, 30123, 30130, 30140, 30156, 30187, 30206, 30207, 30208, 30234, 30243, 30253, 30254, 30265, 30288, 30291, 30321, 30340, 30349, 30388, 30389, 30416, 30432, 30434, 30435, 30437, 30438, 30459, 30472, 30479, 30504, 30506, 30529, 30532, 30540, 30545, 30546, 30556, 30557, 30580, 30660, 30662, 30668, 30690, 30703, 30719, 30728, 30743, 30745, 30764, 30780, 30792, 30799, 30816, 30850, 30893, 30947, 30948, 31008, 31022, 31069, 31075, 31120, 31132, 31144, 31172, 31179, 31190, 31201, 31229, 31250, 31258, 31267, 31270, 31271, 31283, 31306, 31348, 31360, 31368, 31371, 31384, 31394, 31405, 31435, 31471, 31478, 31480, 31486, 31499, 31504, 31527, 31528, 31532, 31557, 31564, 31584, 31603, 31655, 31662, 31674, 31689, 31711, 31774, 31787, 31795, 31796, 31811, 31842, 31853, 31903, 31938, 31963, 31966, 31972, 31994, 32005, 32011, 32027, 32038, 32045, 32069, 32077, 32101, 32118, 32119, 32157, 32165, 32171, 32172, 32237, 32265, 32266, 32270, 32275, 32302, 32308, 32356, 32391, 32421, 32432, 32437, 32442, 32449, 32454, 32496, 32508, 32517, 32521, 32572, 32599, 32620, 32623, 32637, 32646, 32649, 32652, 32676, 32688, 32726, 32755, 32787, 32796, 32819, 32834, 32846, 32858, 32871, 32898, 32942, 32964, 32966, 32968, 32978, 33072, 33073, 33112, 33119, 33124, 33142, 33152, 33181, 33186, 33227, 33249, 33353, 33393, 33408, 33414, 33429, 33461, 33488, 33494, 33557, 33561, 33562, 33567, 33598, 33607, 33620, 33623, 33631, 33635, 33663, 33667, 33703, 33705, 33736, 33741, 33764, 33790, 33795, 33819, 33834, 33836, 33837, 33843, 33850, 33853, 33857, 33894, 33901, 33919, 33946, 33968, 33978, 33992, 33993, 34029, 34055, 34070, 34092, 34112, 34114, 34162, 34207, 34220, 34238, 34257, 34310, 34398, 34413, 34457, 34483, 34495, 34530, 34608, 34609, 34632, 34649, 34670, 34694, 34709, 34720, 34728, 34729, 34748, 34752, 34759, 34805, 34822, 34833, 34875, 34878, 34900, 34904, 34908, 34910, 34930, 34950, 34952, 34963, 35038, 35044, 35063, 35076, 35083, 35104, 35121, 35124, 35126, 35259, 35268, 35289, 35337, 35401, 35458, 35491, 35503, 35518, 35519, 35554, 35561, 35604, 35614, 35635, 35653, 35666, 35668, 35679, 35683, 35705, 35718, 35727, 35733, 35767, 35770, 35775, 35809, 35826, 35848, 35876, 35884, 35984, 36017, 36053, 36097, 36119, 36136, 36137, 36167, 36171, 36173, 36177, 36190, 36202, 36212, 36244, 36247, 36250, 36293, 36311, 36317, 36345, 36347, 36357, 36359, 36457, 36466, 36491, 36498, 36500, 36511, 36536, 36551, 36565, 36583, 36588, 36606, 36615, 36628, 36640, 36680, 36700, 36706, 36712, 36751, 36752, 36773, 36786, 36865, 36872, 36901, 36951, 36952, 36996, 37006, 37008, 37020, 37026, 37045, 37063, 37153, 37181, 37243, 37274, 37294, 37324, 37328, 37377, 37398, 37420, 37487, 37497, 37501, 37514, 37517, 37533, 37567, 37583, 37584, 37652, 37655, 37661, 37690, 37693, 37700, 37718, 37753, 37761, 37777, 37821, 37888, 37944, 37957, 37959, 37979, 37996, 37998, 38014, 38048, 38069, 38111, 38121, 38129, 38165, 38207, 38243, 38249, 38260, 38267, 38287, 38294, 38333, 38337, 38363, 38391, 38402, 38458, 38493, 38509, 38511, 38512, 38528, 38588, 38608, 38613, 38619, 38620, 38654, 38657, 38697, 38797, 38808, 38817, 38838, 38851, 38880, 38930, 38952, 38960, 38963, 38978, 38980, 39046, 39052, 39059, 39067, 39084, 39093, 39111, 39118, 39129, 39179, 39218, 39225, 39298, 39302, 39323, 39375, 39471, 39481, 39488, 39517, 39565, 39579, 39581, 39598, 39680, 39683, 39694, 39726, 39744, 39760, 39812, 39843, 39856, 39859, 39897, 39908, 39937, 39951, 39959, 40042, 40066, 40073, 40082, 40101, 40111, 40132, 40143, 40186, 40329, 40338, 40350, 40385, 40405, 40427, 40463, 40468, 40533, 40549, 40660, 40718, 40733, 40821, 40910, 40988, 41017, 41028, 41092, 41128, 41134, 41156, 41185, 41215, 41345, 41394, 41412, 41482, 41502, 41534, 41577, 41588, 41599, 41600, 41605, 41657, 41662, 41680, 41715, 41757, 41766, 41799, 41811, 41875, 41884, 41905, 41997, 42164, 42194, 42219, 42220, 42226, 42244, 42283, 42285, 42418, 42428, 42550, 42561, 42594, 42619, 42626, 42661, 42683, 42722, 42780, 42876, 42879, 42887, 42971, 42995, 43010, 43027, 43031, 43090, 43115, 43170, 43197, 43205, 43206, 43260, 43279, 43306, 43348, 43355, 43412, 43457, 43509, 43533, 43603, 43681, 43745, 43754, 43823, 43868, 43961, 43983, 44062, 44136, 44167, 44224, 44226, 44241, 44341, 44377, 44389, 44393, 44474, 44485, 44609, 44614, 44638, 44722, 44755, 44782, 44807, 44816, 44864, 44896, 44927, 45046, 45050, 45117, 45122, 45202, 45277, 45313, 45412, 45483, 45505, 45544, 45548, 45714, 45736, 45927, 45940, 46095, 46130, 46252, 46258, 46270, 46285, 46325, 46356, 46389, 46393, 46545, 46634, 46691, 46733, 47037, 47058, 47174, 47652, 47679, 47724, 47817, 47822, 47853, 47902, 47997, 48011, 48037, 48106, 48197, 48267, 48374, 48419, 48426, 48483, 48554, 48717, 48795, 48876, 49204, 49234, 49514, 49595, 49733, 49876, 49978, 50053, 50102, 50440, 50499, 50646, 50662, 50787, 50971, 51537, 51894, 52500, 52938, 52953, 53362, 55153]
    SEQLENS = [7399, 10556, 11930, 13062, 14981, 15632, 16030, 17505, 18201, 19079, 20196, 21722, 22326, 23236, 24109, 25017, 26196, 27153, 28060, 29159, 30206, 31179, 32005, 33112, 34220, 35458, 36097, 37006, 38207, 39059, 40082, 41345, 42285, 43090, 44377, 45202, 46389, 47679, 48197, 49204, 50499, 53362]
    # SEQLENS = [7399, 7828, 7843]
    TOKEN_LEN = 7168
    DTYPES = ["float8_e4m3fn", "bfloat16"]
    EXPERT_NUMS = [16]
    TOPKS = [8]

    def setUp(self):
        """Initialize test environment."""
        paddle.seed(42)  # For reproducibility

    def test_permute_unpermute_consistency(self):
        """Test that permute + unpermute recovers original tensors."""
        # 检查CUDA是否可用
        if paddle.device.is_compiled_with_cuda():
            print("CUDA可用，使用GPU进行测试")
            paddle.set_device('gpu:3')
        else:
            print("警告: CUDA不可用或PyTorch未检测到CUDA")
            if not paddle.device.is_compiled_with_cuda():
                print("  - Paddle CUDA不可用")
            exit(1)

        for dt, seqlen, expert_num, topk in itertools.product(
            self.DTYPES, self.SEQLENS, self.EXPERT_NUMS, self.TOPKS
        ):
            print("\n\n" + "=" * 60)
            print(f"Configuration: SEQLEN={seqlen}, TOKEN_LEN={self.TOKEN_LEN}, dtype={dt}")
            print(f"               expert_num={expert_num}, topk={topk}")
            print("-" * 60)
            with self.subTest(dtype=dt, expert_num=expert_num, topk=topk):
                (
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    tokens_per_expert,
                ) = fabricate_dispatch_result(
                    seqlen,
                    self.TOKEN_LEN,
                    topk,
                    expert_num,
                    data_type=dt,
                    broadcast_ratio=0.1,
                )
                if dt == "bfloat16":
                    scale = None

                # Permute step
                (
                    unzipped_tokens,
                    zipped_expertwise_rowmap,
                    unzipped_probs,
                    unzipped_scales,
                ) = moe_permute(
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_experts=expert_num,
                    tokens_per_expert=tokens_per_expert,
                    padding_alignment=128,
                )
                unpermute_input = (
                    unzipped_tokens.astype("float32")
                    * unzipped_probs.unsqueeze(-1)
                ).astype("bfloat16")

                # print("hidden_states", hidden_states.shape, hidden_states.dtype)
                # print("unzipped_tokens", unzipped_tokens.shape, unzipped_tokens.dtype)

                # Warmup
                for _ in range(100):
                    moe_unpermute(
                        unpermute_input,
                        zipped_expertwise_rowmap,
                        expert_routemap_topk,
                        unzipped_probs,
                        total_zipped_tokens=seqlen,
                        num_experts=expert_num,
                    )
                paddle.device.synchronize()
                    
                #############################################################
                start_time = time.time()
                # paddle.base.core.nvprof_start()
                for i in range(1000):
                    unzipped_tokens_recovered, expert_prob_topk_recovered = (
                        moe_unpermute(
                            unpermute_input,
                            zipped_expertwise_rowmap,
                            expert_routemap_topk,
                            unzipped_probs,
                            total_zipped_tokens=seqlen,
                            num_experts=expert_num,
                        )
                    )
                # paddle.base.core.nvprof_stop()
                paddle.device.synchronize()
                end_time = time.time()
                #############################################################
                total_time = (end_time - start_time) * 1000.0
                avg_time = total_time / 1000.0
                print("\n\n" + "*" * 50)
                print(f"Total time (1000 runs): {total_time :.3f} ms")
                print(f"Average time per run:   {avg_time :.3f} ms")
                print("*" * 50)
                
                results.append({
                    'SEQLEN Size': seqlen,
                    'TOKEN_LEN Size': self.TOKEN_LEN,
                    'Dtype': dt,
                    "expert_num": expert_num,
                    "topk": topk,
                    'Time(ms)': f"{total_time:.3f}",
                })
                # print("SEQLEN = ", seqlen, "TOKEN_LEN = ", self.TOKEN_LEN)
                # print("unpermute_input", unpermute_input.shape, unpermute_input.dtype)
                # print("zipped_expertwise_rowmap", zipped_expertwise_rowmap.shape, zipped_expertwise_rowmap.dtype)
                # print("expert_routemap_topk", expert_routemap_topk.shape, expert_routemap_topk.dtype)
                # print("unzipped_probs", unzipped_probs.shape, unzipped_probs.dtype)
                # print("total_zipped_tokens", seqlen)
                # print("num_experts", expert_num)
                # print("unzipped_tokens_recovered", unzipped_tokens_recovered.shape, unzipped_tokens_recovered.dtype)
                # print("expert_prob_topk_recovered", expert_prob_topk_recovered.shape, expert_prob_topk_recovered.dtype)
                # print("\n\n")

        # 保存结果到CSV
        if results:
            csv_file = 'moe_unpermute_paddle_h.csv'
            fieldnames = [
                'SEQLEN Size', 'TOKEN_LEN Size', 'Dtype', 
                'expert_num', 'topk', 'Time(ms)'
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n结果已保存至: {csv_file}")



if __name__ == "__main__":
    unittest.main()