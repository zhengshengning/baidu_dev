#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np
import time
import csv

import paddle
from paddle import core
import FusedQuantOps as FQO


class TestFusedStackTransposeQuantOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.results = []
        # 检查CUDA是否可用
    if paddle.device.is_compiled_with_cuda():
        print("CUDA可用，使用GPU进行测试")
        paddle.set_device('gpu:0')
    else:
        print("警告: CUDA不可用或PyTorch未检测到CUDA")
        if not paddle.device.is_compiled_with_cuda():
            print("  - Paddle CUDA不可用")
        exit(1)

    def run_benchmark(self, N, M, K, dtype):
        x_tensor_list = [
            paddle.randn([M, K], dtype=dtype).clip(min=-50, max=50)
            for _ in range(N)
        ]
        x_fp32 = paddle.stack(x_tensor_list).reshape([-1, K]).astype('float32')

        print(f"\n\nRunning test: N={N}, M={M}, K={K}, dtype={dtype}")

        # Warmup
        for i in range(100):
            # out, scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            #     x_tensor_list, transpose=self.transpose
            # )
            out, scale = FQO.fused_stack_transpose_quant(
                x_tensor_list
            )
        paddle.device.synchronize()

        #############################################################
        start_time = time.time()
        # paddle.base.core.nvprof_start()
        for i in range(1000):
            # out, scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            #     x_tensor_list, transpose=self.transpose
            # )
            out, scale = FQO.fused_stack_transpose_quant(
                x_tensor_list
            )
        # paddle.base.core.nvprof_stop()
        paddle.device.synchronize()
        end_time = time.time()
        #############################################################

        total_time = (end_time - start_time) * 1000.0
        avg_time = total_time / 1000.0
        print("*" * 50)
        print(f"Total time (1000 runs): {total_time :.3f} ms")
        print(f"Average time per run:   {avg_time:.3f} ms")
        print("*" * 50)

        self.results.append({
            'N Size': N,
            'M Size': M,
            'K Size': K,
            'Dtype': dtype,
            'Time(ms)': f"{total_time:.3f}",
        })


    def test_fused_stack_transpose_quant(self):
        NS = [16]
        MS = [3584, 7168]
        KS = [7168]
        self.dtypes = ['bfloat16']
        self.transpose = True
        for dtype in self.dtypes:
            for N in NS:
                for M in MS:
                    for K in KS:
                        self.run_benchmark(N, M, K, dtype)
        
        # 保存结果到CSV
        if self.results:
            csv_file = 'nlp_fused_stack_transpose_quant_h.csv'
            fieldnames = [
                'N Size', 'M Size', 'K Size', 'Dtype', 'Time(ms)'
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"\n结果已保存至: {csv_file}")


if __name__ == "__main__":
    unittest.main()
