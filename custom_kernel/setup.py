# !/usr/bin/env python3
"""
fused_ln setup
"""
import multiprocessing
import os


def run(func):
    """run"""
    p = multiprocessing.Process(target=func)
    p.start()
    p.join()


def change_pwd():
    """change_pwd"""
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


def setup_custom_ops():
    """setup_custom_ops"""
    from paddle.utils.cpp_extension import CUDAExtension, setup

    change_pwd()
    setup(
        name="custom_ops",
        ext_modules=CUDAExtension(
            sources=[
                "./fused_weighted_swiglu_act_quant_kernel.cu",
                "./fused_stack_transpose_quant.cu",
                "./fused_transpose_split_quant_old.cu",
                "./fused_transpose_split_quant.cu",
                "./per_token_quant_kernel.cu",
                "./fused_swiglu_weighted_bwd_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    # "--use_fast_math",
                    # "-maxrregcount=50",
                ],
            },
        ),
    )


run(setup_custom_ops)
