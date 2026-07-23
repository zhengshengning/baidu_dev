export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com,0.0.0.0,baidu-int.com,aliyun.com,127.0.0.1,.baidu.com,.bcebos.com && export http_proxy=http://agent.baidu.com:8891 && export https_proxy=http://agent.baidu.com:8891
export http_proxy=agent.baidu.com:8891
export https_proxy=agent.baidu.com:8891
export no_proxy="baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com"


source /root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/paddle_env/bin/activate
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=/root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/src/Paddle/build/python/:$PYTHONPATH
export PYTHONPATH=/root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/src/Paddle/build/:$PYTHONPATH

# 【Pytorch编译】
# python -m pip install -e .
# uv pip install --no-build-isolation -v -e .

# 【创建Paddle开发环境】
# virtualenv -p /usr/bin/python3.10 paddle_env
# source paddle_env/bin/activate
# python -m pip install --upgrade pip

# 【Cmake安装】
# pip install cmake==3.27.0

# 【PR提交】
# pip install pre-commit==2.17.0
# pre-commit install
# PIP_INDEX_URL=https://pypi.org/simple git commit -m "fix"

# 【安装官网 paddle】
# uv pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ --force-reinstall --no-deps

# docker exec -it ningzhengsheng_a100_80g_cuda129 bash
# python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/ --force-reinstall

# 【安装torch】
# pip install torch==2.9.1 --force-reinstall --no-deps

# LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.12:/usr/local/cuda/lib64/libcublasLt.so.12 timeout 180 python test.py

# 【传输文件 开启Http服务】
# python -m updog -p 8124

# 【Paddle编译命令】
# cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON -DWITH_SHARED_PHI=ON -DWITH_TENSORRT=ON -DWITH_OPENVINO=OFF -DWITH_ROCM=OFF -DWITH_CINN=ON -DWITH_DISTRIBUTE=ON -DWITH_MKL=OFF -DWITH_AVX=ON -DCUDA_ARCH_NAME=Manual -DNEW_RELEASE_PYPI=OFF -DNEW_RELEASE_ALL=OFF -DNEW_RELEASE_JIT=OFF -DWITH_PYTHON=ON -DWITH_TESTING=OFF -DWITH_COVERAGE=OFF -DWITH_INCREMENTAL_COVERAGE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_INFERENCE_API_TEST=OFF -DPY_VERSION=3.10 -DWITH_PSLIB= -DWITH_GLOO=ON -DWITH_XPU=OFF -DWITH_IPU=OFF -DXPU_SDK_ROOT= -DWITH_XPU_BKCL=OFF -DWITH_XPU_XHPC=OFF -DWITH_XPU_XFT=OFF -DWITH_XPU_XRE5=OFF -DWITH_XPU_FFT=OFF -DWITH_ARM=OFF -DWITH_STRIP=ON -DON_INFER=OFF -DCUDA_ARCH_BIN="80 90 100 103 120" -DWITH_RECORD_BUILDTIME=OFF -DWITH_UNITY_BUILD=OFF -DWITH_ONNXRUNTIME=OFF -DWITH_CUDNN_FRONTEND=OFF -DWITH_CPP_TEST=OFF -DWITH_FA_BUILD_WITH_CACHE=ON


# 【多机模型调试】
# mpirun python -m pip install paddlepaddle_gpu-3.3.1.dev20260330-cp310-cp310-linux_x86_64.whl --no-deps --force-reinstall
# python script/sync_new.py file script/train_gpu.sh
# python script/sync_new.py file third_party/ernie-core/src/ernie_core/models/moe/moe_layer.py

# 【单测运行】
# export PYTHONPATH=/home/ningzhengsheng/src/Paddle/build/python:/home/ningzhengsheng/src/Paddle/test:/home/ningzhengsheng/src/Paddle/test/ir/pir/fused_pass
# python -m paddle.distributed.launch --devices=0,1,2,3,4,5,6,7 xxx.py
# FLAGS_PIR_OPTEST=1 FLAGS_PIR_OPTEST_WHITE_LIST=1   python test/legacy_test/test_fused_adam_op.py

# 【起文件服务】
# python -m updog -p 8124

