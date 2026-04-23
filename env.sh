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
# uv pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ --force-reinstall

# docker exec -it ningzhengsheng_a100_80g_cuda129 bash
