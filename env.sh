export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com,0.0.0.0,baidu-int.com,aliyun.com,127.0.0.1,.baidu.com,.bcebos.com && export http_proxy=http://agent.baidu.com:8891 && export https_proxy=http://agent.baidu.com:8891
export https_proxy="agent.baidu.com:8188"
export http_proxy="agent.baidu.com:8188"
export no_proxy="baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com"


source /root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/paddle_env/bin/activate
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


# pytorch编译
#  python -m pip install -e .
#  uv pip install --no-build-isolation -v -e .

# 【命令行自动补全】
# apt install -y bash-completion
# if [ -f /etc/bash_completion ]; then
#   . /etc/bash_completion
# fi


