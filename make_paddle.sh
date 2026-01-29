
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com,0.0.0.0,baidu-int.com,aliyun.com,127.0.0.1,.baidu.com,.bcebos.com && export http_proxy=http://agent.baidu.com:8891 && export https_proxy=http://agent.baidu.com:8891


cd /root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/src/Paddle/build
cmake .. -DON_INFER=OFF -DPY_VERSION=3.10 -DWITH_GLOO=OFF -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_DISTRIBUTE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_CINN=ON -DWITH_SLEEF=ON


# cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=OFF -DWITH_GPU=ON -DWITH_MKL=OFF -DWITH_TESTING=ON -DWITH_PYTHON=ON -DON_INFER=ON -DWITH_INFERENCE_API_TEST=ON -DWITH_TENSORRT=ON -DWITH_UNITY_BUILD=ON -DCUDA_ARCH_NAME=Auto -DNEW_RELEASE_ALL=ON -DNEW_RELEASE_PYPI=OFF -DNEW_RELEASE_JIT=OFF -DWITH_ONNXRUNTIME=ON -DWITH_CPP_TEST=ON -DWIN_UNITTEST_LEVEL=2 -DWITH_NIGHTLY_BUILD=OFF -DWITH_PIP_CUDA_LIBRARIES=OFF -DWITH_SHARED_PHI=ON

make -j 512

cd ./python/dist
pip uninstall *.whl
pip install *.whl
