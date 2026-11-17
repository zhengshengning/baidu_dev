#!/bin/bash
# 多机启动：任一节点执行，经 script/mpi.sh（mpirun -pernode）派发到全部节点各跑一份本脚本。
# 用法：[NSYS=1 NSYS_PROF_START=.. NSYS_PROF_END=..] bash run_fsdp_test.sh
SELF=$(readlink -f "$0")
cd /root/paddlejob/paddlefleet-public/ningzhengsheng/erniebot || exit 1

# 启动器：mpirun 不透传这些变量，写到共享盘 env 文件让各节点读
if [[ -z "${OMPI_COMM_WORLD_RANK}" ]]; then
  env_file="${PWD}/.run_fsdp_test_env.$$"
  declare -p NSYS NSYS_PROF_START NSYS_PROF_END EXTRA_KWARGS >"${env_file}" 2>/dev/null || true
  rc=0; bash script/mpi.sh bash "${SELF}" --env-file "${env_file}" || rc=$?
  rm -f "${env_file}"; exit ${rc}
fi
[[ "$1" == "--env-file" ]] && source "$2"
[[ -n "${NSYS_PROF_START}" ]] && export NSYS_PROF_START NSYS_PROF_END

# bas
sleep 3s

# train_gpu.sh 会把 ${FLEET_SRC} 插到 PYTHONPATH 最前面，这里指向本地 PaddleFleet 工作副本（fsdp_fp8_fused_linear 分支），覆盖掉默认的 ./third_party/PaddleFleet/src/。
# export FLEET_SRC="/root/paddlejob/paddlefleet-public/ningzhengsheng/FSDP/baidu/ernie/erniebot/PaddleFleet/src"
export PYTHONPATH="/root/paddlejob/paddlefleet-public/ningzhengsheng/erniebot/Paddle/build/python:${PYTHONPATH:-}"

# train_gpu.sh 默认只用节点 0（range(0,1)），两机要显式选 [0,2)，否则 world_size=8 而 pp2*ep8 需要 16 卡
export LAUNCH_NODES="range(0,2)"

# 单机运行：script/selective_launch.py 的 nnodes 完全来自 TRAINER_INSTANCES，
# 平台上是 4 个 IP，会算出 --nnodes 4 并一直等其余 3 个节点注册（卡在 rendezvous）。
# 这里只保留本机 IP，强制 --nnodes 1。
# export TRAINER_INSTANCES=$(python -c "
# import os
# ips = [i.strip() for i in os.getenv('TRAINER_INSTANCES', '').split(',') if i.strip()]
# print(ips[int(os.getenv('POD_INDEX', '0'))] if ips else '127.0.0.1')
# ")
# export POD_INDEX=0

# # 消除非确定性，便于与 stage1 做逐位比对（对齐.md:240 起）
# export FLAGS_embedding_deterministic=1
# export FLAGS_cudnn_deterministic=1
# export FLAGS_deterministic_rng=1

# # stage1 <-> FSDP 逐位对齐诊断开关（见 stage1_fsdp_align_hack.md，纯诊断、非生产语义）
# export MUON_FSDP_BCAST_GRAD=1
# export MUON_FSDP_NS_SINGLE=1
# export MUON_FSDP_STAGE1_CLIP_EPS=1
# # 专家 buffer 不额外除 accum（替代 EXPERT_GRAD_MUL=<accum>：先除再乘回仅在 accum 为 2 的幂时逐位无损）
# export MUON_FSDP_EXPERT_NO_ACCUM_SCALE=1


# nsys 只抓 [NSYS_PROF_START, NSYS_PROF_END) 这几步：pretraining_trainer.py 里按 global_step 调 cudaProfilerStart/Stop
export NSYS_PROF_START=${NSYS_PROF_START:-6}
export NSYS_PROF_END=${NSYS_PROF_END:-10}

# 镜像里没装 nsys，用本机已解压的 Nsight Systems CLI（2026.2.1）
export PATH="/root/paddlejob/paddlefleet-public/lizhenxing/FSDP/nsys/bin:${PATH}"

OUT_DIR=./timeline
TAG=ernielite-T4-fsdp
NODE_RANK=${OMPI_COMM_WORLD_RANK:-${POD_INDEX:-0}}
mkdir -p "${OUT_DIR}"

# NSYS=1：每个节点都采，否则跨机通信不对称会 hang；NSYS=0 直接跑训练
nsys_args=""
if [[ "${NSYS:-0}" == "1" ]]; then
  capture_args=""
  if [[ -n "${NSYS_PROF_START}" ]]; then
    capture_args="--capture-range=cudaProfilerApi --capture-range-end=stop"
  fi
  nsys_args="nsys profile \
  -w true \
  -t cuda,nvtx \
  --wait=primary \
  --trace-fork-before-exec=true \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite true \
  ${capture_args} \
  -o ${OUT_DIR}/${TAG}_n${NODE_RANK}"
fi

${nsys_args} bash script/train.sh \
    --configs conf/experiment/ERNIE-5.5-Flash/V0.2.1/ernie5_5_flash_ladder_v0.2.1_C4_T4_h1280_L20-pretrain-8k-hzz1.yaml \
    --kwargs data_filelist="conf/allinone/eb5/EB5Pretrain-260803-a1b-v19p8-bfd-8k-1pct.zy" \
             data_weights="conf/allinone/eb5/EB5Pretrain-260803-a1b-v19p8-bfd-8k-1pct.ratio" \
             max_steps=20 \
             save_steps=200 \
             internal_medicine_monitor_interval=0 \
             ${EXTRA_KWARGS}
            #  fsdp_prefetch_units=2
            #  fsdp_comm_unit_size=703647744
             #  fsdp_prefetch_units=2

exit
