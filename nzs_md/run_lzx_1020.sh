#!/bin/bash
# 用法：TAG=xxx bash run_lzx_1020.sh（任一节点执行，经 script/mpi.sh 派发到全部节点）
# 可选：YAML / MAX_STEPS / EXTRA_KWARGS="k=v ..." / FORCE=1 / NSYS=1 [NSYS_PROF_START=6 NSYS_PROF_END=9 → 采 step 7~9]
set -e
SELF=$(readlink -f "$0")
cd "$(dirname "${SELF}")"
TAG=${TAG:-1020_T4_fsdp_t1}
OUT_DIR=run_lzx   # log / nsys / output_dir 都放这里
mkdir -p "${OUT_DIR}"

# 启动器：mpirun 不透传变量，写到共享盘 env 文件让各节点读（output_dir 不一致会让 kernel warmup 卡住）
if [[ -z "${OMPI_COMM_WORLD_RANK}" ]]; then
  if compgen -G "${OUT_DIR}/${TAG}_n*.log" >/dev/null && [[ "${FORCE:-0}" != "1" ]]; then
    echo "[run_lzx] ${OUT_DIR}/${TAG}_n*.log 已存在，换个 TAG 或设 FORCE=1" >&2; exit 1
  fi
  env_file="${PWD}/${OUT_DIR}/.run_lzx_env.${TAG}"
  declare -p TAG YAML MAX_STEPS NSYS EXTRA_KWARGS NSYS_PROF_START NSYS_PROF_END >"${env_file}" 2>/dev/null || true
  rc=0; bash script/mpi.sh bash "${SELF}" --env-file "${env_file}" || rc=$?
  rm -f "${env_file}"; exit ${rc}
fi
[[ "$1" == "--env-file" ]] && source "$2"

source /root/paddlejob/paddlefleet-public/lizhenxing/ladder/1020/venv/bin/activate
export FLEET_DEEPGEMM_NUM_SMS=120

YAML=${YAML:-conf/experiment/ERNIE-5.5-Flash/ScalingLadder/ernie5_5_flash_ladder_v90_T4_h1280_L20_34.93BA1.232B-pretrain-8k-hzz1-fsdp.yaml}
NODE_RANK=${OMPI_COMM_WORLD_RANK:-${POD_INDEX:-0}}

# NSYS=1：每个节点都采，否则跨机通信不对称会 hang
nsys_args=""
if [[ "${NSYS:-0}" == "1" ]]; then
  capture_args=""
  if [[ -n "${NSYS_PROF_START}" ]]; then
    export NSYS_PROF_START NSYS_PROF_END
    capture_args="--capture-range=cudaProfilerApi --capture-range-end=stop"
  fi
  nsys_args="/root/paddlejob/paddlefleet-public/lizhenxing/FSDP/nsys/bin/nsys profile \
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
    --configs "${YAML}" \
    --kwargs \
      max_steps=${MAX_STEPS:-20} \
      output_dir=${OUT_DIR}/${TAG} \
      profile=false \
      ${EXTRA_KWARGS} \
  2>&1 | tee ${OUT_DIR}/${TAG}_n${NODE_RANK}.log
