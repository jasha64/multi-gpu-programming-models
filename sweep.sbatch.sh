#!/bin/bash

#SBATCH -J mgpm-sweep
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o sweep.%j.out
#SBATCH -e sweep.%j.err
#SBATCH -A a-g200

# 说明:
# - 在 Slurm 上运行参数扫掠，比较 NCCL / NCCL+Graphs / NVSHMEM 多种实现。
# - 依赖 bench.sh 中的构建逻辑与已有可执行文件布局。

set -euo pipefail

# : "${ENROOT_IMG_PATH:=.}"
: "${LUSTRE:=.}"

# IMG=nvcr.io#nvidia/nvhpc:25.5-devel-cuda12.9-ubuntu22.04
# SQUASHFS_IMG=$ENROOT_IMG_PATH/`echo "$IMG" | md5sum | cut -f1 -d " "`
# CONTAINER_NAME=HPCSDK-CONTAINER
# CONTAINER_MNTS=$LUSTRE/workspace/multi-gpu-programming-models:/mnt

# 默认 sweep 配置，可通过环境变量覆盖
: "${GPU_COUNTS:=2,4}"
: "${NX_LIST:=512,1024,2048,4096,8192}"
: "${NY:=16384}"
: "${NITER:=300}"
: "${NREP:=3}"

start=`date`
echo "[sweep] job $SLURM_JOB_ID started at $start"

# if [[ -f "$SQUASHFS_IMG" ]]; then
    # echo "Using: $SQUASHFS_IMG"
# else
    # echo "Fetching $IMG to $SQUASHFS_IMG"
    # # srun -n 1 -N 1 --ntasks-per-node=1
    # enroot import -o $SQUASHFS_IMG docker://$IMG
    # echo "$IMG" > "${SQUASHFS_IMG}.url"
# fi

# CONTAINER_IMG=$SQUASHFS_IMG
# if [[ ! -f "$CONTAINER_IMG" ]]; then
#     echo "Falling back to $IMG"
#     CONTAINER_IMG=$IMG
# fi

# 预热容器于所有分配节点
srun -N ${SLURM_JOB_NUM_NODES} \
     -n ${SLURM_JOB_NUM_NODES} \
     --mpi=pmix \
     --ntasks-per-node=1 \
     --environment=cscs-nv-hpc-bench \
     true

export SRUN_ARGS="--cpu-bind=none --mpi=pmix --environment=cscs-nv-hpc-bench"  # --no-container-remap-root --container-mounts=$CONTAINER_MNTS --container-workdir=/mnt --container-name=$CONTAINER_NAME"
export OMPI_MCA_coll_hcoll_enable=0
export MPIRUN_ARGS="--oversubscribe"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# 构建全部目标
echo "[sweep] building targets"
srun $SRUN_ARGS -n 1 /bin/bash -c "./test.sh clean; sleep 1; ./test.sh"

outdir="sweep_results.$SLURM_JOB_ID"
mkdir -p "$outdir"

comma_to_array() {
    local IFS=','
    read -r -a arr <<< "$1"
    printf '%s\n' "${arr[@]}"
}

gpu_counts=( $(comma_to_array "$GPU_COUNTS") )
nx_list=( $(comma_to_array "$NX_LIST") )

echo "type,nx,ny,iter_max,num_gpus,variant,opts,runtime,best_raw" | tee "$outdir/summary.csv"

get_runtime_value() {
    local line="$1"
    local f8
    f8=$(awk -F',' '{print $8}' <<<"$line" 2>/dev/null || true)
    if [[ "$f8" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "$f8"
    else
        awk -F',' '{print $NF+0}' <<<"$line"
    fi
}

# NVSHMEM 需要较大对称堆
export NVSHMEM_SYMMETRIC_SIZE=3690987520
export NCCL_DEBUG=WARN

for g in "${gpu_counts[@]}"; do
    for nx in "${nx_list[@]}"; do
        # NCCL baseline
        cmd_nccl_base="srun ${SRUN_ARGS} -n ${g} ./nccl/jacobi -csv -nx ${nx} -ny ${NY} -niter ${NITER}"
        best_raw=""; best_rt=""
        for rep in $(seq 1 $NREP); do
            out=$(eval "$cmd_nccl_base" 2>&1)
            line=$(printf "%s\n" "$out" | grep -E '^(nccl|nccl_graphs|nvshmem),' | tail -n 1)
            rt=$(get_runtime_value "$line")
            if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
                if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
            fi
        done
        echo "NCCL,$nx,${NY},${NITER},$g,baseline,,$best_rt,$best_raw" | tee -a "$outdir/summary.csv"

        # NCCL + CUDA Graphs
        cmd_nccl_graphs="srun ${SRUN_ARGS} -n ${g} ./nccl_graphs/jacobi -csv -nx ${nx} -ny ${NY} -niter ${NITER}"
        best_raw=""; best_rt=""
        for rep in $(seq 1 $NREP); do
            out=$(eval "$cmd_nccl_graphs" 2>&1)
            line=$(printf "%s\n" "$out" | grep -E '^(nccl|nccl_graphs|nvshmem),' | tail -n 1)
            rt=$(get_runtime_value "$line")
            if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
                if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
            fi
        done
        echo "NCCL,$nx,${NY},${NITER},$g,graphs,,$best_rt,$best_raw" | tee -a "$outdir/summary.csv"

        # NVSHMEM variants
        for opts in "" "-neighborhood_sync" "-neighborhood_sync -norm_overlap" "-use_block_comm" "-use_block_comm -neighborhood_sync"; do
            cmd_nvshmem="srun ${SRUN_ARGS} -n ${g} ./nvshmem/jacobi -csv -nx ${nx} -ny ${NY} -niter ${NITER} ${opts}"
            best_raw=""; best_rt=""
            for rep in $(seq 1 $NREP); do
                out=$(eval "$cmd_nvshmem" 2>&1)
                line=$(printf "%s\n" "$out" | grep -E '^(nccl|nccl_graphs|nvshmem),' | tail -n 1)
                rt=$(get_runtime_value "$line")
                if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
                    if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
                fi
            done
            tag=$(echo "$opts" | tr ' ' '+' | sed 's/^$/baseline/')
            echo "NVSHMEM,$nx,${NY},${NITER},$g,${tag},${opts},$best_rt,$best_raw" | tee -a "$outdir/summary.csv"
        done
    done
done

echo "[sweep] results in: $outdir"
end=`date`
echo "[sweep] started at $start, ended at $end"



