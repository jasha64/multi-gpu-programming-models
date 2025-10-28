#!/bin/bash

#SBATCH -J mgpm-sweep
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -n 4
#SBATCH -t 03:00:00
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
: "${NODE_COUNTS:=1}"
: "${TASKS_PER_NODE_LIST:=}"
: "${NX_LIST:=128,256,512,1024,2048,4096,8192}"
: "${NY:=16384}"
: "${NITER:=300}"
: "${NREP:=3}"
# : "${NCCL_ALGOS:=Ring,Tree}"
# : "${NCCL_PROTOS:=LL128,Simple}"

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
srun $SRUN_ARGS -N 1 --ntasks-per-node=1 -n 1 /bin/bash -c "./test.sh clean; sleep 1; ./test.sh"

outdir="sweep_results.$SLURM_JOB_ID"
mkdir -p "$outdir"

comma_to_array() {
    local IFS=','
    read -r -a arr <<< "$1"
    printf '%s\n' "${arr[@]}"
}

nx_list=( $(comma_to_array "$NX_LIST") )
# nccl_algos=( $(comma_to_array "$NCCL_ALGOS") )
# nccl_protos=( $(comma_to_array "$NCCL_PROTOS") )

if [[ -z "$TASKS_PER_NODE_LIST" ]]; then
    TASKS_PER_NODE_LIST="1,2,4"
fi
node_counts=( $(comma_to_array "$NODE_COUNTS") )
tasks_per_node_list=( $(comma_to_array "$TASKS_PER_NODE_LIST") )

echo "type,nx,ny,iter_max,num_nodes,num_gpus,variant,opts,runtime,best_raw" | tee "$outdir/summary.csv"

get_runtime_value() {
    local line="$1"
    local f8
    f8=$(awk -F',' '{print $8}' <<<"$line" 2>/dev/null | sed -e 's/^\s\+//' -e 's/\s\+$//' || true)
    if [[ "$f8" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "$f8"
    else
        local last
        last=$(awk -F',' '{print $NF}' <<<"$line" 2>/dev/null | sed -e 's/^\s\+//' -e 's/\s\+$//' || true)
        echo "$last"
    fi
}

# NVSHMEM 需要较大对称堆
export NVSHMEM_SYMMETRIC_SIZE=3690987520
export NCCL_DEBUG=WARN

for nodes in "${node_counts[@]}"; do
    for tpn in "${tasks_per_node_list[@]}"; do
        np=$(( nodes * tpn ))
        # If job allocation is smaller than requested, skip with a warning
        if (( nodes > ${SLURM_JOB_NUM_NODES:-1} )); then
            echo "[warn] skipping nodes=${nodes} > allocation (${SLURM_JOB_NUM_NODES:-1})" >&2
            continue
        fi
        for nx in "${nx_list[@]}"; do
        # NCCL baseline
        cmd_nccl_base="srun ${SRUN_ARGS} -N ${nodes} --ntasks-per-node=${tpn} -n ${np} --gpus-per-task=1 --gpu-bind=single:1 ./nccl/jacobi -csv -nx ${nx} -ny ${NY} -niter ${NITER}"
        best_raw=""; best_rt=""
        for rep in $(seq 1 $NREP); do
            out=$(eval "$cmd_nccl_base" 2>&1)
            line=$(printf "%s\n" "$out" | grep -E '^((nccl(_graphs|_overlap)?)|nvshmem)[^,]*,' | tail -n 1 || true)
            if [[ -z "$line" ]]; then
                echo "[warn] no CSV line matched for NCCL base (nx=${nx}, nodes=${nodes}, tpn=${tpn}) on rep=${rep}" >&2
                continue
            fi
            rt=$(get_runtime_value "$line")
            if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
                if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
            fi
        done
        if [[ -z "$best_raw" ]]; then best_rt="NA"; best_raw="NA"; fi
        echo "NCCL,$nx,${NY},${NITER},${nodes},${np},baseline,,$best_rt,$best_raw" | tee -a "$outdir/summary.csv"

        # # NCCL + CUDA Graphs
        # cmd_nccl_graphs="srun ${SRUN_ARGS} -N ${nodes} --ntasks-per-node=${tpn} -n ${np} --gpus-per-task=1 --gpu-bind=single:1 ./nccl_graphs/jacobi -csv -fixed_iter -nx ${nx} -ny ${NY} -niter ${NITER}"
        # best_raw=""; best_rt=""
        # for rep in $(seq 1 $NREP); do
        #     out=$(eval "$cmd_nccl_graphs" 2>&1)
        #     line=$(printf "%s\n" "$out" | grep -E '^((nccl(_graphs|_overlap)?)|nvshmem)[^,]*,' | tail -n 1 || true)
        #     if [[ -z "$line" ]]; then
        #         echo "[warn] no CSV line matched for NCCL graphs (nx=${nx}, nodes=${nodes}, tpn=${tpn}) on rep=${rep}" >&2
        #         continue
        #     fi
        #     rt=$(get_runtime_value "$line")
        #     if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
        #         if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
        #     fi
        # done
        # if [[ -z "$best_raw" ]]; then best_rt="NA"; best_raw="NA"; fi
        # echo "NCCL,$nx,${NY},${NITER},${nodes},${np},graphs,,$best_rt,$best_raw" | tee -a "$outdir/summary.csv"

        # NVSHMEM variants
        for opts in "" "-neighborhood_sync" "-neighborhood_sync" "-use_block_comm" "-use_block_comm -neighborhood_sync"; do
                cmd_nvshmem="srun ${SRUN_ARGS} -N ${nodes} --ntasks-per-node=${tpn} -n ${np} --gpus-per-task=1 --gpu-bind=single:1 ./nvshmem/jacobi -csv -nx ${nx} -ny ${NY} -niter ${NITER} ${opts}"
                best_raw=""; best_rt=""
                for rep in $(seq 1 $NREP); do
                    out=$(eval "$cmd_nvshmem" 2>&1)
                    line=$(printf "%s\n" "$out" | grep -E '^((nccl(_graphs|_overlap)?)|nvshmem)[^,]*,' | tail -n 1 || true)
                    if [[ -z "$line" ]]; then
                        echo "[warn] no CSV line matched for NVSHMEM opts='${opts}' (nx=${nx}, nodes=${nodes}, tpn=${tpn}) on rep=${rep}" >&2
                        continue
                    fi
                    rt=$(get_runtime_value "$line")
                    if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
                        if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
                    fi
                done
                if [[ -z "$best_raw" ]]; then best_rt="NA"; best_raw="NA"; fi
                tag=$(echo "$opts" | tr ' ' '+' | sed 's/^$/baseline/')
                echo "NVSHMEM,$nx,${NY},${NITER},${nodes},${np},${tag},${opts},$best_rt,$best_raw" | tee -a "$outdir/summary.csv"
        done

        # NVSHMEM multi-CTA variant
        for opts in "" "-neighborhood_sync" "-neighborhood_sync -norm_overlap" "-use_block_comm" "-use_block_comm -neighborhood_sync"; do
                cmd_nvshmem_multi="srun ${SRUN_ARGS} -N ${nodes} --ntasks-per-node=${tpn} -n ${np} --gpus-per-task=1 --gpu-bind=single:1 ./nvshmem_multi/jacobi -csv -nx ${nx} -ny ${NY} -niter ${NITER} ${opts}"
                best_raw=""; best_rt=""
                for rep in $(seq 1 $NREP); do
                    out=$(eval "$cmd_nvshmem_multi" 2>&1)
                    line=$(printf "%s\n" "$out" | grep -E '^((nccl(_graphs|_overlap)?)|nvshmem)[^,]*,' | tail -n 1 || true)
                    if [[ -z "$line" ]]; then
                        echo "[warn] no CSV line matched for NVSHMEM opts='${opts}' (nx=${nx}, nodes=${nodes}, tpn=${tpn}) on rep=${rep}" >&2
                        continue
                    fi
                    rt=$(get_runtime_value "$line")
                    if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
                        if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
                    fi
                done
                if [[ -z "$best_raw" ]]; then best_rt="NA"; best_raw="NA"; fi
                tag=$(echo "$opts" | tr ' ' '+' | sed 's/^$/multi/')
                echo "NVSHMEM,$nx,${NY},${NITER},${nodes},${np},${tag},${opts},$best_rt,$best_raw" | tee -a "$outdir/summary.csv"
        done

        # # NCCL ALGO/PROTO sweeps: baseline
        # for algo in "${nccl_algos[@]}"; do
        #     for proto in "${nccl_protos[@]}"; do
        #         cmd_nccl_combo="NCCL_ALGO=${algo} NCCL_PROTO=${proto} srun ${SRUN_ARGS} -N ${nodes} --ntasks-per-node=${tpn} -n ${np} ./nccl/jacobi -csv -nx ${nx} -ny ${NY} -niter ${NITER}"
        #         best_raw=""; best_rt=""
        #         for rep in $(seq 1 $NREP); do
        #             out=$(eval "$cmd_nccl_combo" 2>&1)
        #             line=$(printf "%s\n" "$out" | grep -E '^((nccl(_graphs|_overlap)?)|nvshmem)[^,]*,' | tail -n 1 || true)
        #             if [[ -z "$line" ]]; then
        #                 echo "[warn] no CSV line matched for NCCL combo algo=${algo} proto=${proto} (nx=${nx}, nodes=${nodes}, tpn=${tpn}) on rep=${rep}" >&2
        #                 continue
        #             fi
        #             rt=$(get_runtime_value "$line")
        #             if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
        #                 if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
        #             fi
        #         done
        #         if [[ -z "$best_raw" ]]; then best_rt="NA"; best_raw="NA"; fi
        #         echo "NCCL,$nx,${NY},${NITER},${nodes},${np},baseline+algo=${algo}+proto=${proto},NCCL_ALGO=${algo} NCCL_PROTO=${proto},$best_rt,$best_raw" | tee -a "$outdir/summary.csv"
        #     done
        # done

        # # NCCL ALGO/PROTO sweeps: graphs
        # for algo in "${nccl_algos[@]}"; do
        #     for proto in "${nccl_protos[@]}"; do
        #         cmd_nccl_graphs_combo="NCCL_ALGO=${algo} NCCL_PROTO=${proto} srun ${SRUN_ARGS} -N ${nodes} --ntasks-per-node=${tpn} -n ${np} ./nccl_graphs/jacobi -csv -nx ${nx} -ny ${NY} -niter ${NITER}"
        #         best_raw=""; best_rt=""
        #         for rep in $(seq 1 $NREP); do
        #             out=$(eval "$cmd_nccl_graphs_combo" 2>&1)
        #             line=$(printf "%s\n" "$out" | grep -E '^((nccl(_graphs|_overlap)?)|nvshmem)[^,]*,' | tail -n 1 || true)
        #             if [[ -z "$line" ]]; then
        #                 echo "[warn] no CSV line matched for NCCL graphs combo algo=${algo} proto=${proto} (nx=${nx}, nodes=${nodes}, tpn=${tpn}) on rep=${rep}" >&2
        #                 continue
        #             fi
        #             rt=$(get_runtime_value "$line")
        #             if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
        #                 if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
        #             fi
        #         done
        #         if [[ -z "$best_raw" ]]; then best_rt="NA"; best_raw="NA"; fi
        #         echo "NCCL,$nx,${NY},${NITER},${nodes},${np},graphs+algo=${algo}+proto=${proto},NCCL_ALGO=${algo} NCCL_PROTO=${proto},$best_rt,$best_raw" | tee -a "$outdir/summary.csv"
        #     done
        # done

        # # NCCL overlap variant
        # cmd_nccl_overlap="srun ${SRUN_ARGS} -N ${nodes} --ntasks-per-node=${tpn} -n ${np} --gpus-per-task=1 --gpu-bind=single:1 ./nccl_overlap/jacobi -csv -fixed_iter -nx ${nx} -ny ${NY} -niter ${NITER}"
        # best_raw=""; best_rt=""
        # for rep in $(seq 1 $NREP); do
        #     out=$(eval "$cmd_nccl_overlap" 2>&1)
        #     line=$(printf "%s\n" "$out" | grep -E '^((nccl(_graphs|_overlap)?)|nvshmem)[^,]*,' | tail -n 1 || true)
        #     if [[ -z "$line" ]]; then
        #         echo "[warn] no CSV line matched for NCCL overlap (nx=${nx}, nodes=${nodes}, tpn=${tpn}) on rep=${rep}" >&2
        #         continue
        #     fi
        #     rt=$(get_runtime_value "$line")
        #     if [[ -z "$best_raw" ]]; then best_raw="$line"; best_rt="$rt"; else
        #         if (( $(awk -v a="$rt" -v b="$best_rt" 'BEGIN{print (a<b)?1:0}') )); then best_raw="$line"; best_rt="$rt"; fi
        #     fi
        # done
        # if [[ -z "$best_raw" ]]; then best_rt="NA"; best_raw="NA"; fi
        # echo "NCCL,$nx,${NY},${NITER},${nodes},${np},overlap,,$best_rt,$best_raw" | tee -a "$outdir/summary.csv"
        done
    done
done

echo "[sweep] results in: $outdir"
end=`date`
echo "[sweep] started at $start, ended at $end"



