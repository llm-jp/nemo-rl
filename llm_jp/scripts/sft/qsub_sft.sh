#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0227_train
#PBS -l select=1
#PBS -l walltime=168:00:00
#PBS -m n

# Setup logs
cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/train-$JOBID.out
ERRFILE=./logs/train-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

# Setup environment
source llm_jp/environment.sh
# source .venv/bin/activate

# Name model randomly
NAME="sft-$(od -vAn -N4 -tu4 < /dev/urandom | tr -d ' ')"
echo "NAME=${NAME}"

# mpirun \
#   --display-allocation \
#   --report-bindings \
#   --oversubscribe \
#   -np $NUM_GPUS \
#   --npernode $NUM_GPUS_PER_NODE \
#   -x MASTER_ADDR=$MASTER_ADDR \
#   -x MASTER_PORT=$MASTER_PORT \
#   -bind-to none \
#   -map-by slot \
#   -x PATH \
#   python train_sft.py \
#     trainer.num_nodes=${NUM_NODES} \
#     use_mpi=True \
#     use_slurm=True \
#     name=${NAME} \
#     seed=${SEED} \
#     model.restore_from_path=${MODEL_PATH} \
#     "${MODEL_PARAMS[@]}"

uv run python examples/run_sft.py \
  --config ${CONFIG_PATH} \
  policy.model_name=${MODEL_PATH} \
  sft.seed=${SEED}
