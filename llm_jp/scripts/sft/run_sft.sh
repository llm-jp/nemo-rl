#!/bin/bash

set -euxo pipefail

if [ $# -ne 4 ]; then
    >&2 echo "Usage: $0 <config-path> <num-nodes> <model-path> <seed>"
    >&2 echo "Example: $0 llmjp3_1.8b 2 /path/to/model 42"
    exit 1
fi

config_path=$1; shift
num_nodes=$1; shift
model_path=$1; shift
seed=$1; shift

qsub -l select=${num_nodes} \
  -v RTYPE=rt_HF,CONFIG_PATH=${config_path},MODEL_PATH="${model_path},SEED=${seed}" \
  -o /dev/null -e /dev/null \
  -m n \
  llm_jp/scripts/sft/qsub_sft.sh
