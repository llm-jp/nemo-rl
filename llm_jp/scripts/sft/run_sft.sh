#!/bin/bash

set -euxo pipefail

if [ $# -ne 4 ]; then
    >&2 echo "Usage: $0 <param-name> <num-nodes> <model-path> <seed>"
    >&2 echo "Example: $0 llmjp3_1.8b 2 /path/to/model 42"
    exit 1
fi

config_path=$1; shift
num_nodes=$1; shift
model_path=$1; shift
seed=$1; shift

# default: rt_HF, user can override by specifying RTYPE env var
RTYPE=${RTYPE:-rt_HF}

qsub -l select=${num_nodes} \
  -v RTYPE=${RTYPE},CONFIG_PATH=${config_path},MODEL_PATH="${model_path}" \
  -o /dev/null -e /dev/null \
  -m n \
  scripts/sft/qsub_sft.sh