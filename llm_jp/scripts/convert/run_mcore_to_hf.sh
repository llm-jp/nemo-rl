#!/bin/bash

set -euxo pipefail

if [ $# -ne 3 ]; then
    >&2 echo "Usage: $0 <config-path> <ckpt-path> <output-path>"
    >&2 echo "Example: $0 /path/to/config.yaml /path/to/megatron/checkpoint /path/to/output"
    exit 1
fi

config_path=$1; shift
ckpt_path=$1; shift
output_path=$1; shift

RTYPE=${RTYPE:-rt_HG}

qsub \
  -v RTYPE=${RTYPE},CONFIG_PATH=${config_path},CKPT_PATH="${ckpt_path}",OUTPUT_PATH=${output_path} \
  -o /dev/null -e /dev/null \
  -m n \
  scripts/convert/qsub_mcore_to_hf.sh
