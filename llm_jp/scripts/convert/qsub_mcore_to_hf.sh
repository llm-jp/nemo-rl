#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0227_convert
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -m 1

# Setup logs
cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/convert-$JOBID.out
ERRFILE=./logs/convert-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

uv run \
    --extra mcore python3 examples/converters/convert_megatron_to_hf.py \
    --config ${CONFIG_PATH} \
    --megatron-ckpt-path ${CKPT_PATH} \
    --hf-ckpt-path ${OUTPUT_PATH}

# Add "chat_template" attribute to tokenizer_config.json
CHAT_TEMPLATE="{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。' }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"
TOKENIZER_CONFIG_PATH="${OUTPUT_PATH}/tokenizer_config.json"
jq --arg chat_template "$CHAT_TEMPLATE" '. + {chat_template: $chat_template}' \
    "${TOKENIZER_CONFIG_PATH}" > "${TOKENIZER_CONFIG_PATH}.tmp"
mv "${TOKENIZER_CONFIG_PATH}.tmp" "${TOKENIZER_CONFIG_PATH}"
