#!/usr/bin/env bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

set -ex

export PYTHONUNBUFFERED=1
export WEAVE_PRINT_CALL_LINK=false
pip install math_verify weave

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"

CORPUS_FILE="${CORPUS_FILE:?Set CORPUS_FILE to the paper tool-use corpus JSONL.}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spade_tool_use_4b/$(date +%Y%m%d_%H%M%S)}"
GAMES_DIR="${GAMES_DIR:-${OUTPUT_DIR}/spade_games}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spade-workspace}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/spade-workspace}"
if [ -z "${WANDB_ENTITY:-}" ]; then
   echo "ERROR: Set WANDB_ENTITY to your Weights & Biases team/entity." >&2
   exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${GAMES_DIR}"

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

export WORKSPACE_DIR
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_aime.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/eval_aime.yaml" > "${EVAL_CONFIG_FILE}"

GEM_EVAL_CONFIG_FILE="${PROJECT_ROOT}/eval_configs/gem_eval_bfcl_only.yaml"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
   --ref-load "${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist"
   --save "${OUTPUT_DIR}/Qwen3-4B-Instruct-2507_tool_use/"
   --save-interval 16
)

ROLLOUT_ARGS=(
   --data-source-path spade.slime.data_source.SpadeDataSource
   --rollout-function-path spade.slime.spade_rollout.spade_generate_rollout
   --num-rollout 500
   --rollout-batch-size 16
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-temperature 0.7
   --global-batch-size 256
   --use-dynamic-global-batch-size
   --balance-data
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

SPADE_ARGS=(
   --spade-gamma1 0.98
   --spade-gamma2 0.85
   --spade-env-temperature 0.6
   --spade-actor-temperature 0.6
   --spade-actor-max-tokens 8192
   --spade-env-max-tokens 16384
   --spade-max-context-length 32768
   --spade-max-turns 25
   --spade-gamma 0.99
   --spade-game-regeneration-interval 32
   --spade-num-games-per-rollout 16
   --spade-games-dir "${GAMES_DIR}"
   --spade-game-difficulty medium
   --spade-trajectories-per-game 16
   --spade-cache-dir "${OUTPUT_DIR}/spade_games_cache"
   --spade-game-type tool_use
   --spade-skills API_Orchestration Data_Retrieval State_Modification Error_Recovery Tool_Selection Multi-Step_Workflows
   --spade-env-generation-template qwen3_tool_use_game_generation
   --spade-actor-template qwen3_game
   --spade-reward-normalization grpo
   --spade-env-reward-variant regret_based
   --spade-hint-mode self
   --spade-hint-temperature 0.3
   --spade-hint-max-tokens 512
   --spade-hint-plays-per-game 4
   --spade-use-solver-variance-reward
   --spade-proposer-training-delay 32
   --spade-gem-eval-config "${GEM_EVAL_CONFIG_FILE}"
   --spade-corpus-file "${CORPUS_FILE}"
   --spade-corpus-max-doc-tokens 4000
   --spade-corpus-seed 42
   --spade-compact-filter
   --spade-use-env-memory
   --spade-use-env-validator
   --spade-env-memory-max-size 200
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.005
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

WANDB_ARGS=(
   --use-wandb
   --wandb-team "${WANDB_ENTITY}"
   --wandb-project "${WANDB_PROJECT:-spade}"
   --wandb-group "${WANDB_GROUP:-spade-tool-use-4b}"
   --wandb-key "${WANDB_API_KEY:?Set WANDB_API_KEY.}"
)

EVAL_ARGS=(
   --eval-interval 48
   --eval-config ${EVAL_CONFIG_FILE}
   --apply-chat-template
   --rm-type math
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-tool-call-parser qwen
   --router-policy consistent_hashing   # session-affinity routing for multi-turn rollouts
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"LITELLM_LOCAL_MODEL_COST_MAP\": \"True\",
    \"LITELLM_LOG\": \"WARNING\",
    \"SPADE_MULTITURN_ENV_GEN\": \"1\",
    \"SPADE_RESET_GATE\": \"1\",
    \"SPADE_MASK_MALFORMED_TOOLCALL\": \"1\",
    \"SPADE_TURN_CURRICULUM\": \"1\",
    \"LCB_OFFICIAL_DIR\": \"${LCB_OFFICIAL_DIR:-${WORKSPACE_DIR}/LiveCodeBench-official}\",
    \"LCB_V6_JSONL\": \"${LCB_V6_JSONL:-${WORKSPACE_DIR}/livecodebench/test6.jsonl}\",
    \"HF_DATASETS_CACHE\": \"${HF_DATASETS_CACHE:-${WORKSPACE_DIR}/hf_cache/datasets}\"
  }
}"

RAY_JOB_ID="spade_tool_use_4b_$$"

ray job submit --address="http://127.0.0.1:8265" \
   --submission-id="${RAY_JOB_ID}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --no-wait \
   -- python3 -m train_spade_slime \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${SPADE_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}"

ray job logs --address="http://127.0.0.1:8265" --follow "${RAY_JOB_ID}"
