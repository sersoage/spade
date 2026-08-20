#!/usr/bin/env bash
set -euo pipefail

TOOL_MODEL_SIZE="${TOOL_MODEL_SIZE:?Set TOOL_MODEL_SIZE to 8b or 30b.}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/spade-workspace}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/spade-workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/spade_paper_tool_use_${TOOL_MODEL_SIZE}/$(date +%Y%m%d_%H%M%S)}"
GAMES_DIR="${GAMES_DIR:-${OUTPUT_DIR}/spade_games}"
CORPUS_FILE="${CORPUS_FILE:?Set CORPUS_FILE to the paper tool-use corpus JSONL.}"

if [[ -z "${WANDB_ENTITY:-}" ]]; then
   echo "ERROR: Set WANDB_ENTITY to your Weights & Biases team/entity." >&2
   exit 1
fi

case "${TOOL_MODEL_SIZE}" in
   8b)
      MODEL_CONFIG="${PROJECT_ROOT}/cmd/models/qwen3-8B.sh"
      MODEL_NAME="Qwen3-8B"
      ENV_MAX_TOKENS="${ENV_MAX_TOKENS:-20000}"
      ENABLE_THINKING=1
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
      SGLANG_ARGS=(
         --rollout-num-gpus-per-engine 1
         --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}"
         --sglang-tool-call-parser qwen
      )
      OPTIMIZER_EXTRA_ARGS=()
      ;;
   30b)
      MODEL_CONFIG="${PROJECT_ROOT}/cmd/models/qwen3-30B-A3B.sh"
      MODEL_NAME="Qwen3-30B-A3B-Instruct-2507"
      ENV_MAX_TOKENS="${ENV_MAX_TOKENS:-16384}"
      ENABLE_THINKING=0
      PERF_ARGS=(
         --tensor-model-parallel-size 4
         --sequence-parallel
         --pipeline-model-parallel-size 1
         --context-parallel-size 1
         --expert-model-parallel-size 8
         --expert-tensor-parallel-size 1
         --recompute-granularity full
         --recompute-method uniform
         --recompute-num-layers 1
         --use-dynamic-batch-size
         --max-tokens-per-gpu 8192
      )
      SGLANG_ARGS=(
         --rollout-num-gpus-per-engine 8
         --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.85}"
         --sglang-ep-size 8
         --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
         --sglang-tool-call-parser qwen
         --sglang-moe-runner-backend triton
         --sglang-disable-custom-all-reduce
      )
      OPTIMIZER_EXTRA_ARGS=(
         --optimizer-cpu-offload
         --overlap-cpu-optimizer-d2h-h2d
         --use-precision-aware-optimizer
      )
      ;;
   *)
      echo "ERROR: TOOL_MODEL_SIZE must be 8b or 30b; got ${TOOL_MODEL_SIZE}." >&2
      exit 1
      ;;
esac

echo "[TOOL-USE SPADE ${TOOL_MODEL_SIZE}]"
echo "  model: ${MODEL_NAME}; both roles thinking: ${ENABLE_THINKING}; KL: 0.005"
echo "  reward: 0.6 plateau + 0.4 floored hint regret"
echo "  budget: ${NUM_ROLLOUT:-400} rollouts, rollout batch 24, global batch 192"
echo "  regeneration/delay: ${REGEN_INTERVAL:-8}/${PROPOSER_DELAY:-8}"
echo "  hint plays: 16"
echo "  corpus: ${CORPUS_FILE}"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
   exit 0
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
   echo "ERROR: tool-use corpus not found: ${CORPUS_FILE}." >&2
   exit 1
fi
if [[ ! -f "${MODEL_CONFIG}" ]]; then
   echo "ERROR: model config not found: ${MODEL_CONFIG}. Initialize the Slime submodule if required." >&2
   exit 1
fi

# shellcheck source=/dev/null
source "${MODEL_CONFIG}"

for process_name in sglang ray; do
   pkill -9 "${process_name}" 2>/dev/null || true
done
ray stop --force >/dev/null 2>&1 || true
sleep 3

export PYTHONUNBUFFERED=1
export WEAVE_PRINT_CALL_LINK=false
pip install math_verify weave

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l | tr -d ' ' || true)"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi

mkdir -p "${OUTPUT_DIR}" "${GAMES_DIR}"
export WORKSPACE_DIR
EVAL_CONFIG_FILE="${OUTPUT_DIR}/eval_aime.yaml"
envsubst < "${PROJECT_ROOT}/eval_configs/eval_aime_avg32.yaml" > "${EVAL_CONFIG_FILE}"

GEM_EVAL_CONFIG_FILE="${OUTPUT_DIR}/gem_eval_bfcl_only.yaml"
sed "s|/workspace/spade-workspace|${WORKSPACE_DIR}|g" \
   "${PROJECT_ROOT}/eval_configs/gem_eval_bfcl_only.yaml" > "${GEM_EVAL_CONFIG_FILE}"
for bfcl_dir in "${WORKSPACE_DIR}/bfcl/data" "${WORKSPACE_DIR}/bfcl/data_v3"; do
   if [[ ! -d "${bfcl_dir}" ]]; then
      echo "ERROR: BFCL data directory missing: ${bfcl_dir}." >&2
      exit 1
   fi
done

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/${MODEL_NAME}"
   --ref-load "${MODEL_ROOT}/${MODEL_NAME}_torch_dist"
)
if [[ -n "${LOAD_DIR:-}" ]]; then
   CKPT_ARGS+=( --load "${LOAD_DIR}" )
fi
CKPT_ARGS+=(
   --save "${OUTPUT_DIR}/${MODEL_NAME}_tool_use_blend/"
   --save-interval "${SAVE_INTERVAL:-16}"
)

ROLLOUT_ARGS=(
   --data-source-path spade.slime.data_source.SpadeDataSource
   --rollout-function-path spade.slime.spade_rollout.spade_generate_rollout
   --num-rollout "${NUM_ROLLOUT:-400}"
   --rollout-batch-size 24
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --global-batch-size 192
   --use-dynamic-global-batch-size
   --balance-data
   --apply-chat-template-kwargs '{"enable_thinking":false}'
)

SPADE_ARGS=(
   --spade-gamma1 0.98
   --spade-gamma2 0.85
   --spade-env-temperature 0.6
   --spade-actor-temperature 0.6
   --spade-actor-max-tokens "${ACTOR_MAX_TOKENS:-8192}"
   --spade-env-max-tokens "${ENV_MAX_TOKENS}"
   --spade-max-context-length "${MAX_CONTEXT_LENGTH:-32768}"
   --spade-max-turns "${MAX_TURNS:-25}"
   --spade-gamma 0.99
   --spade-game-regeneration-interval "${REGEN_INTERVAL:-8}"
   --spade-proposer-training-delay "${PROPOSER_DELAY:-8}"
   --spade-num-games-per-rollout 24
   --spade-games-dir "${GAMES_DIR}"
   --spade-game-difficulty medium
   --spade-trajectories-per-game 16
   --spade-cache-dir "${OUTPUT_DIR}/spade_games_cache"
   --spade-game-type tool_use
   --spade-skills API_Orchestration Data_Retrieval State_Modification Error_Recovery Tool_Selection Multi-Step_Workflows
   --spade-env-generation-template qwen3_tool_use_game_generation
   --spade-actor-template qwen3_game
   --spade-use-env-validator
   --spade-reward-normalization grpo
   --spade-env-reward-variant blend
   --spade-plateau-weight 0.6
   --spade-regret-weight 0.4
   --spade-regret-floor
   --spade-regret-scale 0.15
   --spade-micro-lp-weight 0.0
   --spade-frontier-weight 0.0
   --spade-plateau-lo 0.4
   --spade-plateau-hi 0.6
   --spade-plateau-ramp 0.25
   --spade-hint-mode self
   --spade-hint-temperature 0.3
   --spade-hint-max-tokens "${HINT_MAX_TOKENS:-2048}"
   --spade-hint-plays-per-game 16
   --spade-gem-eval-config "${GEM_EVAL_CONFIG_FILE}"
   --spade-compact-filter
   --spade-corpus-file "${CORPUS_FILE}"
   --spade-corpus-max-doc-tokens 4000
   --spade-corpus-seed 42
   --spade-use-env-memory
   --spade-env-memory-max-size 200
)
if [[ "${ENABLE_THINKING}" == "1" ]]; then
   SPADE_ARGS+=( --spade-env-enable-thinking --spade-actor-enable-thinking )
fi

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
   --wandb-group "${WANDB_GROUP:-paper-tool-use-${TOOL_MODEL_SIZE}-blend}"
   --wandb-key "${WANDB_API_KEY:?Set WANDB_API_KEY.}"
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-32}"
   --eval-config "${EVAL_CONFIG_FILE}"
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
   "${OPTIMIZER_EXTRA_ARGS[@]}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_DISABLE_VERSION_CHECK\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"LITELLM_LOCAL_MODEL_COST_MAP\": \"True\",
    \"LITELLM_LOG\": \"WARNING\",
    \"SPADE_MULTITURN_ENV_GEN\": \"1\",
    \"SPADE_RESET_GATE\": \"1\",
    \"SPADE_MASK_MALFORMED_TOOLCALL\": \"${SPADE_MASK_MALFORMED_TOOLCALL:-1}\",
    \"SPADE_TURN_CURRICULUM\": \"0\",
    \"HF_DATASETS_CACHE\": \"${HF_DATASETS_CACHE:-${WORKSPACE_DIR}/hf_cache/datasets}\"
  }
}"

RAY_JOB_ID="spade_tool_use_blend_${TOOL_MODEL_SIZE}_$$"
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
