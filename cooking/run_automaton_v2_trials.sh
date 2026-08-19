#!/usr/bin/env bash
set -u
set -o pipefail

# Sequential five-trial launcher for the automaton-liveness-v2 branch.
# Defaults: 6 experiments x 2 models x 2 shield modes x 5 trials = 120 runs.

TRIALS="${TRIALS:-5}"
OLLAMA_URL="${OLLAMA_URL:-http://volta13:11434}"
MODELS="${MODELS:-qwen25,qwen3}"
MODES="${MODES:-shielded,unshielded}"
EXPERIMENTS="${EXPERIMENTS:-01,05,07,08,09,10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-automaton_v2_runs}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/reasoning" "$OUTPUT_ROOT/images"
MANIFEST="$OUTPUT_ROOT/manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'run_id\tmodel\tmode\texperiment\ttrial\texit_code\tstatus\tlog\n' > "$MANIFEST"

declare -A EXPERIMENT_SCRIPTS=(
  [01]="cooking_stars.py"
  [05]="cooking_exp05_chili_distractor.py"
  [07]="cooking_exp07_human_completed_garlic.py"
  [08]="cooking_exp08_cumin_appears_required_minimal_prompt.py"
  [09]="cooking_exp09_onion_rollback.py"
  [10]="cooking_exp10_onion_rollback_plus_cumin.py"
)

IFS=',' read -r -a EXPERIMENT_LIST <<< "$EXPERIMENTS"
experiments=()
for EXP in "${EXPERIMENT_LIST[@]}"; do
  if [[ -z "${EXPERIMENT_SCRIPTS[$EXP]+x}" ]]; then
    echo "Unknown experiment: $EXP" >&2
    exit 2
  fi
  experiments+=("${EXP}:${EXPERIMENT_SCRIPTS[$EXP]}")
done

model_profile() {
  case "$1" in
    qwen25)
      MODEL_ID="qwen2.5vl:3b"
      MODEL_PREFIX="qwen25"
      NUM_PREDICT=1024
      ;;
    qwen3)
      MODEL_ID="qwen3-vl:32b"
      MODEL_PREFIX="qwen3"
      NUM_PREDICT=4096
      ;;
    *)
      echo "Unknown model profile: $1" >&2
      return 2
      ;;
  esac
}

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a MODE_LIST <<< "$MODES"

TOTAL=$(( ${#MODEL_LIST[@]} * ${#MODE_LIST[@]} * ${#experiments[@]} * TRIALS ))
RUN_NUMBER=0

for MODEL_PROFILE in "${MODEL_LIST[@]}"; do
  model_profile "$MODEL_PROFILE" || exit $?

  for MODE in "${MODE_LIST[@]}"; do
    case "$MODE" in
      shielded|unshielded) ;;
      *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
    esac

    for entry in "${experiments[@]}"; do
      EXP="${entry%%:*}"
      SCRIPT="${entry#*:}"

      for TRIAL in $(seq 1 "$TRIALS"); do
        RUN_NUMBER=$((RUN_NUMBER + 1))
        RUN="${MODEL_PREFIX}_exp${EXP}_${MODE}_trial${TRIAL}_$(date +%Y%m%d_%H%M%S_%N)"
        LOG="$OUTPUT_ROOT/logs/${RUN}.log"
        TRACE="$OUTPUT_ROOT/reasoning/${RUN}.jsonl"
        SHIELD_TRACE="$OUTPUT_ROOT/reasoning/${RUN}_shield.jsonl"
        IMAGE_DIR="$OUTPUT_ROOT/images/${RUN}"

        echo
        echo "======================================================================"
        echo "RUN $RUN_NUMBER/$TOTAL"
        echo "Run ID: $RUN"
        echo "Model: $MODEL_ID"
        echo "Mode: $MODE"
        echo "Experiment: $EXP ($SCRIPT)"
        echo "Trial: $TRIAL/$TRIALS"
        echo "Log: $LOG"
        echo "======================================================================"

        common_args=(
          --planner qwen_ollama
          --ollama-url "$OLLAMA_URL"
          --ollama-model "$MODEL_ID"
          --ollama-timeout 1800
          --ollama-http-retries 0
          --ollama-num-predict "$NUM_PREDICT"
          --ollama-keep-alive 2h
          --no-ollama-think
          --salt-max-additions 3
          --seed 0
          --run-id "$RUN"
          --trace-jsonl "$TRACE"
          --image-dir "$IMAGE_DIR"
        )

        if [[ "$MODE" == "shielded" ]]; then
          mode_args=(
            --shield-mode template_guidance
            --shield-gamma 0.20
            --shield-theta-prune 0.10
            --shield-guidance-threshold 0.95
            --shield-reprompt-limit 12
            --max-steps 50
            --shield-trace-jsonl "$SHIELD_TRACE"
          )
        else
          mode_args=(
            --shield-mode off
            --unshielded-max-steps 50
          )
        fi

        set +e
        python -u "$SCRIPT" "${common_args[@]}" "${mode_args[@]}" 2>&1 | tee "$LOG"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        if [[ $EXIT_CODE -eq 0 ]]; then
          STATUS="PASS"
        else
          STATUS="FAIL"
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$RUN" "$MODEL_ID" "$MODE" "$EXP" "$TRIAL" \
          "$EXIT_CODE" "$STATUS" "$LOG" >> "$MANIFEST"

        echo "Completed $RUN with status $STATUS (exit $EXIT_CODE)"
      done
    done
  done
done

echo
echo "All requested runs finished."
echo "Manifest: $MANIFEST"
