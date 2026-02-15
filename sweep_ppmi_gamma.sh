#!/usr/bin/env bash
set -euo pipefail

PY=python3
SCRIPT=spectralbpe_sanity_v3.py

TRAIN=data/train.txt
EVAL=data/eval.txt
K=16000

OUTROOT=pareto_sweep_gamma
SEED=0

# How many runs to execute concurrently (set to your CPU count or less)
N_JOBS=${N_JOBS:-8}

# Sweep knob
GAMMA_LIST=(0.0 0.25 0.50 0.75 1.00 1.25 1.50)

# Fixed settings
BATCH=50
TAU=5
ALPHA=1.0
BETA=0.05
BPE_WARMSTART=500

SIGMA_PERCENTILE=90
COH_LAMBDA=0.15
RERANK_TOP=20000
EMBED_ALPHA=0.0
EMBED_BETA=0.0
CHECKPOINT_EVERY=1000

mkdir -p "$OUTROOT"

run_one () {
  local gamma="$1"
  local tag="${gamma/./p}"          # 0.25 -> 0p25
  local outdir="$OUTROOT/gamma_${tag}"
  local logfile="$outdir/run.log"
  local cmdfile="$outdir/cmd.txt"

  mkdir -p "$outdir"

  if [[ -f "$logfile" ]]; then
    echo "[skip] gamma=$gamma (found $logfile)"
    return 0
  fi

  {
    echo "$PY $SCRIPT \\"
    echo "  --train_text $TRAIN --eval_text $EVAL \\"
    echo "  --vocab_size $K \\"
    echo "  --batch_size $BATCH \\"
    echo "  --tau $TAU --alpha $ALPHA --beta $BETA \\"
    echo "  --sigma_auto --sigma_percentile $SIGMA_PERCENTILE \\"
    echo "  --coh_lambda $COH_LAMBDA --rerank_top $RERANK_TOP --ppmi_gamma $gamma \\"
    echo "  --embed_alpha $EMBED_ALPHA --embed_beta $EMBED_BETA \\"
    echo "  --bpe_warmstart $BPE_WARMSTART \\"
    echo "  --seed $SEED --deterministic_ties \\"
    echo "  --checkpoint_dir $outdir --checkpoint_every $CHECKPOINT_EVERY \\"
    echo "  --train_lm \\"
    echo "  --lm_robust_eval --lm_noise_mode swap --lm_noise_prob 0.10"
  } > "$cmdfile"

  echo "[run] gamma=$gamma -> $outdir"

  # Capture everything to logfile (no console spam; remove ">" if you want also stdout)
  $PY $SCRIPT \
    --train_text "$TRAIN" --eval_text "$EVAL" \
    --vocab_size "$K" \
    --batch_size "$BATCH" \
    --tau "$TAU" --alpha "$ALPHA" --beta "$BETA" \
    --sigma_auto --sigma_percentile "$SIGMA_PERCENTILE" \
    --coh_lambda "$COH_LAMBDA" --rerank_top "$RERANK_TOP" --ppmi_gamma "$gamma" \
    --embed_alpha "$EMBED_ALPHA" --embed_beta "$EMBED_BETA" \
    --bpe_warmstart "$BPE_WARMSTART" \
    --seed "$SEED" --deterministic_ties \
    --checkpoint_dir "$outdir" --checkpoint_every "$CHECKPOINT_EVERY" \
    --train_lm \
    --lm_robust_eval --lm_noise_mode swap --lm_noise_prob 0.10 \
    >"$logfile" 2>&1
}

export -f run_one
export PY SCRIPT TRAIN EVAL K OUTROOT SEED
export BATCH TAU ALPHA BETA BPE_WARMSTART SIGMA_PERCENTILE COH_LAMBDA RERANK_TOP EMBED_ALPHA EMBED_BETA CHECKPOINT_EVERY

printf "%s\n" "${GAMMA_LIST[@]}" | xargs -n 1 -P "$N_JOBS" -I {} bash -lc 'run_one "$@"' _ {}

echo "[done] Logs in: $OUTROOT"
