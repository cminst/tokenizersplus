#!/usr/bin/env bash
set -euo pipefail

# ---- EDIT THESE IF NEEDED ----
PY=python3
SCRIPT=spectralbpe_sanity_v3.py

TRAIN=data/train.txt
EVAL=data/eval.txt
K=16000

OUTROOT=pareto_sweep_v3
SEED=0

# Pareto knob
LAM_LIST=(0.00 0.05 0.10 0.15 0.20 0.25)

# Base hyperparams (from your working run)
BATCH=50
TAU=5
ALPHA=1.0
BETA=0.05
BPE_WARMSTART=500

SIGMA_PERCENTILE=90
COH_LAMBDA_DEFAULT=0.15      # overridden in loop
RERANK_TOP=20000
PPMI_GAMMA=0.5
EMBED_ALPHA=0.0
EMBED_BETA=0.0

CHECKPOINT_EVERY=1000

mkdir -p "$OUTROOT"

for lam in "${LAM_LIST[@]}"; do
  tag="${lam/./p}"  # 0.15 -> 0p15
  outdir="$OUTROOT/lam_${tag}"
  mkdir -p "$outdir"

  logfile="$outdir/run.log"
  cmdfile="$outdir/cmd.txt"

  if [[ -f "$logfile" ]]; then
    echo "[skip] $lam (found $logfile)"
    continue
  fi

  echo "[run] coh_lambda=$lam -> $outdir"
  {
    echo "$PY $SCRIPT \\"
    echo "  --train_text $TRAIN --eval_text $EVAL \\"
    echo "  --vocab_size $K \\"
    echo "  --batch_size $BATCH \\"
    echo "  --tau $TAU --alpha $ALPHA --beta $BETA \\"
    echo "  --sigma_auto --sigma_percentile $SIGMA_PERCENTILE \\"
    echo "  --coh_lambda $lam --rerank_top $RERANK_TOP --ppmi_gamma $PPMI_GAMMA \\"
    echo "  --embed_alpha $EMBED_ALPHA --embed_beta $EMBED_BETA \\"
    echo "  --bpe_warmstart $BPE_WARMSTART \\"
    echo "  --seed $SEED --deterministic_ties \\"
    echo "  --checkpoint_dir $outdir --checkpoint_every $CHECKPOINT_EVERY \\"
    echo "  --train_lm"
  } > "$cmdfile"

  # Run and capture everything
  $PY $SCRIPT \
    --train_text "$TRAIN" --eval_text "$EVAL" \
    --vocab_size "$K" \
    --batch_size "$BATCH" \
    --tau "$TAU" --alpha "$ALPHA" --beta "$BETA" \
    --sigma_auto --sigma_percentile "$SIGMA_PERCENTILE" \
    --coh_lambda "$lam" --rerank_top "$RERANK_TOP" --ppmi_gamma "$PPMI_GAMMA" \
    --embed_alpha "$EMBED_ALPHA" --embed_beta "$EMBED_BETA" \
    --bpe_warmstart "$BPE_WARMSTART" \
    --seed "$SEED" --deterministic_ties \
    --checkpoint_dir "$outdir" --checkpoint_every "$CHECKPOINT_EVERY" \
    --train_lm \
    2>&1 | tee "$logfile"
done

echo "[done] Logs in: $OUTROOT"
