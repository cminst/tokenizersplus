export PY=python3
export SCRIPT=spectralbpe_sanity_v3.py
export TRAIN=data/train.txt
export EVAL=data/eval.txt
export K=16000
export SEED=0

# match your current sweep defaults
export BATCH=50
export TAU=5
export ALPHA=1.0
export BETA=0.05
export BPE_WARMSTART=500
export SIGMA_PERCENTILE=90
export RERANK_TOP=20000
export EMBED_ALPHA=0.0
export EMBED_BETA=0.0
export CHECKPOINT_EVERY=1000

export N_JOBS=16

OUTROOT=pareto_sweep_lambda
mkdir -p "$OUTROOT"

LAMBDA_LIST=(0.0 0.05 0.10 0.15 0.2 0.25 0.30)

run_one_lambda () {
  local lam="$1"
  local tag="${lam/./p}"
  local outdir="$OUTROOT/lambda_${tag}"
  local logfile="$outdir/run.log"
  mkdir -p "$outdir"
  if [[ -f "$logfile" ]]; then
    echo "[skip] lambda=$lam (found $logfile)"
    return 0
  fi

  $PY $SCRIPT \
    --train_text "$TRAIN" --eval_text "$EVAL" \
    --vocab_size "$K" --methods bpe,spectralbpe \
    --batch_size "$BATCH" --tau "$TAU" --alpha "$ALPHA" --beta "$BETA" \
    --sigma_auto --sigma_percentile "$SIGMA_PERCENTILE" \
    --coh_lambda "$lam" --ppmi_gamma 0.50 --rerank_top "$RERANK_TOP" \
    --embed_alpha "$EMBED_ALPHA" --embed_beta "$EMBED_BETA" \
    --bpe_warmstart "$BPE_WARMSTART" \
    --seed "$SEED" --deterministic_ties \
    --checkpoint_dir "$outdir" --checkpoint_every "$CHECKPOINT_EVERY" \
    --train_lm --lm_robust_eval --lm_noise_mode swap --lm_noise_prob 0.10 \
    --out_json "$outdir/out.json" \
    >"$logfile" 2>&1

  echo "[done] lambda=$lam -> $outdir"
}

export -f run_one_lambda
export PY SCRIPT TRAIN EVAL K SEED
export BATCH TAU ALPHA BETA BPE_WARMSTART SIGMA_PERCENTILE RERANK_TOP EMBED_ALPHA EMBED_BETA CHECKPOINT_EVERY OUTROOT

printf "%s\n" "${LAMBDA_LIST[@]}" | xargs -n 1 -P "$N_JOBS" -I {} bash -lc 'run_one_lambda "$@"' _ {}
echo "[done] lambda sweep logs in: $OUTROOT"

python3 parse_and_plot_lambda.py
python3 plot_lambda_figs.py
