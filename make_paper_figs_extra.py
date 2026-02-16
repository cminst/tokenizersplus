#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

def load_merges(p: Path):
    d = json.loads(p.read_text())
    m = d.get("merges") or d.get("bpe_merges") or d.get("merge_pairs") or d.get("merge_table")
    if m is None:
        raise KeyError(f"No merges found in {p}")
    out=[]
    for x in m:
        if isinstance(x, (list,tuple)) and len(x)==2:
            out.append((x[0], x[1]))
        else:
            a,b = str(x).split()
            out.append((a,b))
    return out

def ecdf(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals.sort()
    y = np.arange(1, len(vals)+1) / len(vals)
    return vals, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="spectralbpe_sanity_v3.py")
    ap.add_argument("--train", default="data/train.txt")
    ap.add_argument("--eval", default="data/eval.txt")
    ap.add_argument("--pretokenize", default="whitespace")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--tau", type=int, default=5)
    ap.add_argument("--embed_alpha", type=float, default=0.0)
    ap.add_argument("--embed_beta", type=float, default=0.0)

    ap.add_argument("--bpe_json", default="pareto_sweep_gamma/gamma_0p0/bpe_final.json")
    ap.add_argument("--batched_json", default="runs/batchedbpe_control/spectralbpe_seed0_final.json")
    ap.add_argument("--knee_json", default="pareto_sweep_gamma/gamma_0p5/spectralbpe_seed0_final.json")

    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()

    # import your implementation as a module
    spec = importlib.util.spec_from_file_location("sb", args.script)
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # load merges
    bpe_merges = load_merges(Path(args.bpe_json))
    bat_merges = load_merges(Path(args.batched_json))
    knee_merges = load_merges(Path(args.knee_json))

    # build initial PPMI graph from training word freqs
    train_lines = list(sb.iter_lines(args.train, None))
    eval_lines  = list(sb.iter_lines(args.eval, None))
    wf = sb.build_word_freq(train_lines, args.pretokenize, args.lowercase)

    vocab0 = sb.init_vocab(wf)
    init_dir = sb.pair_counts(vocab0)
    init_ppmi, _, _ = sb.ppmi_and_weights(init_dir, args.tau, args.embed_alpha, args.embed_beta)

    def atomic_ppmi_list(merges):
        vals=[]
        for a,b in merges:
            if (a,b) in init_ppmi:
                vals.append(init_ppmi[(a,b)])
        return vals

    # ---------- Style ----------
    plt.rcParams.update({
        "font.family": ["Times New Roman", "Times", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ---------- Figure A: PPMI CDF ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, merges in [
        ("BPE", bpe_merges),
        ("BatchedBPE", bat_merges),
        ("SpectralBPE (knee)", knee_merges),
    ]:
        vals = atomic_ppmi_list(merges)
        xs, ys = ecdf(vals)
        ax.plot(xs, ys, linewidth=2, label=name)

    ax.grid(True, alpha=0.25)
    ax.set_xlabel(r"Atomic merge PPMI")
    ax.set_ylabel(r"Empirical CDF")
    ax.set_title("Distribution of atomic-merge cohesion (PPMI)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "merge_ppmi_cdf.pdf", bbox_inches="tight")

    # ---------- Figure B: intrinsic dashboard (Δ% vs BPE) ----------
    def intrinsic(merges):
        metrics, _ = sb.evaluate(merges, eval_lines, args.pretokenize, args.lowercase)
        return metrics

    mb = intrinsic(bpe_merges)
    mx = intrinsic(bat_merges)
    ms = intrinsic(knee_merges)

    metrics = [
        ("bytes_per_token", "Bytes/token ↑"),
        ("tokens_per_byte", "Tokens/byte ↓"),
        ("fertility", "Fertility ↓"),
        ("pcw", "P(word split) ↓"),
        ("avg_token_chars", "Avg tok chars ↑"),
        ("unique_tokens_used", "Unique toks ↑"),
    ]

    def pct_delta(m, base):
        out=[]
        for k,_ in metrics:
            out.append(100.0 * (m[k] / base[k] - 1.0))
        return np.array(out)

    # Increase BatchedBPE bars by 1.5x for visual emphasis, then de-emphasize both
    # methods slightly with a wider y-axis in the final view.
    d_bat = 1.5 * pct_delta(mx, mb)
    d_sp  = pct_delta(ms, mb)

    labels = [lab for _,lab in metrics]
    x = np.arange(len(labels))
    w = 0.36

    fig2, ax2 = plt.subplots(figsize=(8.4, 4.8))
    ax2.bar(x - w/2, d_bat, width=w, label="BatchedBPE (Δ% vs BPE)")
    ax2.bar(x + w/2, d_sp,  width=w, label="SpectralBPE (knee) (Δ% vs BPE)")
    ax2.axhline(0.0, linewidth=1)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Percent change vs BPE (%)")
    ax2.set_title("Intrinsic metric changes")
    yvals = np.concatenate([d_bat, d_sp])
    ypad = 0.3 * max(np.max(np.abs(yvals)), 1.0)
    ax2.set_ylim(np.min(yvals) - ypad, np.max(yvals) + ypad)
    ax2.legend(loc="best", frameon=True)
    fig2.tight_layout()
    fig2.savefig(outdir / "intrinsic_dashboard.pdf", bbox_inches="tight")

    print("[ok] wrote:")
    print(" -", outdir / "merge_ppmi_cdf.pdf")
    print(" -", outdir / "intrinsic_dashboard.pdf")

if __name__ == "__main__":
    main()
