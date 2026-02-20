import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("pareto_sweep_gamma")
LOG_GLOB = "gamma_*/run.log"

RE_BPB  = re.compile(r"^BPB \(eval\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
RE_BPB_NOISY  = re.compile(r"^BPB \(eval noisy\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
RE_TOKS = re.compile(r"^Avg tokens/sent\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
RE_PPMI = re.compile(r"^Avg PPMI of Merges.*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*$")

def parse_one(log_path: Path):
    m = re.search(r"gamma_(\d+p\d+)", str(log_path.parent))
    if not m:
        return None
    gamma = float(m.group(1).replace("p", "."))

    bpe_bpb = spec_bpb = None
    bpe_bpb_noisy = spec_bpb_noisy = None
    bpe_tok = spec_tok = None
    bpe_ppmi = spec_ppmi = None

    for line in log_path.read_text(errors="ignore").splitlines():
        line = line.strip()

        m = RE_BPB.match(line)
        if m:
            bpe_bpb = float(m.group(1))
            spec_bpb = float(m.group(2))
            continue

        m = RE_BPB_NOISY.match(line)
        if m:
            bpe_bpb_noisy = float(m.group(1))
            spec_bpb_noisy = float(m.group(2))
            continue

        m = RE_TOKS.match(line)
        if m:
            bpe_tok = float(m.group(1))
            spec_tok = float(m.group(2))
            continue

        m = RE_PPMI.match(line)
        if m:
            bpe_ppmi = float(m.group(1))
            spec_ppmi = float(m.group(2))
            continue

    if None in (bpe_bpb, spec_bpb, bpe_tok, spec_tok, bpe_ppmi, spec_ppmi):
        return None

    return {
        "ppmi_gamma": gamma,
        "bpe_bpb": bpe_bpb,
        "spec_bpb": spec_bpb,
        "bpb_delta_pct": 100.0 * (spec_bpb / bpe_bpb - 1.0),
        "bpe_tokens_sent": bpe_tok,
        "spec_tokens_sent": spec_tok,
        "tokens_delta_pct": 100.0 * (spec_tok / bpe_tok - 1.0),
        "bpe_ppmi": bpe_ppmi,
        "spec_ppmi": spec_ppmi,
        "ppmi_gain_pct": 100.0 * (spec_ppmi / bpe_ppmi - 1.0),
        "bpe_bpb_noisy": (bpe_bpb_noisy if bpe_bpb_noisy is not None else float("nan")),
        "spec_bpb_noisy": (spec_bpb_noisy if spec_bpb_noisy is not None else float("nan")),
        "bpe_noise_increase_pct": (100.0 * (bpe_bpb_noisy / bpe_bpb - 1.0) if (bpe_bpb_noisy is not None and bpe_bpb) else float("nan")),
        "spec_noise_increase_pct": (100.0 * (spec_bpb_noisy / spec_bpb - 1.0) if (spec_bpb_noisy is not None and spec_bpb) else float("nan")),
        "robust_delta_pct": (
            (100.0 * (spec_bpb_noisy / spec_bpb - 1.0) if (spec_bpb_noisy is not None and spec_bpb) else float("nan"))
            - (100.0 * (bpe_bpb_noisy / bpe_bpb - 1.0) if (bpe_bpb_noisy is not None and bpe_bpb) else float("nan"))
        ),
        "log_path": str(log_path),
    }

def main():
    logs = sorted(ROOT.glob(LOG_GLOB))
    rows = []
    for lp in logs:
        r = parse_one(lp)
        if r is None:
            print(f"[warn] could not parse: {lp}")
            continue
        rows.append(r)

    rows.sort(key=lambda x: x["ppmi_gamma"])
    if not rows:
        raise SystemExit(f"No parseable logs found under {ROOT / LOG_GLOB}")

    # ---- CSV ----
    csv_path = ROOT / "pareto.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {csv_path}")

    # ---- Plot ----
    xs = [r["ppmi_gain_pct"] for r in rows]
    ys = [r["bpb_delta_pct"] for r in rows]

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

    scale = 0.8
    fig, ax = plt.subplots(figsize=(7.2*scale, 5.0*scale))
    ax.scatter(xs, ys, color="C2")
    ax.plot(xs, ys, linestyle="--", linewidth=1.0, color="C2")

    x_min, x_max = min(xs), max(xs)
    x_span = x_max - x_min if x_max > x_min else 1.0
    x_left_pad = 0.08 * x_span
    x_right_trim = -0.05 * x_span
    ax.set_xlim(x_min - x_left_pad, x_max - x_right_trim)

    for i, r in enumerate(rows):
        label_offset = (0, 3)
        if i == 0:
            label_offset = (6, 3)
        elif i == len(rows) - 1:
            label_offset = (-6, -2)

        ax.annotate(
            rf"${r['ppmi_gamma']:.2f}$",
            (r["ppmi_gain_pct"], r["bpb_delta_pct"]),
            fontsize=12,
            xytext=label_offset,
            textcoords="offset points",
            ha="right",
            va="bottom",
        )

    ax.axhline(0.0, linewidth=1)
    ax.grid(True, alpha=0.25)

    ax.set_xlabel(r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%) $\uparrow$")
    ax.set_ylabel(r"BPB change $\Delta_{\mathrm{BPB}}$ (%) $\downarrow$")
    ax.set_title(r"SpectralBPE Pareto sweep over $\beta$ (PPMI exponent)")

    fig.tight_layout()

    out_pdf = ROOT / "pareto_gamma.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[ok] wrote {out_pdf}")

    # ---- Robustness Plot (optional; requires --lm_robust_eval logs) ----
    rrows = [r for r in rows if math.isfinite(r.get("robust_delta_pct", float("nan")))]
    if rrows:
        xs_r = [r["ppmi_gain_pct"] for r in rrows]
        ys_r = [r["robust_delta_pct"] for r in rrows]

        fig2, ax2 = plt.subplots(figsize=(7.2, 5.0))
        ax2.scatter(xs_r, ys_r, color="C2")
        ax2.plot(xs_r, ys_r, linestyle="--", linewidth=1.0, color="C2")
        ax2.axhline(0.0, linewidth=1)
        ax2.grid(True, alpha=0.25)

        ax2.set_xlabel(r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%) $\uparrow$")
        ax2.set_ylabel(r"Robustness delta $\Delta_{\mathrm{rob}}$ (%) (more negative is better)")
        ax2.set_title(r"Robustness under word-level swap noise (p=0.10)")

        for i, r in enumerate(rrows):
            ax2.annotate(
                rf"${r['ppmi_gamma']:.2f}$",
                (r["ppmi_gain_pct"], r["robust_delta_pct"]),
                fontsize=12,
                xytext=(0, 3),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )

        fig2.tight_layout()
        out_pdf2 = ROOT / "robust_gamma.pdf"
        fig2.savefig(out_pdf2, bbox_inches="tight")
        print(f"[ok] wrote {out_pdf2}")

        corr = float(np.corrcoef(np.array(xs_r), np.array(ys_r))[0, 1]) if len(xs_r) >= 2 else float("nan")
        print(f"[robust] corr(ppmi_gain_pct, robust_delta_pct) = {corr:.4f}")

if __name__ == "__main__":
    main()
