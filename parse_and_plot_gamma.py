#!/usr/bin/env python3
import re
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("pareto_sweep_gamma")
LOG_GLOB = "gamma_*/run.log"

RE_BPB  = re.compile(r"^BPB \(eval\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
RE_TOKS = re.compile(r"^Avg tokens/sent\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
RE_PPMI = re.compile(r"^Avg PPMI of Merges.*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*$")

def parse_one(log_path: Path):
    m = re.search(r"gamma_(\d+p\d+)", str(log_path.parent))
    if not m:
        return None
    gamma = float(m.group(1).replace("p", "."))

    bpe_bpb = spec_bpb = None
    bpe_tok = spec_tok = None
    bpe_ppmi = spec_ppmi = None

    for line in log_path.read_text(errors="ignore").splitlines():
        line = line.strip()

        m = RE_BPB.match(line)
        if m:
            bpe_bpb = float(m.group(1))
            spec_bpb = float(m.group(2))
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
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {csv_path}")

    # ---- Plot ----
    xs = [r["ppmi_gain_pct"] for r in rows]
    ys = [r["bpb_delta_pct"] for r in rows]

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(xs, ys)

    for r in rows:
        ax.annotate(
            rf"${r['ppmi_gamma']:.2f}$",
            (r["ppmi_gain_pct"], r["bpb_delta_pct"]),
            fontsize=12,
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.axhline(0.0, linewidth=1)
    ax.grid(True, alpha=0.25)

    ax.set_xlabel(r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%) $\uparrow$")
    ax.set_ylabel(r"BPB change $\Delta_{\mathrm{BPB}}$ (%) $\downarrow$")
    ax.set_title(r"SpectralBPE Pareto sweep over $\gamma$ (PPMI exponent)")

    fig.tight_layout()

    out_pdf = ROOT / "pareto_gamma.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[ok] wrote {out_pdf}")

if __name__ == "__main__":
    main()
