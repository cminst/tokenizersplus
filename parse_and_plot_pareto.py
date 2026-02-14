#!/usr/bin/env python3
import re
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("pareto_sweep_v3")   # must match OUTROOT in the bash script
LOG_GLOB = "lam_*/run.log"

# Regex helpers (robust to spacing/alignment)
RE_BPB = re.compile(r"^BPB \(eval\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
RE_TOKS = re.compile(r"^Avg tokens/sent\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
RE_PPMI = re.compile(r"^Avg PPMI of Merges.*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*$")

def parse_one(log_path: Path):
    # lam_0p15 -> 0.15
    m = re.search(r"lam_(\d+p\d+)", str(log_path.parent))
    if not m:
        return None
    lam = float(m.group(1).replace("p", "."))

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

    # require all three to build a good pareto point
    if None in (bpe_bpb, spec_bpb, bpe_tok, spec_tok, bpe_ppmi, spec_ppmi):
        return None

    bpb_delta_pct = 100.0 * (spec_bpb / bpe_bpb - 1.0)
    tok_delta_pct = 100.0 * (spec_tok / bpe_tok - 1.0)
    ppmi_gain_pct = 100.0 * (spec_ppmi / bpe_ppmi - 1.0)

    return {
        "coh_lambda": lam,
        "bpe_bpb": bpe_bpb,
        "spec_bpb": spec_bpb,
        "bpb_delta_pct": bpb_delta_pct,
        "bpe_tokens_sent": bpe_tok,
        "spec_tokens_sent": spec_tok,
        "tokens_delta_pct": tok_delta_pct,
        "bpe_ppmi": bpe_ppmi,
        "spec_ppmi": spec_ppmi,
        "ppmi_gain_pct": ppmi_gain_pct,
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

    rows.sort(key=lambda x: x["coh_lambda"])
    if not rows:
        raise SystemExit(f"No parseable logs found under {ROOT / LOG_GLOB}")

    # Write CSV
    csv_path = ROOT / "pareto.csv"
    fieldnames = [
        "coh_lambda",
        "bpe_bpb", "spec_bpb", "bpb_delta_pct",
        "bpe_tokens_sent", "spec_tokens_sent", "tokens_delta_pct",
        "bpe_ppmi", "spec_ppmi", "ppmi_gain_pct",
        "log_path",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {csv_path}")

    # Plot: x = cohesion gain (%), y = BPB delta (%)
    xs = [r["ppmi_gain_pct"] for r in rows]
    ys = [r["bpb_delta_pct"] for r in rows]

    plt.figure()
    plt.scatter(xs, ys)

    for r in rows:
        plt.annotate(
            f'{r["coh_lambda"]:.2f}',
            (r["ppmi_gain_pct"], r["bpb_delta_pct"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Cohesion gain: 100*(PPMI_spec/PPMI_bpe - 1) [%] (higher is better)")
    plt.ylabel("BPB delta: 100*(BPB_spec/BPB_bpe - 1) [%] (lower is better)")
    plt.title("SpectralBPE Pareto Sweep over coh_lambda")

    out_png = ROOT / "pareto.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[ok] wrote {out_png}")

if __name__ == "__main__":
    main()
