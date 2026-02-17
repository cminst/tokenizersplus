import csv
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

END_WORD = "</w>"
LADEC_URL = "https://huggingface.co/datasets/cminst/LADECv1/resolve/main/LADECv1-2019.csv"
DEFAULT_LADEC_PATH = os.path.join("data", "LADECv1-2019.csv")


def load_merges(path: Path) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle dict-with-"merges" and raw-list formats
    raw = data.get("merges", []) if isinstance(data, dict) else data
    merges = []
    for pair in raw:
        if isinstance(pair, list) or isinstance(pair, tuple):
            if len(pair) == 2:
                merges.append((pair[0], pair[1]))
        else:
            parts = str(pair).split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    if not merges:
        raise ValueError(f"No merges parsed from {path}")
    return merges


def ensure_ladec_csv(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(LADEC_URL) as resp:
        raw = resp.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw.decode("utf-8", errors="ignore"))
    print(f"[download] wrote {path}", file=sys.stderr)


def encode(word: str, rank: Dict[Tuple[str, str], int]) -> List[str]:
    """Standard BPE encoding over characters with </w> on last char."""
    if not word:
        return []
    toks = list(word)
    toks[-1] += END_WORD

    while True:
        min_rank = float("inf")
        best_pair = None

        # Find best ranked adjacent pair
        for i in range(len(toks) - 1):
            pair = (toks[i], toks[i + 1])
            r = rank.get(pair)
            if r is not None and r < min_rank:
                min_rank = r
                best_pair = pair

        if best_pair is None:
            break

        # Merge best_pair
        a, b = best_pair
        merged = a + b
        new_toks = []
        i = 0
        while i < len(toks):
            if i < len(toks) - 1 and (toks[i], toks[i + 1]) == best_pair:
                new_toks.append(merged)
                i += 2
            else:
                new_toks.append(toks[i])
                i += 1
        toks = new_toks

    return toks


def check_boundary(tokens: List[str], gold_split_index: int) -> bool:
    # Success 1: whole word token
    if len(tokens) == 1:
        return True

    # Success 2: exact boundary alignment
    current_len = 0
    for t in tokens:
        clean_t = t.replace(END_WORD, "")
        current_len += len(clean_t)
        if current_len == gold_split_index:
            return True
        if current_len > gold_split_index:
            return False
    return False


def load_compounds(ladec_path: str) -> List[Tuple[str, int]]:
    ensure_ladec_csv(ladec_path)
    compounds = []
    with open(ladec_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1 = (row.get("c1") or "").strip()
            stim = (row.get("stim") or "").strip()
            if not c1 or not stim:
                continue
            compounds.append((stim, len(c1)))
    return compounds


def discover_gamma_dirs(root: Path) -> List[Tuple[float, Path]]:
    out = []
    for d in root.glob("gamma_*"):
        m = re.search(r"gamma_(\d+p\d+)", d.name)
        if not m:
            continue
        gamma = float(m.group(1).replace("p", "."))
        out.append((gamma, d))
    out.sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="pareto_sweep_gamma",
                    help="Root directory containing gamma_*/ subfolders (default: pareto_sweep_gamma)")
    ap.add_argument("--bpe_name", type=str, default="bpe_final.json",
                    help="Filename for BPE merges inside each gamma dir (default: bpe_final.json)")
    ap.add_argument("--spec_name", type=str, default="spectralbpe_seed0_final.json",
                    help="Filename for SpectralBPE merges inside each gamma dir (default: spectralbpe_seed0_final.json)")
    ap.add_argument("--ladec_path", type=str, default=DEFAULT_LADEC_PATH,
                    help=f"Path to LADEC csv (default: {DEFAULT_LADEC_PATH})")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Optional output CSV path (default: <root>/boundary_gamma.csv)")
    ap.add_argument("--fig_dir", type=str, default="figs",
                    help="Directory to write the plot PDF (default: figs)")
    ap.add_argument("--fig_name", type=str, default="boundary_gamma.pdf",
                    help="Plot filename (default: boundary_gamma.pdf)")
    ap.add_argument("--print_examples", action="store_true",
                    help="Also print differing examples for each gamma (can be large).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"[error] root not found: {root}")

    gamma_dirs = discover_gamma_dirs(root)
    if not gamma_dirs:
        raise SystemExit(f"[error] no gamma_* dirs found under {root}")

    compounds = load_compounds(args.ladec_path)
    n = len(compounds)
    print(f"[dataset] loaded {n} compounds from {args.ladec_path}", file=sys.stderr)

    # Load BPE baseline once from the first available gamma dir
    bpe_rank = None
    bpe_hits = 0
    bpe_rate = 0.0
    bpe_path_str = "N/A"

    for _, d in gamma_dirs:
        possible_bpe = d / args.bpe_name
        if possible_bpe.exists():
            print(f"[info] Loading BPE baseline from {possible_bpe}", file=sys.stderr)
            bpe_rank = {p: i for i, p in enumerate(load_merges(possible_bpe))}
            bpe_path_str = str(possible_bpe)
            # Compute stats once
            for word, split_idx in compounds:
                if check_boundary(encode(word, bpe_rank), split_idx):
                    bpe_hits += 1
            bpe_rate = bpe_hits / n if n else float("nan")
            break

    if bpe_rank is None:
        print("[warn] Could not find any BPE file in any gamma directory!", file=sys.stderr)

    rows = []
    table_rows = []

    for gamma, d in gamma_dirs:
        spec_path = d / args.spec_name
        if not spec_path.exists():
            print(f"[warn] missing {spec_path}, skipping gamma={gamma}", file=sys.stderr)
            continue

        # SpectralBPE (per gamma)
        spec_rank = {p: i for i, p in enumerate(load_merges(spec_path))}
        spec_hits = 0

        diff_examples = []
        for word, split_idx in compounds:
            ok_spec = check_boundary(encode(word, spec_rank), split_idx)
            if ok_spec:
                spec_hits += 1

            if args.print_examples and bpe_rank is not None:
                ok_bpe = check_boundary(encode(word, bpe_rank), split_idx)
                if ok_bpe != ok_spec:
                    diff_examples.append((word, ok_bpe, ok_spec))

        spec_rate = spec_hits / n if n else float("nan")
        delta = spec_rate - bpe_rate

        rows.append({
            "ppmi_gamma": gamma,
            "bpe_hits": bpe_hits,
            "bpe_rate": bpe_rate,
            "spec_hits": spec_hits,
            "spec_rate": spec_rate,
            "delta_rate": delta,
            "bpe_path": bpe_path_str,
            "spec_path": str(spec_path),
        })

        table_rows.append([
            f"{gamma:.2f}",
            f"{bpe_hits}/{n}",
            f"{bpe_rate*100:.2f}%",
            f"{spec_hits}/{n}",
            f"{spec_rate*100:.2f}%",
            f"{delta*100:+.2f}%",
        ])

        if args.print_examples and diff_examples:
            print(f"\n[gamma={gamma:.2f}] differing examples (first 20):")
            for w, ok_b, ok_s in diff_examples[:20]:
                print(f"  {w!r}: BPE={ok_b}  Spec={ok_s}")

    if not rows:
        raise SystemExit("[error] no rows computed (missing jsons?)")

    # ---- ASCII table ----
    print("\n" + tabulate(
        table_rows,
        headers=["gamma", "BPE hits", "BPE rate", "Spec hits", "Spec rate", "Δ (Spec-BPE)"],
        tablefmt="github",
    ))

    # ---- CSV ----
    out_csv = Path(args.out_csv) if args.out_csv else (root / "boundary_gamma.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] wrote {out_csv}", file=sys.stderr)

    # ---- Plot ----
    os.makedirs(args.fig_dir, exist_ok=True)
    fig_path = Path(args.fig_dir) / args.fig_name

    gammas = [r["ppmi_gamma"] for r in rows]
    bpe_rates = [r["bpe_rate"] for r in rows]
    spec_rates = [r["spec_rate"] for r in rows]

    x = np.arange(len(gammas), dtype=np.float64)
    width = 0.38

    plt.rcParams.update({
        "font.family": ["Times New Roman", "Times", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x - width/2, bpe_rates, width, label="BPE") # default color
    ax.bar(x + width/2, spec_rates, width, label="SpectralBPE", color="C2")  # green for consistency

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g:.2f}" for g in gammas])
    ax.set_ylabel("Boundary accuracy")
    ax.set_xlabel(r"$\gamma$ (PPMI exponent)")
    ax.set_title("LADEC compound boundary accuracy vs. $\gamma$")
    ax.set_ylim(bottom=0.77, top=0.84)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"[out] wrote {fig_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
