#!/usr/bin/env python3
"""Post-hoc *exact* replication of SpectralBPE's in-run cohesion metric, with an added Unigram column.

This script reproduces the numbers printed by spectralbpe_sanity_v3.py:

  == Vocabulary Quality (Statistical Cohesion) ==
  Avg PPMI of Merges (↑) | BPE | SpectralBPE
  (Computed on min(b_n, s_n) common atomic pairs)

by:
  1) Recomputing init_dir and init_ppmi_dir from the training text using the *same* helper
     functions from spectralbpe_sanity_v3.py.
  2) Running the same calc_vocab_quality() on BPE/Spectral merge lists.

For Unigram, since there is no merge list, we construct a deterministic "pseudo-merge" list:
  - Collect all adjacent character pairs that appear inside any SentencePiece piece in the
    provided .model (excluding meta/control pieces).
  - Use that set of pairs as the pseudo merge list, and evaluate with the same init_ppmi.

No approximations: PPMI is computed exactly from the corpus counts.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import List, Tuple, Set


def load_sanity_module(path: str):
    spec = importlib.util.spec_from_file_location("spectralbpe_sanity_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def load_merges_json(path: str) -> List[Tuple[str, str]]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    merges = obj.get("merges")
    if merges is None:
        raise ValueError(f"No 'merges' field in {path}")
    out: List[Tuple[str, str]] = []
    for p in merges:
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError(f"Bad merge entry {p} in {path}")
        out.append((str(p[0]), str(p[1])))
    return out


def unigram_pairs(model_path: str) -> List[Tuple[str, str]]:
    try:
        import sentencepiece as spm
    except Exception as e:
        raise RuntimeError("Need sentencepiece installed: pip install sentencepiece") from e

    sp = spm.SentencePieceProcessor()
    if not sp.Load(model_path):
        raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")

    pairs: Set[Tuple[str, str]] = set()

    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        # Skip meta pieces
        if piece in ("<unk>", "<s>", "</s>", "<pad>"):
            continue
        # SentencePiece marks word boundary with ▁; remove it for char-pair extraction.
        if piece.startswith("▁"):
            piece = piece[1:]
        if not piece:
            continue
        # Skip any pieces that are just whitespace-like after stripping.
        # Keep punctuation etc.
        chars = list(piece)
        if len(chars) < 2:
            continue
        for a, b in zip(chars[:-1], chars[1:]):
            pairs.add((a, b))

    # Deterministic ordering (like merge list ordering isn't important for averaging, but nice for reproducibility)
    return sorted(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_text", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--spectral", required=True)
    ap.add_argument("--unigram_model", required=True)
    ap.add_argument("--tau", type=int, default=5)
    ap.add_argument("--pretokenize", default="basic", choices=["whitespace", "basic"])
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument(
        "--sanity_py",
        default=os.path.join(os.path.dirname(__file__), "spectralbpe_sanity_v3.py"),
        help="Path to spectralbpe_sanity_v3.py (used as ground-truth implementation)",
    )
    args = ap.parse_args()

    mod = load_sanity_module(args.sanity_py)

    # Build init_dir and init_ppmi_dir exactly as in train_spectral_bpe debug.
    lines = list(mod.iter_lines(args.train_text, None))
    word_freq = mod.build_word_freq(lines, args.pretokenize, args.lowercase)
    vocab = mod.init_vocab(word_freq)
    init_dir = mod.pair_counts(vocab)
    init_ppmi_dir, _, _ = mod.ppmi_and_weights(init_dir, args.tau, 0.0, 0.0)  # embed_alpha/beta don't matter for ppmi table

    bpe_merges = load_merges_json(args.bpe)
    sp_merges = load_merges_json(args.spectral)

    b_score, b_hit, b_miss = mod.calc_vocab_quality(bpe_merges, init_dir, init_ppmi_dir)
    s_score, s_hit, s_miss = mod.calc_vocab_quality(sp_merges, init_dir, init_ppmi_dir)

    u_pairs = unigram_pairs(args.unigram_model)
    u_score, u_hit, u_miss = mod.calc_vocab_quality(u_pairs, init_dir, init_ppmi_dir)

    print("== Vocabulary Quality (Statistical Cohesion) ==")
    print(f"Metric                       | {'BPE':>10} | {'SpectralBPE':>12} | {'Unigram':>10}")
    print("-" * 72)
    print(f"Avg PPMI of Merges (↑)       | {b_score:10.4f} | {s_score:12.4f} | {u_score:10.4f}")
    print(f"(BPE hit={b_hit}, miss={b_miss}; Spectral hit={s_hit}, miss={s_miss}; Unigram pairs hit={u_hit}, miss={u_miss}; tau={args.tau})")
    print(f"(Computed on {min(b_hit, s_hit)} common atomic pairs)")


if __name__ == "__main__":
    main()
