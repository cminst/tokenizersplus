#!/usr/bin/env python3
"""
Stratify tokens-per-word by word frequency bins for two tokenizers.
Outputs a table and (optionally) a bar chart.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

END_WORD = "</w>"


def iter_lines(path: str, max_lines: Optional[int]) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            yield line.rstrip("\n")


def pretokenize(line: str, mode: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    if mode == "whitespace":
        return line.split()
    if mode == "basic":
        return re.findall(r"\w+|[^\w\s]", line, flags=re.UNICODE)
    raise ValueError(f"Unknown --pretokenize={mode}")


def build_word_freq(lines: Iterable[str], mode: str, lowercase: bool) -> Counter:
    c = Counter()
    for line in lines:
        if lowercase:
            line = line.lower()
        for w in pretokenize(line, mode):
            if w:
                c[w] += 1
    return c


def strip_end(tokens: Sequence[str]) -> List[str]:
    out = []
    for t in tokens:
        out.append(t[:-len(END_WORD)] if t.endswith(END_WORD) else t)
    return out


def adj_pairs(tokens: Sequence[str]) -> set:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def merge_once(tokens: Sequence[str], pair: Tuple[str, str]) -> List[str]:
    a, b = pair
    ab = a + b
    out = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def encode_word(word: str, rank: Dict[Tuple[str, str], int]) -> List[str]:
    if not word:
        return []
    toks = list(word)
    toks[-1] = toks[-1] + END_WORD
    while True:
        pairs = adj_pairs(toks)
        best = None
        best_r = None
        for p in pairs:
            r = rank.get(p)
            if r is None:
                continue
            if best_r is None or r < best_r:
                best_r, best = r, p
        if best is None:
            break
        toks = merge_once(toks, best)
    return strip_end(toks)


def load_merges(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        merges = data
    elif isinstance(data, dict) and "merges" in data:
        merges = data["merges"]
    else:
        raise ValueError(f"Unrecognized merges file format: {path}")
    out: List[Tuple[str, str]] = []
    for p in merges:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"Bad merge pair in {path}: {p}")
        out.append((str(p[0]), str(p[1])))
    return out


def bin_by_rank(words: List[str], freq: Dict[str, int], bins: int) -> List[List[str]]:
    words_sorted = sorted(words, key=lambda w: (-freq.get(w, 0), w))
    n = len(words_sorted)
    if n == 0:
        return []
    out = []
    for i in range(bins):
        start = i * n // bins
        end = (i + 1) * n // bins
        if start >= end:
            continue
        out.append(words_sorted[start:end])
    return out


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_text", required=True)
    ap.add_argument("--eval_text", default=None, help="If provided, use its vocab for evaluation (default).")
    ap.add_argument("--bpe_merges", required=True)
    ap.add_argument("--spectral_merges", required=True)
    ap.add_argument("--pretokenize", choices=["whitespace", "basic"], default="whitespace")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--max_train_lines", type=int, default=None)
    ap.add_argument("--max_eval_lines", type=int, default=None)
    ap.add_argument("--freq_bins", type=int, default=10)
    ap.add_argument("--max_words", type=int, default=None, help="Limit number of words (by train rank) for speed")
    ap.add_argument("--min_train_freq", type=int, default=1, help="Filter words below this train frequency")
    ap.add_argument("--weight_by_freq", action="store_true", help="Weight averages by train frequency")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_png", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    train_lines = list(iter_lines(args.train_text, args.max_train_lines))
    train_freq = build_word_freq(train_lines, args.pretokenize, args.lowercase)
    if not train_freq:
        raise RuntimeError("No words found in training data after pretokenization.")

    if args.eval_text:
        eval_lines = list(iter_lines(args.eval_text, args.max_eval_lines))
        eval_freq = build_word_freq(eval_lines, args.pretokenize, args.lowercase)
        vocab = list(eval_freq.keys())
        vocab_source = "eval"
    else:
        eval_freq = Counter()
        vocab = list(train_freq.keys())
        vocab_source = "train"

    if args.min_train_freq > 1:
        vocab = [w for w in vocab if train_freq.get(w, 0) >= args.min_train_freq]

    if args.max_words is not None:
        vocab = sorted(vocab, key=lambda w: (-train_freq.get(w, 0), w))[: args.max_words]

    if not vocab:
        raise RuntimeError("No words left after filtering.")

    bpe_merges = load_merges(args.bpe_merges)
    sp_merges = load_merges(args.spectral_merges)
    bpe_rank = {p: i for i, p in enumerate(bpe_merges)}
    sp_rank = {p: i for i, p in enumerate(sp_merges)}

    bins = bin_by_rank(vocab, train_freq, args.freq_bins)
    if not bins:
        raise RuntimeError("No bins created; check --freq_bins or input data.")

    rows = []
    for i, words in enumerate(bins):
        if not words:
            continue
        b_sum = 0.0
        s_sum = 0.0
        w_sum = 0.0
        f_min = None
        f_max = None
        for w in words:
            f = train_freq.get(w, 0)
            if f_min is None or f < f_min:
                f_min = f
            if f_max is None or f > f_max:
                f_max = f
            weight = float(f) if args.weight_by_freq else 1.0
            if weight <= 0:
                continue
            b_sum += len(encode_word(w, bpe_rank)) * weight
            s_sum += len(encode_word(w, sp_rank)) * weight
            w_sum += weight

        b_avg = b_sum / w_sum if w_sum else 0.0
        s_avg = s_sum / w_sum if w_sum else 0.0
        lo = int(round(i * 100 / len(bins)))
        hi = int(round((i + 1) * 100 / len(bins)))
        label = f"Top {hi}%" if i == 0 else f"{lo}-{hi}%"
        rows.append(
            {
                "bin": i,
                "label": label,
                "words": len(words),
                "freq_min": int(f_min or 0),
                "freq_max": int(f_max or 0),
                "bpe_tokens_per_word": b_avg,
                "spectral_tokens_per_word": s_avg,
                "delta": s_avg - b_avg,
            }
        )

    print(f"== Tokens per word by word frequency ({vocab_source} vocab) ==")
    print(
        f"{'bin':>3s} | {'label':>9s} | {'words':>7s} | {'f_min':>6s} | {'f_max':>6s} | "
        f"{'BPE':>7s} | {'Spectral':>9s} | {'delta':>7s}"
    )
    print("-" * 72)
    for r in rows:
        print(
            f"{r['bin']:3d} | {r['label']:>9s} | {r['words']:7d} | {r['freq_min']:6d} | {r['freq_max']:6d} | "
            f"{r['bpe_tokens_per_word']:7.3f} | {r['spectral_tokens_per_word']:9.3f} | {r['delta']:7.3f}"
        )

    if args.out_csv:
        ensure_parent(args.out_csv)
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write(
                "bin,label,words,freq_min,freq_max,bpe_tokens_per_word,spectral_tokens_per_word,delta\n"
            )
            for r in rows:
                f.write(
                    f"{r['bin']},{r['label']},{r['words']},{r['freq_min']},{r['freq_max']},"
                    f"{r['bpe_tokens_per_word']:.6f},{r['spectral_tokens_per_word']:.6f},{r['delta']:.6f}\n"
                )
        print(f"[out] wrote {args.out_csv}", file=sys.stderr)

    if args.out_png:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required for --out_png (pip install matplotlib)") from e

        ensure_parent(args.out_png)
        labels = [r["label"] for r in rows]
        b_vals = [r["bpe_tokens_per_word"] for r in rows]
        s_vals = [r["spectral_tokens_per_word"] for r in rows]
        x = list(range(len(rows)))
        width = 0.38

        plt.figure(figsize=(max(6.0, len(rows) * 0.7), 4.2))
        plt.bar([i - width / 2 for i in x], b_vals, width, label="BPE")
        plt.bar([i + width / 2 for i in x], s_vals, width, label="SpectralBPE")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Tokens per word")
        title = args.title or "Tokens per Word by Word Frequency (rank bins)"
        if args.weight_by_freq:
            title += " (freq-weighted)"
        plt.title(title)
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=200)
        print(f"[out] wrote {args.out_png}", file=sys.stderr)


if __name__ == "__main__":
    main()
