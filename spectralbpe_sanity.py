#!/usr/bin/env python3
# LM-free sanity-check: compare standard frequency-BPE vs SpectralBPE on intrinsic tokenizer metrics
# (bytes/token, fertility, PCW, NSL vs BPE) + a small interpretability dump.
#
# Requirements:
#   pip install numpy scipy

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Set, Any

import numpy as np

try:
    from scipy.sparse import csr_matrix, diags, identity
    from scipy.sparse.linalg import eigsh
except Exception as e:
    raise RuntimeError("This script needs scipy (scipy.sparse + eigsh). Install: pip install scipy") from e

END_WORD = "</w>"


# ---------- I/O ----------
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


def word_to_syms(word: str) -> Tuple[str, ...]:
    if not word:
        return tuple()
    chars = list(word)
    chars[-1] = chars[-1] + END_WORD
    return tuple(chars)


def strip_end(tokens: Sequence[str]) -> List[str]:
    out = []
    for t in tokens:
        out.append(t[:-len(END_WORD)] if t.endswith(END_WORD) else t)
    return out


# ---------- BPE training ----------
def init_vocab(word_freq: Counter) -> Dict[Tuple[str, ...], int]:
    vocab = defaultdict(int)
    for w, f in word_freq.items():
        syms = word_to_syms(w)
        if syms:
            vocab[syms] += int(f)
    return dict(vocab)


def symbol_set(vocab: Dict[Tuple[str, ...], int]) -> Set[str]:
    s: Set[str] = set()
    for seq in vocab:
        s.update(seq)
    return s


def pair_counts(vocab: Dict[Tuple[str, ...], int]) -> Counter:
    c = Counter()
    for seq, freq in vocab.items():
        for i in range(len(seq) - 1):
            c[(seq[i], seq[i + 1])] += freq
    return c


def merge_seq(seq: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    a, b = pair
    ab = a + b
    out: List[str] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def apply_merge(vocab: Dict[Tuple[str, ...], int], pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
    new = defaultdict(int)
    for seq, freq in vocab.items():
        new[merge_seq(seq, pair)] += freq
    return dict(new)


def train_bpe(word_freq: Counter, vocab_size: int, max_merges: Optional[int]) -> List[Tuple[str, str]]:
    vocab = init_vocab(word_freq)
    init_syms = symbol_set(vocab)
    target = vocab_size - len(init_syms)
    if max_merges is not None:
        target = min(target, max_merges)

    merges: List[Tuple[str, str]] = []
    for it in range(max(0, target)):
        pc = pair_counts(vocab)
        if not pc:
            break
        pair, _ = pc.most_common(1)[0]
        vocab = apply_merge(vocab, pair)
        merges.append(pair)
        if (it + 1) % 200 == 0:
            print(f"[BPE] merges={it+1}/{target}", file=sys.stderr)
    return merges


# ---------- SpectralBPE training ----------
def sym_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def ppmi_and_weights(
    N_dir: Counter, tau: int, alpha: float
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    From directed adjacency counts N(i,j), compute:
      ppmi_dir[(i,j)] = max(0, log p(i,j) / (p(i)p(j)) )
      W_dir[(i,j)]    = 1{N>=tau} * ppmi_dir * N^alpha   (for merge scoring)
      W_und[{i,j}]    = W_dir(i,j) + W_dir(j,i)          (for spectral embedding)
    """
    total = float(sum(N_dir.values()))
    if total <= 0:
        return {}, {}, {}

    out_m = Counter()
    in_m = Counter()
    for (a, b), c in N_dir.items():
        out_m[a] += c
        in_m[b] += c

    ppmi_dir: Dict[Tuple[str, str], float] = {}
    W_dir: Dict[Tuple[str, str], float] = {}
    W_und: Dict[Tuple[str, str], float] = defaultdict(float)

    for (a, b), c in N_dir.items():
        p_ab = c / total
        p_a = out_m[a] / total
        p_b = in_m[b] / total
        denom = p_a * p_b
        if p_ab <= 0 or denom <= 0:
            continue

        pmi = math.log(p_ab / denom)
        ppmi = max(0.0, pmi)
        ppmi_dir[(a, b)] = ppmi

        if c >= tau and ppmi > 0.0:
            w = ppmi * (float(c) ** alpha)
            W_dir[(a, b)] = w
            W_und[sym_key(a, b)] += w

    return ppmi_dir, W_dir, dict(W_und)


def fiedler(tokens: List[str], W_und: Dict[Tuple[str, str], float], eig_k: int, eig_eps: float) -> Dict[str, float]:
    n = len(tokens)
    if n == 0:
        return {}
    if n == 1:
        return {tokens[0]: 0.0}

    idx = {t: i for i, t in enumerate(tokens)}
    rows, cols, data = [], [], []
    for (a, b), w in W_und.items():
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        if i == j:
            continue
        rows += [i, j]
        cols += [j, i]
        data += [w, w]
    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    L = identity(n, format="csr") - (diags(inv_sqrt) @ A @ diags(inv_sqrt))

    k = min(eig_k, n - 1) if n > 2 else 1
    vals, vecs = eigsh(L, k=k, which="SM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    j = 0
    while j < len(vals) and vals[j] <= eig_eps:
        j += 1
    j = min(j, vecs.shape[1] - 1)
    v = vecs[:, j]
    return {t: float(v[i]) for i, t in enumerate(tokens)}


def select_conflict_free(scored: List[Tuple[float, Tuple[str, str]]], batch_size: int, rng: random.Random) -> List[Tuple[str, str]]:
    """
    Conservative 'conflict-free' selection:
      no token type can appear in more than one selected pair within the batch.
    """
    selected: List[Tuple[str, str]] = []
    used: Set[str] = set()
    i = 0
    while i < len(scored) and len(selected) < batch_size:
        s0 = scored[i][0]
        j = i
        while j < len(scored) and scored[j][0] == s0:
            j += 1
        group = scored[i:j]
        rng.shuffle(group)
        for _, (a, b) in group:
            if len(selected) >= batch_size:
                break
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            selected.append((a, b))
        i = j
    return selected


def train_spectral_bpe(
    word_freq: Counter,
    vocab_size: int,
    tau: int,
    alpha: float,
    sigma: float,
    batch_size: int,
    eig_k: int,
    eig_eps: float,
    seed: int,
    max_merges: Optional[int],
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    rng = random.Random(seed)
    vocab = init_vocab(word_freq)
    init_syms = symbol_set(vocab)
    target = vocab_size - len(init_syms)
    if max_merges is not None:
        target = min(target, max_merges)

    # Debug: initial graph stats for interpretability
    init_dir = pair_counts(vocab)
    init_ppmi_dir, _, init_W_und = ppmi_and_weights(init_dir, tau, alpha)
    init_tokens = sorted(symbol_set(vocab))
    init_z = fiedler(init_tokens, init_W_und, eig_k=eig_k, eig_eps=eig_eps)
    debug = {"init_dir": init_dir, "init_ppmi_dir": init_ppmi_dir, "init_z": init_z}

    merges: List[Tuple[str, str]] = []
    sigma2 = float(sigma) ** 2

    outer = 0
    while len(merges) < max(0, target):
        outer += 1
        N_dir = pair_counts(vocab)
        if not N_dir:
            break

        _, W_dir, W_und = ppmi_and_weights(N_dir, tau, alpha)
        tokens = sorted(symbol_set(vocab))
        z = fiedler(tokens, W_und, eig_k=eig_k, eig_eps=eig_eps)

        scored: List[Tuple[float, Tuple[str, str]]] = []
        for (a, b), n_ab in N_dir.items():
            w = W_dir.get((a, b), 0.0)
            if w <= 0.0:
                continue
            dz2 = (z.get(a, 0.0) - z.get(b, 0.0)) ** 2
            coh = math.exp(-dz2 / sigma2) if sigma2 > 0 else 1.0
            s = float(n_ab) * w * coh
            if s > 0 and math.isfinite(s):
                scored.append((s, (a, b)))

        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        batch = select_conflict_free(scored, batch_size=batch_size, rng=rng)
        if not batch:
            break

        for pair in batch:
            if len(merges) >= target:
                break
            vocab = apply_merge(vocab, pair)
            merges.append(pair)

        if outer % 10 == 0:
            print(f"[SpectralBPE] outer={outer} merges={len(merges)}/{target}", file=sys.stderr)

    return merges, debug


# ---------- Encoding ----------
def adj_pairs(tokens: Sequence[str]) -> Set[Tuple[str, str]]:
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


# ---------- Metrics ----------
def evaluate(
    merges: List[Tuple[str, str]],
    eval_lines: Iterable[str],
    pre_mode: str,
    lowercase: bool,
    max_examples: int = 20000,
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    rank = {p: i for i, p in enumerate(merges)}
    total_bytes = 0
    total_words = 0
    total_tokens = 0
    continued = 0
    uniq: Set[str] = set()
    tok_char_sum = 0
    examples: Dict[str, List[str]] = {}

    for line in eval_lines:
        if lowercase:
            line = line.lower()
        total_bytes += len(line.encode("utf-8", errors="ignore"))
        for w in pretokenize(line, pre_mode):
            if not w:
                continue
            t = encode_word(w, rank)
            total_words += 1
            total_tokens += len(t)
            if len(t) >= 2:
                continued += 1
            for x in t:
                uniq.add(x)
                tok_char_sum += len(x)
            if len(examples) < max_examples and w not in examples:
                examples[w] = t

    fert = total_tokens / total_words if total_words else 0.0
    pcw = continued / total_words if total_words else 0.0
    bpt = total_bytes / total_tokens if total_tokens else 0.0
    tpb = total_tokens / total_bytes if total_bytes else 0.0
    atl = tok_char_sum / total_tokens if total_tokens else 0.0

    metrics = {
        "bytes": float(total_bytes),
        "words": float(total_words),
        "tokens": float(total_tokens),
        "bytes_per_token": float(bpt),
        "tokens_per_byte": float(tpb),
        "fertility": float(fert),
        "pcw": float(pcw),
        "avg_token_chars": float(atl),
        "unique_tokens_used": float(len(uniq)),
    }
    return metrics, examples


def print_table(bpe: Dict[str, float], sp: Dict[str, float]) -> None:
    nsl = (sp["tokens"] / bpe["tokens"]) if bpe["tokens"] else float("nan")
    print("\n== Intrinsic sanity-check metrics ==")
    print(f"{'Metric':38s} | {'BPE':>12s} | {'SpectralBPE':>12s}")
    print("-" * 70)

    def row(name, key, fmt="{:.6f}"):
        print(f"{name:38s} | {fmt.format(bpe[key]):>12s} | {fmt.format(sp[key]):>12s}")

    row("Bytes per token (↑)", "bytes_per_token")
    row("Tokens per byte (↓)", "tokens_per_byte")
    row("Fertility (tokens/word) (↓)", "fertility")
    row("PCW = P(word split) (↓)", "pcw")
    row("Avg token length (chars) (↑)", "avg_token_chars")
    row("Unique tokens used on eval", "unique_tokens_used", fmt="{:.0f}")
    print(f"{'NSL vs BPE (tokens_sp/tokens_bpe) (↓)':38s} | {1.0:12.6f} | {nsl:12.6f}")
    print("-" * 70)
    print(
        f"Totals: bytes={int(bpe['bytes'])} words={int(bpe['words'])} "
        f"BPE_tokens={int(bpe['tokens'])} Spectral_tokens={int(sp['tokens'])}"
    )


# ---------- Interpretability ----------
def top_diff_words(ex_b: Dict[str, List[str]], ex_s: Dict[str, List[str]], k: int = 12):
    diffs = []
    for w, tb in ex_b.items():
        ts = ex_s.get(w)
        if ts is None:
            continue
        if tb != ts:
            diffs.append((abs(len(tb) - len(ts)), w, tb, ts))
    diffs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return diffs[:k]


def interpretability(
    bpe_merges: List[Tuple[str, str]],
    sp_merges: List[Tuple[str, str]],
    ex_b: Dict[str, List[str]],
    ex_s: Dict[str, List[str]],
    debug: Dict[str, Any],
    sigma: float,
    top_merges: int = 20,
) -> None:
    print("\n== Interpretability ==")
    print(f"\nTop {top_merges} merges:")
    print(f"{'r':>3s} | {'BPE':24s} | {'SpectralBPE':24s}")
    print("-" * 60)
    for i in range(top_merges):
        bm = bpe_merges[i] if i < len(bpe_merges) else None
        sm = sp_merges[i] if i < len(sp_merges) else None
        print(f"{i:3d} | {str(bm):24s} | {str(sm):24s}")

    print("\nWords with largest |Δtokens| in stored examples:")
    diffs = top_diff_words(ex_b, ex_s, k=12)
    if not diffs:
        print("  (No differences found in stored examples.)")
    for gap, w, tb, ts in diffs:
        print(f"  - {w!r} (|Δ|={gap}):")
        print(f"      BPE        : {tb}")
        print(f"      SpectralBPE: {ts}")

    init_dir: Counter = debug.get("init_dir", Counter())
    init_ppmi: Dict[Tuple[str, str], float] = debug.get("init_ppmi_dir", {})
    init_z: Dict[str, float] = debug.get("init_z", {})
    bpe_rank = {p: i for i, p in enumerate(bpe_merges)}
    sp_rank = {p: i for i, p in enumerate(sp_merges)}
    sigma2 = sigma * sigma

    cand = []
    for (a, b), c in init_dir.most_common(500):
        pp = init_ppmi.get((a, b), 0.0)
        coh = (
            math.exp(-((init_z.get(a, 0.0) - init_z.get(b, 0.0)) ** 2) / sigma2)
            if sigma2 > 0
            else 1.0
        )
        cand.append((c, pp, coh, (a, b)))
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))

    print("\nHigh-frequency 'bridge-like' initial pairs (high count, low PPMI, low coherence):")
    print(f"{'pair':24s} | {'cnt':>6s} | {'PPMI':>7s} | {'coh':>6s} | {'BPE_r':>6s} | {'Spec_r':>6s}")
    print("-" * 70)
    shown = 0
    for c, pp, coh, p in cand:
        if shown >= 15:
            break
        if pp > 0.25 and coh > 0.6:
            continue
        print(
            f"{str(p):24s} | {c:6d} | {pp:7.3f} | {coh:6.3f} | "
            f"{str(bpe_rank.get(p,'-')):>6s} | {str(sp_rank.get(p,'-')):>6s}"
        )
        shown += 1

    def merge_stats(merges):
        ppv, chv = [], []
        for (a, b) in merges[:200]:
            if (a, b) not in init_dir:
                continue
            ppv.append(init_ppmi.get((a, b), 0.0))
            chv.append(
                math.exp(-((init_z.get(a, 0.0) - init_z.get(b, 0.0)) ** 2) / sigma2)
                if sigma2 > 0
                else 1.0
            )
        if not ppv:
            return None
        return float(np.mean(ppv)), float(np.mean(chv)), float(np.median(ppv)), float(np.median(chv))

    bs = merge_stats(bpe_merges)
    ss = merge_stats(sp_merges)
    print("\nEarly-merge association/coherence (computed on initial graph, over merges present at char-level):")
    print("  format: mean_ppmi, mean_coh, median_ppmi, median_coh")
    print(f"  BPE        : {bs}")
    print(f"  SpectralBPE: {ss}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_text", required=True)
    ap.add_argument("--eval_text", required=True)
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--pretokenize", choices=["whitespace", "basic"], default="whitespace")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--max_train_lines", type=int, default=None)
    ap.add_argument("--max_eval_lines", type=int, default=None)
    ap.add_argument("--max_merges", type=int, default=None)

    # SpectralBPE params
    ap.add_argument("--tau", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--eig_k", type=int, default=8)
    ap.add_argument("--eig_eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    train_lines = list(iter_lines(args.train_text, args.max_train_lines))
    eval_lines = list(iter_lines(args.eval_text, args.max_eval_lines))
    wf = build_word_freq(train_lines, args.pretokenize, args.lowercase)
    if not wf:
        raise RuntimeError("No words found in training data after pretokenization.")

    print("[train] BPE...", file=sys.stderr)
    bpe_merges = train_bpe(wf, args.vocab_size, args.max_merges)

    print("[train] SpectralBPE...", file=sys.stderr)
    sp_merges, debug = train_spectral_bpe(
        wf,
        args.vocab_size,
        tau=args.tau,
        alpha=args.alpha,
        sigma=args.sigma,
        batch_size=args.batch_size,
        eig_k=args.eig_k,
        eig_eps=args.eig_eps,
        seed=args.seed,
        max_merges=args.max_merges,
    )

    bpe_metrics, ex_b = evaluate(bpe_merges, eval_lines, args.pretokenize, args.lowercase)
    sp_metrics, ex_s = evaluate(sp_merges, eval_lines, args.pretokenize, args.lowercase)

    print_table(bpe_metrics, sp_metrics)
    interpretability(bpe_merges, sp_merges, ex_b, ex_s, debug, sigma=args.sigma)

    if args.out_json:
        payload = {
            "config": vars(args),
            "metrics": {"bpe": bpe_metrics, "spectralbpe": sp_metrics},
            "num_merges": {"bpe": len(bpe_merges), "spectralbpe": len(sp_merges)},
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[out] wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
