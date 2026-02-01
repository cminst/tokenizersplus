#!/usr/bin/env python3
# Sanity-check: compare standard frequency-BPE vs SpectralBPE on intrinsic tokenizer metrics
# (bytes/token, fertility, PCW, NSL vs BPE) + a small interpretability dump.
# Optional: train a tiny LM for a generalization test (requires torch).
#
# Requirements:
#   pip install numpy scipy
#   pip install torch  # only needed for --train_lm

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
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
    N_dir: Counter, tau: int, alpha: float, beta: float = 0.0
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    From directed adjacency counts N(i,j), compute:
      ppmi_dir[(i,j)] = max(0, log p(i,j) / (p(i)p(j)) )
      W_dir[(i,j)]    = 1{N>=tau} * (ppmi_dir + beta) * N^alpha   (for merge scoring)
      W_und[{i,j}]    = W_dir(i,j) + W_dir(j,i)          (for spectral embedding)

    beta: baseline added to PPMI to prevent hard gating (beta=0 recovers original behavior)
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

        if c >= tau:
            ppmi_eff = ppmi + beta
            w = ppmi_eff * (float(c) ** alpha)
            W_dir[(a, b)] = w
            W_und[sym_key(a, b)] += w

    return ppmi_dir, W_dir, dict(W_und)


def fiedler(tokens: List[str], W_und: Dict[Tuple[str, str], float], eig_k: int, eig_eps: float) -> Dict[str, float]:
    n = len(tokens)
    if n <= 1:
        return {t: 0.0 for t in tokens}

    # 1. Build Adjacency Matrix
    idx = {t: i for i, t in enumerate(tokens)}
    rows, cols, data = [], [], []
    for (a, b), w in W_und.items():
        if a in idx and b in idx and a != b:
            i, j = idx[a], idx[b]
            rows.append(i); cols.append(j); data.append(w)
            rows.append(j); cols.append(i); data.append(w)

    if not data:
        return {t: 0.0 for t in tokens}

    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    # 2. Extract Largest Connected Component (LCC)
    # Spectral geometry is only defined within a component.
    # Distances between components are infinite/undefined.
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)

    # If fragmented, only solve for the largest chunk
    counts = Counter(labels)
    largest_cc_label = counts.most_common(1)[0][0]
    lcc_indices = np.where(labels == largest_cc_label)[0]

    if len(lcc_indices) < 2:
        return {t: 0.0 for t in tokens}

    # Create sub-matrix for LCC
    A_lcc = A[lcc_indices][:, lcc_indices]
    n_lcc = A_lcc.shape[0]

    # 3. Compute Laplacian for LCC
    deg = np.asarray(A_lcc.sum(axis=1)).ravel()
    # Add epsilon to degree to prevent div/0 in degenerate cases
    deg[deg == 0] = 1e-10
    inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = diags(inv_sqrt)
    L = identity(n_lcc, format="csr") - (D_inv_sqrt @ A_lcc @ D_inv_sqrt)

    # 4. Robust Eigensolve
    # We only need the first non-trivial vector.
    # Since we isolated LCC, the only 0 eigenvalue is the first one.
    # We want the second one (Fiedler).
    k_target = min(eig_k, n_lcc - 1) if n_lcc > 2 else 1

    try:
        # 'SA' (Smallest Algebraic) is often more stable than 'SM' for Laplacians
        # We ask for k+1 to safely skip the null vector
        vals, vecs = eigsh(L, k=k_target+1, which="SA", tol=eig_eps)

        # Sort and pick the first vector that is definitely non-zero
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]

        # Skip eigenvalues near zero (tolerance 1e-5 usually safe for normalized laplacian)
        sel = 0
        while sel < len(vals) and vals[sel] < 1e-5:
            sel += 1

        if sel < len(vals):
            v_lcc = vecs[:, sel]
        else:
            # Fallback if solver returns all zeros
            v_lcc = np.zeros(n_lcc)

    except Exception as e:
        print(f"[WARNING] Solver failed on LCC (size {n_lcc}): {e}", file=sys.stderr)
        v_lcc = np.zeros(n_lcc)

    # 5. Map back to tokens
    # Tokens outside LCC get 0.0 embedding (neutral)
    z_map = {t: 0.0 for t in tokens}
    for local_idx, global_idx in enumerate(lcc_indices):
        z_map[tokens[global_idx]] = float(v_lcc[local_idx])

    return z_map


def select_conflict_free(scored: List[Tuple[float, Tuple[str, str]]], batch_size: int, rng: random.Random, deterministic: bool = False) -> List[Tuple[str, str]]:
    """
    Conservative 'conflict-free' selection:
      no token type can appear in more than one selected pair within the batch.

    deterministic: if True, skip the random shuffle of equal-score ties
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
        if not deterministic:
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
    beta: float = 0.0,
    sigma_auto: bool = False,
    deterministic_ties: bool = False,
    bpe_warmstart: int = 0,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    rng = random.Random(seed)
    vocab = init_vocab(word_freq)
    init_syms = symbol_set(vocab)
    target = vocab_size - len(init_syms)
    if max_merges is not None:
        target = min(target, max_merges)

    # Debug: initial graph stats for interpretability
    init_dir = pair_counts(vocab)
    init_ppmi_dir, _, init_W_und = ppmi_and_weights(init_dir, tau, alpha, beta)
    init_tokens = sorted(symbol_set(vocab))
    init_z = fiedler(init_tokens, init_W_und, eig_k=eig_k, eig_eps=eig_eps)
    debug = {"init_dir": init_dir, "init_ppmi_dir": init_ppmi_dir, "init_z": init_z}

    merges: List[Tuple[str, str]] = []

    # BPE warm-start: run plain BPE for first N merges
    if bpe_warmstart > 0:
        print(f"[SpectralBPE] warm-start: running {bpe_warmstart} BPE merges first", file=sys.stderr)
        warmstart_target = min(bpe_warmstart, target)
        while len(merges) < warmstart_target:
            N_dir = pair_counts(vocab)
            if not N_dir:
                break
            best_pair = max(N_dir.items(), key=lambda x: x[1])[0]
            vocab = apply_merge(vocab, best_pair)
            merges.append(best_pair)
        print(f"[SpectralBPE] warm-start complete: {len(merges)} merges done", file=sys.stderr)

    # Auto-calibrate sigma if requested
    if sigma_auto:
        N_dir = pair_counts(vocab)
        _, _, W_und = ppmi_and_weights(N_dir, tau, alpha, beta)
        tokens = sorted(symbol_set(vocab))
        z = fiedler(tokens, W_und, eig_k=eig_k, eig_eps=eig_eps)
        dz_vals = []
        for (a, b) in N_dir.keys():
            if a in z and b in z:
                dz_vals.append(abs(z[a] - z[b]))
        if dz_vals:
            sigma = float(np.percentile(dz_vals, 75))
            print(f"[SpectralBPE] auto-calibrated sigma = {sigma:.4f} (p75 of |dz|)", file=sys.stderr)
        else:
            sigma = 1.0
            print("[SpectralBPE] auto-calibration failed, using sigma = 1.0", file=sys.stderr)
        debug["auto_sigma"] = sigma

    sigma2 = float(sigma) ** 2

    outer = 0
    coherence_stats = []
    while len(merges) < max(0, target):
        outer += 1
        N_dir = pair_counts(vocab)
        if not N_dir:
            break

        _, W_dir, W_und = ppmi_and_weights(N_dir, tau, alpha, beta)
        tokens = sorted(symbol_set(vocab))
        z = fiedler(tokens, W_und, eig_k=eig_k, eig_eps=eig_eps)

        scored: List[Tuple[float, Tuple[str, str]]] = []
        coherences = []
        for (a, b), n_ab in N_dir.items():
            w = W_dir.get((a, b), 0.0)
            if w <= 0.0:
                continue
            dz2 = (z.get(a, 0.0) - z.get(b, 0.0)) ** 2
            coh = math.exp(-dz2 / sigma2) if sigma2 > 0 else 1.0
            coherences.append(coh)
            s = float(n_ab) * w * coh
            if s > 0 and math.isfinite(s):
                scored.append((s, (a, b)))

        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        batch = select_conflict_free(scored, batch_size=batch_size, rng=rng, deterministic=deterministic_ties)
        if not batch:
            break

        for pair in batch:
            if len(merges) >= target:
                break
            vocab = apply_merge(vocab, pair)
            merges.append(pair)

        # Track coherence statistics
        if coherences:
            coherence_stats.append({
                "outer": outer,
                "median": float(np.median(coherences)),
                "p10": float(np.percentile(coherences, 10)),
                "p90": float(np.percentile(coherences, 90)),
            })

        if outer % 10 == 0:
            if coherences:
                coh_median = np.median(coherences)
                coh_p10 = np.percentile(coherences, 10)
                coh_p90 = np.percentile(coherences, 90)
                print(f"[SpectralBPE] outer={outer} merges={len(merges)}/{target} coherence: median={coh_median:.4f} p10={coh_p10:.4f} p90={coh_p90:.4f}", file=sys.stderr)
            else:
                print(f"[SpectralBPE] outer={outer} merges={len(merges)}/{target}", file=sys.stderr)

    debug["coherence_stats"] = coherence_stats

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


def encode_word(word: str, rank: Dict[Tuple[str, str], int], keep_end: bool = False) -> List[str]:
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
    return toks if keep_end else strip_end(toks)


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


# ---------- LM Generalization (optional) ----------
def tokens_from_lines(
    lines: Sequence[str],
    merges: List[Tuple[str, str]],
    pre_mode: str,
    lowercase: bool,
) -> Tuple[List[str], List[int]]:
    rank = {p: i for i, p in enumerate(merges)}
    tokens: List[str] = []
    lengths: List[int] = []
    for line in lines:
        if lowercase:
            line = line.lower()
        line_tokens: List[str] = []
        for w in pretokenize(line, pre_mode):
            if not w:
                continue
            line_tokens.extend(encode_word(w, rank))
        if line_tokens:
            lengths.append(len(line_tokens))
            tokens.extend(line_tokens)
    return tokens, lengths


def build_vocab(tokens: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    uniq = [t for t in sorted(set(tokens)) if t != "<unk>"]
    vocab = ["<unk>"] + uniq
    stoi = {t: i for i, t in enumerate(vocab)}
    return vocab, stoi


def iter_lm_batches(
    ids_tensor,
    block_size: int,
    batch_size: int,
    rng: Optional[random.Random],
):
    n = int(ids_tensor.shape[0])
    num_blocks = (n - 1) // block_size
    if num_blocks <= 0:
        return
    starts = list(range(0, num_blocks * block_size, block_size))
    if rng is not None:
        rng.shuffle(starts)
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i:i + batch_size]
        x = np.stack([ids_tensor[s:s + block_size] for s in batch_starts], axis=0)
        y = np.stack([ids_tensor[s + 1:s + block_size + 1] for s in batch_starts], axis=0)
        yield x, y


def train_and_eval_lm(
    train_lines: Sequence[str],
    eval_lines: Sequence[str],
    merges: List[Tuple[str, str]],
    pre_mode: str,
    lowercase: bool,
    eval_bytes: int,
    epochs: int,
    batch_size: int,
    block_size: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    lr: float,
    seed: int,
) -> Tuple[float, float, float]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError("Torch is required for --train_lm. Install: pip install torch") from e

    train_tokens, _ = tokens_from_lines(train_lines, merges, pre_mode, lowercase)
    eval_tokens, eval_lengths = tokens_from_lines(eval_lines, merges, pre_mode, lowercase)
    avg_seq_len = (sum(eval_lengths) / len(eval_lengths)) if eval_lengths else 0.0

    if len(train_tokens) < block_size + 1:
        raise RuntimeError("Not enough training tokens for the requested --lm_block_size.")

    vocab, stoi = build_vocab(train_tokens)
    unk_id = stoi["<unk>"]
    train_ids = np.array([stoi.get(t, unk_id) for t in train_tokens], dtype=np.int64)
    eval_ids = np.array([stoi.get(t, unk_id) for t in eval_tokens], dtype=np.int64)

    device = torch.device("cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    class MiniTransformerLM(nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, n_embd)
            self.pos_emb = nn.Embedding(block_size, n_embd)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
            self.ln = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size, bias=False)

        def forward(self, idx):
            bsz, t = idx.shape
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
            x = self.token_emb(idx) + self.pos_emb(pos)
            mask = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()
            x = self.encoder(x, mask)
            x = self.ln(x)
            return self.head(x)

    model = MiniTransformerLM(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        for x_np, y_np in iter_lm_batches(train_ids, block_size, batch_size, rng):
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0

    model.eval()
    eval_block_size = min(block_size, len(eval_ids) - 1) if len(eval_ids) > 1 else 0
    if eval_block_size <= 0:
        raise RuntimeError("Not enough evaluation tokens for LM evaluation.")

    total_loss = 0.0
    with torch.no_grad():
        for x_np, y_np in iter_lm_batches(eval_ids, eval_block_size, batch_size, None):
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_loss += float(loss.item())

    if eval_bytes <= 0:
        bpb = float("nan")
    else:
        bpb = (total_loss / float(eval_bytes)) / math.log(2.0)

    return bpb, avg_seq_len, train_time


def print_lm_table(bpe_res: Tuple[float, float, float], sp_res: Tuple[float, float, float]) -> None:
    b_bpb, b_seq, b_time = bpe_res
    s_bpb, s_seq, s_time = sp_res
    print("\n== LM generalization (BPB) ==")
    print(f"{'Metric':24s} | {'BPE':>12s} | {'SpectralBPE':>12s}")
    print("-" * 56)
    print(f"{'BPB (eval)':24s} | {b_bpb:12.6f} | {s_bpb:12.6f}")
    print(f"{'Avg tokens/sent':24s} | {b_seq:12.2f} | {s_seq:12.2f}")
    print(f"{'Training time (s)':24s} | {b_time:12.2f} | {s_time:12.2f}")


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
    ap.add_argument("--tau", type=int, default=2, help="Minimum pair count threshold (default: 2, was 5)")
    ap.add_argument("--alpha", type=float, default=0.25, help="Frequency exponent (default: 0.25, was 0.5)")
    ap.add_argument("--sigma", type=float, default=1.0, help="Coherence bandwidth (ignored if --sigma_auto)")
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--eig_k", type=int, default=8)
    ap.add_argument("--eig_eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)

    # New params for improvements
    ap.add_argument("--beta", type=float, default=0.05, help="PPMI baseline to avoid hard gating (default: 0.05)")
    ap.add_argument("--sigma_auto", action="store_true", help="Auto-calibrate sigma from spectral coordinate differences")
    ap.add_argument("--deterministic_ties", action="store_true", help="Disable random shuffle of equal-score ties")
    ap.add_argument("--bpe_warmstart", type=int, default=0, help="Number of plain BPE merges before spectral guidance")
    ap.add_argument("--num_seeds", type=int, default=1, help="Run with multiple seeds and report stats")

    # Optional LM generalization test
    ap.add_argument("--train_lm", action="store_true", help="Train a tiny LM and report BPB on eval.txt")
    ap.add_argument("--lm_epochs", type=int, default=2)
    ap.add_argument("--lm_batch_size", type=int, default=16)
    ap.add_argument("--lm_block_size", type=int, default=128)
    ap.add_argument("--lm_dim", type=int, default=128)
    ap.add_argument("--lm_heads", type=int, default=4)
    ap.add_argument("--lm_layers", type=int, default=2)
    ap.add_argument("--lm_lr", type=float, default=3e-4)

    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    train_lines = list(iter_lines(args.train_text, args.max_train_lines))
    eval_lines = list(iter_lines(args.eval_text, args.max_eval_lines))
    wf = build_word_freq(train_lines, args.pretokenize, args.lowercase)
    if not wf:
        raise RuntimeError("No words found in training data after pretokenization.")

    print("[train] BPE...", file=sys.stderr)
    bpe_merges = train_bpe(wf, args.vocab_size, args.max_merges)

    # Multi-seed evaluation if requested
    if args.num_seeds > 1:
        print(f"[train] SpectralBPE with {args.num_seeds} seeds...", file=sys.stderr)
        all_sp_metrics = []
        for seed_i in range(args.num_seeds):
            current_seed = args.seed + seed_i
            print(f"[train] SpectralBPE seed {seed_i+1}/{args.num_seeds} (seed={current_seed})...", file=sys.stderr)
            sp_merges_i, debug_i = train_spectral_bpe(
                wf,
                args.vocab_size,
                tau=args.tau,
                alpha=args.alpha,
                sigma=args.sigma,
                batch_size=args.batch_size,
                eig_k=args.eig_k,
                eig_eps=args.eig_eps,
                seed=current_seed,
                max_merges=args.max_merges,
                beta=args.beta,
                sigma_auto=args.sigma_auto,
                deterministic_ties=args.deterministic_ties,
                bpe_warmstart=args.bpe_warmstart,
            )
            sp_metrics_i, ex_s_i = evaluate(sp_merges_i, eval_lines, args.pretokenize, args.lowercase)
            all_sp_metrics.append(sp_metrics_i)
            if seed_i == 0:
                sp_merges, debug, ex_s = sp_merges_i, debug_i, ex_s_i

        # Compute mean and std across seeds
        keys = ["nsl", "bytes_per_token", "tokens_per_byte", "fertility", "pcw"]
        sp_metrics_mean = {}
        sp_metrics_std = {}
        for key in keys:
            values = [m[key] for m in all_sp_metrics]
            sp_metrics_mean[key] = float(np.mean(values))
            sp_metrics_std[key] = float(np.std(values))

        print("\n=== Multi-seed Results ===", file=sys.stderr)
        print(f"Seeds: {args.num_seeds}", file=sys.stderr)
        for key in keys:
            print(f"  {key}: {sp_metrics_mean[key]:.6f} ± {sp_metrics_std[key]:.6f}", file=sys.stderr)

        sp_metrics = all_sp_metrics[0]  # Use first seed for comparison table
    else:
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
            beta=args.beta,
            sigma_auto=args.sigma_auto,
            deterministic_ties=args.deterministic_ties,
            bpe_warmstart=args.bpe_warmstart,
        )
        sp_metrics, ex_s = evaluate(sp_merges, eval_lines, args.pretokenize, args.lowercase)

    bpe_metrics, ex_b = evaluate(bpe_merges, eval_lines, args.pretokenize, args.lowercase)

    print_table(bpe_metrics, sp_metrics)
    interpretability(bpe_merges, sp_merges, ex_b, ex_s, debug, sigma=args.sigma if not args.sigma_auto else debug.get("auto_sigma", args.sigma))

    lm_results = None
    if args.train_lm:
        if args.max_eval_lines is None:
            try:
                eval_bytes = len(open(args.eval_text, "rb").read())
            except Exception:
                eval_bytes = sum(len((line + "\n").encode("utf-8", errors="ignore")) for line in eval_lines)
        else:
            eval_bytes = sum(len((line + "\n").encode("utf-8", errors="ignore")) for line in eval_lines)

        print("\n[train] LM on BPE tokens...", file=sys.stderr)
        bpe_lm = train_and_eval_lm(
            train_lines,
            eval_lines,
            bpe_merges,
            args.pretokenize,
            args.lowercase,
            eval_bytes=eval_bytes,
            epochs=args.lm_epochs,
            batch_size=args.lm_batch_size,
            block_size=args.lm_block_size,
            n_embd=args.lm_dim,
            n_head=args.lm_heads,
            n_layer=args.lm_layers,
            lr=args.lm_lr,
            seed=args.seed,
        )

        print("[train] LM on SpectralBPE tokens...", file=sys.stderr)
        sp_lm = train_and_eval_lm(
            train_lines,
            eval_lines,
            sp_merges,
            args.pretokenize,
            args.lowercase,
            eval_bytes=eval_bytes,
            epochs=args.lm_epochs,
            batch_size=args.lm_batch_size,
            block_size=args.lm_block_size,
            n_embd=args.lm_dim,
            n_head=args.lm_heads,
            n_layer=args.lm_layers,
            lr=args.lm_lr,
            seed=args.seed,
        )

        print_lm_table(bpe_lm, sp_lm)
        lm_results = {"bpe": bpe_lm, "spectralbpe": sp_lm, "eval_bytes": eval_bytes}

    if args.out_json:
        payload = {
            "config": vars(args),
            "metrics": {"bpe": bpe_metrics, "spectralbpe": sp_metrics},
            "num_merges": {"bpe": len(bpe_merges), "spectralbpe": len(sp_merges)},
        }
        if lm_results is not None:
            payload["lm"] = {
                "bpe_bpb": lm_results["bpe"][0],
                "bpe_avg_seq_len": lm_results["bpe"][1],
                "bpe_train_time_sec": lm_results["bpe"][2],
                "spectral_bpb": lm_results["spectralbpe"][0],
                "spectral_avg_seq_len": lm_results["spectralbpe"][1],
                "spectral_train_time_sec": lm_results["spectralbpe"][2],
                "eval_bytes": lm_results["eval_bytes"],
            }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[out] wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
