#!/usr/bin/env python3
"""
SpectralBPE minimal experiment (toy "proof/intuition" + optional micro-LM).

What you get (toy stage, fully self-contained):
  - A synthetic corpus where each word = STEM+SUFFIX, with a known gold boundary.
  - Train 4 tokenizers (batched, conflict-free):
      1) freq-BPE
      2) PPMI-BPE (association-only)
      3) SpectralBPE (PPMI + spectral coherence)
      4) SpectralBPE-shuffled (same, but the spectral embedding is permuted as a control)
  - Plots:
      - boundary-retention curve vs merge steps
      - boundary-retention vs avg tokens/word (matched compression view)
      - cross-boundary merge fraction vs merge steps
      - spectral distance of selected merges (mechanism plot)

Optional real-text stage (requires torch; optionally huggingface `datasets`):
  - Train the same 4 tokenizers on a small real corpus (WikiText-2 by default if `datasets` installed,
    otherwise you can pass --real_text_path).
  - Run a tiny LM for a small number of steps and report bits-per-byte (normalizes tokenizer length effects).
  - Run a quick English affix boundary probe (heuristic).

Designed so you can run the toy stage in seconds and (optionally) keep the real LM stage under ~30 minutes
on a fast GPU by choosing small steps/model sizes.

Example:
  python run_spectralbpe_miniexp.py --out_dir out --toy_vocab_size 2000 --toy_batch_size 50

Real LM (GPU):
  pip install torch --upgrade
  pip install datasets  # optional
  python run_spectralbpe_miniexp.py --out_dir out --run_real_lm --device cuda --lm_steps 1500

If you don't have `datasets`, use a local text file:
  python run_spectralbpe_miniexp.py --out_dir out --run_real_lm --real_text_path ./my_corpus.txt

Notes:
- The BPE training here is a *batched conflict-free* variant for speed.
  For the minimal experiment, that's fine: the point is comparing merge *scoring rules* and showing
  the spectral mechanism.
- The synthetic corpus uses disjoint alphabets for stems and suffixes so "cross-boundary merges" are
  easy to detect (a merged token that contains characters from both alphabets).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np

# --- SciPy for spectral embedding (fallback to dense if eigsh fails) ---
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh

# --- plotting ---
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ==========================
# Utilities
# ==========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def pretty_table(rows: List[List[Any]], headers: List[str]) -> str:
    cols = len(headers)
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))
    sep = " | "
    out = []
    out.append(sep.join(str(headers[i]).ljust(widths[i]) for i in range(cols)))
    out.append(sep.join("-"*widths[i] for i in range(cols)))
    for r in rows:
        out.append(sep.join(str(r[i]).ljust(widths[i]) for i in range(cols)))
    return "\n".join(out)


# ==========================
# Synthetic corpus generator
# ==========================

def generate_synthetic_corpus(
    n_word_types: int,
    seed: int,
    stem_len_range: Tuple[int, int],
    suffix_len_range: Tuple[int, int],
    stem_chars: str,
    suffix_chars: str,
    end_chars: str,
    start_chars: str,
    zipf_a: float,
    max_count: int,
    val_frac: float,
) -> Tuple[Dict[str, int], List[Tuple[str, int, int]], Dict[str, Set[str]]]:
    """
    Synthetic word types of the form stem + suffix with a known boundary.
    Disjoint alphabets => cross-boundary merges are detectable.
    A small mapping end_char -> start_char encourages boundary bigrams with high PMI/PPMI.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    stem_set = set(stem_chars)
    suffix_set = set(suffix_chars)
    end_list = list(end_chars)
    start_list = list(start_chars)
    assert len(end_list) == len(start_list), "end_chars and start_chars must have same length"
    mapping = {end_list[i]: start_list[i] for i in range(len(end_list))}

    samples: List[Tuple[str, int, int]] = []
    for _ in range(n_word_types):
        stem_len = rng.randint(*stem_len_range)
        suf_len = rng.randint(*suffix_len_range)

        end_ch = rng.choice(end_list)

        stem_body_choices = [c for c in stem_chars if c not in end_list] or list(stem_chars)
        stem_body = "".join(rng.choice(stem_body_choices) for _ in range(stem_len - 1))
        stem = stem_body + end_ch

        start_ch = mapping[end_ch]
        suffix_body_choices = [c for c in suffix_chars if c not in start_list] or list(suffix_chars)
        suffix_body = "".join(rng.choice(suffix_body_choices) for _ in range(suf_len - 1))
        suffix = start_ch + suffix_body

        word = stem + suffix
        boundary = len(stem)

        c = int(min(max_count, max(1, np_rng.zipf(zipf_a))))
        samples.append((word, boundary, c))

    rng.shuffle(samples)
    split = int(len(samples) * (1.0 - val_frac))
    train = samples[:split]
    val = samples[split:]

    train_word_counts = collections.Counter()
    for w, _, c in train:
        train_word_counts[w] += c

    alph = {"stem_set": stem_set, "suffix_set": suffix_set}
    return dict(train_word_counts), val, alph


# ==========================
# BPE training + logging
# ==========================

def compute_bigram_counts(
    word_tokens: Dict[str, List[str]],
    word_counts: Dict[str, int],
) -> Tuple[collections.Counter, collections.Counter, collections.Counter, int]:
    pair_counts = collections.Counter()
    first_counts = collections.Counter()
    second_counts = collections.Counter()
    total_pairs = 0

    for w, toks in word_tokens.items():
        cnt = word_counts.get(w, 0)
        if cnt <= 0 or len(toks) < 2:
            continue
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            pair_counts[(a, b)] += cnt
            first_counts[a] += cnt
            second_counts[b] += cnt
            total_pairs += cnt

    return pair_counts, first_counts, second_counts, total_pairs

def compute_ppmi_weights(
    pair_counts: Dict[Tuple[str, str], int],
    first_counts: Dict[str, int],
    second_counts: Dict[str, int],
    total_pairs: int,
    tau: int,
    alpha: float,
    eps: float = 1e-12,
) -> Dict[Tuple[str, str], float]:
    """
    W_ij = 1{N>=tau} * PPMI(i,j) * N(i,j)^alpha.
    Here PMI uses directed bigram marginals:
      p(i) = sum_y N(i,y)/Z , p(j)=sum_x N(x,j)/Z, p(i,j)=N(i,j)/Z
    """
    weights: Dict[Tuple[str, str], float] = {}
    if total_pairs <= 0:
        return weights

    inv_total = 1.0 / total_pairs
    for (a, b), nij in pair_counts.items():
        if nij < tau:
            continue
        pij = nij * inv_total
        pi = first_counts[a] * inv_total
        pj = second_counts[b] * inv_total
        pmi = math.log((pij + eps) / (pi * pj + eps))
        if pmi <= 0:
            continue
        weights[(a, b)] = float(pmi) * float(nij ** alpha)
    return weights

def compute_fiedler_embedding(
    vocab: List[str],
    weights: Dict[Tuple[str, str], float],
    symmetrize: bool = True,
    eigen_tol: float = 1e-4,
    max_iter: int = 2000,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    1D Fiedler embedding = eigenvector of the smallest positive eigenvalue of the normalized Laplacian.
    """
    n = len(vocab)
    if n <= 2:
        return np.zeros(n, dtype=np.float32)

    idx = {t: i for i, t in enumerate(vocab)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for (a, b), w in weights.items():
        if w <= 0:
            continue
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None:
            continue
        rows.append(ia)
        cols.append(ib)
        data.append(float(w))
        if symmetrize:
            rows.append(ib)
            cols.append(ia)
            data.append(float(w))

    if not data:
        return np.zeros(n, dtype=np.float32)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W = (W + W.T) * 0.5
    d = np.asarray(W.sum(axis=1)).reshape(-1)
    d_safe = np.where(d > eps, d, eps)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    D_inv_sqrt = diags(d_inv_sqrt)
    L = identity(n, format="csr") - (D_inv_sqrt @ W @ D_inv_sqrt)

    k = min(3, n - 1)  # ask for a few smallest
    try:
        evals, evecs = eigsh(L, k=k, which="SM", tol=eigen_tol, maxiter=max_iter)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        chosen = None
        for i in range(len(evals)):
            if evals[i] > 1e-7:
                chosen = i
                break
        if chosen is None:
            chosen = min(1, evecs.shape[1] - 1)
        z = evecs[:, chosen]
    except Exception:
        # dense fallback
        Ld = L.toarray()
        evals, evecs = np.linalg.eigh(Ld)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        chosen = None
        for i in range(len(evals)):
            if evals[i] > 1e-7:
                chosen = i
                break
        if chosen is None:
            chosen = 1 if n > 1 else 0
        z = evecs[:, chosen]

    z = (z - z.mean()) / (z.std() + 1e-9)
    return z.astype(np.float32)

def token_domain(token: str, stem_set: Set[str], suffix_set: Set[str]) -> int:
    """
    0 = other, 1 = stem-only, 2 = suffix-only, 3 = cross (both).
    """
    has_stem = any(ch in stem_set for ch in token)
    has_suf = any(ch in suffix_set for ch in token)
    if has_stem and has_suf:
        return 3
    if has_stem:
        return 1
    if has_suf:
        return 2
    return 0

@dataclass
class MergeLog:
    step: int
    a: str
    b: str
    new_token: str
    score: float
    batch_id: int
    is_cross_boundary: Optional[bool] = None
    z_dist: Optional[float] = None

@dataclass
class TrainedBPETokenizer:
    method: str
    merges: List[Tuple[str, str]]  # ordered by rank
    vocab: Set[str]
    merge_logs: List[MergeLog] = field(default_factory=list)

    def ranks(self) -> Dict[Tuple[str, str], int]:
        return {pair: i for i, pair in enumerate(self.merges)}

def train_batched_bpe(
    word_counts: Dict[str, int],
    method: str,
    target_vocab_size: int,
    batch_size: int,
    tau: int,
    alpha: float,
    sigma: float,
    recompute_every: int,
    stem_set: Optional[Set[str]] = None,
    suffix_set: Optional[Set[str]] = None,
    log_z_for_all: bool = True,
    max_batches: int = 10000,
    verbose: bool = True,
) -> TrainedBPETokenizer:
    """
    Batched (conflict-free) BPE training with different merge scores.

    method:
      - 'freq'
      - 'ppmi'
      - 'spectral'
      - 'spectral_shuffled' (control)
    """
    assert method in ("freq", "ppmi", "spectral", "spectral_shuffled")

    word_tokens = {w: list(w) for w in word_counts.keys()}
    vocab: Set[str] = set()
    for toks in word_tokens.values():
        vocab.update(toks)

    merges: List[Tuple[str, str]] = []
    logs: List[MergeLog] = []

    last_z: Optional[np.ndarray] = None
    last_vocab_list: Optional[List[str]] = None
    last_sigma: Optional[float] = None

    t0 = time.time()
    batch_id = 0

    while len(vocab) < target_vocab_size and batch_id < max_batches:
        pair_counts, first_counts, second_counts, total_pairs = compute_bigram_counts(word_tokens, word_counts)
        if not pair_counts:
            if verbose:
                print(f"[{method}] no more pairs; stopping")
            break

        need_weights = method in ("ppmi", "spectral", "spectral_shuffled") or log_z_for_all
        weights = compute_ppmi_weights(pair_counts, first_counts, second_counts, total_pairs, tau=tau, alpha=alpha) if need_weights else {}

        need_z = method in ("spectral", "spectral_shuffled") or log_z_for_all
        if need_z and (last_z is None or last_vocab_list is None or len(last_vocab_list) != len(vocab) or (batch_id % max(1, recompute_every) == 0)):
            vocab_list = sorted(vocab)
            z = compute_fiedler_embedding(vocab_list, weights)
            if method == "spectral_shuffled":
                rng = np.random.default_rng(12345 + batch_id)
                z = z.copy()
                rng.shuffle(z)
            last_z = z
            last_vocab_list = vocab_list

            # sigma: if <=0, estimate from median |z_i-z_j| on weighted edges
            if sigma <= 0:
                idx = {t: i for i, t in enumerate(vocab_list)}
                dists = []
                for (a, b), w in weights.items():
                    ia = idx.get(a)
                    ib = idx.get(b)
                    if ia is None or ib is None:
                        continue
                    dists.append(abs(float(z[ia] - z[ib])))
                last_sigma = (float(np.median(dists)) + 1e-6) if dists else 1.0
            else:
                last_sigma = float(sigma)
        else:
            vocab_list = last_vocab_list if last_vocab_list is not None else sorted(vocab)
            z = last_z
            last_sigma = last_sigma if last_sigma is not None else (float(sigma) if sigma > 0 else 1.0)

        idx_map = {t: i for i, t in enumerate(vocab_list)} if (need_z and z is not None) else {}

        # score pairs
        scored: List[Tuple[float, Tuple[str, str]]] = []
        for (a, b), nij in pair_counts.items():
            if method == "freq":
                score = float(nij)
            else:
                w = float(weights.get((a, b), 0.0))
                if w <= 0:
                    continue
                score = float(nij) * w
                if method in ("spectral", "spectral_shuffled"):
                    ia = idx_map.get(a)
                    ib = idx_map.get(b)
                    if ia is None or ib is None or z is None:
                        continue
                    dz = float(z[ia] - z[ib])
                    coh = math.exp(-(dz * dz) / (last_sigma * last_sigma))
                    score *= coh
            if score > 0:
                scored.append((score, (a, b)))

        if not scored:
            if verbose:
                print(f"[{method}] no positive-scoring pairs; stopping")
            break

        scored.sort(reverse=True, key=lambda x: x[0])

        # conflict-free selection
        used: Set[str] = set()
        selected: List[Tuple[str, str, str, float]] = []
        for score, (a, b) in scored:
            if a in used or b in used:
                continue
            new_tok = a + b
            selected.append((a, b, new_tok, float(score)))
            used.add(a)
            used.add(b)
            if len(selected) >= batch_size:
                break

        if not selected:
            if verbose:
                print(f"[{method}] couldn't select any merges; stopping")
            break

        # apply all selected merges simultaneously (single pass per word)
        pair_to_new = {(a, b): new_tok for a, b, new_tok, _ in selected}
        for w, toks in word_tokens.items():
            if len(toks) < 2:
                continue
            new: List[str] = []
            i = 0
            while i < len(toks):
                if i < len(toks) - 1 and (toks[i], toks[i + 1]) in pair_to_new:
                    new.append(pair_to_new[(toks[i], toks[i + 1])])
                    i += 2
                else:
                    new.append(toks[i])
                    i += 1
            word_tokens[w] = new

        # update merges/vocab/logs
        for a, b, new_tok, score in selected:
            merges.append((a, b))
            is_cross = None
            if stem_set is not None and suffix_set is not None:
                is_cross = (token_domain(new_tok, stem_set, suffix_set) == 3)
            z_dist = None
            if need_z and z is not None:
                ia = idx_map.get(a)
                ib = idx_map.get(b)
                if ia is not None and ib is not None:
                    z_dist = float(abs(float(z[ia] - z[ib])))
            logs.append(
                MergeLog(
                    step=len(merges),
                    a=a,
                    b=b,
                    new_token=new_tok,
                    score=score,
                    batch_id=batch_id,
                    is_cross_boundary=is_cross,
                    z_dist=z_dist,
                )
            )
            vocab.add(new_tok)

        batch_id += 1

        if verbose and (batch_id % 10 == 0 or len(vocab) >= target_vocab_size):
            elapsed = time.time() - t0
            print(f"[{method}] batch {batch_id:4d} | merges {len(merges):5d} | vocab {len(vocab):5d} | elapsed {elapsed:5.1f}s")

    return TrainedBPETokenizer(method=method, merges=merges, vocab=vocab, merge_logs=logs)


# ==========================
# BPE encoding + evaluation
# ==========================

def bpe_encode_symbols(symbols: Tuple[str, ...], ranks: Dict[Tuple[str, str], int]) -> Tuple[str, ...]:
    """
    Standard BPE encoding: repeatedly merge the best-ranked pair until no merge applies.
    New token is always concatenation a+b (consistent with our training).
    """
    if len(symbols) < 2:
        return symbols

    def get_pairs(seq: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        return {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}

    pairs = get_pairs(symbols)
    while True:
        best_pair = None
        best_rank = None
        for p in pairs:
            r = ranks.get(p)
            if r is None:
                continue
            if best_rank is None or r < best_rank:
                best_rank = r
                best_pair = p
        if best_pair is None:
            break

        a, b = best_pair
        new: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b)
                i += 2
            else:
                new.append(symbols[i])
                i += 1

        symbols = tuple(new)
        if len(symbols) < 2:
            break
        pairs = get_pairs(symbols)

    return symbols

def make_bpe_encoder(merges: List[Tuple[str, str]]):
    ranks = {pair: i for i, pair in enumerate(merges)}
    cache: Dict[str, List[str]] = {}

    def encode(word: str) -> List[str]:
        if word in cache:
            return cache[word]
        syms = tuple(list(word))
        out = list(bpe_encode_symbols(syms, ranks))
        cache[word] = out
        return out

    return encode

def evaluate_boundary_f1(
    samples: List[Tuple[str, int, int]],
    encode_fn,
) -> Dict[str, float]:
    """
    samples: list of (word, boundary_idx, count)
    We compute micro precision/recall/F1 on boundary positions.
    Each word has exactly 1 gold boundary.
    """
    total_tp = 0
    total_pred = 0
    total_gold = 0
    total_tokens = 0

    for word, boundary, cnt in samples:
        toks = encode_fn(word)
        total_tokens += len(toks) * cnt
        pos = 0
        pred = set()
        for t in toks[:-1]:
            pos += len(t)
            pred.add(pos)
        total_pred += len(pred) * cnt
        total_gold += 1 * cnt
        if boundary in pred:
            total_tp += 1 * cnt

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = total_tp / total_gold if total_gold > 0 else 0.0
    avg_toks = total_tokens / total_gold if total_gold > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "boundary_acc": float(acc),
        "avg_tokens_per_word": float(avg_toks),
    }

def boundary_curve_with_tokens(
    samples: List[Tuple[str, int, int]],
    merges: List[Tuple[str, str]],
    max_merges: int,
    step: int,
) -> List[Dict[str, float]]:
    curve: List[Dict[str, float]] = []
    max_merges = min(max_merges, len(merges))
    for m in range(0, max_merges + 1, step):
        enc = make_bpe_encoder(merges[:m])
        met = evaluate_boundary_f1(samples, enc)
        curve.append({
            "merges": int(m),
            "boundary_acc": float(met["boundary_acc"]),
            "boundary_f1": float(met["f1"]),
            "avg_tokens_per_word": float(met["avg_tokens_per_word"]),
        })
    return curve

def boundary_acc_curve(
    samples: List[Tuple[str, int, int]],
    merges: List[Tuple[str, str]],
    max_merges: int,
    step: int,
) -> List[Tuple[int, float]]:
    curve = boundary_curve_with_tokens(samples, merges, max_merges, step)
    return [(int(pt["merges"]), pt["boundary_acc"]) for pt in curve]

def auc_boundary_vs_tokens(curve: List[Dict[str, float]], min_tokens_per_word: float) -> Optional[float]:
    if not curve or min_tokens_per_word <= 0:
        return None
    pts = sorted([(pt["avg_tokens_per_word"], pt["boundary_acc"]) for pt in curve], key=lambda x: -x[0])
    if len(pts) < 2:
        return None
    start_x = pts[0][0]
    if min_tokens_per_word >= start_x:
        return None
    area = 0.0
    prev_x, prev_y = pts[0]
    for x, y in pts[1:]:
        if x >= min_tokens_per_word:
            dx = prev_x - x
            area += 0.5 * (prev_y + y) * dx
            prev_x, prev_y = x, y
            continue
        # partial segment to threshold
        if prev_x > min_tokens_per_word > x:
            frac = (prev_x - min_tokens_per_word) / max(1e-9, (prev_x - x))
            y_t = prev_y + (y - prev_y) * frac
            dx = prev_x - min_tokens_per_word
            area += 0.5 * (prev_y + y_t) * dx
        break
    denom = start_x - min_tokens_per_word
    if denom <= 0:
        return None
    return float(area / denom)


# ==========================
# Plotting helpers
# ==========================

def plot_boundary_curves(curves: Dict[str, List[Tuple[int, float]]], out_path: str, title: str) -> None:
    plt.figure()
    for name, pts in curves.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, label=name)
    plt.xlabel("Number of merges applied")
    plt.ylabel("Boundary retention (accuracy)")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_boundary_vs_tokens(curves: Dict[str, List[Tuple[float, float]]], out_path: str, title: str) -> None:
    plt.figure()
    for name, pts in curves.items():
        pts_sorted = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        plt.plot(xs, ys, label=name)
    plt.xlabel("Avg tokens per word (fragmentation)")
    plt.ylabel("Boundary retention (accuracy)")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_cross_boundary_frac(logs_by_method: Dict[str, List[MergeLog]], out_path: str, title: str, max_merges: int) -> None:
    plt.figure()
    for name, logs in logs_by_method.items():
        xs=[]
        ys=[]
        cross=0
        total=0
        for log in logs[:max_merges]:
            total += 1
            if log.is_cross_boundary:
                cross += 1
            xs.append(total)
            ys.append(cross / total)
        plt.plot(xs, ys, label=name)
    plt.xlabel("Merge step")
    plt.ylabel("Cumulative fraction of cross-boundary merges")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_zdist(logs_by_method: Dict[str, List[MergeLog]], out_path: str, title: str, max_merges: int, smooth: int = 5) -> None:
    """
    Plot (smoothed) z-distance of chosen merges vs merge step.
    Only meaningful if z_dist was logged (we log for all methods by default).
    """
    plt.figure()
    for name, logs in logs_by_method.items():
        vals = [log.z_dist for log in logs[:max_merges] if log.z_dist is not None]
        if not vals:
            continue
        # align to steps
        steps = [log.step for log in logs[:max_merges] if log.z_dist is not None]
        ys = np.array(vals, dtype=np.float32)
        xs = np.array(steps, dtype=np.int32)

        if smooth > 1 and len(ys) >= smooth:
            # moving average
            kernel = np.ones(smooth, dtype=np.float32) / smooth
            ys_s = np.convolve(ys, kernel, mode="valid")
            xs_s = xs[smooth - 1 :]
            plt.plot(xs_s, ys_s, label=name)
        else:
            plt.plot(xs, ys, label=name)

    plt.xlabel("Merge step")
    plt.ylabel("Spectral distance |z_i - z_j| (smoothed)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ==========================
# Optional: real text + micro-LM
# ==========================

def load_real_corpus(args) -> Optional[Tuple[str, str]]:
    """
    Returns (train_text, val_text) or None if not available.
    Priority:
      1) --real_text_path (local file)
      2) huggingface datasets WikiText-2 (if installed)
    """
    if args.real_text_path:
        with open(args.real_text_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        if args.real_max_chars > 0:
            txt = txt[: args.real_max_chars]
        # split by lines (90/10)
        lines = txt.splitlines()
        if len(lines) < 10:
            split = int(len(txt) * 0.9)
            return txt[:split], txt[split:]
        split = int(len(lines) * 0.9)
        train = "\n".join(lines[:split])
        val = "\n".join(lines[split:])
        return train, val

    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(ds["train"]["text"])
    val_text = "\n".join(ds["validation"]["text"])
    if args.real_max_chars > 0:
        train_text = train_text[: args.real_max_chars]
        val_text = val_text[: max(1, args.real_max_chars // 10)]
    return train_text, val_text

def build_word_counts_from_text(text: str, lower: bool = False) -> Dict[str, int]:
    """
    Simple pre-tokenization: whitespace split, then prefix each word with '▁' like SentencePiece.
    """
    if lower:
        text = text.lower()
    words = text.split()
    c = collections.Counter()
    for w in words:
        c["▁" + w] += 1
    return dict(c)

def encode_text_to_ids(text: str, tokenizer: TrainedBPETokenizer, lower: bool = False) -> Tuple[List[int], int]:
    """
    Encode text with a trained BPE tokenizer, using the same pre-tokenization scheme (whitespace + '▁').
    Returns:
      ids: token id sequence
      n_bytes: length of the original raw text in bytes (utf-8)
    """
    if lower:
        text = text.lower()
    ranks = tokenizer.ranks()
    # token ids
    vocab_list = sorted(tokenizer.vocab)
    tok2id = {t: i for i, t in enumerate(vocab_list)}
    unk_id = tok2id.get("▁<unk>", None)
    cache: Dict[str, List[str]] = {}

    def encode_word_token(wtok: str) -> List[str]:
        if wtok in cache:
            return cache[wtok]
        syms = tuple(list(wtok))
        out = list(bpe_encode_symbols(syms, ranks))
        cache[wtok] = out
        return out

    ids: List[int] = []
    for w in text.split():
        wtok = "▁" + w
        pieces = encode_word_token(wtok)
        for p in pieces:
            pid = tok2id.get(p, unk_id)
            if pid is None:
                # should not happen; fallback to char-level pieces
                for ch in p:
                    pid2 = tok2id.get(ch)
                    if pid2 is None:
                        continue
                    ids.append(pid2)
            else:
                ids.append(pid)

    return ids, len(text.encode("utf-8", errors="ignore"))

# --- micro LM (tiny transformer) ---
def train_micro_lm_bits_per_byte(
    train_ids: List[int],
    val_ids: List[int],
    vocab_size: int,
    val_bytes: int,
    device: str,
    seed: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    lr: float,
    eval_batches: int = 50,
) -> Dict[str, float]:
    """
    Train a tiny causal LM for a small number of steps and return bits-per-byte on val.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_t = torch.tensor(train_ids, dtype=torch.long, device=device)
    val_t = torch.tensor(val_ids, dtype=torch.long, device=device)

    if len(train_t) < seq_len + 2 or len(val_t) < seq_len + 2:
        raise RuntimeError("Not enough tokens for the requested seq_len")

    class TinyCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            # causal mask (seq_len x seq_len)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.register_buffer("causal_mask", mask, persistent=False)

        def forward(self, x):
            B, T = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            h = self.tok_emb(x) + self.pos_emb(pos)
            # attn_mask: True means "disallow"
            h = self.tr(h, mask=self.causal_mask[:T, :T])
            h = self.ln(h)
            logits = self.head(h)
            return logits

    model = TinyCausalLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    offsets = torch.arange(seq_len, device=device).unsqueeze(0)  # (1,T)

    def sample_batch(data_t: "torch.Tensor"):
        max_start = data_t.shape[0] - seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,), device=device)
        x = data_t[starts.unsqueeze(1) + offsets]           # (B,T)
        y = data_t[starts.unsqueeze(1) + offsets + 1]       # (B,T)
        return x, y

    # train
    model.train()
    for _ in range(steps):
        x, y = sample_batch(train_t)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # eval
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = sample_batch(val_t)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            losses.append(float(loss.item()))
    val_loss_nats_per_token = float(np.mean(losses))
    tokens_per_byte = float(len(val_ids) / max(1, val_bytes))
    bits_per_byte = float((val_loss_nats_per_token / math.log(2.0)) * tokens_per_byte)
    return {
        "val_loss_nats_per_token": val_loss_nats_per_token,
        "tokens_per_byte": tokens_per_byte,
        "bits_per_byte": bits_per_byte,
    }

def affix_probe_samples_from_text(
    text: str,
    prefixes: List[str],
    suffixes: List[str],
    min_base_count: int = 5,
    min_derived_count: int = 2,
    min_extra_len: int = 2,
    exclude_affixed_base: bool = False,
    lower: bool = True,
    max_samples: int = 5000,
) -> List[Tuple[str, int, int]]:
    """
    Build a heuristic boundary probe set:
      - If word = prefix + base and base appears as a standalone word often enough, boundary at prefix.
      - If word = base + suffix and base appears often enough, boundary at base.
      - Require derived word appears at least min_derived_count times.
      - Require derived length >= base length + min_extra_len.
      - Optionally exclude bases that are themselves affixed forms.
    We encode with '▁'+word, so boundary indices include that leading '▁' (index shift by 1).
    Returns samples (word_token, boundary_idx, count) where word_token includes leading '▁'.
    """
    if lower:
        text = text.lower()
    words = re.findall(r"[a-z]+", text)
    counts = collections.Counter(words)

    def base_is_affixed(base: str) -> bool:
        if not exclude_affixed_base:
            return False
        for p in prefixes:
            if base.startswith(p) and len(base) > len(p) + min_extra_len:
                return True
        for s in suffixes:
            if base.endswith(s) and len(base) > len(s) + min_extra_len:
                return True
        return False

    samples: List[Tuple[str, int, int]] = []
    for w, c in counts.items():
        if c < min_derived_count:
            continue
        skip_word = False
        # prefix cases
        for p in prefixes:
            if w.startswith(p):
                base = w[len(p):]
                if len(base) >= 1 and len(w) >= len(base) + min_extra_len:
                    if base_is_affixed(base):
                        skip_word = True
                        break
                    if counts.get(base, 0) >= min_base_count:
                        # boundary after '▁'+prefix
                        samples.append(("▁" + w, 1 + len(p), c))
                        break
        if skip_word:
            continue
        # suffix cases
        for s in suffixes:
            if w.endswith(s):
                base = w[: -len(s)]
                if len(base) >= 1 and len(w) >= len(base) + min_extra_len:
                    if base_is_affixed(base):
                        skip_word = True
                        break
                    if counts.get(base, 0) >= min_base_count:
                        samples.append(("▁" + w, 1 + len(base), c))
                        break

    # subsample if huge
    if len(samples) > max_samples:
        rng = random.Random(0)
        rng.shuffle(samples)
        samples = samples[:max_samples]
    return samples


# ==========================
# Main experiment runner
# ==========================

def run_toy_stage(args) -> Dict[str, Any]:
    print("\n=== Toy stage: synthetic STEM+SUFFIX corpus ===")
    train_counts, val_samples, alph = generate_synthetic_corpus(
        n_word_types=args.toy_word_types,
        seed=args.seed,
        stem_len_range=(args.toy_stem_min, args.toy_stem_max),
        suffix_len_range=(args.toy_suffix_min, args.toy_suffix_max),
        stem_chars=args.toy_stem_chars,
        suffix_chars=args.toy_suffix_chars,
        end_chars=args.toy_end_chars,
        start_chars=args.toy_start_chars,
        zipf_a=args.toy_zipf_a,
        max_count=args.toy_max_count,
        val_frac=args.toy_val_frac,
    )
    stem_set = alph["stem_set"]
    suffix_set = alph["suffix_set"]
    print(f"Train word types: {len(train_counts):,} | total word occurrences: {sum(train_counts.values()):,}")
    print(f"Val word types:   {len(val_samples):,}")

    methods = ["freq", "ppmi", "spectral", "spectral_shuffled"]
    trained: Dict[str, TrainedBPETokenizer] = {}

    for m in methods:
        print(f"\n--- Training tokenizer: {m} ---")
        tok = train_batched_bpe(
            train_counts,
            method=m,
            target_vocab_size=args.toy_vocab_size,
            batch_size=args.toy_batch_size,
            tau=args.tau,
            alpha=args.alpha,
            sigma=args.sigma,
            recompute_every=args.recompute_every,
            stem_set=stem_set,
            suffix_set=suffix_set,
            log_z_for_all=True,
            verbose=True,
        )
        trained[m] = tok
        # quick summary
        first_cross = next((log for log in tok.merge_logs if log.is_cross_boundary), None)
        print(f"[{m}] merges: {len(tok.merges):,} | vocab: {len(tok.vocab):,}")
        if first_cross is not None:
            print(f"[{m}] first cross-boundary merge at step {first_cross.step}: ({first_cross.a},{first_cross.b}) score={first_cross.score:.2f}")
        else:
            print(f"[{m}] no cross-boundary merges observed (unexpected on this toy)")

    # Boundary retention curve (key "intuition" plot)
    curves_by_merge = {}
    curves_by_tokens = {}
    curves_full = {}
    for m in methods:
        curve = boundary_curve_with_tokens(
            val_samples,
            trained[m].merges,
            max_merges=args.plot_max_merges,
            step=args.plot_step,
        )
        curves_full[m] = curve
        curves_by_merge[m] = [(int(pt["merges"]), pt["boundary_acc"]) for pt in curve]
        curves_by_tokens[m] = [(pt["avg_tokens_per_word"], pt["boundary_acc"]) for pt in curve]

    min_tokens_by_method = [min(pt["avg_tokens_per_word"] for pt in curves_full[m]) for m in methods]
    max_tokens_by_method = [max(pt["avg_tokens_per_word"] for pt in curves_full[m]) for m in methods]
    common_min_tokens = max(min_tokens_by_method)
    common_max_tokens = min(max_tokens_by_method)
    if args.matched_tokens_per_word > 0:
        matched_tokens = args.matched_tokens_per_word
        if matched_tokens < common_min_tokens:
            matched_tokens = common_min_tokens
        if matched_tokens > common_max_tokens:
            matched_tokens = common_max_tokens
    else:
        matched_tokens = common_min_tokens

    auc_by_method = {}
    for m in methods:
        auc_by_method[m] = auc_boundary_vs_tokens(curves_full[m], matched_tokens)

    # Final tokenization quality (often will be 0 boundary retention after enough merges; curve is the point)
    final_rows = []
    for m in methods:
        enc = make_bpe_encoder(trained[m].merges)
        met = evaluate_boundary_f1(val_samples, enc)
        cross_frac = None
        logs = trained[m].merge_logs
        if any(log.is_cross_boundary is not None for log in logs):
            cross_frac = sum(1 for log in logs if log.is_cross_boundary) / max(1, len(logs))
        mean_z = np.mean([log.z_dist for log in logs[:min(50, len(logs))] if log.z_dist is not None]) if logs else 0.0
        final_rows.append([
            m,
            f"{met['boundary_acc']:.3f}",
            f"{met['f1']:.3f}",
            f"{met['avg_tokens_per_word']:.2f}",
            f"{auc_by_method[m]:.3f}" if auc_by_method[m] is not None else "n/a",
            f"{cross_frac:.3f}" if cross_frac is not None else "n/a",
            f"{mean_z:.4f}",
        ])

    print("\nToy summary (final tokenizer at target vocab):")
    print(pretty_table(final_rows, headers=["method", "boundary_acc", "boundary_F1", "avg_toks/word", f"auc_acc@toks<= {matched_tokens:.2f}", "cross_merge_frac", "mean|z_i-z_j| (first50)"]))

    # Save plots
    ensure_dir(args.out_dir)
    plot_boundary_curves(curves_by_merge, os.path.join(args.out_dir, "toy_boundary_retention.png"),
                         title="Toy: boundary retention vs merges (higher is better)")
    plot_boundary_vs_tokens(curves_by_tokens, os.path.join(args.out_dir, "toy_boundary_retention_vs_tokens.png"),
                            title="Toy: boundary retention vs avg tokens/word (matched compression)")
    logs_by_method = {m: trained[m].merge_logs for m in methods}
    plot_cross_boundary_frac(logs_by_method, os.path.join(args.out_dir, "toy_cross_boundary_merge_frac.png"),
                            title="Toy: cumulative cross-boundary merges (lower is better)",
                            max_merges=args.plot_max_merges)
    plot_zdist(logs_by_method, os.path.join(args.out_dir, "toy_spectral_distance.png"),
               title="Toy: spectral distance of chosen merges (mechanism plot)",
               max_merges=args.plot_max_merges,
               smooth=max(1, args.plot_smooth))

    # Save raw results
    results = {
        "toy": {
            "config": {
                "toy_word_types": args.toy_word_types,
                "toy_vocab_size": args.toy_vocab_size,
                "toy_batch_size": args.toy_batch_size,
                "tau": args.tau,
                "alpha": args.alpha,
                "sigma": args.sigma,
                "recompute_every": args.recompute_every,
            },
            "curves": {m: curves_by_merge[m] for m in methods},
            "curves_by_merge": {m: curves_by_merge[m] for m in methods},
            "curves_full": {m: curves_full[m] for m in methods},
            "curves_by_tokens": {m: curves_by_tokens[m] for m in methods},
            "matched_tokens_per_word": float(matched_tokens),
            "final_summary": {row[0]: {
                "boundary_acc": float(row[1]),
                "boundary_f1": float(row[2]),
                "avg_toks_per_word": float(row[3]),
                "auc_boundary_vs_tokens": None if row[4] == "n/a" else float(row[4]),
                "cross_merge_frac": None if row[5]=="n/a" else float(row[5]),
                "mean_zdist_first50": float(row[6]),
            } for row in final_rows},
            "first_20_merges": {
                m: [{
                    "step": log.step,
                    "pair": [log.a, log.b],
                    "is_cross_boundary": log.is_cross_boundary,
                    "score": log.score,
                    "z_dist": log.z_dist,
                } for log in trained[m].merge_logs[:20]]
                for m in methods
            }
        }
    }
    save_json(results, os.path.join(args.out_dir, "toy_results.json"))
    print(f"\nSaved toy plots + results to: {args.out_dir}")
    return results

def run_real_lm_stage(args) -> Optional[Dict[str, Any]]:
    print("\n=== Real-text stage: tokenizers + micro-LM (optional) ===")
    corp = load_real_corpus(args)
    if corp is None:
        print("Real corpus unavailable: install `datasets` or pass --real_text_path. Skipping.")
        return None
    train_text, val_text = corp
    print(f"Loaded real corpus: train chars={len(train_text):,}, val chars={len(val_text):,}")

    # Build word counts for tokenizer training
    train_counts = build_word_counts_from_text(train_text, lower=args.real_lower)
    print(f"Real tokenizer training word types: {len(train_counts):,} | total word occurrences: {sum(train_counts.values()):,}")

    methods = ["freq", "ppmi", "spectral", "spectral_shuffled"]
    trained: Dict[str, TrainedBPETokenizer] = {}

    for m in methods:
        print(f"\n--- Training real tokenizer: {m} ---")
        tok = train_batched_bpe(
            train_counts,
            method=m,
            target_vocab_size=args.real_vocab_size,
            batch_size=args.real_batch_size,
            tau=args.tau,
            alpha=args.alpha,
            sigma=args.sigma,
            recompute_every=args.recompute_every,
            stem_set=None,
            suffix_set=None,
            log_z_for_all=False,
            verbose=True,
        )
        trained[m] = tok
        print(f"[{m}] merges: {len(tok.merges):,} | vocab: {len(tok.vocab):,}")

    # Affix probe (heuristic)
    prefixes = [
        "un", "re", "dis", "mis", "pre", "non", "anti", "inter", "trans", "sub", "super",
        "over", "under", "semi", "auto", "micro", "macro", "post", "pro", "de", "co",
    ]
    suffixes = [
        "ing", "ed", "ly", "tion", "sion", "ment", "ness", "able", "less", "ist", "ive", "ous",
        "ize", "ise", "al", "er", "or", "ship", "hood", "ward", "wise", "ful", "dom",
    ]
    if args.probe_text_source == "train":
        probe_text = train_text
    elif args.probe_text_source == "val":
        probe_text = val_text
    else:
        probe_text = train_text + "\n" + val_text
    probe_samples = affix_probe_samples_from_text(
        probe_text,
        prefixes=prefixes,
        suffixes=suffixes,
        min_base_count=args.probe_min_base_count,
        min_derived_count=args.probe_min_derived_count,
        min_extra_len=args.probe_min_extra_len,
        exclude_affixed_base=args.probe_exclude_affixed_base,
        lower=args.real_lower,
        max_samples=args.probe_max_samples,
    )
    print(f"Affix probe samples: {len(probe_samples):,} (source={args.probe_text_source})")

    # Encode train/val and run micro LM for each tokenizer
    try:
        import torch  # noqa: F401
    except Exception:
        print("PyTorch not available; skipping micro-LM.")
        return None

    real_rows = []
    real_metrics = {}

    for m in methods:
        print(f"\n--- Encoding + micro-LM: {m} ---")
        train_ids, _train_bytes = encode_text_to_ids(train_text, trained[m], lower=args.real_lower)
        val_ids, val_bytes = encode_text_to_ids(val_text, trained[m], lower=args.real_lower)
        vocab_list = sorted(trained[m].vocab)
        vocab_size = len(vocab_list)

        # Affix boundary metric
        enc = make_bpe_encoder(trained[m].merges)
        probe_met = evaluate_boundary_f1(probe_samples, enc) if probe_samples else {"boundary_acc": 0.0, "f1": 0.0}

        # Micro-LM
        lm_met = train_micro_lm_bits_per_byte(
            train_ids=train_ids,
            val_ids=val_ids,
            vocab_size=vocab_size,
            val_bytes=val_bytes,
            device=args.device,
            seed=args.seed,
            steps=args.lm_steps,
            batch_size=args.lm_batch_size,
            seq_len=args.lm_seq_len,
            d_model=args.lm_d_model,
            n_layers=args.lm_layers,
            n_heads=args.lm_heads,
            lr=args.lm_lr,
            eval_batches=args.lm_eval_batches,
        )

        real_metrics[m] = {
            "vocab_size": vocab_size,
            "train_tokens": len(train_ids),
            "val_tokens": len(val_ids),
            "val_bytes": val_bytes,
            "probe_boundary_acc": probe_met["boundary_acc"],
            "probe_boundary_f1": probe_met["f1"],
            **lm_met,
        }
        real_rows.append([
            m,
            vocab_size,
            f"{lm_met['tokens_per_byte']:.3f}",
            f"{lm_met['bits_per_byte']:.3f}",
            f"{probe_met['boundary_acc']:.3f}",
            f"{probe_met['f1']:.3f}",
        ])

    print("\nReal-text summary:")
    print(pretty_table(real_rows, headers=["method", "vocab", "tokens/byte", "bits/byte (lower better)", "probe_acc", "probe_F1"]))

    out = {"real": {"config": {
        "real_vocab_size": args.real_vocab_size,
        "real_batch_size": args.real_batch_size,
        "lm_steps": args.lm_steps,
        "lm_batch_size": args.lm_batch_size,
        "lm_seq_len": args.lm_seq_len,
        "lm_d_model": args.lm_d_model,
        "lm_layers": args.lm_layers,
        "lm_heads": args.lm_heads,
        "lm_lr": args.lm_lr,
        "device": args.device,
        "real_lower": args.real_lower,
    }, "metrics": real_metrics}}
    save_json(out, os.path.join(args.out_dir, "real_lm_results.json"))
    print(f"\nSaved real-text results to: {os.path.join(args.out_dir, 'real_lm_results.json')}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="spectralbpe_miniexp_out")
    p.add_argument("--seed", type=int, default=0)

    # Shared method params
    p.add_argument("--tau", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=0.2, help="<=0 => auto sigma from median |z_i-z_j|")
    p.add_argument("--recompute_every", type=int, default=1)

    # Toy dataset
    p.add_argument("--toy_word_types", type=int, default=8000)
    p.add_argument("--toy_val_frac", type=float, default=0.1)
    p.add_argument("--toy_zipf_a", type=float, default=1.2)
    p.add_argument("--toy_max_count", type=int, default=200)
    p.add_argument("--toy_vocab_size", type=int, default=2000)
    p.add_argument("--toy_batch_size", type=int, default=50)
    p.add_argument("--toy_stem_min", type=int, default=5)
    p.add_argument("--toy_stem_max", type=int, default=9)
    p.add_argument("--toy_suffix_min", type=int, default=3)
    p.add_argument("--toy_suffix_max", type=int, default=6)
    p.add_argument("--toy_stem_chars", type=str, default="abcdefghijklm")
    p.add_argument("--toy_suffix_chars", type=str, default="nopqrstuvwxyz")
    p.add_argument("--toy_end_chars", type=str, default="abcde")
    p.add_argument("--toy_start_chars", type=str, default="nopqr")

    # Plotting
    p.add_argument("--plot_max_merges", type=int, default=200)
    p.add_argument("--plot_step", type=int, default=5)
    p.add_argument("--plot_smooth", type=int, default=7)

    # Real LM stage
    p.add_argument("--run_real_lm", action="store_true")
    p.add_argument("--real_text_path", type=str, default="")
    p.add_argument("--real_max_chars", type=int, default=2_000_000)
    p.add_argument("--real_vocab_size", type=int, default=4000)
    p.add_argument("--real_batch_size", type=int, default=50)
    p.add_argument("--real_lower", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")

    # Micro-LM config
    p.add_argument("--lm_steps", type=int, default=1200)
    p.add_argument("--lm_batch_size", type=int, default=64)
    p.add_argument("--lm_seq_len", type=int, default=128)
    p.add_argument("--lm_d_model", type=int, default=256)
    p.add_argument("--lm_layers", type=int, default=4)
    p.add_argument("--lm_heads", type=int, default=4)
    p.add_argument("--lm_lr", type=float, default=3e-4)
    p.add_argument("--lm_eval_batches", type=int, default=30)

    # Probe
    p.add_argument("--probe_min_base_count", type=int, default=5)
    p.add_argument("--probe_min_derived_count", type=int, default=2)
    p.add_argument("--probe_min_extra_len", type=int, default=2)
    p.add_argument("--probe_exclude_affixed_base", action="store_true")
    p.add_argument("--probe_max_samples", type=int, default=5000)
    p.add_argument("--probe_text_source", type=str, default="train", choices=["train", "val", "both"])
    p.add_argument("--matched_tokens_per_word", type=float, default=0.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    toy_results = run_toy_stage(args)
    all_results = dict(toy_results)

    if args.run_real_lm:
        real_results = run_real_lm_stage(args)
        if real_results is not None:
            all_results.update(real_results)

    save_json(all_results, os.path.join(args.out_dir, "all_results.json"))
    print(f"\nWrote combined results to: {os.path.join(args.out_dir, 'all_results.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
