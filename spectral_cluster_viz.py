"""
Repurposed from the old SpectralBPE sanity script.

Goal: run *plain* BPE training over word types (character + </w> sentinel), and
periodically compute *spectral clusters* of a token graph.

Outputs: interactive, zoomable HTML scatter plots (Plotly WebGL) plus JSON snapshots.

Design choice for visualization:
  - We do NOT try to render the full graph (edges explode).
  - Pipeline: graph (adjacency-PPMI by default; optionally adjacency log-count via --adj_log_counts, or distributional similarity via --distributional_similarity) → normalized Laplacian → d-dim spectral embedding → k-means clustering (in d-dim space) → PCA to 2D → visualization
  - We render nodes in 2D (PCA of spectral embedding), colored by cluster.
  - Hover shows token + stats; zoom/pan are unlimited.

Dependencies: numpy, scipy.

Example:
  python spectral_cluster_viz.py \
    --train_text data/train.txt \
    --vocab_size 16000 \
    --snapshot_every 500 \
    --out_dir out/cluster_viz

Then open:
  out/cluster_viz/index.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix, diags, identity
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.csgraph import connected_components
except Exception as e:
    raise RuntimeError("This script needs scipy (scipy.sparse + eigsh). Install: pip install scipy") from e


END_WORD = "</w>"


# ---------------- I/O + pretokenization ----------------
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


def strip_end(tok: str) -> str:
    return tok[:-len(END_WORD)] if tok.endswith(END_WORD) else tok


# ---------------- BPE training state ----------------
def init_vocab(word_freq: Counter) -> Dict[Tuple[str, ...], int]:
    vocab = defaultdict(int)
    for w, f in word_freq.items():
        syms = word_to_syms(w)
        if syms:
            vocab[syms] += int(f)
    return dict(vocab)


def symbol_set(vocab: Dict[Tuple[str, ...], int]) -> List[str]:
    s = set()
    for seq in vocab:
        s.update(seq)
    return sorted(s)


def pair_counts(vocab: Dict[Tuple[str, ...], int]) -> Counter:
    c = Counter()
    for seq, freq in vocab.items():
        for i in range(len(seq) - 1):
            c[(seq[i], seq[i + 1])] += freq
    return c


def token_counts(vocab: Dict[Tuple[str, ...], int]) -> Counter:
    c = Counter()
    for seq, freq in vocab.items():
        for t in seq:
            c[t] += freq
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


# ---------------- Graph builders ----------------
def sym_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_ppmi_undirected(
    N_dir: Counter,
    tau: int,
    ppmi_beta: float,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Return (ppmi_dir, W_und) where
    - ppmi_dir[(i,j)] = max(0, PMI(i,j))
    - W_und[{i,j}] = 1{N>=tau} * ((ppmi(i,j)+ppmi_beta) + (ppmi(j,i)+ppmi_beta))

    Note: ppmi_beta gives the graph some connectivity even when PMI<=0.
    """
    total = float(sum(N_dir.values()))
    if total <= 0:
        return {}, {}

    out_m = Counter()
    in_m = Counter()
    for (a, b), c in N_dir.items():
        out_m[a] += c
        in_m[b] += c

    ppmi_dir: Dict[Tuple[str, str], float] = {}
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
            w = ppmi + float(ppmi_beta)
            if w > 0:
                W_und[sym_key(a, b)] += w

    return ppmi_dir, dict(W_und)


def build_log_count_undirected(N_dir: Counter) -> Dict[Tuple[str, str], float]:
    """Undirected adjacency graph with smoothed log-count edge weights.

    For each directed bigram count N(i,j), contribute log(1 + N(i,j))
    to undirected edge {i,j}. This keeps high-frequency connective edges
    while damping very large counts.
    """
    W_und: Dict[Tuple[str, str], float] = defaultdict(float)
    for (a, b), c in N_dir.items():
        if c <= 0:
            continue
        W_und[sym_key(a, b)] += math.log1p(float(c))
    return dict(W_und)


def build_distributional_similarity_undirected(
    tokens: List[str],
    ppmi_dir: Dict[Tuple[str, str], float],
    knn_k: int,
    min_cos: float,
    batch_size: int,
) -> Dict[Tuple[str, str], float]:
    """Undirected KNN graph from cosine similarity of PPMI context rows.

    Context vector for token i is c_i = [PPMI(i, j)]_j.
    Edge weight is max(0, cos(c_i, c_j)); with nonnegative PPMI this is just cosine.
    """
    n = len(tokens)
    if n < 2 or not ppmi_dir:
        return {}

    k = int(knn_k)
    if k <= 0:
        return {}
    k = min(k, n - 1)
    min_cos = float(min_cos)
    bsz = max(1, int(batch_size))

    idx = {t: i for i, t in enumerate(tokens)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for (a, b), v in ppmi_dir.items():
        if not (v > 0 and math.isfinite(v)):
            continue
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None:
            continue
        rows.append(ia)
        cols.append(ib)
        data.append(float(v))

    if not data:
        return {}

    C_raw = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    row_sq = np.asarray(C_raw.multiply(C_raw).sum(axis=1)).ravel()
    row_norm = np.sqrt(row_sq)
    nonzero = row_norm > 1e-12
    if not np.any(nonzero):
        return {}

    inv_norm = np.zeros((n,), dtype=np.float64)
    inv_norm[nonzero] = 1.0 / row_norm[nonzero]
    C = csr_matrix(diags(inv_norm) @ C_raw)

    W_idx: Dict[Tuple[int, int], float] = {}
    for start in range(0, n, bsz):
        end = min(n, start + bsz)
        # Sparse block similarities: only rows with overlapping contexts appear.
        S = csr_matrix(C[start:end] @ C.T)
        for r in range(end - start):
            i = start + r
            if not nonzero[i]:
                continue

            lo = int(S.indptr[r])
            hi = int(S.indptr[r + 1])
            if hi <= lo:
                continue

            nbr = S.indices[lo:hi]
            sim = S.data[lo:hi]
            mask = (nbr != i) & (sim > min_cos) & np.isfinite(sim)
            if not np.any(mask):
                continue

            nbr = nbr[mask]
            sim = sim[mask]
            if sim.size > k:
                keep = np.argpartition(sim, -k)[-k:]
                nbr = nbr[keep]
                sim = sim[keep]

            for j, w in zip(nbr.tolist(), sim.tolist()):
                a, b = (i, j) if i < j else (j, i)
                old = W_idx.get((a, b))
                if old is None or w > old:
                    W_idx[(a, b)] = float(w)

    return {(tokens[i], tokens[j]): float(w) for (i, j), w in W_idx.items() if w > min_cos}


# ---------------- Spectral embedding + clustering ----------------
def _build_sparse_adjacency(tokens: List[str], W_und: Dict[Tuple[str, str], float]) -> csr_matrix:
    n = len(tokens)
    idx = {t: i for i, t in enumerate(tokens)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for (a, b), w in W_und.items():
        if a == b:
            continue
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None:
            continue
        if not (w > 0 and math.isfinite(w)):
            continue
        rows.append(ia)
        cols.append(ib)
        data.append(float(w))
        rows.append(ib)
        cols.append(ia)
        data.append(float(w))
    if not data:
        return csr_matrix((n, n), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def spectral_embedding_lcc(
    tokens: List[str],
    W_und: Dict[Tuple[str, str], float],
    d: int,
    eig_eps: float,
    eig_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a d-dim spectral embedding on the largest connected component.

    Returns:
      Z_all: (n, d) embedding for all tokens (zeros outside LCC)
      in_lcc: (n,) bool mask
      evals: eigenvalues returned by eigsh (sorted)
      lcc_idx: indices of LCC nodes in the original token list
    """
    n = len(tokens)
    if n == 0:
        return np.zeros((0, d), dtype=np.float64), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    A = _build_sparse_adjacency(tokens, W_und)
    if A.nnz == 0:
        return np.zeros((n, d), dtype=np.float64), np.zeros((n,), dtype=bool), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)
    if n_components <= 0:
        return np.zeros((n, d), dtype=np.float64), np.zeros((n,), dtype=bool), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    # Largest connected component
    comp_sizes = Counter(labels)
    lcc_label = comp_sizes.most_common(1)[0][0]
    lcc_idx = np.where(labels == lcc_label)[0]
    in_lcc = (labels == lcc_label)
    if lcc_idx.size < 2:
        return np.zeros((n, d), dtype=np.float64), in_lcc, np.zeros((0,), dtype=np.float64), lcc_idx.astype(np.int64)

    A_lcc = A[lcc_idx][:, lcc_idx]
    n_lcc = A_lcc.shape[0]
    # Normalized Laplacian
    deg = np.asarray(A_lcc.sum(axis=1)).ravel()
    deg[deg == 0] = 1e-12
    inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = diags(inv_sqrt)
    L = identity(n_lcc, format="csr") - (D_inv_sqrt @ A_lcc @ D_inv_sqrt)

    # We need d non-trivial eigenvectors. We ask for extra to skip near-zeros safely.
    # On a connected component, the first eigenvalue should be ~0.
    extra = 6
    k_req = min(n_lcc - 1, max(d + extra, d + 1))
    if eig_k is not None:
        k_req = min(k_req, max(2, eig_k))

    try:
        vals, vecs = eigsh(L, k=k_req, which="SA", tol=eig_eps)
    except Exception as e:
        print(f"[warn] eigsh failed (n_lcc={n_lcc}, k={k_req}): {e}", file=sys.stderr)
        return np.zeros((n, d), dtype=np.float64), in_lcc, np.zeros((0,), dtype=np.float64), lcc_idx.astype(np.int64)

    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    # Skip eigenvalues near zero
    keep_cols: List[int] = []
    for j in range(len(vals)):
        if vals[j] < 1e-6:
            continue
        keep_cols.append(j)
        if len(keep_cols) >= d:
            break

    if len(keep_cols) < d:
        # If we couldn't get enough (rare), pad with zeros
        Z_lcc = np.zeros((n_lcc, d), dtype=np.float64)
        if keep_cols:
            Z_lcc[:, : len(keep_cols)] = vecs[:, keep_cols]
    else:
        Z_lcc = vecs[:, keep_cols]

    Z_all = np.zeros((n, d), dtype=np.float64)
    Z_all[lcc_idx, :] = Z_lcc
    return Z_all, in_lcc.astype(bool), vals.astype(np.float64), lcc_idx.astype(np.int64)


def row_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def pca_2d(X: np.ndarray) -> np.ndarray:
    """PCA to 2D via SVD. X is (n, d) with small d."""
    if X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    X0 = X - X.mean(axis=0, keepdims=True)
    # SVD on (n,d) where d is small -> fast
    _, _, vt = np.linalg.svd(X0, full_matrices=False)
    comps = vt[:2].T  # (d,2)
    return X0 @ comps


def kmeans_lloyd(
    X: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 100,
    n_init: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simple k-means. Returns (labels, centers, inertia)."""
    n, d = X.shape
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, d), dtype=np.float64), 0.0
    k = int(k)
    k = max(1, min(k, n))

    rng = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_labels = None
    best_centers = None

    for init in range(n_init):
        # init centers: sample without replacement
        idx = rng.choice(n, size=k, replace=False)
        centers = X[idx].copy()
        labels = np.zeros((n,), dtype=np.int64)

        for _ in range(max_iter):
            # assign
            # distances squared: (n,k)
            # Using (x-c)^2 = x^2 + c^2 - 2x·c
            x2 = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
            c2 = np.sum(centers * centers, axis=1, keepdims=True).T  # (1,k)
            dist2 = x2 + c2 - 2.0 * (X @ centers.T)
            new_labels = np.argmin(dist2, axis=1).astype(np.int64)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # update
            for j in range(k):
                mask = labels == j
                if not np.any(mask):
                    # reinit empty cluster to a random point
                    centers[j] = X[rng.integers(0, n)]
                else:
                    centers[j] = X[mask].mean(axis=0)

        # inertia
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T
        dist2 = x2 + c2 - 2.0 * (X @ centers.T)
        inertia = float(np.sum(dist2[np.arange(n), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia


def choose_k_by_eigengap(evals: np.ndarray, k_max: int, skip_zeros: bool = True) -> int:
    """Very lightweight eigengap heuristic.

    evals should be sorted ascending.
    Returns k in [2, k_max].
    """
    if evals.size < 4:
        return 2
    vals = evals
    if skip_zeros:
        vals = vals[vals >= 1e-6]
    if vals.size < 4:
        return 2
    m = min(int(k_max), vals.size - 1)
    if m < 2:
        return 2
    gaps = vals[1 : m + 1] - vals[0:m]
    k = int(np.argmax(gaps) + 1)  # gap after k-th eigenvalue (1-indexed)
    return max(2, min(k, m))


# ---------------- Visualization I/O ----------------
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.30.0.min.js"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_snapshot_html(path: str, title: str, payload: Dict[str, Any]) -> None:
    """Write a self-contained HTML file with embedded data + Plotly from CDN."""

    data_js = _json_dumps(payload)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; }}
    #top {{ padding: 10px 12px; border-bottom: 1px solid #ddd; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    #plot {{ width: 100vw; height: calc(100vh - 56px); }}
    input {{ padding: 6px 8px; font-size: 14px; }}
    button {{ padding: 6px 10px; font-size: 14px; cursor: pointer; }}
    .meta {{ color: #444; font-size: 13px; }}
  </style>
  <script src=\"{PLOTLY_CDN}\"></script>
</head>
<body>
  <div id=\"top\">
    <div><b>{title}</b></div>
    <div class=\"meta\" id=\"meta\"></div>
    <div style=\"flex: 1\"></div>
    <label class=\"meta\">Search token:</label>
    <input id=\"q\" placeholder=\"substring (e.g., 'ing' or 'tion')\" size=\"28\"/>
    <button id=\"btn\">Highlight</button>
    <button id=\"clr\">Clear</button>
  </div>
  <div id=\"plot\"></div>
  <script>
    const payload = {data_js};
    const tokens = payload.tokens;
    const x = payload.x;
    const y = payload.y;
    const cluster = payload.cluster;
    const freq = payload.freq;
    const in_lcc = payload.in_lcc;

    // Marker size from log-frequency (precomputed in Python, but we keep it simple here).
    const size = freq.map(f => Math.max(3, Math.min(18, 3 + Math.log1p(f) * 1.6)));
    const baseOpacity = 0.75;
    const opacity = tokens.map(_ => baseOpacity);

    // Use cluster id as a numeric color; it's fine for quick qualitative inspection.
    // (Discrete palettes get annoying to manage for many clusters.)
    const trace = {{
      type: 'scattergl',
      mode: 'markers',
      x: x,
      y: y,
      text: tokens,
      customdata: tokens.map((t, i) => [cluster[i], freq[i], in_lcc[i] ? 1 : 0]),
      hovertemplate:
        '<b>%{{text}}</b><br>' +
        'cluster=%{{customdata[0]}}<br>' +
        'freq=%{{customdata[1]}}<br>' +
        'in_lcc=%{{customdata[2]}}<extra></extra>',
      marker: {{
        size: size,
        color: cluster,
        opacity: opacity,
      }}
    }};

    const layout = {{
      margin: {{l: 10, r: 10, t: 10, b: 10}},
      hovermode: 'closest',
      dragmode: 'pan',
      xaxis: {{zeroline: false, showgrid: false}},
      yaxis: {{zeroline: false, showgrid: false, scaleanchor: 'x', scaleratio: 1}},
    }};

    Plotly.newPlot('plot', [trace], layout, {{responsive: true, scrollZoom: true}});

    // Meta
    const m = payload.meta || {{}};
    const metaEl = document.getElementById('meta');
    const graphMeta = (m.graph_mode === 'distributional_similarity')
      ? `graph=dist_sim knn=${{m.dist_knn_k}} min_cos=${{m.dist_min_cos}}`
      : (m.graph_mode === 'adjacency_log_count')
        ? `graph=adj_log_count w=log1p(N)`
        : `graph=adj_ppmi tau=${{m.tau}} ppmi_beta=${{m.ppmi_beta}}`;
    metaEl.textContent = `step=${{m.step}} merges=${{m.merges}} | tokens=${{m.num_tokens}} | LCC=${{m.lcc_size}} (${{(100*m.lcc_frac).toFixed(1)}}%) | k=${{m.k}} d=${{m.d}} | ${{graphMeta}}`;

    // Search highlight
    function highlight(query) {{
      const q = (query || '').toLowerCase();
      const newOpacity = tokens.map((t, i) => {{
        if (!q) return baseOpacity;
        const hit = t.toLowerCase().includes(q);
        return hit ? 0.95 : 0.06;
      }});
      const newSize = tokens.map((t, i) => {{
        if (!q) return size[i];
        const hit = t.toLowerCase().includes(q);
        return hit ? Math.min(26, size[i] + 6) : Math.max(2, size[i] - 1);
      }});
      Plotly.restyle('plot', {{'marker.opacity': [newOpacity], 'marker.size': [newSize]}}, [0]);
    }}

    document.getElementById('btn').onclick = () => highlight(document.getElementById('q').value);
    document.getElementById('clr').onclick = () => {{ document.getElementById('q').value=''; highlight(''); }};
    document.getElementById('q').addEventListener('keydown', (e) => {{ if (e.key === 'Enter') highlight(e.target.value); }});
  </script>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def write_index_html(out_dir: str, entries: List[Tuple[int, str, str]]) -> None:
    """entries: list of (step, html_file, json_file)"""
    items = "\n".join(
        f"<li><a href=\"{html}\">step {step}</a> &nbsp; <span style=\"color:#666\">({jsonf})</span></li>"
        for step, html, jsonf in entries
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>Spectral cluster snapshots</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 18px; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <h2>Spectral cluster snapshots</h2>
  <p>Open a snapshot and zoom/scroll. Hover points to see token text. Use the search box to highlight substrings.</p>
  <ul>
    {items}
  </ul>
</body>
</html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ---------------- Main training loop (plain BPE + periodic clustering) ----------------
def snapshot_clusters(
    vocab: Dict[Tuple[str, ...], int],
    step: int,
    merges_done: int,
    out_dir: str,
    tau: int,
    ppmi_beta: float,
    adj_log_counts: bool,
    distributional_similarity: bool,
    dist_knn_k: int,
    dist_min_cos: float,
    dist_batch_size: int,
    d: int,
    k: int,
    k_auto: bool,
    k_max: int,
    eig_eps: float,
    eig_k: Optional[int],
    kmeans_seed: int,
    kmeans_n_init: int,
) -> Tuple[str, str, Dict[str, Any]]:
    """Compute clusters + write JSON and HTML. Returns (html_name, json_name, meta)."""
    tokens = symbol_set(vocab)
    N_dir = pair_counts(vocab)
    freqs = token_counts(vocab)

    if distributional_similarity:
        ppmi_dir, _ = build_ppmi_undirected(N_dir, tau=tau, ppmi_beta=ppmi_beta)
        W_und = build_distributional_similarity_undirected(
            tokens=tokens,
            ppmi_dir=ppmi_dir,
            knn_k=dist_knn_k,
            min_cos=dist_min_cos,
            batch_size=dist_batch_size,
        )
    elif adj_log_counts:
        W_und = build_log_count_undirected(N_dir)
    else:
        _, W_und = build_ppmi_undirected(N_dir, tau=tau, ppmi_beta=ppmi_beta)

    Z_all, in_lcc, evals, lcc_idx = spectral_embedding_lcc(tokens, W_und, d=d, eig_eps=eig_eps, eig_k=eig_k)

    lcc_size = int(np.sum(in_lcc))
    lcc_frac = float(lcc_size / len(tokens)) if tokens else 0.0

    # Choose k if requested
    k_eff = int(k)
    if k_auto:
        k_eff = choose_k_by_eigengap(evals, k_max=k_max, skip_zeros=True)
        k_eff = max(2, min(k_eff, d))

    # Cluster only LCC nodes.
    labels_all = np.full((len(tokens),), -1, dtype=np.int64)
    coords_all = np.zeros((len(tokens), 2), dtype=np.float64)

    if lcc_size >= 2:
        Z_lcc = Z_all[in_lcc]
        Z_lcc = row_normalize(Z_lcc)
        labels_lcc, _, _ = kmeans_lloyd(
            Z_lcc, k=k_eff, seed=kmeans_seed + step, n_init=kmeans_n_init, max_iter=100
        )
        labels_all[in_lcc] = labels_lcc
        coords_lcc = pca_2d(Z_lcc)
        coords_all[in_lcc] = coords_lcc

    # Prepare arrays for output
    token_text = []
    for t in tokens:
        # Show both raw and stripped for readability
        raw = t
        surf = strip_end(t)
        token_text.append(raw if raw == surf else f"{surf}  [{raw}]")
    # Hide non-LCC points in the scatter (otherwise they pile up at (0,0)).
    x = [float(coords_all[i, 0]) if bool(in_lcc[i]) else None for i in range(len(tokens))]
    y = [float(coords_all[i, 1]) if bool(in_lcc[i]) else None for i in range(len(tokens))]
    cluster = labels_all.astype(int).tolist()
    freq = [int(freqs.get(t, 0)) for t in tokens]
    in_lcc_list = in_lcc.astype(bool).tolist() if len(tokens) else []

    meta = {
        "step": int(step),
        "merges": int(merges_done),
        "num_tokens": int(len(tokens)),
        "lcc_size": int(lcc_size),
        "lcc_frac": float(lcc_frac),
        "k": int(k_eff),
        "d": int(d),
        "tau": int(tau),
        "ppmi_beta": float(ppmi_beta),
        "graph_mode": (
            "distributional_similarity"
            if distributional_similarity
            else ("adjacency_log_count" if adj_log_counts else "adjacency_ppmi")
        ),
        "dist_knn_k": int(dist_knn_k),
        "dist_min_cos": float(dist_min_cos),
    }

    payload = {
        "tokens": token_text,
        "x": x,
        "y": y,
        "cluster": cluster,
        "freq": freq,
        "in_lcc": in_lcc_list,
        "meta": meta,
    }

    json_name = f"snapshot_{step:06d}.json"
    html_name = f"snapshot_{step:06d}.html"
    tsv_name = f"snapshot_{step:06d}.tsv"
    summary_name = f"snapshot_{step:06d}_cluster_summary.txt"
    with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    # TSV for quick ad-hoc inspection (Excel/pandas): raw_token, cluster, freq, in_lcc, x, y
    with open(os.path.join(out_dir, tsv_name), "w", encoding="utf-8") as f:
        f.write("token\tcluster\tfreq\tin_lcc\tx\ty\n")
        for i, raw in enumerate(tokens):
            f.write(
                f"{raw}\t{labels_all[i]}\t{freqs.get(raw, 0)}\t{1 if in_lcc[i] else 0}\t{coords_all[i,0]:.6g}\t{coords_all[i,1]:.6g}\n"
            )

    # Human-readable cluster summary (top tokens by frequency per cluster)
    if lcc_size >= 2:
        by_c: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for i, raw in enumerate(tokens):
            c = int(labels_all[i])
            if c < 0:
                continue
            by_c[c].append((int(freqs.get(raw, 0)), raw))
        with open(os.path.join(out_dir, summary_name), "w", encoding="utf-8") as f:
            graph_desc = (
                f"graph=dist_sim knn_k={dist_knn_k} min_cos={dist_min_cos}"
                if distributional_similarity
                else ("graph=adj_log_count w=log1p(N)" if adj_log_counts else f"graph=adj_ppmi tau={tau} ppmi_beta={ppmi_beta}")
            )
            f.write(f"step={step} merges={merges_done} k={k_eff} d={d} {graph_desc}\n")
            f.write(f"tokens={len(tokens)} lcc={lcc_size} ({100*lcc_frac:.2f}%)\n\n")
            for c in sorted(by_c.keys()):
                items = sorted(by_c[c], key=lambda x: (-x[0], x[1]))
                f.write(f"[cluster {c}] size={len(items)}\n")
                for freq_i, tok in items[:80]:
                    surf = strip_end(tok)
                    show = tok if tok == surf else f"{surf} [{tok}]"
                    f.write(f"  {freq_i:8d}  {show}\n")
                f.write("\n")

    title = f"Spectral clusters @ step {step}"
    write_snapshot_html(os.path.join(out_dir, html_name), title=title, payload=payload)

    return html_name, json_name, meta


def run(
    train_text: str,
    out_dir: str,
    vocab_size: int,
    max_merges: Optional[int],
    pretokenize_mode: str,
    lowercase: bool,
    max_train_lines: Optional[int],
    snapshot_every: int,
    tau: int,
    ppmi_beta: float,
    adj_log_counts: bool,
    distributional_similarity: bool,
    dist_knn_k: int,
    dist_min_cos: float,
    dist_batch_size: int,
    d: int,
    k: int,
    k_auto: bool,
    k_max: int,
    eig_eps: float,
    eig_k: Optional[int],
    kmeans_seed: int,
    kmeans_n_init: int,
    seed: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"[load] reading training text: {train_text}", file=sys.stderr)
    lines = list(iter_lines(train_text, max_train_lines))
    if not lines:
        raise RuntimeError("No training lines found.")
    wf = build_word_freq(lines, pretokenize_mode, lowercase)
    if not wf:
        raise RuntimeError("No words found after pretokenization.")

    vocab = init_vocab(wf)
    init_syms = symbol_set(vocab)
    target_merges = vocab_size - len(init_syms)
    if max_merges is not None:
        target_merges = min(target_merges, int(max_merges))
    target_merges = max(0, int(target_merges))
    print(
        f"[init] word_types={len(wf)} init_syms={len(init_syms)} target_merges={target_merges} (vocab_size={vocab_size})",
        file=sys.stderr,
    )

    rng = random.Random(seed)
    entries: List[Tuple[int, str, str]] = []

    # Snapshot at step 0
    html0, json0, meta0 = snapshot_clusters(
        vocab,
        step=0,
        merges_done=0,
        out_dir=out_dir,
        tau=tau,
        ppmi_beta=ppmi_beta,
        adj_log_counts=adj_log_counts,
        distributional_similarity=distributional_similarity,
        dist_knn_k=dist_knn_k,
        dist_min_cos=dist_min_cos,
        dist_batch_size=dist_batch_size,
        d=d,
        k=k,
        k_auto=k_auto,
        k_max=k_max,
        eig_eps=eig_eps,
        eig_k=eig_k,
        kmeans_seed=kmeans_seed,
        kmeans_n_init=kmeans_n_init,
    )
    entries.append((0, html0, json0))
    print(f"[snapshot] step=0 tokens={meta0['num_tokens']} lcc={meta0['lcc_size']} k={meta0['k']}", file=sys.stderr)

    merges_done = 0
    for it in range(target_merges):
        pc = pair_counts(vocab)
        if not pc:
            break
        # plain BPE = most frequent pair
        pair, _ = pc.most_common(1)[0]
        vocab = apply_merge(vocab, pair)
        merges_done += 1

        if snapshot_every > 0 and (merges_done % snapshot_every == 0 or merges_done == target_merges):
            htmlf, jsonf, meta = snapshot_clusters(
                vocab,
                step=merges_done,
                merges_done=merges_done,
                out_dir=out_dir,
                tau=tau,
                ppmi_beta=ppmi_beta,
                adj_log_counts=adj_log_counts,
                distributional_similarity=distributional_similarity,
                dist_knn_k=dist_knn_k,
                dist_min_cos=dist_min_cos,
                dist_batch_size=dist_batch_size,
                d=d,
                k=k,
                k_auto=k_auto,
                k_max=k_max,
                eig_eps=eig_eps,
                eig_k=eig_k,
                kmeans_seed=kmeans_seed,
                kmeans_n_init=kmeans_n_init,
            )
            entries.append((merges_done, htmlf, jsonf))
            print(
                f"[snapshot] step={merges_done} tokens={meta['num_tokens']} lcc={meta['lcc_size']} ({100*meta['lcc_frac']:.1f}%) k={meta['k']} d={meta['d']}",
                file=sys.stderr,
            )
        if (it + 1) % 200 == 0:
            print(f"[BPE] merges={it+1}/{target_merges}", file=sys.stderr)

    write_index_html(out_dir, entries)
    print(f"[done] wrote {len(entries)} snapshots to {out_dir}/index.html", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_text", required=True, help="Training text (one sentence/line).")
    ap.add_argument("--out_dir", required=True, help="Output directory for snapshots (HTML + JSON).")
    ap.add_argument("--vocab_size", type=int, default=16000, help="Target vocab size (like BPE).")
    ap.add_argument("--max_merges", type=int, default=None, help="Cap merges (debugging).")
    ap.add_argument("--pretokenize", choices=["whitespace", "basic"], default="whitespace")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--max_train_lines", type=int, default=None)

    ap.add_argument("--snapshot_every", type=int, default=500, help="Write a cluster snapshot every N merges.")

    # Graph construction
    ap.add_argument(
        "--tau",
        type=int,
        default=5,
        help="Min bigram count to keep an adjacency PPMI edge (ignored with --distributional_similarity and --adj_log_counts).",
    )
    ap.add_argument(
        "--ppmi_beta",
        type=float,
        default=0.05,
        help="Additive baseline for adjacency PPMI edges (ignored with --distributional_similarity and --adj_log_counts).",
    )
    graph_mode_group = ap.add_mutually_exclusive_group()
    graph_mode_group.add_argument(
        "--distributional_similarity",
        action="store_true",
        help="Use a distributional-similarity graph (KNN by cosine over PPMI context vectors) instead of adjacency edges.",
    )
    graph_mode_group.add_argument(
        "--adj_log_counts",
        action="store_true",
        help="Use adjacency log-count weights w(i,j)=log(1+N(i,j)) instead of adjacency PPMI.",
    )
    ap.add_argument(
        "--dist_knn_k",
        type=int,
        default=20,
        help="If --distributional_similarity, connect each token to up to this many nearest neighbors.",
    )
    ap.add_argument(
        "--dist_min_cos",
        type=float,
        default=1e-6,
        help="If --distributional_similarity, keep cosine edges greater than this threshold.",
    )
    ap.add_argument(
        "--dist_batch_size",
        type=int,
        default=256,
        help="Batch size for sparse cosine computation in distributional-similarity mode.",
    )

    # Spectral clustering
    ap.add_argument("--embed_dim", type=int, default=16, help="Spectral embedding dimension d.")
    ap.add_argument(
        "--clusters_k",
        type=int,
        default=0,
        help="Number of clusters. If 0, choose via eigengap (see --k_max).",
    )
    ap.add_argument("--k_max", type=int, default=32, help="Max k to consider for eigengap.")
    ap.add_argument("--eig_eps", type=float, default=1e-8)
    ap.add_argument(
        "--eig_k",
        type=int,
        default=0,
        help="Override the number of eigenpairs requested (0 = auto based on embed_dim).",
    )
    ap.add_argument("--kmeans_seed", type=int, default=0)
    ap.add_argument("--kmeans_n_init", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="BPE training seed (only affects tie-breaking if added later).")
    args = ap.parse_args()

    k_auto = args.clusters_k == 0
    k = args.clusters_k if args.clusters_k > 0 else 2
    eig_k = args.eig_k if args.eig_k and args.eig_k > 0 else None

    run(
        train_text=args.train_text,
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        max_merges=args.max_merges,
        pretokenize_mode=args.pretokenize,
        lowercase=args.lowercase,
        max_train_lines=args.max_train_lines,
        snapshot_every=args.snapshot_every,
        tau=args.tau,
        ppmi_beta=args.ppmi_beta,
        adj_log_counts=args.adj_log_counts,
        distributional_similarity=args.distributional_similarity,
        dist_knn_k=args.dist_knn_k,
        dist_min_cos=args.dist_min_cos,
        dist_batch_size=args.dist_batch_size,
        d=args.embed_dim,
        k=k,
        k_auto=k_auto,
        k_max=args.k_max,
        eig_eps=args.eig_eps,
        eig_k=eig_k,
        kmeans_seed=args.kmeans_seed,
        kmeans_n_init=args.kmeans_n_init,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
