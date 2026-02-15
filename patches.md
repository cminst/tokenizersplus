### 1) What is the “main knee config” for (and what does it fill in the paper)?

In *your current paper*, the “knee” configuration is the **single representative SpectralBPE setting** you *commit to* once you’ve shown the full Pareto sweep.

Concretely, it fills:

* **Table `\ref{tab:main_results}`** (“Main results at (K{=}16k)”)
  That table is explicitly described as the **knee-point configuration** from the Pareto sweep (you already wrote: (\gamma{=}0.50,\ \lambda{=}0.15)).
  So your “run everything” script needs a **one-shot, reproducible run** that produces exactly the numbers that go in that table.

* **Section `\subsection{Main point at the knee}` narrative**
  The paragraph that says “Compared to greedy BPE, SpectralBPE improves cohesion by … while preserving sequence length and slightly improving BPB …” is *about that one config*. The knee run produces the deltas you quote there.

* **Anything qualitative / interpretability that requires a *single* merge table**
  Your qualitative examples / “top wins/losses” and any “interpretability dump” (top merges, bridge-like pairs, etc.) can’t be done “from the sweep” without picking a concrete model. The knee config is the natural one to inspect.

* **The anchor point for other ablations**
  If you do a (\lambda) sweep later, you typically hold (\gamma) fixed at the knee. That’s the “default setting” anchor.

So: the knee config exists so your paper has (1) a **frontier** plot/table and (2) a **single “default” point** you can describe, tabulate, and analyze.

---

### 2) Yes — here’s a downstream “robustness” task you can actually run (and it ties directly to cohesion)

You already have a “tiny LM BPB proxy” in `spectralbpe_sanity_v3.py`. The cleanest downstream robustness test you can add *without downloading new datasets* is:

**Train the same tiny LM on clean token streams, then evaluate on:**

* clean eval text (**existing BPB**), and
* a **noisy eval** where each pre-tokenized word has probability (p) of a *typo-like adjacent character swap*.

This gives you a robustness metric that’s easy to interpret:

* **BPB(eval noisy)**: absolute performance under noise
* **ΔBPB noisy-clean (%)**: how much performance degrades under noise
* Then across your (\gamma) sweep you can compute whether higher cohesion corresponds to better robustness (e.g., smaller degradation relative to BPE).

This is also aligned with what you literally wrote (“confirm cohesion gains correlate with downstream robustness”) because you’ll be able to plot:

* x-axis: cohesion gain (\Delta_{\mathrm{PPMI}})
* y-axis: robustness delta (Spectral degradation − BPE degradation)

Your existing gamma sweep script already produces per-gamma logs and a parser/plotter pipeline, so this fits cleanly into your workflow. 

---

## Patch: add noisy-eval robustness to the LM in `spectralbpe_sanity_v3.py`

This patch adds:

* new CLI flags:

  * `--lm_robust_eval`
  * `--lm_noise_mode swap`
  * `--lm_noise_prob 0.10`
  * `--lm_noise_seed` (optional; defaults to seed+12345)
* tokenization helpers for **noisy eval**
* LM training that evaluates **both clean and noisy** with the same trained model
* a new printed table:

```
== LM robustness (noisy eval) ==
BPB (eval noisy) | ...
ΔBPB noisy-clean (%) | ...
Avg tokens/sent (noisy) | ...
```

> Apply with `git apply` or manually.

```diff
--- spectralbpe_sanity_v3.py
+++ spectralbpe_sanity_v3.py
@@ -1014,6 +1014,80 @@
     return tokens, lengths
 
 
+
+
+def corrupt_word_swap(word: str, rng: random.Random) -> str:
+    """Adjacent character swap (typo-like noise). Length-preserving."""
+    if not word or len(word) < 2:
+        return word
+    i = rng.randrange(0, len(word) - 1)
+    return word[:i] + word[i + 1] + word[i] + word[i + 2:]
+
+
+def corrupt_word(word: str, rng: random.Random, prob: float, mode: str) -> str:
+    if prob <= 0.0 or not word:
+        return word
+    if rng.random() >= prob:
+        return word
+    if mode == "swap":
+        return corrupt_word_swap(word, rng)
+    return word  # fallback
+
+
+def tokens_from_lines_noisy(
+    lines: Sequence[str],
+    merges: List[Tuple[str, str]],
+    pre_mode: str,
+    lowercase: bool,
+    noise_prob: float,
+    noise_mode: str,
+    noise_seed: int,
+) -> Tuple[List[str], List[int]]:
+    """Tokenize eval text after applying deterministic word-level noise."""
+    rank = {p: i for i, p in enumerate(merges)}
+    rng = random.Random(noise_seed)
+    tokens: List[str] = []
+    lengths: List[int] = []
+    for line in lines:
+        if lowercase:
+            line = line.lower()
+        line_tokens: List[str] = []
+        for w in pretokenize(line, pre_mode):
+            if not w:
+                continue
+            w2 = corrupt_word(w, rng, prob=noise_prob, mode=noise_mode)
+            line_tokens.extend(encode_word(w2, rank))
+        if line_tokens:
+            lengths.append(len(line_tokens))
+            tokens.extend(line_tokens)
+    return tokens, lengths
+
+
+def tokens_from_lines_sp_noisy(
+    lines: Sequence[str],
+    sp,
+    pre_mode: str,
+    lowercase: bool,
+    noise_prob: float,
+    noise_mode: str,
+    noise_seed: int,
+) -> Tuple[List[str], List[int]]:
+    rng = random.Random(noise_seed)
+    tokens: List[str] = []
+    lengths: List[int] = []
+    for line in lines:
+        if lowercase:
+            line = line.lower()
+        line_tokens: List[str] = []
+        for w in pretokenize(line, pre_mode):
+            if not w:
+                continue
+            w2 = corrupt_word(w, rng, prob=noise_prob, mode=noise_mode)
+            line_tokens.extend(encode_word_sp(sp, w2))
+        if line_tokens:
+            lengths.append(len(line_tokens))
+            tokens.extend(line_tokens)
+    return tokens, lengths
 def build_vocab(tokens: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
     uniq = [t for t in sorted(set(tokens)) if t != "<unk>"]
     vocab = ["<unk>"] + uniq
@@ -1109,6 +1183,103 @@
     )
 
 
+
+
+def train_and_eval_lm_robust(
+    train_lines: Sequence[str],
+    eval_lines: Sequence[str],
+    merges: List[Tuple[str, str]],
+    pre_mode: str,
+    lowercase: bool,
+    eval_bytes: int,
+    epochs: int,
+    batch_size: int,
+    block_size: int,
+    n_embd: int,
+    n_head: int,
+    n_layer: int,
+    lr: float,
+    seed: int,
+    noise_prob: float,
+    noise_mode: str,
+    noise_seed: int,
+) -> Tuple[float, float, float, float, float]:
+    """Train LM on clean tokens; evaluate on clean + noisy eval token streams."""
+    train_tokens, _ = tokens_from_lines(train_lines, merges, pre_mode, lowercase)
+    eval_tokens, eval_lengths = tokens_from_lines(eval_lines, merges, pre_mode, lowercase)
+    eval_tokens_noisy, eval_lengths_noisy = tokens_from_lines_noisy(
+        eval_lines,
+        merges,
+        pre_mode,
+        lowercase,
+        noise_prob=noise_prob,
+        noise_mode=noise_mode,
+        noise_seed=noise_seed,
+    )
+    return train_and_eval_lm_from_tokens_robust(
+        train_tokens=train_tokens,
+        eval_tokens=eval_tokens,
+        eval_lengths=eval_lengths,
+        eval_tokens_noisy=eval_tokens_noisy,
+        eval_lengths_noisy=eval_lengths_noisy,
+        eval_bytes=eval_bytes,
+        epochs=epochs,
+        batch_size=batch_size,
+        block_size=block_size,
+        n_embd=n_embd,
+        n_head=n_head,
+        n_layer=n_layer,
+        lr=lr,
+        seed=seed,
+    )
+
+
+def train_and_eval_lm_sp_robust(
+    train_lines: Sequence[str],
+    eval_lines: Sequence[str],
+    sp,
+    pre_mode: str,
+    lowercase: bool,
+    eval_bytes: int,
+    epochs: int,
+    batch_size: int,
+    block_size: int,
+    n_embd: int,
+    n_head: int,
+    n_layer: int,
+    lr: float,
+    seed: int,
+    noise_prob: float,
+    noise_mode: str,
+    noise_seed: int,
+) -> Tuple[float, float, float, float, float]:
+    train_tokens, _ = tokens_from_lines_sp(train_lines, sp, pre_mode, lowercase)
+    eval_tokens, eval_lengths = tokens_from_lines_sp(eval_lines, sp, pre_mode, lowercase)
+    eval_tokens_noisy, eval_lengths_noisy = tokens_from_lines_sp_noisy(
+        eval_lines,
+        sp,
+        pre_mode,
+        lowercase,
+        noise_prob=noise_prob,
+        noise_mode=noise_mode,
+        noise_seed=noise_seed,
+    )
+    return train_and_eval_lm_from_tokens_robust(
+        train_tokens=train_tokens,
+        eval_tokens=eval_tokens,
+        eval_lengths=eval_lengths,
+        eval_tokens_noisy=eval_tokens_noisy,
+        eval_lengths_noisy=eval_lengths_noisy,
+        eval_bytes=eval_bytes,
+        epochs=epochs,
+        batch_size=batch_size,
+        block_size=block_size,
+        n_embd=n_embd,
+        n_head=n_head,
+        n_layer=n_layer,
+        lr=lr,
+        seed=seed,
+    )
+
+
 def train_and_eval_lm_from_tokens(
     train_tokens: Sequence[str],
     eval_tokens: Sequence[str],
@@ -1259,6 +1430,143 @@
 
     return bpb, avg_seq_len, train_time
 
 
+
+def train_and_eval_lm_from_tokens_robust(
+    train_tokens: Sequence[str],
+    eval_tokens: Sequence[str],
+    eval_lengths: Sequence[int],
+    eval_tokens_noisy: Sequence[str],
+    eval_lengths_noisy: Sequence[int],
+    eval_bytes: int,
+    epochs: int,
+    batch_size: int,
+    block_size: int,
+    n_embd: int,
+    n_head: int,
+    n_layer: int,
+    lr: float,
+    seed: int,
+) -> Tuple[float, float, float, float, float]:
+    """Identical to train_and_eval_lm_from_tokens, but evaluates the same trained model on a noisy eval stream."""
+    try:
+        import torch
+        import torch.nn as nn
+        import torch.nn.functional as F
+    except Exception as e:
+        raise RuntimeError("Torch is required for --train_lm. Install: pip install torch") from e
+
+    avg_seq_len = (sum(eval_lengths) / len(eval_lengths)) if eval_lengths else 0.0
+    avg_seq_len_noisy = (sum(eval_lengths_noisy) / len(eval_lengths_noisy)) if eval_lengths_noisy else 0.0
+
+    if len(train_tokens) < block_size + 1:
+        raise RuntimeError("Not enough training tokens for the requested --lm_block_size.")
+
+    vocab, stoi = build_vocab(train_tokens)
+    unk_id = stoi["<unk>"]
+    train_ids = np.array([stoi.get(t, unk_id) for t in train_tokens], dtype=np.int64)
+    eval_ids = np.array([stoi.get(t, unk_id) for t in eval_tokens], dtype=np.int64)
+    eval_ids_noisy = np.array([stoi.get(t, unk_id) for t in eval_tokens_noisy], dtype=np.int64)
+
+    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
+    torch.manual_seed(seed)
+    random.seed(seed)
+
+    class MiniTransformerLM(nn.Module):
+        def __init__(self, vocab_size: int):
+            super().__init__()
+            self.token_emb = nn.Embedding(vocab_size, n_embd)
+            self.pos_emb = nn.Embedding(block_size, n_embd)
+            enc_layer = nn.TransformerEncoderLayer(
+                d_model=n_embd,
+                nhead=n_head,
+                dim_feedforward=4 * n_embd,
+                dropout=0.1,
+                batch_first=True,
+            )
+            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
+            self.ln = nn.LayerNorm(n_embd)
+            self.head = nn.Linear(n_embd, vocab_size, bias=False)
+
+        def forward(self, idx):
+            bsz, t = idx.shape
+            pos = torch.arange(t, device=idx.device).unsqueeze(0)
+            x = self.token_emb(idx) + self.pos_emb(pos)
+            mask = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()
+            x = self.encoder(x, mask)
+            x = self.ln(x)
+            return self.head(x)
+
+    def eval_total_loss(model: nn.Module, ids: np.ndarray, eval_block_size: int) -> float:
+        if eval_block_size <= 0:
+            return float("nan")
+        total = 0.0
+        model.eval()
+        with torch.no_grad():
+            for x_np, y_np in iter_lm_batches(ids, eval_block_size, batch_size, None):
+                x = torch.from_numpy(x_np).to(device)
+                y = torch.from_numpy(y_np).to(device)
+                logits = model(x)
+                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
+                total += float(loss.item())
+        return total
+
+    model = MiniTransformerLM(len(vocab)).to(device)
+    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
+
+    # ---- Train ----
+    model.train()
+    t0 = time.perf_counter()
+    for epoch in range(epochs):
+        rng = random.Random(seed + epoch)
+        for x_np, y_np in iter_lm_batches(train_ids, block_size, batch_size, rng):
+            x = torch.from_numpy(x_np).to(device)
+            y = torch.from_numpy(y_np).to(device)
+            logits = model(x)
+            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
+            optimizer.zero_grad(set_to_none=True)
+            loss.backward()
+            optimizer.step()
+    train_time = time.perf_counter() - t0
+
+    # ---- Eval (clean) ----
+    eval_block_size = min(block_size, len(eval_ids) - 1) if len(eval_ids) > 1 else 0
+    total_loss = eval_total_loss(model, eval_ids, eval_block_size)
+
+    # ---- Eval (noisy) ----
+    eval_block_size_noisy = min(block_size, len(eval_ids_noisy) - 1) if len(eval_ids_noisy) > 1 else 0
+    total_loss_noisy = eval_total_loss(model, eval_ids_noisy, eval_block_size_noisy)
+
+    if eval_bytes <= 0:
+        bpb = float("nan")
+        bpb_noisy = float("nan")
+    else:
+        bpb = (total_loss / float(eval_bytes)) / math.log(2.0)
+        bpb_noisy = (total_loss_noisy / float(eval_bytes)) / math.log(2.0)
+
+    return bpb, bpb_noisy, avg_seq_len, avg_seq_len_noisy, train_time
+
+
 def print_lm_table(lm_results: Dict[str, Tuple[float, float, float]]) -> None:
@@ -1294,6 +1602,45 @@
     print("-" * max(56, len(header)))
 
 
+
+def print_lm_robust_table(
+    lm_clean: Dict[str, Tuple[float, float, float]],
+    lm_noisy: Dict[str, Tuple[float, float, float]],
+) -> None:
+    """Print robustness metrics: noisy BPB and degradation relative to clean."""
+    methods = [m for m in ALLOWED_METHODS if (m in lm_clean and m in lm_noisy)]
+    if not methods:
+        print("\n== LM robustness (noisy eval) ==\n(no methods selected)")
+        return
+
+    print("\n== LM robustness (noisy eval) ==")
+    header = f"{'Metric':24s} | " + " | ".join(f"{METHOD_LABELS[m]:>12s}" for m in methods)
+    print(header)
+    print("-" * max(56, len(header)))
+
+    def row_vals(name: str, vals: List[str]):
+        print(f"{name:24s} | " + " | ".join(f"{v:>12s}" for v in vals))
+
+    bpb_noisy_vals = [f"{lm_noisy[m][0]:.6f}" for m in methods]
+    row_vals("BPB (eval noisy)", bpb_noisy_vals)
+
+    deg_vals = []
+    for m in methods:
+        clean = lm_clean[m][0]
+        noisy = lm_noisy[m][0]
+        if not (math.isfinite(clean) and math.isfinite(noisy)) or clean <= 0:
+            deg_vals.append("nan")
+        else:
+            deg_vals.append(f"{100.0 * (noisy / clean - 1.0):.2f}")
+    row_vals("ΔBPB noisy-clean (%)", deg_vals)
+
+    toks_noisy_vals = [f"{lm_noisy[m][1]:.2f}" for m in methods]
+    row_vals("Avg tokens/sent (noisy)", toks_noisy_vals)
+
+    print("-" * max(56, len(header)))
+
+
 # ---------- Main ----------
@@ -1325,6 +1672,16 @@
     ap.add_argument("--lm_layers", type=int, default=2)
     ap.add_argument("--lm_lr", type=float, default=3e-4)
+
+    # Optional downstream robustness test: evaluate the same trained LM on a noisy eval set
+    ap.add_argument("--lm_robust_eval", action="store_true",
+                    help="Also evaluate the trained LM on a noisy version of eval text (robustness)")
+    ap.add_argument("--lm_noise_mode", choices=["swap"], default="swap",
+                    help="Noise type for robust eval (default: swap = adjacent character swap within a word)")
+    ap.add_argument("--lm_noise_prob", type=float, default=0.10,
+                    help="Probability of corrupting each pre-tokenized word in robust eval (default: 0.10)")
+    ap.add_argument("--lm_noise_seed", type=int, default=None,
+                    help="Seed for noise generation (default: seed+12345)")
@@ -1502,51 +1859,118 @@
-    lm_results: Dict[str, Tuple[float, float, float]] = {}
-    lm_eval_bytes = None
-    if args.train_lm:
-        if args.max_eval_lines is None:
-            try:
-                eval_bytes = len(open(args.eval_text, "rb").read())
-            except Exception:
-                eval_bytes = sum(len((line + "\n").encode("utf-8", errors="ignore")) for line in eval_lines)
-        else:
-            eval_bytes = sum(len((line + "\n").encode("utf-8", errors="ignore")) for line in eval_lines)
-        lm_eval_bytes = eval_bytes
-
-        if do_bpe and bpe_merges is not None:
-            print("\n[train] LM on BPE tokens...", file=sys.stderr)
-            bpe_lm = train_and_eval_lm(
-                train_lines,
-                eval_lines,
-                bpe_merges,
-                args.pretokenize,
-                args.lowercase,
-                eval_bytes=eval_bytes,
-                epochs=args.lm_epochs,
-                batch_size=args.lm_batch_size,
-                block_size=args.lm_block_size,
-                n_embd=args.lm_dim,
-                n_head=args.lm_heads,
-                n_layer=args.lm_layers,
-                lr=args.lm_lr,
-                seed=args.seed,
-            )
-            lm_results["bpe"] = bpe_lm
-
-        if do_sp and sp_merges is not None:
-            print("[train] LM on SpectralBPE tokens...", file=sys.stderr)
-            sp_lm = train_and_eval_lm(
-                train_lines,
-                eval_lines,
-                sp_merges,
-                args.pretokenize,
-                args.lowercase,
-                eval_bytes=eval_bytes,
-                epochs=args.lm_epochs,
-                batch_size=args.lm_batch_size,
-                block_size=args.lm_block_size,
-                n_embd=args.lm_dim,
-                n_head=args.lm_heads,
-                n_layer=args.lm_layers,
-                lr=args.lm_lr,
-                seed=args.seed,
-            )
-            lm_results["spectralbpe"] = sp_lm
-
-        if do_uni and sp_model is not None:
-            print("[train] LM on Unigram tokens...", file=sys.stderr)
-            uni_lm = train_and_eval_lm_sp(
-                train_lines,
-                eval_lines,
-                sp_model,
-                args.pretokenize,
-                args.lowercase,
-                eval_bytes=eval_bytes,
-                epochs=args.lm_epochs,
-                batch_size=args.lm_batch_size,
-                block_size=args.lm_block_size,
-                n_embd=args.lm_dim,
-                n_head=args.lm_heads,
-                n_layer=args.lm_layers,
-                lr=args.lm_lr,
-                seed=args.seed,
-            )
-            lm_results["unigram"] = uni_lm
-
-        if lm_results:
-            print_lm_table(lm_results)
+    lm_results: Dict[str, Tuple[float, float, float]] = {}
+    lm_noisy_results: Dict[str, Tuple[float, float, float]] = {}
+    lm_eval_bytes = None
+    lm_noise_cfg: Optional[Dict[str, Any]] = None
+    if args.train_lm:
+        if args.max_eval_lines is None:
+            try:
+                eval_bytes = len(open(args.eval_text, "rb").read())
+            except Exception:
+                eval_bytes = sum(len((line + "\n").encode("utf-8", errors="ignore")) for line in eval_lines)
+        else:
+            eval_bytes = sum(len((line + "\n").encode("utf-8", errors="ignore")) for line in eval_lines)
+        lm_eval_bytes = eval_bytes
+
+        noise_seed = args.lm_noise_seed if args.lm_noise_seed is not None else (args.seed + 12345)
+        if args.lm_robust_eval:
+            lm_noise_cfg = {"mode": args.lm_noise_mode, "prob": float(args.lm_noise_prob), "seed": int(noise_seed)}
+
+        if do_bpe and bpe_merges is not None:
+            print("\n[train] LM on BPE tokens...", file=sys.stderr)
+            if args.lm_robust_eval:
+                bpb, bpb_noisy, asl, asl_noisy, tsec = train_and_eval_lm_robust(
+                    train_lines,
+                    eval_lines,
+                    bpe_merges,
+                    args.pretokenize,
+                    args.lowercase,
+                    eval_bytes=eval_bytes,
+                    epochs=args.lm_epochs,
+                    batch_size=args.lm_batch_size,
+                    block_size=args.lm_block_size,
+                    n_embd=args.lm_dim,
+                    n_head=args.lm_heads,
+                    n_layer=args.lm_layers,
+                    lr=args.lm_lr,
+                    seed=args.seed,
+                    noise_prob=args.lm_noise_prob,
+                    noise_mode=args.lm_noise_mode,
+                    noise_seed=noise_seed,
+                )
+                lm_results["bpe"] = (bpb, asl, tsec)
+                lm_noisy_results["bpe"] = (bpb_noisy, asl_noisy, tsec)
+            else:
+                bpe_lm = train_and_eval_lm(
+                    train_lines,
+                    eval_lines,
+                    bpe_merges,
+                    args.pretokenize,
+                    args.lowercase,
+                    eval_bytes=eval_bytes,
+                    epochs=args.lm_epochs,
+                    batch_size=args.lm_batch_size,
+                    block_size=args.lm_block_size,
+                    n_embd=args.lm_dim,
+                    n_head=args.lm_heads,
+                    n_layer=args.lm_layers,
+                    lr=args.lm_lr,
+                    seed=args.seed,
+                )
+                lm_results["bpe"] = bpe_lm
+
+        if do_sp and sp_merges is not None:
+            print("[train] LM on SpectralBPE tokens...", file=sys.stderr)
+            if args.lm_robust_eval:
+                bpb, bpb_noisy, asl, asl_noisy, tsec = train_and_eval_lm_robust(
+                    train_lines,
+                    eval_lines,
+                    sp_merges,
+                    args.pretokenize,
+                    args.lowercase,
+                    eval_bytes=eval_bytes,
+                    epochs=args.lm_epochs,
+                    batch_size=args.lm_batch_size,
+                    block_size=args.lm_block_size,
+                    n_embd=args.lm_dim,
+                    n_head=args.lm_heads,
+                    n_layer=args.lm_layers,
+                    lr=args.lm_lr,
+                    seed=args.seed,
+                    noise_prob=args.lm_noise_prob,
+                    noise_mode=args.lm_noise_mode,
+                    noise_seed=noise_seed,
+                )
+                lm_results["spectralbpe"] = (bpb, asl, tsec)
+                lm_noisy_results["spectralbpe"] = (bpb_noisy, asl_noisy, tsec)
+            else:
+                sp_lm = train_and_eval_lm(
+                    train_lines,
+                    eval_lines,
+                    sp_merges,
+                    args.pretokenize,
+                    args.lowercase,
+                    eval_bytes=eval_bytes,
+                    epochs=args.lm_epochs,
+                    batch_size=args.lm_batch_size,
+                    block_size=args.lm_block_size,
+                    n_embd=args.lm_dim,
+                    n_head=args.lm_heads,
+                    n_layer=args.lm_layers,
+                    lr=args.lm_lr,
+                    seed=args.seed,
+                )
+                lm_results["spectralbpe"] = sp_lm
+
+        if do_uni and sp_model is not None:
+            print("[train] LM on Unigram tokens...", file=sys.stderr)
+            if args.lm_robust_eval:
+                bpb, bpb_noisy, asl, asl_noisy, tsec = train_and_eval_lm_sp_robust(
+                    train_lines,
+                    eval_lines,
+                    sp_model,
+                    args.pretokenize,
+                    args.lowercase,
+                    eval_bytes=eval_bytes,
+                    epochs=args.lm_epochs,
+                    batch_size=args.lm_batch_size,
+                    block_size=args.lm_block_size,
+                    n_embd=args.lm_dim,
+                    n_head=args.lm_heads,
+                    n_layer=args.lm_layers,
+                    lr=args.lm_lr,
+                    seed=args.seed,
+                    noise_prob=args.lm_noise_prob,
+                    noise_mode=args.lm_noise_mode,
+                    noise_seed=noise_seed,
+                )
+                lm_results["unigram"] = (bpb, asl, tsec)
+                lm_noisy_results["unigram"] = (bpb_noisy, asl_noisy, tsec)
+            else:
+                uni_lm = train_and_eval_lm_sp(
+                    train_lines,
+                    eval_lines,
+                    sp_model,
+                    args.pretokenize,
+                    args.lowercase,
+                    eval_bytes=eval_bytes,
+                    epochs=args.lm_epochs,
+                    batch_size=args.lm_batch_size,
+                    block_size=args.lm_block_size,
+                    n_embd=args.lm_dim,
+                    n_head=args.lm_heads,
+                    n_layer=args.lm_layers,
+                    lr=args.lm_lr,
+                    seed=args.seed,
+                )
+                lm_results["unigram"] = uni_lm
+
+        if lm_results:
+            print_lm_table(lm_results)
+            if args.lm_robust_eval and lm_noisy_results:
+                print_lm_robust_table(lm_results, lm_noisy_results)
@@ -1559,6 +1983,16 @@
         if lm_eval_bytes is not None:
             lm_payload = {"eval_bytes": lm_eval_bytes}
+            if lm_noise_cfg is not None:
+                lm_payload['noise'] = lm_noise_cfg
+            if lm_noisy_results:
+                if 'bpe' in lm_noisy_results:
+                    lm_payload['bpe_bpb_noisy'] = lm_noisy_results['bpe'][0]
+                    lm_payload['bpe_avg_seq_len_noisy'] = lm_noisy_results['bpe'][1]
+                if 'spectralbpe' in lm_noisy_results:
+                    lm_payload['spectral_bpb_noisy'] = lm_noisy_results['spectralbpe'][0]
+                    lm_payload['spectral_avg_seq_len_noisy'] = lm_noisy_results['spectralbpe'][1]
+                if 'unigram' in lm_noisy_results:
+                    lm_payload['unigram_bpb_noisy'] = lm_noisy_results['unigram'][0]
+                    lm_payload['unigram_avg_seq_len_noisy'] = lm_noisy_results['unigram'][1]
             if "bpe" in lm_results:
                 lm_payload["bpe_bpb"] = lm_results["bpe"][0]
                 lm_payload["bpe_avg_seq_len"] = lm_results["bpe"][1]
```

---

## Patch: make the gamma sweep actually run the robustness eval

Your existing sweep script already iterates gammas and writes logs. 
This patch just adds the robustness flags.

```diff
--- sweep_ppmi_gamma.sh
+++ sweep_ppmi_gamma.sh
@@ -59,7 +59,8 @@
     echo "  --bpe_warmstart $BPE_WARMSTART \\"
     echo "  --seed $SEED --deterministic_ties \\"
     echo "  --checkpoint_dir $outdir --checkpoint_every $CHECKPOINT_EVERY \\"
-    echo "  --train_lm"
+    echo "  --train_lm \\"
+    echo "  --lm_robust_eval --lm_noise_mode swap --lm_noise_prob 0.10"
   } > "$cmdfile"
@@ -77,6 +78,7 @@
     --seed "$SEED" --deterministic_ties \
     --checkpoint_dir "$outdir" --checkpoint_every "$CHECKPOINT_EVERY" \
     --train_lm \
+    --lm_robust_eval --lm_noise_mode swap --lm_noise_prob 0.10 \
     >"$logfile" 2>&1
 }
```

---

## Patch: update `parse_and_plot_gamma.py` to also produce `robust_gamma.pdf` + correlation

This adds parsing for `BPB (eval noisy)` and produces a second plot + prints Pearson corr.

```diff
--- parse_and_plot_gamma.py
+++ parse_and_plot_gamma.py
@@ -3,11 +3,14 @@
 import csv
 from pathlib import Path
 import matplotlib.pyplot as plt
+import math
+import numpy as np
@@
 RE_BPB  = re.compile(r"^BPB \(eval\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
+RE_BPB_NOISY  = re.compile(r"^BPB \(eval noisy\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
@@
     bpe_bpb = spec_bpb = None
+    bpe_bpb_noisy = spec_bpb_noisy = None
@@
         m = RE_BPB.match(line)
         if m:
             bpe_bpb = float(m.group(1))
             spec_bpb = float(m.group(2))
+            continue
+
+        m = RE_BPB_NOISY.match(line)
+        if m:
+            bpe_bpb_noisy = float(m.group(1))
+            spec_bpb_noisy = float(m.group(2))
             continue
@@
     return {
@@
         "ppmi_gain_pct": 100.0 * (spec_ppmi / bpe_ppmi - 1.0),
+        "bpe_bpb_noisy": (bpe_bpb_noisy if bpe_bpb_noisy is not None else float("nan")),
+        "spec_bpb_noisy": (spec_bpb_noisy if spec_bpb_noisy is not None else float("nan")),
+        "bpe_noise_increase_pct": (100.0 * (bpe_bpb_noisy / bpe_bpb - 1.0) if (bpe_bpb_noisy is not None and bpe_bpb) else float("nan")),
+        "spec_noise_increase_pct": (100.0 * (spec_bpb_noisy / spec_bpb - 1.0) if (spec_bpb_noisy is not None and spec_bpb) else float("nan")),
+        "robust_delta_pct": (
+            (100.0 * (spec_bpb_noisy / spec_bpb - 1.0) if (spec_bpb_noisy is not None and spec_bpb) else float("nan"))
+            - (100.0 * (bpe_bpb_noisy / bpe_bpb - 1.0) if (bpe_bpb_noisy is not None and bpe_bpb) else float("nan"))
+        ),
         "log_path": str(path),
     }
@@
-    fieldnames = list(rows[0].keys())
+    fieldnames = sorted({k for r in rows for k in r.keys()})
@@
     fig.savefig(out_pdf, bbox_inches="tight")
     print(f"[ok] wrote {out_pdf}")
+
+    # ---- Robustness Plot (optional; requires --lm_robust_eval logs) ----
+    rrows = [r for r in rows if math.isfinite(r.get("robust_delta_pct", float("nan")))]
+    if rrows:
+        xs_r = [r["ppmi_gain_pct"] for r in rrows]
+        ys_r = [r["robust_delta_pct"] for r in rrows]
+
+        fig2, ax2 = plt.subplots(figsize=(7.2, 5.0))
+        ax2.scatter(xs_r, ys_r)
+        ax2.plot(xs_r, ys_r, linestyle="--", linewidth=1.0, color="tab:orange")
+        ax2.axhline(0.0, linewidth=1)
+        ax2.grid(True, alpha=0.25)
+
+        ax2.set_xlabel(r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%) $\uparrow$")
+        ax2.set_ylabel(r"Robustness delta $\Delta_{\mathrm{rob}}$ (%) (more negative is better)")
+        ax2.set_title(r"Robustness under word-level swap noise (p=0.10)")
+
+        for i, r in enumerate(rrows):
+            ax2.annotate(
+                rf"${r['ppmi_gamma']:.2f}$",
+                (r["ppmi_gain_pct"], r["robust_delta_pct"]),
+                fontsize=12,
+                xytext=(0, 3),
+                textcoords="offset points",
+                ha="right",
+                va="bottom",
+            )
+
+        fig2.tight_layout()
+        out_pdf2 = ROOT / "robust_gamma.pdf"
+        fig2.savefig(out_pdf2, bbox_inches="tight")
+        print(f"[ok] wrote {out_pdf2}")
+
+        corr = float(np.corrcoef(np.array(xs_r), np.array(ys_r))[0, 1]) if len(xs_r) >= 2 else float("nan")
+        print(f"[robust] corr(ppmi_gain_pct, robust_delta_pct) = {corr:.4f}")
```

---

## How this changes your “run everything” flow

* **Knee run** now can produce:

  * Table 1 metrics (cohesion, BPB, tokens/sent)
  * **PLUS** robustness metrics (`BPB (eval noisy)`, `ΔBPB noisy-clean (%)`)
* **Gamma sweep** now produces:

  * `pareto_gamma.pdf` (existing)
  * `robust_gamma.pdf` (new)
  * plus a printed correlation between cohesion gain and robustness delta

And importantly: it doesn’t require any external dataset download.

---

## If you want one sentence to drop into the paper (so this isn’t just code)

Add a short subsection like:

> *Robustness to typographical noise.* We train the same small LM on each tokenizer’s clean token stream and evaluate on a noisy evaluation set where each pre-tokenized word undergoes an adjacent character swap with probability (p=0.1). We report BPB on noisy text and the relative degradation (\Delta)BPB(noisy–clean). Across the (\gamma) sweep, we also report the correlation between cohesion gain and robustness delta.

That’s exactly what the new pipeline will produce.

---

If you want, I can also update your “run everything” script so it copies `pareto_sweep_gamma/robust_gamma.pdf` into `figs/` alongside `pareto_gamma.pdf` and fails loudly if robustness lines are missing.

