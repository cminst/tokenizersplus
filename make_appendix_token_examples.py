#!/usr/bin/env python3
import argparse
import json
import random
import re
import urllib.request
from pathlib import Path
import importlib.util
from typing import List, Tuple, Dict, Optional

DEFAULT_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice (UTF-8)

# ---------- utilities ----------

def load_text(url: Optional[str], path: Optional[str]) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    if not url:
        raise ValueError("Need either --text_path or --text_url")
    with urllib.request.urlopen(url) as f:
        return f.read().decode("utf-8", errors="ignore")

def strip_gutenberg_header_footer(text: str) -> str:
    # Many Gutenberg files contain START/END markers. We keep only the body.
    start = re.search(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.I)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.I)
    if end:
        text = text[:end.start()]
    return text

def split_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", text)
    paras = []
    for p in parts:
        p = " ".join(p.strip().split())
        if p:
            paras.append(p)
    return paras

def is_good_paragraph(p: str, min_chars: int, max_chars: int) -> bool:
    if len(p) < min_chars or len(p) > max_chars:
        return False
    # Skip headings / chapter markers / metadata-ish lines.
    if re.match(r"^(chapter|CHAPTER)\b", p):
        return False
    if re.match(r"^\[", p):
        return False
    # Require mostly alphabetic/punctuation (avoid tables/garbage)
    alpha = sum(ch.isalpha() for ch in p)
    if alpha / max(1, len(p)) < 0.60:
        return False
    return True

def latex_escape(s: str) -> str:
    # Escape LaTeX special chars.
    repl = {
        "\\": r"\textbackslash{}",
        "{": r"\{", "}": r"\}",
        "#": r"\#", "$": r"\$", "%": r"\%",
        "&": r"\&", "_": r"\_",
        "^": r"\^{}", "~": r"\~{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)

def load_merges(json_path: str) -> List[Tuple[str, str]]:
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    m = d.get("merges") or d.get("bpe_merges") or d.get("merge_pairs") or d.get("merge_table")
    if m is None:
        raise KeyError(f"Could not find merges list in {json_path}")
    out: List[Tuple[str, str]] = []
    for x in m:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            out.append((x[0], x[1]))
        else:
            a, b = str(x).split()
            out.append((a, b))
    return out

# ---------- rendering ----------

def render_word_tokens(tokens: List[str], red: bool) -> str:
    # Token-level macros to allow wrapping:
    # \tok{...} \tsep \tok{...} ...
    macro = r"\tokd" if red else r"\tok"
    parts = []
    for i, t in enumerate(tokens):
        parts.append(f"{macro}{{{latex_escape(t)}}}")
        if i + 1 < len(tokens):
            parts.append(r"\tsep")
    return "".join(parts)

def render_line(words: List[str], toks_per_word: List[List[str]], red_mask: List[bool]) -> str:
    # Render a full paragraph tokenization line with per-word spacing.
    out = []
    for w, toks, red in zip(words, toks_per_word, red_mask):
        if not toks:
            continue
        out.append(render_word_tokens(toks, red))
        out.append(r"\wsep")
    return "".join(out).rstrip()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectral_script", default="spectralbpe_sanity_v3.py")
    ap.add_argument("--bpe_json", required=True)
    ap.add_argument("--spectral_json", required=True)

    ap.add_argument("--text_url", default=DEFAULT_URL)
    ap.add_argument("--text_path", default=None)
    ap.add_argument("--strip_gutenberg", action="store_true", default=True)

    ap.add_argument("--n_paragraphs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pretokenize", choices=["whitespace", "basic"], default="whitespace")
    ap.add_argument("--lowercase", action="store_true")

    ap.add_argument("--min_chars", type=int, default=300)
    ap.add_argument("--max_chars", type=int, default=650)
    ap.add_argument("--max_words", type=int, default=120)

    # To avoid “everything red”, keep paragraphs with moderate diff fraction
    ap.add_argument("--min_diff_frac", type=float, default=0.05)
    ap.add_argument("--max_diff_frac", type=float, default=0.30)

    # Only mark a word red if tokenizations differ AND token-count delta >= threshold
    ap.add_argument("--diff_threshold", type=int, default=1)

    ap.add_argument("--out_tex", default="appendix_tokenization_examples.tex")
    ap.add_argument("--title", default="Qualitative tokenization examples (public-domain text)")
    ap.add_argument("--source_note", default="Text excerpts sampled from Project Gutenberg eBook #1342 (Pride and Prejudice).")

    args = ap.parse_args()

    # Import your tokenizer code
    spec = importlib.util.spec_from_file_location("sb", args.spectral_script)
    sb = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(sb)

    bpe_merges = load_merges(args.bpe_json)
    sp_merges = load_merges(args.spectral_json)

    rank_b = {p: i for i, p in enumerate(bpe_merges)}
    rank_s = {p: i for i, p in enumerate(sp_merges)}

    text = load_text(args.text_url, args.text_path)
    if args.strip_gutenberg:
        text = strip_gutenberg_header_footer(text)

    paras = [p for p in split_paragraphs(text) if is_good_paragraph(p, args.min_chars, args.max_chars)]
    if not paras:
        raise SystemExit("No paragraphs passed the filters. Try lowering --min_chars or disabling --strip_gutenberg.")

    rng = random.Random(args.seed)

    # Scan paragraphs and keep those with moderate diff fraction
    candidates = []
    for p in paras[:600]:  # cap scan for speed
        line = p.lower() if args.lowercase else p
        words = sb.pretokenize(line, args.pretokenize)
        if len(words) < 20:
            continue
        words = words[:args.max_words]

        b_toks = [sb.encode_word(w, rank_b) for w in words]
        s_toks = [sb.encode_word(w, rank_s) for w in words]

        red_mask = []
        diffs = 0
        for bt, st in zip(b_toks, s_toks):
            is_diff = (bt != st) and (abs(len(bt) - len(st)) >= args.diff_threshold)
            red_mask.append(is_diff)
            diffs += int(is_diff)

        frac = diffs / max(1, len(words))
        if args.min_diff_frac <= frac <= args.max_diff_frac:
            candidates.append((frac, p, words, b_toks, s_toks, red_mask))

    if len(candidates) < args.n_paragraphs:
        # fallback: just sample from all good paras
        candidates = []
        for p in paras[:800]:
            line = p.lower() if args.lowercase else p
            words = sb.pretokenize(line, args.pretokenize)[:args.max_words]
            if len(words) < 20:
                continue
            b_toks = [sb.encode_word(w, rank_b) for w in words]
            s_toks = [sb.encode_word(w, rank_s) for w in words]
            red_mask = [(bt != st) and (abs(len(bt) - len(st)) >= args.diff_threshold) for bt, st in zip(b_toks, s_toks)]
            frac = sum(red_mask) / max(1, len(words))
            candidates.append((frac, p, words, b_toks, s_toks, red_mask))

    rng.shuffle(candidates)
    chosen = candidates[:args.n_paragraphs]

    out = []
    out.append("% Auto-generated. Do not edit by hand.\n")
    out.append(r"\begingroup" + "\n")
    out.append(r"\setlength{\parskip}{0.6em}" + "\n")
    out.append(r"\setlength{\parindent}{0pt}" + "\n")
    out.append(r"\newcommand{\tok}[1]{\texttt{#1}}" + "\n")
    out.append(r"\newcommand{\tokd}[1]{\textcolor{red}{\texttt{#1}}}" + "\n")
    out.append(r"\newcommand{\tsep}{\texttt{|}\allowbreak}" + "\n")
    out.append(r"\newcommand{\wsep}{\hspace{0.35em}}" + "\n\n")
    out.append(r"\subsection{" + latex_escape(args.title) + "}\n")
    out.append(r"\textit{" + latex_escape(args.source_note) + "}\n\n")

    for i, (frac, p, words, b_toks, s_toks, red_mask) in enumerate(chosen, 1):
        out.append(r"\paragraph{Example " + str(i) + r".} " +
                   r"\emph{Red highlights indicate words whose tokenization differs between methods. "
                   r"(diff-frac=" + f"{frac:.2f}" + r")}" + "\n")
        out.append(r"\begin{quote}\small " + latex_escape(p) + r"\end{quote}" + "\n")
        out.append(r"\textbf{BPE:}\\")
        out.append(r"{\footnotesize " + render_line(words, b_toks, red_mask) + "}\n\n")
        out.append(r"\textbf{SpectralBPE:}\\")
        out.append(r"{\footnotesize " + render_line(words, s_toks, red_mask) + "}\n\n")

    out.append(r"\endgroup" + "\n")

    Path(args.out_tex).write_text("".join(out), encoding="utf-8")
    print(f"[ok] wrote {args.out_tex}")

if __name__ == "__main__":
    main()
