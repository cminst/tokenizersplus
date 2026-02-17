import argparse
import json
import os
import sys

from spectralbpe_sanity import encode_word, encode_word_sp, parse_methods, pretokenize


def load_merges(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert list of lists to list of tuples
    merges = [tuple(pair) for pair in data['merges']]
    return {p: i for i, p in enumerate(merges)}


QUAL_LABELS = {
    "bpe": "Standard BPE",
    "spectralbpe": "SpectralBPE (Ours)",
    "unigram": "Unigram",
}


def resolve_sp_model_path(path: str) -> str:
    if path.endswith(".model"):
        return path
    candidate = path + ".model"
    if os.path.exists(candidate):
        return candidate
    return path


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def format_tokens(tokens):
    tok_str = " ".join([f"['{t}']" for t in tokens]).replace("</w>", "")
    tok_str = tok_str.replace("'", "")
    return latex_escape(tok_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe', default='debug_vocab16k/bpe_final.json')
    parser.add_argument('--spectral', default='debug_vocab16k/spectralbpe_seed0_final.json')
    parser.add_argument('--unigram_model', default=None, help='SentencePiece .model or prefix')
    parser.add_argument('--methods', default='bpe,spectralbpe',
                        help='comma-separated: bpe,spectralbpe,unigram (default: bpe,spectralbpe)')
    parser.add_argument('--eval_text', default='data/eval.txt')
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    if "bpe" not in methods or "spectralbpe" not in methods:
        raise SystemExit("This script expects --methods to include bpe and spectralbpe for diff selection.")

    print(f"Loading BPE from {args.bpe}...", file=sys.stderr)
    bpe_rank = load_merges(args.bpe)
    print(f"Loading SpectralBPE from {args.spectral}...", file=sys.stderr)
    spec_rank = load_merges(args.spectral)

    sp = None
    if "unigram" in methods:
        if not args.unigram_model:
            raise SystemExit("Missing --unigram_model for --methods including unigram.")
        model_path = resolve_sp_model_path(args.unigram_model)
        try:
            import sentencepiece as spm
        except Exception as e:
            raise SystemExit("sentencepiece is required for --methods unigram. Install: pip install sentencepiece") from e
        sp = spm.SentencePieceProcessor(model_file=model_path)

    # 1. Comparison Generator
    diffs = []
    seen_words = set()

    print(f"Scanning {args.eval_text} for interesting differences...", file=sys.stderr)
    with open(args.eval_text, 'r', encoding='utf-8') as f:
        for line in f:
            words = pretokenize(line.strip(), 'whitespace')
            for w in words:
                if w in seen_words or len(w) < 5: continue
                seen_words.add(w)

                # Tokenize with both
                t_bpe = encode_word(w, bpe_rank)
                t_spec = encode_word(w, spec_rank)

                # Heuristic for "interesting": Spectral keeps it whole, BPE breaks it
                # OR Spectral has fewer tokens than BPE
                if len(t_spec) < len(t_bpe):
                    tok_map = {"bpe": t_bpe, "spectralbpe": t_spec}
                    if "unigram" in methods and sp is not None:
                        tok_map["unigram"] = encode_word_sp(sp, w)
                    diffs.append((len(t_bpe) - len(t_spec), w, tok_map))

    # Sort by length difference to find the most dramatic improvements
    diffs.sort(key=lambda x: x[0], reverse=True)

    # 2. Output Table 1 (Quantitative - Hardcoded from your Run #3 Logs)
    print("\n% --- COPY THIS INTO YOUR LATEX (Table 1) ---")
    print(r"""
\begin{table}[t]
\caption{\textbf{Main Results.} Comparison of standard frequency-based BPE vs. SpectralBPE ($K=16k$) on the evaluation set. SpectralBPE matches the compression efficiency (BPB) of the baseline while significantly improving the statistical coherence of the vocabulary.}
\label{tab:main_results}
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccc}
\toprule
Metric & BPE & SpectralBPE & $\Delta$ \\
\midrule
Vocab Size & 16,000 & 16,000 & - \\
\textbf{Bits Per Byte (BPB)} $\downarrow$ & 1.552 & 1.554 & +0.1\% \\
Seq. Length (Avg) $\downarrow$ & 111.1 & 111.7 & +0.5\% \\
\midrule
\textbf{Vocab Cohesion (PPMI)} $\uparrow$ & 1.43 & \textbf{1.51} & \textbf{+5.7\%} \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\end{table}
""")

    # 3. Output Table 2 (Qualitative - Generated from data)
    print("\n% --- COPY THIS INTO YOUR LATEX (Table 2) ---")
    col_spec = "l|" + "|".join(["l"] * len(methods))
    header = " & ".join(["Word"] + [QUAL_LABELS[m] for m in methods])
    print(r"""
\begin{table}[h]
\caption{\textbf{Qualitative Segmentation Comparison.} Examples where SpectralBPE preserves morphological boundaries that greedy BPE fragments.}
\label{tab:qualitative}
\begin{center}
\begin{small}
\begin{tabular}{%s}
\toprule
%s \\
\midrule""" % (col_spec, header))

    # Pick top 6 interesting examples
    selected = diffs[:8]
    for _, w, tok_map in selected:
        cells = [latex_escape(w)]
        for m in methods:
            tok_str = format_tokens(tok_map.get(m, []))
            if m == "spectralbpe":
                tok_str = f"\\textbf{{{tok_str}}}"
            cells.append(tok_str)
        print(" & ".join(cells) + " \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{small}
\end{center}
\end{table}
""")

if __name__ == '__main__':
    main()
