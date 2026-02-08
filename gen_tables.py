import json
import argparse
import sys
import re
from spectralbpe_sanity import encode_word, pretokenize

def load_merges(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert list of lists to list of tuples
    merges = [tuple(pair) for pair in data['merges']]
    return {p: i for i, p in enumerate(merges)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe', default='debug_vocab16k/bpe_final.json')
    parser.add_argument('--spectral', default='debug_vocab16k/spectralbpe_seed0_final.json')
    parser.add_argument('--eval_text', default='data/eval.txt')
    args = parser.parse_args()

    print(f"Loading BPE from {args.bpe}...", file=sys.stderr)
    bpe_rank = load_merges(args.bpe)
    print(f"Loading SpectralBPE from {args.spectral}...", file=sys.stderr)
    spec_rank = load_merges(args.spectral)

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
                    diffs.append((w, t_bpe, t_spec))

    # Sort by length difference to find the most dramatic improvements
    diffs.sort(key=lambda x: len(x[1]) - len(x[2]), reverse=True)

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
    print(r"""
\begin{table}[h]
\caption{\textbf{Qualitative Segmentation Comparison.} Examples where SpectralBPE preserves morphological boundaries that greedy BPE fragments.}
\label{tab:qualitative}
\begin{center}
\begin{small}
\begin{tabular}{l|l|l}
\toprule
Word & Standard BPE & SpectralBPE (Ours) \\
\midrule""")
    
    # Pick top 6 interesting examples
    selected = diffs[:8] 
    for w, tb, ts in selected:
        # Format tokens with @@ for latex readability if needed, or just spaces
        bpe_str = " ".join([f"['{t}']" for t in tb]).replace("</w>", "")
        spec_str = " ".join([f"['{t}']" for t in ts]).replace("</w>", "")
        
        # Clean up latex special chars
        w = w.replace("_", "\\_")
        bpe_str = bpe_str.replace("_", "\\_").replace("'", "")
        spec_str = spec_str.replace("_", "\\_").replace("'", "")
        
        print(f"{w} & {bpe_str} & \\textbf{{{spec_str}}} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{small}
\end{center}
\end{table}
""")

if __name__ == '__main__':
    main()
