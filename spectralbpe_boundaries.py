import argparse
import json
import sys

from tabulate import tabulate

END_WORD = "</w>"


def load_merges(path):
    merges = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Handle both list-of-lists and list-of-strings formats
        raw = data.get("merges", []) if isinstance(data, dict) else data
        for pair in raw:
            if isinstance(pair, list):
                if len(pair) == 2:
                    merges.append((pair[0], pair[1]))
            else:
                # Parse "a b" string format if necessary
                parts = str(pair).split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
    return merges


def encode(word, rank):
    # Standard BPE encoding logic
    if not word:
        return []
    toks = list(word)
    toks[-1] += END_WORD
    while True:
        min_rank = float("inf")
        best_pair = None
        for i in range(len(toks) - 1):
            pair = (toks[i], toks[i + 1])
            if pair in rank:
                r = rank[pair]
                if r < min_rank:
                    min_rank = r
                    best_pair = pair
        if best_pair is None:
            break

        # Merge
        new_toks = []
        i = 0
        while i < len(toks):
            if i < len(toks) - 1 and (toks[i], toks[i + 1]) == best_pair:
                new_toks.append(toks[i] + toks[i + 1])
                i += 2
            else:
                new_toks.append(toks[i])
                i += 1
        toks = new_toks
    return toks


def check_boundary(tokens, gold_split_index):
    # Success Case 1: The tokenizer found the whole word (e.g. ['rainforests'])
    if len(tokens) == 1:
        return True

    # Success Case 2: The tokenizer split exactly at the gold index
    current_len = 0
    for t in tokens:
        clean_t = t.replace('</w>', '')
        current_len += len(clean_t)
        if current_len == gold_split_index:
            return True
        if current_len > gold_split_index:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--spectral", required=True)
    args = ap.parse_args()

    # Load tokenizers
    bpe_rank = {p: i for i, p in enumerate(load_merges(args.bpe))}
    spec_rank = {p: i for i, p in enumerate(load_merges(args.spectral))}

    # A tiny dataset of compound words and their "Golden Split" index
    # Format: (Word, Index where the split SHOULD be)
    # e.g. "rainforests": "rain" (4) | "forests"
    compounds = [
        ("rainforests", 4),
        ("waterfall", 5),
        ("blackboard", 5),
        ("playground", 4),
        ("sunflower", 3),
        ("moonlight", 4),
        ("cheeseburger", 6),
        ("firefighter", 4),
        ("newspapers", 4),
        ("grandmother", 5),
        ("underground", 5),
        ("skyscraper", 3),
        ("bedroom", 3),
        ("toothbrush", 5),
        ("haircut", 4),
        ("airport", 3),
        ("notebook", 4),
        ("superman", 5),
        ("interchange", 5),
        ("background", 4),
        ("subway", 3),
        ("output", 3),
        ("wildlife", 4),
        ("outside", 3),
        ("inside", 2),
        ("without", 4),
        ("within", 4),
        ("outcome", 3),
        ("income", 2),
        ("upstream", 2),
        ("downstream", 4),
        ("midstream", 3),
        ("overcome", 4),
        ("understand", 5),
        ("software", 4),
        ("hardware", 4),
        ("spacewalk", 5),
        ("starfish", 4),
        ("jellyfish", 5),
        ("lifeguard", 4),
        ("classmate", 5),
        ("teammate", 4),
        ("roommate", 4),
        ("soulmate", 4),
        ("become", 2),
        ("became", 2),
        ("because", 2),
        ("becoming", 2),
    ]

    rows = []

    bpe_hits = 0
    spec_hits = 0

    for word, split_idx in compounds:
        t_bpe = encode(word, bpe_rank)
        t_spec = encode(word, spec_rank)

        ok_bpe = check_boundary(t_bpe, split_idx)
        ok_spec = check_boundary(t_spec, split_idx)

        if ok_bpe:
            bpe_hits += 1
        if ok_spec:
            spec_hits += 1

        # Only keep rows if they differ (to save space)
        if ok_bpe != ok_spec:
            rows.append([word, str(t_bpe), str(t_spec), ok_bpe, ok_spec])

    if rows:
        print(
            tabulate(
                rows,
                headers=["Word", "BPE Tokens", "Spectral Tokens", "BPE_Ok", "Spec_Ok"],
                tablefmt="github",
            )
        )
    else:
        print("(No differences found.)")

    print()
    print(
        tabulate(
            [
                [
                    "Boundary Accuracy (BPE)",
                    f"{bpe_hits}/{len(compounds)}",
                    f"{(bpe_hits/len(compounds)):.2%}",
                ],
                [
                    "Boundary Accuracy (Spectral)",
                    f"{spec_hits}/{len(compounds)}",
                    f"{(spec_hits/len(compounds)):.2%}",
                ],
            ],
            headers=["Metric", "Hits", "Rate"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
