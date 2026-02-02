import argparse
import json
import os
import sys
import urllib.request

from tabulate import tabulate

END_WORD = "</w>"
DICT_URL = "https://raw.githubusercontent.com/david47k/top-english-wordlists/master/top_english_words_lower_100000.txt"
DICT_PATH = os.path.join("data", "dictionary.txt")


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


def ensure_dictionary(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(DICT_URL) as resp:
        raw = resp.read().decode("utf-8", errors="ignore").splitlines()
    words = []
    for w in raw:
        w = w.strip().lower()
        if len(w) >= 3:
            words.append(w)
    with open(path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")
    print(f"[download] wrote {path} ({len(words)} words)", file=sys.stderr)


def load_dictionary(path: str) -> set:
    ensure_dictionary(path)
    with open(path, "r", encoding="utf-8") as f:
        return set(x.strip().lower() for x in f if x.strip())


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

    vocab = load_dictionary(DICT_PATH)
    print(f"[dict] loaded {len(vocab)} words from {DICT_PATH}", file=sys.stderr)

    compounds = []

    # Algorithm: O(N * L) where L is max word length. Very fast.
    # Iterate through every word in the dictionary.
    # Check every possible split point.
    # If left_part is a word AND right_part is a word -> Match.
    for w in vocab:
        if len(w) < 10:
            continue  # User constraint: Total length > 10
        if not w.isalpha():
            continue

        # Check all splits
        # We need subparts to be at least 3 chars
        # So split index goes from 3 to len(w)-3
        for i in range(3, len(w) - 3):
            head = w[:i]
            tail = w[i:]

            if head in vocab and tail in vocab:
                # Found a valid compound!
                compounds.append((w, len(head)))
                # Break to avoid duplicates (e.g. some words might split multiple ways)
                break

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
