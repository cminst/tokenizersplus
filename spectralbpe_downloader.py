import argparse
from pathlib import Path
from datasets import load_dataset

def write_split(ds, split_name: str, out_path: str, max_lines: int | None):
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds[split_name]:
            text = ex.get("text", "")
            if text is None:
                continue
            # Keep lines "as text" but drop empty lines to reduce noise
            text = text.replace("\r", "").strip("\n")
            if text.strip() == "":
                continue
            f.write(text + "\n")
            n += 1
            if max_lines is not None and n >= max_lines:
                break
    print(f"[ok] wrote {n} lines to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="wikitext-103-raw-v1",
                    help="wikitext-103-raw-v1 (bigger) or wikitext-2-raw-v1 (smaller)")
    ap.add_argument("--max_train_lines", type=int, default=None)
    ap.add_argument("--max_eval_lines", type=int, default=None)
    ap.add_argument("--data_root", default="data",
                    help="Directory to place output files when paths are relative")
    ap.add_argument("--out_train", default="train.txt")
    ap.add_argument("--out_eval", default="eval.txt")
    args = ap.parse_args()

    def resolve_out(path_str: str, data_root: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = Path(data_root).expanduser() / path
        return path.resolve()

    out_train = resolve_out(args.out_train, args.data_root)
    out_eval = resolve_out(args.out_eval, args.data_root)

    # Ensure output directory exists.
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_eval.parent.mkdir(parents=True, exist_ok=True)

    # Dataset is hosted under Salesforce/wikitext on HF
    ds = load_dataset("Salesforce/wikitext", args.config)  # splits: train, validation, test

    write_split(ds, "train", str(out_train), args.max_train_lines)
    write_split(ds, "validation", str(out_eval), args.max_eval_lines)

if __name__ == "__main__":
    main()
