#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SweepConfig:
    tau: int
    alpha: float
    sigma: float
    beta: float
    bpe_warmstart: int


def parse_list_int(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_list_float(raw: Optional[str]) -> List[float]:
    if raw is None:
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def fmt_float(value: float) -> str:
    # Keep stable filenames: 0.4 -> 0p4, 1.0 -> 1p0, 0.01 -> 0p01
    s = f"{value:.6g}"
    return s.replace(".", "p")


def run_id(idx: int, cfg: SweepConfig) -> str:
    return (
        f"r{idx:04d}_"
        f"tau{cfg.tau}_alpha{fmt_float(cfg.alpha)}_sigma{fmt_float(cfg.sigma)}_"
        f"beta{fmt_float(cfg.beta)}_bpe{cfg.bpe_warmstart}"
    )


def build_cmd(
    python: str,
    script: str,
    base_args: Sequence[str],
    cfg: SweepConfig,
    out_json: str,
) -> List[str]:
    return [
        python,
        script,
        *base_args,
        "--tau",
        str(cfg.tau),
        "--alpha",
        str(cfg.alpha),
        "--sigma",
        str(cfg.sigma),
        "--beta",
        str(cfg.beta),
        "--bpe_warmstart",
        str(cfg.bpe_warmstart),
        "--out_json",
        out_json,
    ]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_one(
    idx: int,
    cfg: SweepConfig,
    python: str,
    script: str,
    base_args: Sequence[str],
    out_dir: str,
    env: Dict[str, str],
) -> Dict[str, str]:
    rid = run_id(idx, cfg)
    out_json = os.path.join(out_dir, "json", f"{rid}.json")
    out_log = os.path.join(out_dir, "logs", f"{rid}.out")
    err_log = os.path.join(out_dir, "logs", f"{rid}.err")
    cmd = build_cmd(python, script, base_args, cfg, out_json)

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    dt = time.perf_counter() - t0

    with open(out_log, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    with open(err_log, "w", encoding="utf-8") as f:
        f.write(proc.stderr)

    return {
        "run_id": rid,
        "returncode": str(proc.returncode),
        "out_json": out_json,
        "out_log": out_log,
        "err_log": err_log,
        "seconds": f"{dt:.3f}",
        "tau": str(cfg.tau),
        "alpha": str(cfg.alpha),
        "sigma": str(cfg.sigma),
        "beta": str(cfg.beta),
        "bpe_warmstart": str(cfg.bpe_warmstart),
    }


def read_metrics(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    bpe = payload.get("metrics", {}).get("bpe", {})
    sp = payload.get("metrics", {}).get("spectralbpe", {})
    out = {}
    for k, v in bpe.items():
        out[f"bpe_{k}"] = v
    for k, v in sp.items():
        out[f"spec_{k}"] = v
    if "tokens" in bpe and "tokens" in sp and bpe["tokens"]:
        out["spec_nsl_vs_bpe"] = float(sp["tokens"]) / float(bpe["tokens"])
    return out


def write_summary(rows: List[Dict[str, str]], out_csv: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_print_table(rows: List[Dict[str, str]], columns: Sequence[str]) -> None:
    try:
        from tabulate import tabulate
    except Exception:
        return
    table = [[row.get(c, "") for c in columns] for row in rows]
    print(tabulate(table, headers=list(columns), tablefmt="github"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_text", required=True)
    ap.add_argument("--eval_text", required=True)
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--pretokenize", choices=["whitespace", "basic"], default="whitespace")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--max_train_lines", type=int, default=None)
    ap.add_argument("--max_eval_lines", type=int, default=None)
    ap.add_argument("--max_merges", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=None, help="Parallel jobs (default: all CPU cores)")
    ap.add_argument("--threads_per_job", type=int, default=1, help="BLAS/OpenMP threads per job")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--script", type=str, default="spectralbpe_sanity.py")

    # Sweep lists (comma-separated). Defaults match the user request.
    ap.add_argument("--tau", type=str, default="5,10,15")
    ap.add_argument("--alpha", type=str, default="0.4,0.6,0.8")
    ap.add_argument("--sigma", type=str, default="0.4,0.6,0.8,1.0")
    ap.add_argument("--beta", type=str, default="0.01,0.1,0.15,0.2")
    ap.add_argument("--bpe_warmstart", type=str, default="100,200,300,400")

    args = ap.parse_args()

    tau_list = parse_list_int(args.tau)
    alpha_list = parse_list_float(args.alpha)
    sigma_list = parse_list_float(args.sigma)
    beta_list = parse_list_float(args.beta)
    bpe_list = parse_list_int(args.bpe_warmstart)

    if not (tau_list and alpha_list and sigma_list and beta_list and bpe_list):
        raise SystemExit("All sweep lists must be non-empty.")

    jobs = args.jobs or os.cpu_count() or 1
    out_dir = args.out_dir or os.path.join("sweeps", time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "json"))
    ensure_dir(os.path.join(out_dir, "logs"))

    base_args = [
        "--train_text",
        args.train_text,
        "--eval_text",
        args.eval_text,
        "--vocab_size",
        str(args.vocab_size),
        "--batch_size",
        str(args.batch_size),
        "--pretokenize",
        args.pretokenize,
    ]
    if args.lowercase:
        base_args.append("--lowercase")
    if args.max_train_lines is not None:
        base_args.extend(["--max_train_lines", str(args.max_train_lines)])
    if args.max_eval_lines is not None:
        base_args.extend(["--max_eval_lines", str(args.max_eval_lines)])
    if args.max_merges is not None:
        base_args.extend(["--max_merges", str(args.max_merges)])

    env = os.environ.copy()
    threads = str(max(1, int(args.threads_per_job)))
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[k] = threads

    configs = [
        SweepConfig(t, a, s, b, w)
        for t, a, s, b, w in product(tau_list, alpha_list, sigma_list, beta_list, bpe_list)
    ]

    print(f"[sweep] jobs={jobs} total_runs={len(configs)} out_dir={out_dir}")

    results: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = [
            ex.submit(run_one, i, cfg, args.python, args.script, base_args, out_dir, env)
            for i, cfg in enumerate(configs, start=1)
        ]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "ok" if res["returncode"] == "0" else "fail"
            print(f"[{status}] {res['run_id']} ({res['seconds']}s)")

    # Merge metrics into summary rows
    summary_rows: List[Dict[str, str]] = []
    for res in results:
        row = dict(res)
        if res["returncode"] == "0" and os.path.exists(res["out_json"]):
            try:
                metrics = read_metrics(res["out_json"])
                for k, v in metrics.items():
                    row[k] = f"{v:.6f}" if isinstance(v, float) else str(v)
            except Exception:
                row["metrics_error"] = "1"
        summary_rows.append(row)

    summary_csv = os.path.join(out_dir, "summary.csv")
    write_summary(summary_rows, summary_csv)
    print(f"[sweep] wrote {summary_csv}")

    # Optional table preview
    preview_cols = [
        "run_id",
        "tau",
        "alpha",
        "sigma",
        "beta",
        "bpe_warmstart",
        "spec_nsl_vs_bpe",
        "spec_bytes_per_token",
        "spec_tokens_per_byte",
        "spec_fertility",
    ]
    maybe_print_table(summary_rows, preview_cols)


if __name__ == "__main__":
    main()
