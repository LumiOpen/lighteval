#!/usr/bin/env python3
"""Build/publish the BFCL datasets used by the ``bfcl_*`` tasks.

Reads BFCL's raw prompt files (``data/BFCL_v4_<cat>.json``) and ground-truth files
(``data/possible_answer/BFCL_v4_<cat>.json``) from a gorilla checkout, joins them by
``id``, and writes one JSONL per category plus a merged JSONL for the collection.
Complex fields (``question``/``function``/``ground_truth``) are stored as JSON
strings so the Arrow loader never has to unify their heterogeneous nested schemas;
the task's ``record_to_sample`` calls ``json.loads`` on them.

Two collections: ``nonlive`` (the 7 AST categories) and ``live`` (real user-contributed
data: 4 AST categories + relevance/irrelevance). AST categories carry ground truth;
relevance/irrelevance do not (scored by whether a call is/isn't emitted).

With ``--push-to-hub`` it uploads the collection as a multi-config HF dataset
(one config per category + a merged config named after the collection, split ``train``).

Usage:
  python build_data.py --bfcl-root /path/to/berkeley-function-call-leaderboard \\
      --collection nonlive --out-dir ./bfcl_data
  python build_data.py --bfcl-root ... --collection live --push-to-hub ezosa/bfcl
"""

import argparse
import json
import os


# AST-scored non-live categories (plus `irrelevance`, which has no ground truth).
NON_LIVE_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]
# Live (real user-contributed) categories: 4 AST + relevance/irrelevance (no ground truth).
LIVE_CATEGORIES = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_relevance",
    "live_irrelevance",
]
COLLECTIONS = {"nonlive": NON_LIVE_CATEGORIES, "live": LIVE_CATEGORIES}
VERSION_PREFIX = "BFCL_v4"


def _read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_category(bfcl_root, category):
    data_dir = os.path.join(bfcl_root, "bfcl_eval", "data")
    prompts = _read_jsonl(os.path.join(data_dir, f"{VERSION_PREFIX}_{category}.json"))
    answer_path = os.path.join(data_dir, "possible_answer", f"{VERSION_PREFIX}_{category}.json")
    answers = {}
    if os.path.exists(answer_path):
        answers = {r["id"]: r.get("ground_truth") for r in _read_jsonl(answer_path)}

    rows = []
    for entry in prompts:
        gt = answers.get(entry["id"])
        rows.append(
            {
                "id": entry["id"],
                "category": category,
                "question": json.dumps(entry["question"], ensure_ascii=False),
                "function": json.dumps(entry["function"], ensure_ascii=False),
                "ground_truth": json.dumps(gt, ensure_ascii=False) if gt is not None else None,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-root", required=True, help="path to berkeley-function-call-leaderboard")
    ap.add_argument("--collection", choices=sorted(COLLECTIONS), default="nonlive")
    ap.add_argument("--out-dir", default="./bfcl_data")
    ap.add_argument("--push-to-hub", metavar="REPO", default=None, help="e.g. ezosa/bfcl")
    args = ap.parse_args()

    categories = COLLECTIONS[args.collection]
    os.makedirs(args.out_dir, exist_ok=True)
    per_category = {}
    merged = []
    for category in categories:
        rows = build_category(args.bfcl_root, category)
        per_category[category] = rows
        merged.extend(rows)
        with open(os.path.join(args.out_dir, f"bfcl_{category}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{category:22s} {len(rows):5d} rows")
    with open(os.path.join(args.out_dir, f"bfcl_{args.collection}.jsonl"), "w") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{'bfcl_' + args.collection + ' (merged)':22s} {len(merged):5d} rows")

    if args.push_to_hub:
        from datasets import Dataset

        for category, rows in per_category.items():
            Dataset.from_list(rows).push_to_hub(args.push_to_hub, config_name=category, split="train")
            print(f"pushed config={category}")
        Dataset.from_list(merged).push_to_hub(args.push_to_hub, config_name=args.collection, split="train")
        print(f"pushed config={args.collection} -> {args.push_to_hub}")


if __name__ == "__main__":
    main()
