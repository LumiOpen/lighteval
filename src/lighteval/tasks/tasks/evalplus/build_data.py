#!/usr/bin/env python3
"""Build/publish the EvalPlus datasets used by the ``humaneval_plus`` / ``mbpp_plus`` tasks.

Reads the canonical EvalPlus HF datasets (``evalplus/humanevalplus`` and
``evalplus/mbppplus``, split ``test``) and normalizes both into a single uniform
schema so the task's ``record_to_sample`` stays trivial and the Arrow loader never
has to unify heterogeneous fields:

  kind             "humaneval" | "mbpp"
  task_id          original task id
  prompt           instruct source text (HumanEval: signature+docstring;
                   MBPP: natural-language description)
  entry_point      function name the model must define (derived from ``code`` for MBPP)
  test             EvalPlus ``check(candidate)`` harness (the augmented "+" tests)
  example          MBPP only: one example assert (conveys the signature); "" otherwise
  canonical_solution  reference implementation (for local sanity checks / tests)

With ``--push-to-hub`` it uploads a multi-config HF dataset (config ``humaneval_plus``
and ``mbpp_plus``, split ``train``).

Usage:
  python build_data.py --collection humaneval_plus --out-dir ./evalplus_data
  python build_data.py --collection mbpp_plus --push-to-hub ezosa/evalplus
  python build_data.py --collection all --push-to-hub ezosa/evalplus
"""

import argparse
import json
import os
import re


_SOURCES = {
    "humaneval_plus": ("evalplus/humanevalplus", "test"),
    "mbpp_plus": ("evalplus/mbppplus", "test"),
}


def _entry_point_from_code(code: str) -> str:
    m = re.findall(r"^\s*def\s+(\w+)\s*\(", code, re.MULTILINE)
    return m[-1] if m else ""


def build_collection(collection: str) -> list:
    from datasets import load_dataset

    repo, split = _SOURCES[collection]
    ds = load_dataset(repo, split=split)
    rows = []
    if collection == "humaneval_plus":
        for r in ds:
            rows.append(
                {
                    "kind": "humaneval",
                    "task_id": r["task_id"],
                    "prompt": r["prompt"],
                    "entry_point": r["entry_point"],
                    "test": r["test"],
                    "example": "",
                    "canonical_solution": r["prompt"] + r["canonical_solution"],
                }
            )
    else:  # mbpp_plus
        for r in ds:
            entry = _entry_point_from_code(r.get("code", ""))
            test_list = r.get("test_list") or []
            example = test_list[0] if test_list else ""
            rows.append(
                {
                    "kind": "mbpp",
                    "task_id": str(r["task_id"]),
                    "prompt": r["prompt"],
                    "entry_point": entry,
                    "test": r["test"],
                    "example": example,
                    "canonical_solution": r.get("code", ""),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", choices=[*sorted(_SOURCES), "all"], default="all")
    ap.add_argument("--out-dir", default="./evalplus_data")
    ap.add_argument("--push-to-hub", metavar="REPO", default=None, help="e.g. ezosa/evalplus")
    args = ap.parse_args()

    collections = sorted(_SOURCES) if args.collection == "all" else [args.collection]
    os.makedirs(args.out_dir, exist_ok=True)
    for collection in collections:
        rows = build_collection(collection)
        with open(os.path.join(args.out_dir, f"{collection}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        missing = [r["task_id"] for r in rows if not r["entry_point"]]
        print(f"{collection:16s} {len(rows):5d} rows" + (f"  WARN missing entry_point: {missing}" if missing else ""))
        if args.push_to_hub:
            from datasets import Dataset

            Dataset.from_list(rows).push_to_hub(args.push_to_hub, config_name=collection, split="train")
            print(f"pushed config={collection} -> {args.push_to_hub}")


if __name__ == "__main__":
    main()
