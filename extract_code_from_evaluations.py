#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any, Iterable, Optional, Tuple

DEFAULT_OUT_DIR = "extracted_code_after_evaluated"

# Supported filename patterns:
#   {model}-{dataset}-{technique}-merged.jsonl
#   {model}-{dataset}-{technique}-merged_sanitized.jsonl
#   {model}-{dataset}-{technique}-merged_eval_results.json
#   {model}-{dataset}-{technique}-merged_sanitized_eval_results.json
#   results/{model}-{dataset}-{technique}.jsonl


def parse_meta_from_filename(filename: str) -> Optional[Tuple[str, str, str, bool]]:
    """Parse (model, dataset, technique, is_sanitized) from filename.

    Returns None if not matched.
    """
    base = os.path.basename(filename)
    is_sanitized = "sanitized" in base

    suffixes = [
        "-merged_sanitized_eval_results.json",
        "-merged_eval_results.json",
    ]
    matched = False
    for suf in suffixes:
        if base.endswith(suf):
            base = base[: -len(suf)]
            matched = True
            break
    if not matched:
        return None

    try:
        model, dataset, technique = base.rsplit("-", 2)
        return model, dataset, technique, is_sanitized
    except ValueError:
        return None


def iter_records_from_file(path: str) -> Iterable[Tuple[str, str]]:
    """Yield (task_id, solution) pairs from a JSONL or JSON eval file.

    - JSONL: each line has {"task_id": str, "solution": str}
    - JSON:  has {"eval": {task_id: [{"solution": str, ...}, ...]}}
    """
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = obj.get("task_id")
                solution = obj.get("solution")
                if isinstance(task_id, str) and isinstance(solution, str):
                    yield task_id, solution
    elif path.endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        eval_block = data.get("eval", {})
        if isinstance(eval_block, dict):
            for task_id, entries in eval_block.items():
                if not isinstance(entries, list) or not entries:
                    continue
                # take the first entry
                entry = entries[0]
                solution = entry.get("solution")
                if isinstance(task_id, str) and isinstance(solution, str):
                    yield task_id, solution


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def task_id_to_filename(task_id: str) -> str:
    # Expect format like "BigCodeBench/1040" -> "1040.py"
    part = task_id.split("/")[-1]
    # Fallback if not clean numeric
    if not part:
        part = task_id.replace("/", "_")
    return f"{part}.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract solution code to files")
    parser.add_argument(
        "input",
        help="Path to a JSONL/JSON file or a directory containing such files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Base directory to write files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--only-bigcodebench",
        action="store_true",
        help="Process only tasks whose task_id contains BigCodeBench (recommended)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )

    args = parser.parse_args()

    inputs: Iterable[str]
    if os.path.isdir(args.input):
        inputs = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith((".jsonl", ".json"))
        ]
    else:
        inputs = [args.input]

    total_files = 0
    total_written = 0

    for path in sorted(inputs):
        meta = parse_meta_from_filename(path)
        if not meta:
            # Skip files we cannot parse into (model, dataset, technique)
            continue
        model, dataset, technique, is_sanitized = meta

        for task_id, solution in iter_records_from_file(path):
            if args.only_bigcodebench and "bigcodebench" not in task_id.lower():
                continue

            out_dir = os.path.join(args.output_dir, model, dataset, technique)
            ensure_dir(out_dir)
            out_file = os.path.join(out_dir, task_id_to_filename(task_id))

            total_files += 1
            if os.path.exists(out_file) and not args.overwrite:
                continue

            try:
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(solution)
                total_written += 1
            except Exception as e:
                print(f"Failed to write {out_file}: {e}")

    print(f"Processed inputs: {len(list(inputs))}")
    print(f"Total code records seen: {total_files}")
    print(f"Files written: {total_written}")
    print(f"Output base: {args.output_dir}")


if __name__ == "__main__":
    main()
