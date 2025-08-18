#!/usr/bin/env python3
import os
import json
import csv
from typing import Dict, Set, Tuple, List

ORIGINAL_BASE = "analysis_results/original"
SANITIZED_BASE = "analysis_results/evaluated"

OUTPUT_CSV = "pylint_set_differences.csv"
SUMMARY_TXT = "pylint_set_differences_summary.txt"


def latest_analysis_dir(base_dir: str) -> str:
    candidates = [d for d in os.listdir(base_dir) if d.startswith("analysis_")]
    if not candidates:
        raise FileNotFoundError(f"No analysis_* directory under {base_dir}")
    return os.path.join(base_dir, sorted(candidates)[-1])


essential_fields = [
    "model",
    "dataset",
    "technique",
    "task_id",
    "status",
    "orig_present",
    "san_present",
    "orig_code_path",
    "san_code_path",
]


def load_all_results(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_key(key: str) -> Tuple[str, str]:
    # key format: "dataset/technique"
    parts = key.split("/")
    if len(parts) == 2:
        return parts[0], parts[1]
    # Fallback
    return key, ""


def compare_pylint_sets() -> Tuple[List[Dict[str, str]], List[str]]:
    orig_dir = latest_analysis_dir(ORIGINAL_BASE)
    san_dir = latest_analysis_dir(SANITIZED_BASE)

    orig_all_path = os.path.join(orig_dir, "all_results.json")
    san_all_path = os.path.join(san_dir, "all_results.json")

    orig_all = load_all_results(orig_all_path)
    san_all = load_all_results(san_all_path)

    rows: List[Dict[str, str]] = []
    summary_lines: List[str] = []

    # Iterate over union of models
    models = set(orig_all.get("results", {}).keys()) | set(
        san_all.get("results", {}).keys()
    )

    for model in sorted(models):
        orig_model = orig_all.get("results", {}).get(model, {})
        san_model = san_all.get("results", {}).get(model, {})

        keys = set(orig_model.keys()) | set(san_model.keys())  # dataset/technique keys
        for key in sorted(keys):
            dataset, technique = split_key(key)
            orig_res = orig_model.get(key, {})
            san_res = san_model.get(key, {})

            orig_pylint = orig_res.get("pylint", {}) or {}
            san_pylint = san_res.get("pylint", {}) or {}

            orig_tasks: Set[str] = set(orig_pylint.keys())
            san_tasks: Set[str] = set(san_pylint.keys())

            # Improvements: present only after sanitization (were missing before -> likely syntax errors originally)
            improvements = sorted(san_tasks - orig_tasks)
            # Regressions: present originally but missing after sanitization
            regressions = sorted(orig_tasks - san_tasks)

            if improvements or regressions:
                summary_lines.append(
                    f"{model}/{dataset}/{technique}: +{len(improvements)} improvements, -{len(regressions)} regressions (orig {len(orig_tasks)} -> san {len(san_tasks)})"
                )

            for task_id in improvements:
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "technique": technique,
                        "task_id": task_id,
                        "status": "improved",
                        "orig_present": "no",
                        "san_present": "yes",
                        "orig_code_path": os.path.join(
                            "extracted_code", model, dataset, technique, f"{task_id}.py"
                        ),
                        "san_code_path": os.path.join(
                            "extracted_code_sanitized",
                            model,
                            dataset,
                            technique,
                            f"{task_id}.py",
                        ),
                    }
                )

            for task_id in regressions:
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "technique": technique,
                        "task_id": task_id,
                        "status": "regressed",
                        "orig_present": "yes",
                        "san_present": "no",
                        "orig_code_path": os.path.join(
                            "extracted_code", model, dataset, technique, f"{task_id}.py"
                        ),
                        "san_code_path": os.path.join(
                            "extracted_code_sanitized",
                            model,
                            dataset,
                            technique,
                            f"{task_id}.py",
                        ),
                    }
                )

    return rows, summary_lines


def main() -> None:
    rows, summary_lines = compare_pylint_sets()

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=essential_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Write summary
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nTotal rows: {len(rows)}\n")

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
