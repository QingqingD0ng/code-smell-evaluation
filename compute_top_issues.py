import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def iter_issues(entry: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield all issue-like dicts from a single problem entry.

    The JSON shows two containers: "fatal_errors" (list) and "other_issues" (list).
    We count both to reflect the total issues per model+dataset, if present.
    """
    for key in ("fatal_errors", "other_issues"):
        items = entry.get(key)
        if isinstance(items, list):
            for issue in items:
                if isinstance(issue, dict):
                    yield issue


def descend_and_iter_issues(blob: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively walk nested dicts until finding entries that contain
    fatal_errors/other_issues, then yield those issues.
    """
    if not isinstance(blob, dict):
        return
    # If this dict looks like an entry with issues, yield them
    if "fatal_errors" in blob or "other_issues" in blob:
        yield from iter_issues(blob)
        return
    # Otherwise, descend into values
    for value in blob.values():
        if isinstance(value, dict):
            yield from descend_and_iter_issues(value)


def load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError("Unexpected JSON structure: missing top-level 'results'")
    return data["results"]


def _extract_prompt(dataset_key: str) -> str:
    """Return the prompt part from dataset key like 'bigcodebench/baseline'."""
    if not isinstance(dataset_key, str):
        return str(dataset_key)
    parts = dataset_key.split("/", 1)
    return parts[1] if len(parts) == 2 else dataset_key


def _extract_dataset_root(dataset_key: str) -> str:
    """Return the dataset root part before the slash, e.g., 'bigcodebench'."""
    if not isinstance(dataset_key, str):
        return str(dataset_key)
    parts = dataset_key.split("/", 1)
    return parts[0]


def aggregate_issues(
    results: Dict[str, Any], group_by: str = "dataset"
) -> Dict[Tuple[str, str], Counter]:
    """
    Build a mapping: (model, dataset) -> Counter of issue symbols.
    Prefer the 'symbol' field; fall back to 'message-id' or 'message' if missing.
    """
    agg: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for model, model_blob in results.items():
        if not isinstance(model_blob, dict):
            continue
        for dataset, dataset_blob in model_blob.items():
            if not isinstance(dataset_blob, dict):
                continue
            # dataset_blob may contain extra levels (e.g., linter name -> problem_id -> entry)
            for issue in descend_and_iter_issues(dataset_blob):
                symbol = (
                    issue.get("symbol")
                    or issue.get("message-id")
                    or issue.get("message")
                    or "<unknown>"
                )
                if not isinstance(symbol, str):
                    symbol = str(symbol)
                if group_by == "prompt":
                    key = (model, _extract_prompt(dataset))
                elif group_by == "dataset_root":
                    key = (model, _extract_dataset_root(dataset))
                else:
                    # default: group by dataset (original behavior)
                    key = (model, dataset)
                agg[key][symbol] += 1

    return agg


def compute_top_k(
    agg: Dict[Tuple[str, str], Counter], k: int
) -> List[Tuple[str, str, str, int, float]]:
    """
    For each (model, dataset), compute top-k issues and their frequency.
    Returns list of rows: (model, dataset, issue_symbol, count, frequency)
    where frequency = count / total_issues for that model+dataset.
    """
    rows: List[Tuple[str, str, str, int, float]] = []
    for (model, dataset_or_prompt), counter in agg.items():
        total = sum(counter.values())
        if total == 0:
            continue
        for issue_symbol, count in counter.most_common(k):
            freq = count / total
            rows.append((model, dataset_or_prompt, issue_symbol, count, freq))
    return rows


def write_csv(
    rows: List[Tuple[str, str, str, int, float]], out_path: Path, second_col_name: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", second_col_name, "issue_symbol", "count", "frequency"]
        )
        for model, second, issue_symbol, count, freq in rows:
            writer.writerow([model, second, issue_symbol, count, f"{freq:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute top-K issues per (model, dataset) from analysis JSON and output CSV."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to all_results.json",
    )
    parser.add_argument(
        "--group-by",
        choices=["dataset", "prompt", "dataset_root"],
        default="dataset",
        help=(
            "Aggregate by dataset (default), by prompt (merge datasets), or by dataset_root (before slash)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top issues to include per (model, dataset). Default: 5",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis_results/top_issues_per_model_dataset.csv"),
        help="Output CSV path",
    )

    args = parser.parse_args()

    results = load_results(args.json_path)
    agg = aggregate_issues(results, group_by=args.group_by)
    rows = compute_top_k(agg, args.top_k)
    second_col_name = (
        "prompt"
        if args.group_by == "prompt"
        else ("dataset_root" if args.group_by == "dataset_root" else "dataset")
    )
    write_csv(rows, args.out, second_col_name)

    # Also print a brief summary to stdout for quick inspection
    # Group back by (model, dataset)
    grouped: Dict[Tuple[str, str], List[Tuple[str, str, int, float]]] = defaultdict(
        list
    )
    for model, dataset, issue_symbol, count, freq in rows:
        grouped[(model, dataset)].append((issue_symbol, model, count, freq))

    label_name = (
        "Prompt"
        if args.group_by == "prompt"
        else ("Dataset root" if args.group_by == "dataset_root" else "Dataset")
    )
    for (model, label), items in sorted(grouped.items()):
        print(f"\nModel: {model} | {label_name}: {label}")
        for issue_symbol, _m, count, freq in items:
            print(f"  - {issue_symbol}: {count} ({freq:.2%})")
    print(f"\nWrote CSV to: {args.out}")


if __name__ == "__main__":
    main()
