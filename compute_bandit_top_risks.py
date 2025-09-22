import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_bandit(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError("Unexpected Bandit JSON: missing 'results' list")
    return data


def parse_path_fields(
    filename: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to extract (model, dataset_root, prompt) from a Bandit filename path.
    Examples:
      ./extracted_code_bigcodebench/qwen/bigcodebench/persona/100.py
      ./extracted_code/phi-3/codereval/baseline/124.py
    Returns (model, dataset_root, prompt) or (None, None, None) if not matched.
    """
    parts = Path(filename).parts
    # Normalize by removing leading '.' if present
    if parts and parts[0] == ".":
        parts = parts[1:]
    try:
        # Find the segment that matches known model names
        known_models = {"phi-3", "phi-4", "qwen"}
        for i, p in enumerate(parts):
            if p in known_models:
                model = p
                dataset_root = parts[i + 1] if i + 1 < len(parts) else None
                prompt = parts[i + 2] if i + 2 < len(parts) else None
                return model, dataset_root, prompt
    except Exception:
        pass
    return None, None, None


def aggregate_bandit(
    data: Dict[str, Any], group_by: Optional[str] = None
) -> Dict[Tuple[str, ...], Counter]:
    """
    Build counters of Bandit findings by test_id (with test_name).
    group_by can be None, 'model', 'dataset_root', 'prompt'.
    Key is a tuple of grouping values (or ("ALL",)) mapping to Counter of "test_id|test_name".
    """
    counters: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    results = data.get("results", [])
    for item in results:
        if not isinstance(item, dict):
            continue
        test_id = item.get("test_id") or "<unknown>"
        test_name = item.get("test_name") or ""
        key_symbol = f"{test_id}|{test_name}"

        filename = item.get("filename", "")
        model, dataset_root, prompt = parse_path_fields(filename)

        if group_by == "model" and model is not None:
            key = (model,)
        elif group_by == "dataset_root" and dataset_root is not None:
            key = (dataset_root,)
        elif group_by == "prompt" and prompt is not None:
            key = (prompt,)
        else:
            key = ("ALL",)

        counters[key][key_symbol] += 1

    return counters


def compute_top_k(
    grouped: Dict[Tuple[str, ...], Counter], k: int
) -> List[Tuple[str, str, str, int, float]]:
    """
    Produce rows: (group_label, risk, test_id, count, frequency).
    risk is test_name; test_id is kept separately for clarity.
    """
    rows: List[Tuple[str, str, str, int, float]] = []
    for key_tuple, counter in grouped.items():
        group_label = " | ".join(key_tuple)
        total = sum(counter.values())
        if total == 0:
            continue
        for sym, count in counter.most_common(k):
            test_id, test_name = sym.split("|", 1)
            freq = count / total
            rows.append((group_label, test_name, test_id, count, freq))
    return rows


def write_csv(rows: List[Tuple[str, str, str, int, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "risk_name", "test_id", "count", "frequency"])
        for group, risk_name, test_id, count, freq in rows:
            writer.writerow([group, risk_name, test_id, count, f"{freq:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute top-K Bandit security risks across all models/datasets/prompts"
    )
    parser.add_argument(
        "bandit_json",
        type=Path,
        default=Path("bandit_results.json"),
        help="Path to bandit_results.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top risks to include (default: 5)",
    )
    parser.add_argument(
        "--group-by",
        choices=["none", "model", "dataset_root", "prompt"],
        default="none",
        help="Optional grouping for breakdowns (default: none for global top-5)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis_results/top_bandit_risks.csv"),
        help="Output CSV path",
    )

    args = parser.parse_args()

    data = load_bandit(args.bandit_json)
    group = None if args.group_by == "none" else args.group_by
    grouped = aggregate_bandit(data, group_by=group)
    rows = compute_top_k(grouped, args.top_k)
    write_csv(rows, args.out)

    for group_label, risk_name, test_id, count, freq in rows:
        print(
            f"{group_label:20s}  {test_id:>5s}  {risk_name:40s}  {count:5d}  {freq:6.2%}"
        )
    print(f"\nWrote CSV to: {args.out}")


if __name__ == "__main__":
    main()



