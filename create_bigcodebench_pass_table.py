#!/usr/bin/env python3
"""
Create a table of BigCodeBench pass@1 rates across models and prompts.

Scans a results directory for files named:
  <model>-bigcodebench-<prompt>-merged_pass_at_k.json

Extracts the pass@1 field and produces a pivot table with rows as models
and columns as prompts. Outputs a Markdown table to stdout by default,
and can optionally also write CSV/Markdown to files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FILENAME_PATTERN = re.compile(
    r"^(?P<model>.+)-bigcodebench-(?P<prompt>.+)-merged_pass_at_k\.json$"
)


PROMPT_ORDER = [
    "baseline",
    "cot",
    "persona",
    "quality_focused",
    "rci",
]


@dataclass
class ResultEntry:
    model_name: str
    prompt_name: str
    pass_at_1: float


def discover_results(results_dir: Path) -> List[Tuple[Path, str, str]]:
    """Find result files and parse model/prompt from filename."""
    files: List[Tuple[Path, str, str]] = []
    for path in results_dir.glob("*-bigcodebench-*-merged_pass_at_k.json"):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        model = match.group("model").strip()
        prompt = match.group("prompt").strip()
        files.append((path, model, prompt))
    return files


def load_pass_at_1(file_path: Path) -> Optional[float]:
    try:
        data = json.loads(file_path.read_text())
    except Exception:
        return None
    value = data.get("pass@1")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_pivot(
    entries: List[ResultEntry],
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Return (sorted_prompts, model_to_prompt_to_value)."""
    model_to_prompt_to_value: Dict[str, Dict[str, float]] = {}
    prompts: set[str] = set()
    for e in entries:
        prompts.add(e.prompt_name)
        model_map = model_to_prompt_to_value.setdefault(e.model_name, {})
        model_map[e.prompt_name] = e.pass_at_1

    # Determine prompt order using PROMPT_ORDER first, then alphabetical for others
    known = [p for p in PROMPT_ORDER if p in prompts]
    unknown = sorted(p for p in prompts if p not in PROMPT_ORDER)
    sorted_prompts = known + unknown

    return sorted_prompts, model_to_prompt_to_value


def format_markdown_table(
    sorted_prompts: List[str],
    pivot: Dict[str, Dict[str, float]],
    decimals: int,
    percent: bool,
) -> str:
    headers = ["Model"] + sorted_prompts
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for model in sorted(pivot.keys(), key=lambda s: s.lower()):
        row: List[str] = [model]
        for prompt in sorted_prompts:
            value = pivot.get(model, {}).get(prompt)
            if value is None:
                row.append("")
            else:
                if percent:
                    row.append(f"{value * 100:.{decimals}f}%")
                else:
                    row.append(f"{value:.{decimals}f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_csv(
    output_csv: Path,
    sorted_prompts: List[str],
    pivot: Dict[str, Dict[str, float]],
    decimals: int,
    percent: bool,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *sorted_prompts])
        for model in sorted(pivot.keys(), key=lambda s: s.lower()):
            row: List[str] = [model]
            for prompt in sorted_prompts:
                value = pivot.get(model, {}).get(prompt)
                if value is None:
                    row.append("")
                else:
                    if percent:
                        row.append(f"{value * 100:.{decimals}f}")
                    else:
                        row.append(f"{value:.{decimals}f}")
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create BigCodeBench pass@1 table from merged results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("extracted_results"),
        help="Directory containing *-bigcodebench-*-merged_pass_at_k.json files",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimals to display",
    )
    parser.add_argument(
        "--no-percent",
        action="store_true",
        help="Display raw fraction instead of percentage",
    )
    parser.add_argument(
        "--write-csv",
        type=Path,
        default=None,
        help="Optional path to write CSV output",
    )
    parser.add_argument(
        "--write-md",
        type=Path,
        default=None,
        help="Optional path to write Markdown output",
    )

    args = parser.parse_args()

    results_dir: Path = args.results_dir
    decimals: int = args.decimals
    percent: bool = not args.no_percent
    write_csv_path: Optional[Path] = args.write_csv
    write_md_path: Optional[Path] = args.write_md

    files = discover_results(results_dir)
    if not files:
        raise SystemExit(f"No result files found in {results_dir}")

    entries: List[ResultEntry] = []
    for file_path, model, prompt in files:
        value = load_pass_at_1(file_path)
        if value is None:
            continue
        entries.append(
            ResultEntry(model_name=model, prompt_name=prompt, pass_at_1=value)
        )

    if not entries:
        raise SystemExit("No pass@1 values found.")

    sorted_prompts, pivot = build_pivot(entries)

    md = format_markdown_table(
        sorted_prompts, pivot, decimals=decimals, percent=percent
    )

    # Print Markdown to stdout
    print(md)

    # Optionally write CSV/MD files
    if write_csv_path is not None:
        write_csv(
            write_csv_path, sorted_prompts, pivot, decimals=decimals, percent=percent
        )
    if write_md_path is not None:
        write_md_path.parent.mkdir(parents=True, exist_ok=True)
        write_md_path.write_text(md)


if __name__ == "__main__":
    main()
