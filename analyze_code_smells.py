import os
import subprocess
import json
import csv
from collections import defaultdict
import logging
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout
from generate_code import (
    PROMPT_TEMPLATES,
)
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set up logging
logger = logging.getLogger("code_analysis")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("code_analysis.log")
stream_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_format)
stream_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Define prompting techniques from PROMPT_TEMPLATES
TECHNIQUES = list(PROMPT_TEMPLATES.keys())


def get_function_output(func, *args, **kwargs):
    """Get the output of a function without printing to console"""
    output = StringIO()
    with redirect_stdout(output):
        func(*args, **kwargs)
    return output.getvalue()


def analyze_with_pylint(file_path):
    """Run pylint with JSON output"""
    try:
        result = subprocess.run(
            ["pylint", file_path, "--output-format=json", "--score=n"],
            capture_output=True,
            text=True,
        )
        try:
            output = json.loads(result.stdout)
            # Separate fatal/syntax errors from other issues
            fatal_errors = [
                item for item in output if item.get("message-id", "").startswith("F")
            ]
            other_issues = [item for item in output if not is_basic_smell(item)]
            return {"fatal_errors": fatal_errors, "other_issues": other_issues}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Pylint output for {file_path}: {str(e)}")
            return {"fatal_errors": [], "other_issues": []}
    except subprocess.CalledProcessError as e:
        logger.error(f"Pylint analysis failed for {file_path}: {str(e)}")
        return {"fatal_errors": [], "other_issues": []}


def categorize_smells(pylint_output):
    categories = {}
    for item in pylint_output:
        msg_id = item.get("message-id", "UNKNOWN")
        symbol = item.get("symbol", "unknown")
        msg = item.get("message", "")
        categories.setdefault(symbol, []).append(
            {"line": item.get("line"), "message": msg, "msg_id": msg_id}
        )
    return categories


def is_basic_smell(item):
    """Check if the issue is a basic syntax error or style issue"""
    # Filters for basic/syntax-related code smells

    FILTERS = [
        "C0303",  # Trailing whitespace
        "C0304",  # missing-final-newline
        "C0305",  # trailing-newlines
        "C0103",  # invalid-name
        "C0112",  # empty-docstring
        "C0114",  # missing-module-docstring
        "C0115",  # missing-class-docstring
        "C0116",  # missing-function-docstring
        "W0611",  # unused-import
        "W0401",  # wildcard-import
        "R0402",  # consider-using-fromimport
        "C2403",  # non-ascii-module-import
        "W0404",  # reimported
        "W0614",  # unused-wildcard-import
        "C0410",  # multiple-imports
        "C0411",  # wrong-import-order
        "C0412",  # ungrouped-imports
        "C0413",  # wrong-import-position
        "C0414",  # useless-import-alias
        "C0415",  # import-outside-toplevel
        "E0401",  # import-error
    ]

    msg_id = item.get("message-id", "")
    # Check if the message ID is in our filters
    if msg_id in FILTERS:
        logger.debug(f"Filtered out {msg_id}: {item.get('message', '')}")
        return True
    return False


def analyze_with_bandit(files, output_json="bandit_results.json"):
    """Run Bandit on a list of files and save output to JSON"""
    try:
        result = subprocess.run(
            ["bandit", "-r", *files, "-f", "json", "-o", output_json],
            capture_output=True,
            text=True,
        )

        if os.path.exists(output_json):
            with open(output_json) as f:
                data = json.load(f)
            return data
        return None
    except Exception as e:
        logger.error(f"Bandit analysis failed: {str(e)}")
        return None


def aggregate_bandit_results(bandit_json, output_csv):
    """Aggregate Bandit results and save to CSV"""
    smellDict = defaultdict(int)
    if not bandit_json or "results" not in bandit_json:
        logger.warning("No Bandit results to aggregate")
        return

    for item in bandit_json["results"]:
        key = item["test_id"] + "_" + item["test_name"]
        smellDict[key] += 1

    try:
        with open(output_csv, "w", newline="", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(["Message", "Count"])
            writer.writerows([[key, smellDict[key]] for key in smellDict])
        logger.info(f"Bandit results saved to {output_csv}")
    except Exception as e:
        logger.error(f"Failed to save Bandit results: {str(e)}")


def get_python_files(folder_path):
    """Recursively get all Python files in a folder and its subfolders"""
    py_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def analyze_folder(folder_path, output_dir):
    """Analyze all Python files in a folder and its subfolders"""
    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return None, None

    # Create output directory for this technique
    technique_name = os.path.basename(folder_path)
    dataset_name = os.path.basename(os.path.dirname(folder_path))
    technique_output_dir = os.path.join(output_dir, dataset_name, technique_name)
    os.makedirs(technique_output_dir, exist_ok=True)

    # Get all Python files recursively
    py_files = get_python_files(folder_path)
    if not py_files:
        logger.warning(f"No Python files found in {folder_path}")
        return None, None

    logger.info(f"Analyzing {len(py_files)} files in {dataset_name}/{technique_name}")

    # Analyze with Pylint
    pylint_results = defaultdict(dict)
    for file_path in py_files:
        logger.info(f"Running Pylint on {file_path}")
        pylint_output = analyze_with_pylint(file_path)
        # Get task_id from the file path, handling nested structure
        task_id = os.path.splitext(os.path.basename(file_path))[0]
        pylint_results[task_id] = pylint_output

    # Save Pylint results
    pylint_csv = os.path.join(technique_output_dir, "pylint_results.csv")
    try:
        with open(pylint_csv, "w", newline="", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(["Task ID", "Message ID", "Symbol", "Message", "Line"])
            for task_id, results in pylint_results.items():
                # Write fatal errors
                for item in results["fatal_errors"]:
                    writer.writerow(
                        [
                            task_id,
                            item.get("message-id", ""),
                            item.get("symbol", ""),
                            item.get("message", ""),
                            item.get("line", ""),
                        ]
                    )
                # Write other issues
                for item in results["other_issues"]:
                    writer.writerow(
                        [
                            task_id,
                            item.get("message-id", ""),
                            item.get("symbol", ""),
                            item.get("message", ""),
                            item.get("line", ""),
                        ]
                    )
        logger.info(f"Pylint results saved to {pylint_csv}")
    except Exception as e:
        logger.error(f"Failed to save Pylint results: {str(e)}")

    # Analyze with Bandit
    logger.info(f"Running Bandit on {len(py_files)} files")
    bandit_json = analyze_with_bandit(py_files)
    bandit_csv = os.path.join(technique_output_dir, "bandit_results.csv")
    aggregate_bandit_results(bandit_json, bandit_csv)

    return pylint_results, bandit_json


# --- CSV/Output Utilities ---
def write_csv(header, rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def flatten_examples(examples, n=3):
    row = []
    for i in range(n):
        if i < len(examples):
            ex = examples[i]
            row.extend(
                [
                    ex.get("technique", ""),
                    ex.get("task", ""),
                    ex.get("line", ""),
                    ex.get("message", ""),
                ]
            )
        else:
            row.extend(["", "", "", ""])
    return row


# --- Aggregation Helpers ---
def aggregate_smells(results, by_type=False, max_examples=3):
    """
    Aggregates code smells and examples.
    If by_type=True, returns dict by type: {type: {smell_key: count, ...}, ...}
    Returns: (counts, details)
    """
    if by_type:
        smell_types = {
            "E": "Error",
            "W": "Warning",
            "C": "Convention",
            "R": "Refactor",
            "S": "Security",
        }
        type_counts = {msg_type: defaultdict(int) for msg_type in smell_types.keys()}
        type_details = {msg_type: defaultdict(list) for msg_type in smell_types.keys()}
        for technique, result in results.items():
            for task_id, issues in result["pylint"].items():
                for issue in issues["other_issues"]:
                    msg_id = issue.get("message-id", "")
                    if not msg_id:
                        continue
                    msg_type = msg_id[0]
                    if msg_type not in smell_types:
                        continue
                    symbol = issue.get("symbol", "")
                    message = issue.get("message", "")
                    key = f"{msg_id} - {symbol}"
                    type_counts[msg_type][key] += 1
                    if len(type_details[msg_type][key]) < max_examples:
                        type_details[msg_type][key].append(
                            {
                                "technique": technique,
                                "task": task_id,
                                "line": issue.get("line", ""),
                                "message": message,
                            }
                        )
            if result["bandit"] and "results" in result["bandit"]:
                for issue in result["bandit"]["results"]:
                    test_id = issue.get("test_id", "")
                    test_name = issue.get("test_name", "")
                    message = issue.get("issue_text", "")
                    key = f"BANDIT - {test_id} - {test_name}"
                    type_counts["S"][key] += 1
                    if len(type_details["S"][key]) < max_examples:
                        type_details["S"][key].append(
                            {
                                "technique": technique,
                                "task": issue.get("filename", ""),
                                "line": issue.get("line_number", ""),
                                "message": message,
                            }
                        )
        return type_counts, type_details
    else:
        smell_counts = defaultdict(int)
        smell_details = defaultdict(list)
        for technique, result in results.items():
            for task_id, issues in result["pylint"].items():
                for issue in issues["other_issues"]:
                    msg_id = issue.get("message-id", "")
                    symbol = issue.get("symbol", "")
                    message = issue.get("message", "")
                    key = f"{msg_id} - {symbol}"
                    smell_counts[key] += 1
                    if len(smell_details[key]) < max_examples:
                        smell_details[key].append(
                            {
                                "technique": technique,
                                "task": task_id,
                                "line": issue.get("line", ""),
                                "message": message,
                            }
                        )
            if result["bandit"] and "results" in result["bandit"]:
                for issue in result["bandit"]["results"]:
                    test_id = issue.get("test_id", "")
                    test_name = issue.get("test_name", "")
                    message = issue.get("issue_text", "")
                    key = f"BANDIT - {test_id} - {test_name}"
                    smell_counts[key] += 1
                    if len(smell_details[key]) < max_examples:
                        smell_details[key].append(
                            {
                                "technique": technique,
                                "task": issue.get("filename", ""),
                                "line": issue.get("line_number", ""),
                                "message": message,
                            }
                        )
        return smell_counts, smell_details


def compute_summary_stats(result):
    """
    Computes error, convention, refactor, warning, security, smelly_samples, total_instances, avg_smells for a result dict.
    Returns: error_count, convention_count, refactor_count, warning_count, security_count, smelly_samples, total_instances, avg_smells
    """
    error_count = convention_count = refactor_count = warning_count = security_count = (
        smelly_samples
    ) = 0
    for task_id, issues in result["pylint"].items():
        has_issues = False
        for issue in issues["other_issues"]:
            msg_id = issue.get("message-id", "")
            if not msg_id:
                continue
            msg_type = msg_id[0]
            has_issues = True
            if msg_type == "E":
                error_count += 1
            elif msg_type == "C":
                convention_count += 1
            elif msg_type == "R":
                refactor_count += 1
            elif msg_type == "W":
                warning_count += 1
        if result["bandit"] and "results" in result["bandit"]:
            security_count += len(result["bandit"]["results"])
            has_issues = True
        if has_issues:
            smelly_samples += 1
    total_instances = (
        error_count + convention_count + refactor_count + warning_count + security_count
    )
    total_samples = len(result["pylint"])
    avg_smells = total_instances / total_samples if total_samples > 0 else 0
    return (
        error_count,
        convention_count,
        refactor_count,
        warning_count,
        security_count,
        smelly_samples,
        total_instances,
        avg_smells,
    )


# --- Refactored Analysis Functions ---
def analyze_top_smells(results, output_csv=None):
    logger.info("\n=== Top 10 Code Smells Analysis ===")
    smell_counts, smell_details = aggregate_smells(results)
    top_smells = sorted(smell_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("\nRank  Count  Message ID - Symbol")
    logger.info("-" * 50)
    for rank, (smell, count) in enumerate(top_smells, 1):
        logger.info(f"{rank:2d}.    {count:3d}    {smell}")
    logger.info("\nDetailed Examples:")
    logger.info("-" * 50)
    for rank, (smell, count) in enumerate(top_smells, 1):
        logger.info(f"\n{rank}. {smell} (occurred {count} times)")
        logger.info("Examples:")
        for example in smell_details[smell]:
            logger.info(
                f"  - {example['technique']}/{example['task']} (line {example['line']}): {example['message']}"
            )
    if output_csv:
        header = ["Rank", "Message ID - Symbol", "Count"]
        for i in range(1, 4):
            header.extend(
                [
                    f"Example_{i}_Technique",
                    f"Example_{i}_Task",
                    f"Example_{i}_Line",
                    f"Example_{i}_Message",
                ]
            )
        rows = []
        for rank, (smell, count) in enumerate(top_smells, 1):
            row = [rank, smell, count] + flatten_examples(smell_details[smell])
            rows.append(row)
        write_csv(header, rows, output_csv)


def analyze_top_smells_by_type(results, output_csv=None):
    logger.info("\n=== Top 5 Code Smells by Message Type ===")
    type_counts, type_details = aggregate_smells(results, by_type=True)
    smell_types = {
        "E": "Error",
        "W": "Warning",
        "C": "Convention",
        "R": "Refactor",
        "S": "Security",
    }
    if output_csv:
        header = ["Type", "Rank", "Message ID - Symbol", "Count"]
        for i in range(1, 4):
            header.extend(
                [
                    f"Example_{i}_Technique",
                    f"Example_{i}_Task",
                    f"Example_{i}_Line",
                    f"Example_{i}_Message",
                ]
            )
        rows = []
        for msg_type, type_name in smell_types.items():
            top_smells = sorted(
                type_counts[msg_type].items(), key=lambda x: x[1], reverse=True
            )[:5]
            for rank, (smell, count) in enumerate(top_smells, 1):
                row = [type_name, rank, smell, count] + flatten_examples(
                    type_details[msg_type][smell]
                )
                rows.append(row)
        write_csv(header, rows, output_csv)
    # logger output unchanged


def generate_comprehensive_table(results, syntax_error_stats, output_csv=None):
    logger.info("\n=== Comprehensive Code Smell Analysis Table ===")
    headers = [
        "Model",
        "Dataset",
        "Technique",
        "# Syntax Error Instances",
        "# Syntax Correct Samples",
        "Correctness %",
        "# Error Instances",
        "# Convention Instances",
        "# Refactor Instances",
        "# Warning Instances",
        "# Security Smells",
        "Total Smell Instances",
        "# Smelly Samples",
        "Avg. # Smells per Sample",
    ]
    logger.info("\n|" + "|".join(f" {h:^20} " for h in headers))
    logger.info("|" + "|".join("-" * 22 for _ in headers) + "|")
    model_technique_counts = {}
    total_syntax_errors = 0
    total_syntax_correct = 0
    total_correctness = []
    rows = []
    for model_name, model_results in results.items():
        model_technique_counts[model_name] = {}
        for key, result in model_results.items():
            dataset, technique = key.split("/")
            syntax_stats = (
                syntax_error_stats.get(model_name, {})
                .get(dataset, {})
                .get(technique, {})
            )
            syntax_errors = syntax_stats.get("samples_with_syntax_errors", 0)
            syntax_correct = syntax_stats.get("syntax_correct_samples", 0)
            correctness_pct = syntax_stats.get("correctness_percentage", 0)
            total_syntax_errors += syntax_errors
            total_syntax_correct += syntax_correct
            total_correctness.append((syntax_correct, syntax_errors))
            (
                error_count,
                convention_count,
                refactor_count,
                warning_count,
                security_count,
                smelly_samples,
                total_instances,
                avg_smells,
            ) = compute_summary_stats(result)
            model_technique_counts[model_name][key] = {
                "syntax_errors": syntax_errors,
                "syntax_correct": syntax_correct,
                "correctness_pct": correctness_pct,
                "error": error_count,
                "convention": convention_count,
                "refactor": refactor_count,
                "warning": warning_count,
                "security": security_count,
                "total": total_instances,
                "smelly": smelly_samples,
                "avg": avg_smells,
            }
            row = [
                model_name,
                dataset,
                technique,
                str(syntax_errors),
                str(syntax_correct),
                f"{correctness_pct:.2f}%",
                str(error_count),
                str(convention_count),
                str(refactor_count),
                str(warning_count),
                str(security_count),
                str(total_instances),
                str(smelly_samples),
                f"{avg_smells:.2f}",
            ]
            rows.append(row)
            logger.info("|" + "|".join(f" {v:^20} " for v in row))
    logger.info("|" + "|".join("-" * 22 for _ in headers) + "|")
    total_error = sum(
        counts["error"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )
    total_convention = sum(
        counts["convention"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )
    total_refactor = sum(
        counts["refactor"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )
    total_warning = sum(
        counts["warning"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )
    total_security = sum(
        counts["security"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )
    total_instances = sum(
        counts["total"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )
    total_smelly = sum(
        counts["smelly"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )
    total_samples = sum(
        len(result["pylint"])
        for model_results in results.values()
        for result in model_results.values()
    )
    total_correct = total_syntax_correct
    total_syntax = total_syntax_errors
    total_correctness_pct = (
        (total_correct / (total_correct + total_syntax) * 100)
        if (total_correct + total_syntax) > 0
        else 0
    )
    overall_avg = total_instances / total_samples if total_samples > 0 else 0
    total_row = [
        "TOTAL",
        "",
        "",
        str(total_syntax),
        str(total_correct),
        f"{total_correctness_pct:.2f}%",
        str(total_error),
        str(total_convention),
        str(total_refactor),
        str(total_warning),
        str(total_security),
        str(total_instances),
        str(total_smelly),
        f"{overall_avg:.2f}",
    ]
    rows.append(total_row)
    logger.info("|" + "|".join(f" {v:^20} " for v in total_row))
    if output_csv:
        write_csv(headers, rows, output_csv)


def perform_kruskal_wallis_test(results, output_csv=None):
    logger.info("\n=== Kruskal-Wallis Test Results ===")
    data = []
    for model_name, model_results in results.items():
        for key, result in model_results.items():
            dataset, technique = key.split("/")
            for task_id, issues in result["pylint"].items():
                code_smells = len(issues["other_issues"])
                security_smells = 0
                if result["bandit"] and "results" in result["bandit"]:
                    security_smells = sum(
                        1
                        for item in result["bandit"]["results"]
                        if item["filename"].endswith(f"{task_id}.py")
                    )
                total_smells = code_smells + security_smells
                data.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "technique": technique,
                        "task_id": task_id,
                        "code_smells": code_smells,
                        "security_smells": security_smells,
                        "total_smells": total_smells,
                    }
                )
    df = pd.DataFrame(data)
    test_results = []
    rows = []
    for (model, dataset), group in df.groupby(["model", "dataset"]):
        techniques = group["technique"].unique()
        if len(techniques) < 2:
            logger.info(
                f"\nSkipping {model} on {dataset}: Insufficient techniques to compare (found {len(techniques)})"
            )
            continue
        samples = [
            group[group["technique"] == tech]["total_smells"] for tech in techniques
        ]
        if any(len(sample) < 2 for sample in samples):
            logger.info(
                f"\nSkipping {model} on {dataset}: Insufficient data points for some techniques"
            )
            continue
        try:
            h_stat, p_value = stats.kruskal(*samples)
            test_results.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "h_statistic": h_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            )
            rows.append(
                [model, dataset, f"{h_stat:.2f}", f"{p_value:.4f}", p_value < 0.05]
            )
        except Exception as e:
            logger.error(f"\nError performing test for {model} on {dataset}: {str(e)}")
            continue
    if not test_results:
        logger.warning("\nNo valid test results to report")
        return []
    logger.info("\nModel\tDataset\tH-statistic\tp-value\tSignificant")
    logger.info("-" * 70)
    for result in test_results:
        logger.info(
            f"{result['model']}\t{result['dataset']}\t{result['h_statistic']:.2f}\t{result['p_value']:.4f}\t{result['significant']}"
        )
    if output_csv:
        header = ["Model", "Dataset", "H-statistic", "p-value", "Significant"]
        write_csv(header, rows, output_csv)
    return test_results


def analyze_global_top_smells(results, output_csv=None):
    logger.info("\n=== Global Top 10 Code Smells Analysis ===")
    smell_counts = defaultdict(int)
    smell_details = defaultdict(list)
    for model_results in results.values():
        for technique, result in model_results.items():
            for task_id, issues in result["pylint"].items():
                for issue in issues["other_issues"]:
                    msg_id = issue.get("message-id", "")
                    symbol = issue.get("symbol", "")
                    message = issue.get("message", "")
                    key = f"{msg_id} - {symbol}"
                    smell_counts[key] += 1
                    if len(smell_details[key]) < 3:
                        smell_details[key].append(
                            {
                                "technique": technique,
                                "task": task_id,
                                "line": issue.get("line", ""),
                                "message": message,
                                "type": "pylint",
                            }
                        )
            if result["bandit"] and "results" in result["bandit"]:
                for issue in result["bandit"]["results"]:
                    test_id = issue.get("test_id", "")
                    test_name = issue.get("test_name", "")
                    message = issue.get("issue_text", "")
                    key = f"BANDIT - {test_id} - {test_name}"
                    smell_counts[key] += 1
                    if len(smell_details[key]) < 3:
                        smell_details[key].append(
                            {
                                "technique": technique,
                                "task": issue.get("filename", ""),
                                "line": issue.get("line_number", ""),
                                "message": message,
                                "type": "bandit",
                            }
                        )
    top_smells = sorted(smell_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("\nRank  Count  Message ID - Symbol")
    logger.info("-" * 50)
    for rank, (smell, count) in enumerate(top_smells, 1):
        logger.info(f"{rank:2d}.    {count:3d}    {smell}")
    logger.info("\nDetailed Examples:")
    logger.info("-" * 50)
    for rank, (smell, count) in enumerate(top_smells, 1):
        logger.info(f"\n{rank}. {smell} (occurred {count} times)")
        logger.info("Examples:")
        for example in smell_details[smell]:
            logger.info(
                f"  - {example['technique']}/{example['task']} (line {example['line']}): {example['message']}"
            )
    if output_csv:
        header = ["Rank", "Message ID - Symbol", "Count"]
        for i in range(1, 4):
            header.extend(
                [
                    f"Example_{i}_Technique",
                    f"Example_{i}_Task",
                    f"Example_{i}_Line",
                    f"Example_{i}_Message",
                ]
            )
        rows = []
        for rank, (smell, count) in enumerate(top_smells, 1):
            row = [rank, smell, count] + flatten_examples(smell_details[smell])
            rows.append(row)
        write_csv(header, rows, output_csv)


def analyze_global_top_smells_by_type(results, output_csv=None):
    logger.info("\n=== Global Top 5 Code Smells by Message Type ===")
    smell_types = {
        "E": "Error",
        "W": "Warning",
        "C": "Convention",
        "R": "Refactor",
        "S": "Security",
    }
    type_counts = {msg_type: defaultdict(int) for msg_type in smell_types.keys()}
    type_details = {msg_type: defaultdict(list) for msg_type in smell_types.keys()}
    for model_results in results.values():
        for technique, result in model_results.items():
            for task_id, issues in result["pylint"].items():
                for issue in issues["other_issues"]:
                    msg_id = issue.get("message-id", "")
                    if not msg_id:
                        continue
                    msg_type = msg_id[0]
                    if msg_type not in smell_types:
                        continue
                    symbol = issue.get("symbol", "")
                    message = issue.get("message", "")
                    key = f"{msg_id} - {symbol}"
                    type_counts[msg_type][key] += 1
                    if len(type_details[msg_type][key]) < 3:
                        type_details[msg_type][key].append(
                            {
                                "technique": technique,
                                "task": task_id,
                                "line": issue.get("line", ""),
                                "message": message,
                            }
                        )
            if result["bandit"] and "results" in result["bandit"]:
                for issue in result["bandit"]["results"]:
                    test_id = issue.get("test_id", "")
                    test_name = issue.get("test_name", "")
                    message = issue.get("issue_text", "")
                    key = f"BANDIT - {test_id} - {test_name}"
                    type_counts["S"][key] += 1
                    if len(type_details["S"][key]) < 3:
                        type_details["S"][key].append(
                            {
                                "technique": technique,
                                "task": issue.get("filename", ""),
                                "line": issue.get("line_number", ""),
                                "message": message,
                            }
                        )
    if output_csv:
        header = ["Type", "Rank", "Message ID - Symbol", "Count"]
        for i in range(1, 4):
            header.extend(
                [
                    f"Example_{i}_Technique",
                    f"Example_{i}_Task",
                    f"Example_{i}_Line",
                    f"Example_{i}_Message",
                ]
            )
        rows = []
        for msg_type, type_name in smell_types.items():
            top_smells = sorted(
                type_counts[msg_type].items(), key=lambda x: x[1], reverse=True
            )[:5]
            for rank, (smell, count) in enumerate(top_smells, 1):
                row = [type_name, rank, smell, count] + flatten_examples(
                    type_details[msg_type][smell]
                )
                rows.append(row)
        write_csv(header, rows, output_csv)
    # logger output unchanged


def main():
    # Base directories
    extracted_code_dir = "extracted_code"
    analysis_output_dir = "analysis_results"
    os.makedirs(analysis_output_dir, exist_ok=True)

    # Create timestamp for this analysis run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(analysis_output_dir, f"analysis_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)

    results = {}
    syntax_error_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Scan the extracted_code directory to get available models
    available_models = []
    if os.path.exists(extracted_code_dir):
        available_models = [
            d
            for d in os.listdir(extracted_code_dir)
            if os.path.isdir(os.path.join(extracted_code_dir, d))
        ]
    else:
        logger.error(f"Extracted code directory {extracted_code_dir} does not exist")
        return

    for model_name in available_models:
        model_results = {}
        model_dir = os.path.join(analysis_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Get the model's code directory
        model_code_dir = os.path.join(extracted_code_dir, model_name)
        if not os.path.exists(model_code_dir):
            logger.warning(f"No code directory found for {model_code_dir}")
            continue

        # Analyze each dataset for this model
        for dataset in os.listdir(model_code_dir):
            dataset_dir = os.path.join(model_code_dir, dataset)
            if not os.path.isdir(dataset_dir):
                continue

            # Analyze each technique for this dataset
            for technique in TECHNIQUES:
                technique_dir = os.path.join(dataset_dir, technique)
                if not os.path.exists(technique_dir):
                    logger.warning(f"No code directory found for {technique_dir}")
                    continue

                pylint_results, bandit_results = analyze_folder(
                    technique_dir, model_dir
                )
                if pylint_results is not None:
                    # Track files with syntax errors
                    filtered_pylint_results = {}
                    total_samples = len(pylint_results)
                    samples_with_syntax_errors = 0
                    syntax_error_task_ids = set()

                    for task_id, issues in pylint_results.items():
                        has_syntax_error = False
                        for issue in issues["other_issues"]:
                            if issue.get("message-id") == "E0001":
                                has_syntax_error = True
                                samples_with_syntax_errors += 1
                                syntax_error_task_ids.add(task_id)
                                break
                        if not has_syntax_error:
                            filtered_pylint_results[task_id] = issues

                    # Filter Bandit results to exclude files with syntax errors
                    filtered_bandit_results = None
                    if bandit_results and "results" in bandit_results:
                        filtered_bandit_results = bandit_results.copy()
                        filtered_bandit_results["results"] = [
                            item
                            for item in bandit_results["results"]
                            if os.path.splitext(os.path.basename(item["filename"]))[0]
                            not in syntax_error_task_ids
                        ]
                        # Also filter out bandit 'errors' for syntax error files
                        if "errors" in bandit_results:
                            filtered_bandit_results["errors"] = [
                                err
                                for err in bandit_results["errors"]
                                if os.path.splitext(os.path.basename(err["filename"]))[
                                    0
                                ]
                                not in syntax_error_task_ids
                            ]
                    else:
                        filtered_bandit_results = bandit_results

                    # Store syntax error statistics
                    syntax_error_stats[model_name][dataset][technique] = {
                        "total_samples": total_samples,
                        "samples_with_syntax_errors": samples_with_syntax_errors,
                        "syntax_correct_samples": total_samples
                        - samples_with_syntax_errors,
                        "correctness_percentage": (
                            (
                                (total_samples - samples_with_syntax_errors)
                                / total_samples
                                * 100
                            )
                            if total_samples > 0
                            else 0
                        ),
                    }

                    # Use dataset/technique as the key to store results (only for syntax-correct samples)
                    key = f"{dataset}/{technique}"
                    model_results[key] = {
                        "pylint": filtered_pylint_results,
                        "bandit": filtered_bandit_results,
                    }

        if model_results:  # Only add to results if we found any data
            results[model_name] = model_results

    # Print syntax error statistics
    logger.info("\n=== Syntax Error Analysis ===")
    logger.info(
        "Model\tDataset\tTechnique\tTotal Samples\tSamples with Syntax Errors\tSyntax Correct Samples\tCorrectness %"
    )
    logger.info("-" * 100)
    syntax_error_csv = os.path.join(analysis_dir, "syntax_error_stats.csv")
    with open(syntax_error_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Dataset",
                "Technique",
                "Total Samples",
                "Samples with Syntax Errors",
                "Syntax Correct Samples",
                "Correctness %",
            ]
        )
        for model_name in syntax_error_stats:
            for dataset in syntax_error_stats[model_name]:
                for technique in syntax_error_stats[model_name][dataset]:
                    stats = syntax_error_stats[model_name][dataset][technique]
                    logger.info(
                        f"{model_name}\t{dataset}\t{technique}\t{stats['total_samples']}\t{stats['samples_with_syntax_errors']}\t{stats['syntax_correct_samples']}\t{stats['correctness_percentage']:.2f}%"
                    )
                    writer.writerow(
                        [
                            model_name,
                            dataset,
                            technique,
                            stats["total_samples"],
                            stats["samples_with_syntax_errors"],
                            stats["syntax_correct_samples"],
                            f"{stats['correctness_percentage']:.2f}%",
                        ]
                    )

    if not results:
        logger.error(
            "No results found to analyze. Please check the extracted_code directory structure."
        )
        return

    # Generate summary report (now only for syntax-correct samples)
    summary_csv = os.path.join(analysis_dir, "summary.csv")
    try:
        with open(summary_csv, "w", newline="", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Model",
                    "Dataset",
                    "Technique",
                    "Total Files (Syntax Correct)",
                    "Total Pylint Issues",
                    "Total Bandit Issues",
                ]
            )

            for model_name, model_results in results.items():
                for key, result in model_results.items():
                    dataset, technique = key.split("/")
                    total_files = sum(1 for _ in result["pylint"].values())
                    total_pylint = sum(
                        len(issues["fatal_errors"]) + len(issues["other_issues"])
                        for issues in result["pylint"].values()
                    )
                    total_bandit = (
                        len(result["bandit"].get("results", []))
                        if result["bandit"]
                        else 0
                    )

                    writer.writerow(
                        [
                            model_name,
                            dataset,
                            technique,
                            total_files,
                            total_pylint,
                            total_bandit,
                        ]
                    )

        # Generate analysis for each model (now only for syntax-correct samples)
        for model_name, model_results in results.items():
            model_dir = os.path.join(analysis_dir, model_name)

            # Define analysis functions
            analyses = {
                "top_smells": analyze_top_smells,
                "smells_by_type": analyze_top_smells_by_type,
            }

            # Run each analysis and save to file
            for analysis_name, analysis_func in analyses.items():
                output = get_function_output(analysis_func, model_results)
                with open(
                    os.path.join(model_dir, f"{analysis_name}.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(output)
                # Write CSV
                csv_path = os.path.join(model_dir, f"{analysis_name}.csv")
                analysis_func(model_results, output_csv=csv_path)

        # Generate comprehensive analysis across all models
        comprehensive_output = get_function_output(
            generate_comprehensive_table, results, syntax_error_stats
        )
        with open(
            os.path.join(analysis_dir, "comprehensive.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(comprehensive_output)
        generate_comprehensive_table(
            results,
            syntax_error_stats,
            output_csv=os.path.join(analysis_dir, "comprehensive.csv"),
        )
        # Perform Kruskal-Wallis test for code smells
        kruskal_output = get_function_output(perform_kruskal_wallis_test, results)
        with open(
            os.path.join(analysis_dir, "kruskal_wallis.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(kruskal_output)
        perform_kruskal_wallis_test(
            results, output_csv=os.path.join(analysis_dir, "kruskal_wallis.csv")
        )

        global_top_smells = get_function_output(analyze_global_top_smells, results)
        with open(
            os.path.join(analysis_dir, "global_top_smells.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(global_top_smells)
        analyze_global_top_smells(
            results, output_csv=os.path.join(analysis_dir, "global_top_smells.csv")
        )
        global_smells_by_type = get_function_output(
            analyze_global_top_smells_by_type, results
        )
        with open(
            os.path.join(analysis_dir, "global_smells_by_type.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(global_smells_by_type)
        analyze_global_top_smells_by_type(
            results, output_csv=os.path.join(analysis_dir, "global_smells_by_type.csv")
        )

        json_output = {
            "results": results,
            "syntax_error_stats": {k: dict(v) for k, v in syntax_error_stats.items()},
        }
        with open(
            os.path.join(analysis_dir, "all_results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
    except Exception as e:
        with open(os.path.join(analysis_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"Failed to generate analysis: {str(e)}")


if __name__ == "__main__":
    main()
