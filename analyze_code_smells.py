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
    MODELS,
    PROMPT_TEMPLATES,
)
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logger = logging.getLogger("code_analysis")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("code_analysis.log")
stream_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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
        "C0303",
        "C0304",
        "C0305",
        "C0103",
        "C0112",
        "C0114",
        "C0115",
        "C0116",
        "W0611",
        "C0415",
        "W0404",
        "C0413",
        "C0411",
        "C0412",
        "W0401",
        "W0614",
        "C0410",
        "C0413",
        "C0414",
        "R0402",
        "C2403",
        "E0401",
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
            writer.writerow(
                ["Task ID", "Message ID", "Symbol", "Message", "Line", "Type"]
            )
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
                            "Fatal Error",
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
                            "Other Issue",
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


def analyze_top_smells(results):
    """Analyze and display the top 10 most common code smells across all techniques"""
    print("\n=== Top 10 Code Smells Analysis ===")

    # Dictionary to store smell counts
    smell_counts = defaultdict(int)
    smell_details = defaultdict(list)

    # Collect all smells and their counts
    for technique, result in results.items():
        for task_id, issues in result["pylint"].items():
            # Count other issues
            for issue in issues["other_issues"]:
                msg_id = issue.get("message-id", "")
                symbol = issue.get("symbol", "")
                message = issue.get("message", "")
                key = f"{msg_id} - {symbol}"
                smell_counts[key] += 1
                if len(smell_details[key]) < 3:  # Keep up to 3 examples
                    smell_details[key].append(
                        {
                            "technique": technique,
                            "task": task_id,
                            "line": issue.get("line", ""),
                            "message": message,
                            "type": "pylint",
                        }
                    )

        # Process Bandit security issues
        if result["bandit"] and "results" in result["bandit"]:
            for issue in result["bandit"]["results"]:
                test_id = issue.get("test_id", "")
                test_name = issue.get("test_name", "")
                message = issue.get("issue_text", "")
                key = f"BANDIT - {test_id} - {test_name}"
                smell_counts[key] += 1
                if len(smell_details[key]) < 3:  # Keep up to 3 examples
                    smell_details[key].append(
                        {
                            "technique": technique,
                            "task": issue.get("filename", ""),
                            "line": issue.get("line_number", ""),
                            "message": message,
                            "type": "bandit",
                        }
                    )

    # Sort smells by count and get top 10
    top_smells = sorted(smell_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Print results
    print("\nRank  Count  Message ID - Symbol")
    print("-" * 50)
    for rank, (smell, count) in enumerate(top_smells, 1):
        print(f"{rank:2d}.    {count:3d}    {smell}")

    print("\nDetailed Examples:")
    print("-" * 50)
    for rank, (smell, count) in enumerate(top_smells, 1):
        print(f"\n{rank}. {smell} (occurred {count} times)")
        print("Examples:")
        for example in smell_details[smell]:
            print(
                f"  - {example['technique']}/{example['task']} (line {example['line']}): {example['message']}"
            )


def analyze_top_smells_by_type(results):
    """Analyze and display the top 5 code smells for each Pylint message type"""
    print("\n=== Top 5 Code Smells by Message Type ===")

    # Dictionary to store smell counts by type
    smell_types = {
        "E": "Error",
        "W": "Warning",
        "C": "Convention",
        "R": "Refactor",
        "S": "Security",
    }

    # Initialize counters for each type
    type_counts = {msg_type: defaultdict(int) for msg_type in smell_types.keys()}
    type_details = {msg_type: defaultdict(list) for msg_type in smell_types.keys()}

    # Collect all smells and their counts by type
    for technique, result in results.items():
        for task_id, issues in result["pylint"].items():
            for issue in issues["other_issues"]:
                msg_id = issue.get("message-id", "")
                if not msg_id:
                    continue

                msg_type = msg_id[0]  # First character of message ID
                if msg_type not in smell_types:
                    continue

                symbol = issue.get("symbol", "")
                message = issue.get("message", "")
                key = f"{msg_id} - {symbol}"

                type_counts[msg_type][key] += 1
                if len(type_details[msg_type][key]) < 3:  # Keep up to 3 examples
                    type_details[msg_type][key].append(
                        {
                            "technique": technique,
                            "task": task_id,
                            "line": issue.get("line", ""),
                            "message": message,
                        }
                    )

        # Process Bandit security issues
        if result["bandit"] and "results" in result["bandit"]:
            for issue in result["bandit"]["results"]:
                test_id = issue.get("test_id", "")
                test_name = issue.get("test_name", "")
                message = issue.get("issue_text", "")
                key = f"BANDIT - {test_id} - {test_name}"

                type_counts["S"][key] += 1
                if len(type_details["S"][key]) < 3:  # Keep up to 3 examples
                    type_details["S"][key].append(
                        {
                            "technique": technique,
                            "task": issue.get("filename", ""),
                            "line": issue.get("line_number", ""),
                            "message": message,
                        }
                    )

    # Print results for each type
    for msg_type, type_name in smell_types.items():
        print(f"\n{type_name} Messages (starting with '{msg_type}'):")
        print("-" * 70)

        # Get top 5 for this type
        top_smells = sorted(
            type_counts[msg_type].items(), key=lambda x: x[1], reverse=True
        )[:5]

        if not top_smells:
            print("No issues found of this type.")
            continue

        print("\nRank  Count  Message ID - Symbol")
        print("-" * 50)
        for rank, (smell, count) in enumerate(top_smells, 1):
            print(f"{rank:2d}.    {count:3d}    {smell}")

        print("\nDetailed Examples:")
        print("-" * 50)
        for rank, (smell, count) in enumerate(top_smells, 1):
            print(f"\n{rank}. {smell} (occurred {count} times)")
            print("Examples:")
            for example in type_details[msg_type][smell]:
                print(
                    f"  - {example['technique']}/{example['task']} "
                    f"(line {example['line']}): {example['message']}"
                )


def generate_comprehensive_table(results):
    """Generate a comprehensive table of all code smell statistics"""
    print("\n=== Comprehensive Code Smell Analysis Table ===")

    # Headers for the table
    headers = [
        "Model",
        "Dataset",
        "Technique",
        "# Error Instances",
        "# Convention Instances",
        "# Refactor Instances",
        "# Warning Instances",
        "# Security Smells",
        "Total Smell Instances",
        "# Smelly Samples",
        "Avg. # Smells per Sample",
    ]

    # Print table header
    print("\n|" + "|".join(f" {h:^30} " for h in headers))
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Store counts for each model and technique
    model_technique_counts = {}

    # Process each model and its techniques
    for model_name, model_results in results.items():
        model_technique_counts[model_name] = {}

        for key, result in model_results.items():
            dataset, technique = key.split("/")
            # Initialize counters
            error_count = 0
            convention_count = 0
            refactor_count = 0
            warning_count = 0
            security_count = 0
            smelly_samples = 0

            # Count issues by type
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

                # Count security smells
                if result["bandit"] and "results" in result["bandit"]:
                    security_count += len(result["bandit"]["results"])
                    has_issues = True

                if has_issues:
                    smelly_samples += 1

            # Calculate totals and averages
            total_instances = (
                error_count
                + convention_count
                + refactor_count
                + warning_count
                + security_count
            )
            total_samples = len(result["pylint"])
            avg_smells = total_instances / total_samples if total_samples > 0 else 0

            # Store counts for this technique
            model_technique_counts[model_name][key] = {
                "error": error_count,
                "convention": convention_count,
                "refactor": refactor_count,
                "warning": warning_count,
                "security": security_count,
                "total": total_instances,
                "smelly": smelly_samples,
                "avg": avg_smells,
            }

            # Print row
            row = [
                model_name,
                dataset,
                technique,
                str(error_count),
                str(convention_count),
                str(refactor_count),
                str(warning_count),
                str(security_count),
                str(total_instances),
                str(smelly_samples),
                f"{avg_smells:.2f}",
            ]
            print("|" + "|".join(f" {v:^30} " for v in row))

    # Print total row
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Calculate totals across all models and techniques
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
    overall_avg = total_instances / total_samples if total_samples > 0 else 0

    # Print total row
    total_row = [
        "TOTAL",
        "",
        "",
        str(total_error),
        str(total_convention),
        str(total_refactor),
        str(total_warning),
        str(total_security),
        str(total_instances),
        str(total_smelly),
        f"{overall_avg:.2f}",
    ]
    print("|" + "|".join(f" {v:^30} " for v in total_row))


def perform_kruskal_wallis_test(results):
    """Perform Kruskal-Wallis test for different prompt techniques"""
    print("\n=== Kruskal-Wallis Test Results ===")

    # Convert results to DataFrame format
    data = []
    for model_name, model_results in results.items():
        for key, result in model_results.items():
            dataset, technique = key.split("/")
            # Count total smells for each file
            for task_id, issues in result["pylint"].items():
                # Count code smells
                code_smells = len(issues["other_issues"])

                # Count security smells for this file
                security_smells = 0
                if result["bandit"] and "results" in result["bandit"]:
                    security_smells = sum(
                        1
                        for item in result["bandit"]["results"]
                        if item["filename"].endswith(f"{task_id}.py")
                    )

                # Calculate total smells
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

    # Perform test for each model and dataset combination
    test_results = []
    for (model, dataset), group in df.groupby(["model", "dataset"]):
        # Check if we have enough techniques to compare
        techniques = group["technique"].unique()
        if len(techniques) < 2:
            print(
                f"\nSkipping {model} on {dataset}: Insufficient techniques to compare (found {len(techniques)})"
            )
            continue

        # Prepare data for Kruskal-Wallis test
        samples = [
            group[group["technique"] == tech]["total_smells"] for tech in techniques
        ]

        # Check if we have enough data points
        if any(len(sample) < 2 for sample in samples):
            print(
                f"\nSkipping {model} on {dataset}: Insufficient data points for some techniques"
            )
            continue

        try:
            # Perform Kruskal-Wallis test
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
        except Exception as e:
            print(f"\nError performing test for {model} on {dataset}: {str(e)}")
            continue

    if not test_results:
        print("\nNo valid test results to report")
        return []

    # Print results in a table format
    print("\nModel\tDataset\tH-statistic\tp-value\tSignificant")
    print("-" * 70)
    for result in test_results:
        print(
            f"{result['model']}\t{result['dataset']}\t{result['h_statistic']:.2f}\t{result['p_value']:.4f}\t{result['significant']}"
        )

    return test_results


def create_smell_visualizations(results, output_dir):
    """Create visualizations of code smell and security smell distributions"""
    print("\nCreating visualizations...")

    # Convert results to DataFrame format
    data = []
    for model_name, model_results in results.items():
        for key, result in model_results.items():
            dataset, technique = key.split("/")
            # Count total smells for each file
            for task_id, issues in result["pylint"].items():
                total_smells = len(issues["other_issues"])
                # Count security smells for this file
                security_smells = 0
                if result["bandit"] and "results" in result["bandit"]:
                    security_smells = sum(
                        1
                        for item in result["bandit"]["results"]
                        if item["filename"].endswith(f"{task_id}.py")
                    )

                data.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "technique": technique,
                        "task_id": task_id,
                        "code_smells": total_smells,
                        "security_smells": security_smells,
                        "total_smells": total_smells + security_smells,
                    }
                )

    df = pd.DataFrame(data)

    # Set style
    plt.style.use("ggplot")

    # Create box plots for each model and dataset combination
    for (model, dataset), group in df.groupby(["model", "dataset"]):
        # Total smells box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=group, x="technique", y="total_smells")
        plt.title(f"Total Smells Distribution - {model} on {dataset}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"total_smells_boxplot_{model}_{dataset}.png")
        )
        plt.close()

    # Create a single comprehensive heatmap showing all types of smells
    plt.figure(figsize=(20, 15))

    # Total smells heatmap
    plt.figure(figsize=(20, 15))
    total_pivot = df.pivot_table(
        values="total_smells",
        index=["model", "dataset"],
        columns="technique",
        aggfunc="mean",
    )
    sns.heatmap(total_pivot, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("Mean Total Smells by Model, Dataset, and Technique")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comprehensive_smells_heatmap.png"))
    plt.close()


def main():
    # Add argument parser for debug mode
    parser = argparse.ArgumentParser(description="Analyze code smells")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with smaller models"
    )
    args = parser.parse_args()

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

    # Select models based on debug flag
    models_to_analyze = MODELS.items()

    # Analyze each model's results
    for model_name, _ in models_to_analyze:
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
                    # Count syntax errors and filter them out
                    filtered_pylint_results = {}
                    total_samples = len(pylint_results)
                    samples_with_syntax_errors = 0

                    for task_id, issues in pylint_results.items():
                        has_syntax_error = False
                        for issue in issues["other_issues"]:
                            if issue.get("message-id") == "E0001":
                                has_syntax_error = True
                                samples_with_syntax_errors += 1
                                break

                        if not has_syntax_error:
                            filtered_pylint_results[task_id] = issues

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
                        "bandit": bandit_results,
                    }

        if model_results:  # Only add to results if we found any data
            results[model_name] = model_results

    # Print syntax error statistics
    print("\n=== Syntax Error Analysis ===")
    print(
        "Model\tDataset\tTechnique\tTotal Samples\tSamples with Syntax Errors\tSyntax Correct Samples\tCorrectness %"
    )
    print("-" * 100)

    for model_name in syntax_error_stats:
        for dataset in syntax_error_stats[model_name]:
            for technique in syntax_error_stats[model_name][dataset]:
                stats = syntax_error_stats[model_name][dataset][technique]
                print(
                    f"{model_name}\t{dataset}\t{technique}\t{stats['total_samples']}\t{stats['samples_with_syntax_errors']}\t{stats['syntax_correct_samples']}\t{stats['correctness_percentage']:.2f}%"
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

            # Create model summary
            summary = f"""Analysis for {model_name}
{'=' * 50}

Analysis completed at: {datetime.now()}

Results directory: {model_dir}
Summary CSV: {summary_csv}

Analysis files:
{chr(10).join(f'- {analysis_name}.txt' for analysis_name in analyses.keys())}
"""
            with open(
                os.path.join(model_dir, "summary.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(summary)

        # Generate comprehensive analysis across all models
        comprehensive_output = get_function_output(
            generate_comprehensive_table, results
        )
        with open(
            os.path.join(analysis_dir, "comprehensive.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(comprehensive_output)

        # Perform Kruskal-Wallis test for code smells
        kruskal_output = get_function_output(perform_kruskal_wallis_test, results)
        with open(
            os.path.join(analysis_dir, "kruskal_wallis.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(kruskal_output)

        # Create visualizations
        create_smell_visualizations(results, analysis_dir)

    except Exception as e:
        with open(os.path.join(analysis_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"Failed to generate analysis: {str(e)}")


if __name__ == "__main__":
    main()
