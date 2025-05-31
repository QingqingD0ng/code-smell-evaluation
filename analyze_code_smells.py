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
    PRODUCTION_MODELS,
    DEBUG_MODELS,
    PROMPT_TEMPLATES,
)
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("code_analysis.log"), logging.StreamHandler()],
)

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
            logging.error(f"Failed to parse Pylint output for {file_path}: {str(e)}")
            return {"fatal_errors": [], "other_issues": []}
    except subprocess.CalledProcessError as e:
        logging.error(f"Pylint analysis failed for {file_path}: {str(e)}")
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
        logging.debug(f"Filtered out {msg_id}: {item.get('message', '')}")
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
        logging.error(f"Bandit analysis failed: {str(e)}")
        return None


def aggregate_bandit_results(bandit_json, output_csv):
    """Aggregate Bandit results and save to CSV"""
    smellDict = defaultdict(int)
    if not bandit_json or "results" not in bandit_json:
        logging.warning("No Bandit results to aggregate")
        return

    for item in bandit_json["results"]:
        key = item["test_id"] + "_" + item["test_name"]
        smellDict[key] += 1

    try:
        with open(output_csv, "w", newline="", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(["Message", "Count"])
            writer.writerows([[key, smellDict[key]] for key in smellDict])
        logging.info(f"Bandit results saved to {output_csv}")
    except Exception as e:
        logging.error(f"Failed to save Bandit results: {str(e)}")


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
        logging.error(f"Folder not found: {folder_path}")
        return None, None

    # Create output directory for this technique
    technique_name = os.path.basename(folder_path)
    dataset_name = os.path.basename(os.path.dirname(folder_path))
    technique_output_dir = os.path.join(output_dir, dataset_name, technique_name)
    os.makedirs(technique_output_dir, exist_ok=True)

    # Get all Python files recursively
    py_files = get_python_files(folder_path)
    if not py_files:
        logging.warning(f"No Python files found in {folder_path}")
        return None, None

    logging.info(f"Analyzing {len(py_files)} files in {dataset_name}/{technique_name}")

    # Analyze with Pylint
    pylint_results = defaultdict(dict)
    for file_path in py_files:
        logging.info(f"Running Pylint on {file_path}")
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
        logging.info(f"Pylint results saved to {pylint_csv}")
    except Exception as e:
        logging.error(f"Failed to save Pylint results: {str(e)}")

    # Analyze with Bandit
    logging.info(f"Running Bandit on {len(py_files)} files")
    bandit_json = analyze_with_bandit(py_files)
    bandit_csv = os.path.join(technique_output_dir, "bandit_results.csv")
    aggregate_bandit_results(bandit_json, bandit_csv)

    return pylint_results, bandit_json


def analyze_syntax_errors(results):
    """Analyze and print findings about fatal/syntax errors in the results"""
    print("\n=== Fatal/Syntax Error Analysis ===")
    print("Technique\tFiles with Fatal Errors\tFiles without Fatal Errors")
    print("-" * 70)

    for technique, result in results.items():
        files_with_errors = 0
        files_without_errors = 0

        for task_id, issues in result["pylint"].items():
            if issues["fatal_errors"]:
                files_with_errors += 1
            else:
                files_without_errors += 1

        print(f"{technique}\t{files_with_errors}\t\t\t{files_without_errors}")

    print("\nDetailed Findings:")
    print("-" * 70)
    for technique, result in results.items():
        print(f"\n{technique.upper()}:")
        for task_id, issues in result["pylint"].items():
            if issues["fatal_errors"]:
                print(f"\nTask {task_id} has fatal/syntax errors:")
                for error in issues["fatal_errors"]:
                    print(
                        f"  - Line {error.get('line')}: [{error.get('message-id')}] {error.get('message')}"
                    )


def analyze_top_smells(results):
    """Analyze and display the top 10 most common code smells across all techniques"""
    print("\n=== Top 10 Code Smells Analysis ===")

    # Dictionary to store smell counts
    smell_counts = defaultdict(int)
    smell_details = defaultdict(list)

    # Collect all smells and their counts
    for technique, result in results.items():
        for task_id, issues in result["pylint"].items():
            # Count fatal errors
            for error in issues["fatal_errors"]:
                msg_id = error.get("message-id", "")
                symbol = error.get("symbol", "")
                message = error.get("message", "")
                key = f"{msg_id} - {symbol}"
                smell_counts[key] += 1
                if len(smell_details[key]) < 3:  # Keep up to 3 examples
                    smell_details[key].append(
                        {
                            "technique": technique,
                            "task": task_id,
                            "line": error.get("line", ""),
                            "message": message,
                        }
                    )

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
    smell_types = {"F": "Fatal", "E": "Error", "W": "Warning", "C": "Convention"}

    # Initialize counters for each type
    type_counts = {msg_type: defaultdict(int) for msg_type in smell_types.keys()}
    type_details = {msg_type: defaultdict(list) for msg_type in smell_types.keys()}

    # Collect all smells and their counts by type
    for technique, result in results.items():
        for task_id, issues in result["pylint"].items():
            # Process both fatal errors and other issues
            all_issues = issues["fatal_errors"] + issues["other_issues"]
            for issue in all_issues:
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


def generate_statistics_table(results):
    """Generate a comprehensive statistics table for code smells"""
    print("\n=== Code Smell Statistics Table ===")

    # Headers for the table
    headers = [
        "Dataset",
        "# Error Instances",
        "# Convention Instances",
        "# Refactor Instances",
        "# Warning Instances",
        "Total Smell Instances",
        "# Smelly Samples",
        "Avg. # Smells per Sample",
    ]

    # Print table header
    print("\n|" + "|".join(f" {h:^30} " for h in headers))
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Store counts for each technique
    technique_counts = {}

    # Process each technique
    for technique, result in results.items():
        # Initialize counters
        error_count = 0
        convention_count = 0
        refactor_count = 0
        warning_count = 0
        smelly_samples = 0

        # Count issues by type
        for task_id, issues in result["pylint"].items():
            all_issues = issues["fatal_errors"] + issues["other_issues"]
            has_issues = False

            for issue in all_issues:
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

            if has_issues:
                smelly_samples += 1

        # Calculate totals and averages
        total_instances = (
            error_count + convention_count + refactor_count + warning_count
        )
        total_samples = len(result["pylint"])
        avg_smells = total_instances / total_samples if total_samples > 0 else 0

        # Store counts for this technique
        technique_counts[technique] = {
            "error": error_count,
            "convention": convention_count,
            "refactor": refactor_count,
            "warning": warning_count,
            "total": total_instances,
            "smelly": smelly_samples,
            "avg": avg_smells,
        }

        # Print row
        row = [
            technique,
            str(error_count),
            str(convention_count),
            str(refactor_count),
            str(warning_count),
            str(total_instances),
            str(smelly_samples),
            f"{avg_smells:.2f}",
        ]
        print("|" + "|".join(f" {v:^30} " for v in row))

    # Print total row
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Calculate totals from stored counts
    total_error = sum(counts["error"] for counts in technique_counts.values())
    total_convention = sum(counts["convention"] for counts in technique_counts.values())
    total_refactor = sum(counts["refactor"] for counts in technique_counts.values())
    total_warning = sum(counts["warning"] for counts in technique_counts.values())
    total_instances = sum(counts["total"] for counts in technique_counts.values())
    total_smelly = sum(counts["smelly"] for counts in technique_counts.values())
    total_samples = sum(len(result["pylint"]) for result in results.values())
    overall_avg = total_instances / total_samples if total_samples > 0 else 0

    # Print total row
    total_row = [
        "TOTAL",
        str(total_error),
        str(total_convention),
        str(total_refactor),
        str(total_warning),
        str(total_instances),
        str(total_smelly),
        f"{overall_avg:.2f}",
    ]
    print("|" + "|".join(f" {v:^30} " for v in total_row))


def generate_security_statistics_table(results):
    """Generate a comprehensive statistics table for security smells from Bandit results"""
    print("\n=== Security Smell Statistics Table (Bandit Results) ===")

    # Headers for the table
    headers = [
        "Dataset",
        "Syntax Error",
        "# Security Smell Type",
        "Total Smell Instances",
        "# Security 'Smelly' Samples",
        "Avg. # Sec. Smell Per Samples",
    ]

    # Print table header
    print("\n|" + "|".join(f" {h:^30} " for h in headers))
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Process each technique
    for technique, result in results.items():
        # Get Bandit results
        bandit_results = result["bandit"]
        if not bandit_results or "results" not in bandit_results:
            # If no Bandit results, print row with zeros
            row = [technique, "0", "0", "0", "0", "0.00"]
            print("|" + "|".join(f" {v:^30} " for v in row))
            continue

        # Count syntax errors from Pylint results
        syntax_errors = sum(
            len(issues["fatal_errors"]) for issues in result["pylint"].values()
        )

        # Count security smells
        security_smells = bandit_results["results"]
        unique_smell_types = len(set(item["test_id"] for item in security_smells))
        total_instances = len(security_smells)

        # Count smelly samples (files with security issues)
        smelly_samples = len(set(item["filename"] for item in security_smells))

        # Calculate average
        total_files = len(result["pylint"])
        avg_smells = total_instances / total_files if total_files > 0 else 0

        # Print row
        row = [
            technique,
            str(syntax_errors),
            str(unique_smell_types),
            str(total_instances),
            str(smelly_samples),
            f"{avg_smells:.2f}",
        ]
        print("|" + "|".join(f" {v:^30} " for v in row))

    # Print total row
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Calculate totals
    total_syntax = sum(
        len(issues["fatal_errors"])
        for result in results.values()
        for issues in result["pylint"].values()
    )

    total_security_types = len(
        set(
            item["test_id"]
            for result in results.values()
            if result["bandit"] and "results" in result["bandit"]
            for item in result["bandit"]["results"]
        )
    )

    total_instances = sum(
        len(result["bandit"]["results"])
        for result in results.values()
        if result["bandit"] and "results" in result["bandit"]
    )

    total_smelly = len(
        set(
            item["filename"]
            for result in results.values()
            if result["bandit"] and "results" in result["bandit"]
            for item in result["bandit"]["results"]
        )
    )

    total_files = sum(len(result["pylint"]) for result in results.values())
    overall_avg = total_instances / total_files if total_files > 0 else 0

    # Print total row
    total_row = [
        "TOTAL",
        str(total_syntax),
        str(total_security_types),
        str(total_instances),
        str(total_smelly),
        f"{overall_avg:.2f}",
    ]
    print("|" + "|".join(f" {v:^30} " for v in total_row))

    # Print detailed security smell types
    print("\nDetailed Security Smell Types:")
    print("-" * 70)

    # Collect all security smell types and their counts
    security_types = defaultdict(int)
    for result in results.values():
        if result["bandit"] and "results" in result["bandit"]:
            for item in result["bandit"]["results"]:
                security_types[item["test_id"]] += 1

    # Print top security smells
    print("\nTop Security Smells:")
    print("|".join(f" {h:^30} " for h in ["Security Smell ID", "Count"]))
    print("|" + "|".join("-" * 32 for _ in range(2)) + "|")

    for smell_id, count in sorted(
        security_types.items(), key=lambda x: x[1], reverse=True
    ):
        print("|" + "|".join(f" {v:^30} " for v in [smell_id, str(count)]))


def generate_top_security_smells_table(results):
    """Generate a table of top 3 security smells detected by Bandit for each dataset"""
    print("\n=== Top 3 Security Smells by Dataset (Bandit Results) ===")

    # Headers for the table
    headers = ["Dataset", "Message", "CWE", "Total"]
    print("\n|" + "|".join(f" {h:^30} " for h in headers))
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Process each technique
    for technique, result in results.items():
        if not result["bandit"] or "results" not in result["bandit"]:
            continue

        # Count total security smells for this dataset
        total_smells = len(result["bandit"]["results"])

        # Group security smells by test_id and collect their details
        security_smells = defaultdict(lambda: {"count": 0, "message": "", "cwe": ""})
        for item in result["bandit"]["results"]:
            test_id = item["test_id"]
            security_smells[test_id]["count"] += 1
            security_smells[test_id]["message"] = item["test_name"]
            security_smells[test_id]["cwe"] = item.get("issue_cwe", {}).get("id", "N/A")

        # Get top 3 security smells
        top_smells = sorted(
            security_smells.items(), key=lambda x: x[1]["count"], reverse=True
        )[:3]

        # Print each top smell
        for i, (test_id, details) in enumerate(top_smells):
            count = details["count"]
            percentage = (count / total_smells * 100) if total_smells > 0 else 0
            row = [
                technique if i == 0 else "",  # Only show dataset name for first row
                f"{test_id}-{details['message']}",
                f"CWE-{details['cwe']}",
                f"{count:,} ({percentage:.2f}%)",
            ]
            print("|" + "|".join(f" {v:^30} " for v in row))

        # Add separator between datasets
        if (
            technique != list(results.keys())[-1]
        ):  # Don't print separator after last dataset
            print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Print total row
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Calculate overall totals
    all_security_smells = defaultdict(lambda: {"count": 0, "message": "", "cwe": ""})
    total_all_smells = 0

    for result in results.values():
        if result["bandit"] and "results" in result["bandit"]:
            total_all_smells += len(result["bandit"]["results"])
            for item in result["bandit"]["results"]:
                test_id = item["test_id"]
                all_security_smells[test_id]["count"] += 1
                all_security_smells[test_id]["message"] = item["test_name"]
                all_security_smells[test_id]["cwe"] = item.get("issue_cwe", {}).get(
                    "id", "N/A"
                )

    # Get top 3 overall security smells
    top_overall = sorted(
        all_security_smells.items(), key=lambda x: x[1]["count"], reverse=True
    )[:3]

    # Print overall totals
    print("\nOverall Top 3 Security Smells:")
    print("|" + "|".join(f" {h:^30} " for h in headers))
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    for test_id, details in top_overall:
        count = details["count"]
        percentage = (count / total_all_smells * 100) if total_all_smells > 0 else 0
        row = [
            "ALL",
            f"{test_id}-{details['message']}",
            f"CWE-{details['cwe']}",
            f"{count:,} ({percentage:.2f}%)",
        ]
        print("|" + "|".join(f" {v:^30} " for v in row))


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
            smelly_samples = 0

            # Count issues by type
            for task_id, issues in result["pylint"].items():
                all_issues = issues["fatal_errors"] + issues["other_issues"]
                has_issues = False

                for issue in all_issues:
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

                if has_issues:
                    smelly_samples += 1

            # Calculate totals and averages
            total_instances = (
                error_count + convention_count + refactor_count + warning_count
            )
            total_samples = len(result["pylint"])
            avg_smells = total_instances / total_samples if total_samples > 0 else 0

            # Store counts for this technique
            model_technique_counts[model_name][key] = {
                "error": error_count,
                "convention": convention_count,
                "refactor": refactor_count,
                "warning": warning_count,
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
        str(total_instances),
        str(total_smelly),
        f"{overall_avg:.2f}",
    ]
    print("|" + "|".join(f" {v:^30} " for v in total_row))


def generate_comprehensive_security_table(results):
    """Generate a comprehensive table of security smells across all models"""
    print("\n=== Comprehensive Security Smell Analysis Table ===")

    # Headers for the table
    headers = [
        "Model",
        "Dataset",
        "Technique",
        "Syntax Error",
        "# Security Smell Type",
        "Total Smell Instances",
        "# Security 'Smelly' Samples",
        "Avg. # Sec. Smell Per Samples",
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
            # Get Bandit results
            bandit_results = result["bandit"]
            if not bandit_results or "results" not in bandit_results:
                # If no Bandit results, print row with zeros
                row = [model_name, dataset, technique, "0", "0", "0", "0", "0.00"]
                print("|" + "|".join(f" {v:^30} " for v in row))
                continue

            # Count syntax errors from Pylint results
            syntax_errors = sum(
                len(issues["fatal_errors"]) for issues in result["pylint"].values()
            )

            # Count security smells
            security_smells = bandit_results["results"]
            unique_smell_types = len(set(item["test_id"] for item in security_smells))
            total_instances = len(security_smells)

            # Count smelly samples (files with security issues)
            smelly_samples = len(set(item["filename"] for item in security_smells))

            # Calculate average
            total_files = len(result["pylint"])
            avg_smells = total_instances / total_files if total_files > 0 else 0

            # Store counts for this technique
            model_technique_counts[model_name][key] = {
                "syntax_errors": syntax_errors,
                "unique_types": unique_smell_types,
                "total_instances": total_instances,
                "smelly_samples": smelly_samples,
                "avg_smells": avg_smells,
            }

            # Print row
            row = [
                model_name,
                dataset,
                technique,
                str(syntax_errors),
                str(unique_smell_types),
                str(total_instances),
                str(smelly_samples),
                f"{avg_smells:.2f}",
            ]
            print("|" + "|".join(f" {v:^30} " for v in row))

    # Print total row
    print("|" + "|".join("-" * 32 for _ in headers) + "|")

    # Calculate totals across all models and techniques
    total_syntax = sum(
        counts["syntax_errors"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )

    total_security_types = len(
        set(
            item["test_id"]
            for model_results in results.values()
            for result in model_results.values()
            if result["bandit"] and "results" in result["bandit"]
            for item in result["bandit"]["results"]
        )
    )

    total_instances = sum(
        counts["total_instances"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )

    total_smelly = sum(
        counts["smelly_samples"]
        for model_counts in model_technique_counts.values()
        for counts in model_counts.values()
    )

    total_files = sum(
        len(result["pylint"])
        for model_results in results.values()
        for result in model_results.values()
    )
    overall_avg = total_instances / total_files if total_files > 0 else 0

    # Print total row
    total_row = [
        "TOTAL",
        "",
        "",
        str(total_syntax),
        str(total_security_types),
        str(total_instances),
        str(total_smelly),
        f"{overall_avg:.2f}",
    ]
    print("|" + "|".join(f" {v:^30} " for v in total_row))


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

    # Select models based on debug flag
    models_to_analyze = DEBUG_MODELS if args.debug else PRODUCTION_MODELS

    # Analyze each model's results
    for model_name, _ in models_to_analyze:
        model_results = {}
        model_dir = os.path.join(analysis_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Get the model's code directory
        model_code_dir = os.path.join(extracted_code_dir, model_name)
        if not os.path.exists(model_code_dir):
            logging.warning(f"No code directory found for {model_code_dir}")
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
                    logging.warning(f"No code directory found for {technique_dir}")
                    continue

                pylint_results, bandit_results = analyze_folder(
                    technique_dir, model_dir
                )
                if pylint_results is not None:
                    # Use dataset/technique as the key to store results
                    key = f"{dataset}/{technique}"
                    model_results[key] = {
                        "pylint": pylint_results,
                        "bandit": bandit_results,
                    }

        if model_results:  # Only add to results if we found any data
            results[model_name] = model_results

    if not results:
        logging.error(
            "No results found to analyze. Please check the extracted_code directory structure."
        )
        return

    # Generate summary report
    summary_csv = os.path.join(analysis_dir, "summary.csv")
    try:
        with open(summary_csv, "w", newline="", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Model",
                    "Dataset",
                    "Technique",
                    "Total Files",
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

        # Generate analysis for each model
        for model_name, model_results in results.items():
            model_dir = os.path.join(analysis_dir, model_name)

            # Define analysis functions
            analyses = {
                "syntax_errors": analyze_syntax_errors,
                "top_smells": analyze_top_smells,
                "smells_by_type": analyze_top_smells_by_type,
                "statistics": generate_statistics_table,
                "security_statistics": generate_security_statistics_table,
                "top_security_smells": generate_top_security_smells_table,
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

        # Generate comprehensive security analysis across all models
        comprehensive_security_output = get_function_output(
            generate_comprehensive_security_table, results
        )
        with open(
            os.path.join(analysis_dir, "comprehensive_security.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(comprehensive_security_output)

        # Create overall summary
        overall_summary = f"""Code Smell Analysis Summary
{'=' * 50}

Analysis completed at: {datetime.now()}

Results directory: {analysis_dir}
Summary CSV: {summary_csv}

Analyzed models:
{chr(10).join(f'- {model_name}' for model_name in results.keys())}

Each model directory contains:
- syntax_errors.txt
- top_smells.txt
- smells_by_type.txt
- statistics.txt
- security_statistics.txt
- top_security_smells.txt
- summary.txt

Overall analysis:
- comprehensive.txt
- comprehensive_security.txt
"""
        with open(
            os.path.join(analysis_dir, "overall_summary.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(overall_summary)

    except Exception as e:
        with open(os.path.join(analysis_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"Failed to generate analysis: {str(e)}")


if __name__ == "__main__":
    main()
