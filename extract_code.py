import os
import json
import re
import argparse
import logging
from collections import defaultdict

# Set up logging for code extraction
logger = logging.getLogger("code_extraction")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("code_extraction.log")
stream_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_format)
stream_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def parse_filename_info(filename):
    """Parse model, technique, and dataset from filename.

    Args:
        filename (str): Filename like "phi-3-bigcodebench-baseline-merged-sanitized.jsonl"

    Returns:
        tuple: (model_name, dataset, technique)
    """
    # Remove the suffix
    base_name = filename.replace("-merged-sanitized.jsonl", "")
    parts = base_name.split("-")

    # Define possible dataset names
    possible_datasets = ["bigcodebench", "codereval"]

    # Find which part contains the dataset name
    dataset = None
    dataset_index = None

    for i, part in enumerate(parts):
        if part in possible_datasets:
            dataset = part
            dataset_index = i
            break

    if dataset is None:
        # No dataset found, assume codereval
        dataset = "codereval"
        dataset_index = 1  # Assume dataset is the second part

    # Determine model name and technique based on dataset position
    if dataset_index == 1:  # dataset is second part
        if parts[0] == "phi" and parts[1] in ["3", "4"]:
            # phi-3-dataset-technique format
            model_name = f"{parts[0]}-{parts[1]}"
            technique = (
                parts[dataset_index + 1]
                if len(parts) > dataset_index + 1
                else "unknown"
            )
        else:
            # model-dataset-technique format
            model_name = parts[0]
            technique = (
                parts[dataset_index + 1]
                if len(parts) > dataset_index + 1
                else "unknown"
            )
    elif dataset_index == 2:  # dataset is third part (phi-3-dataset-technique)
        model_name = f"{parts[0]}-{parts[1]}"
        technique = (
            parts[dataset_index + 1] if len(parts) > dataset_index + 1 else "unknown"
        )
    else:
        # Fallback
        model_name = parts[0] if len(parts) > 0 else "unknown"
        technique = parts[-1] if len(parts) > 1 else "unknown"

    return model_name, dataset, technique


def save_code_to_file(code, output_base_path, model_name, dataset, technique, task_id):
    """Save extracted code to a Python file.

    Args:
        code (str): The code to save
        output_base_path (str): Base path for output
        model_name (str): Name of the model
        dataset (str): Name of the dataset
        technique (str): Name of the technique used
        task_id (str): ID of the task
    """
    if not code.strip():
        logger.warning(f"Empty code extracted for {task_id} ({technique}) - skipping")
        return False

    # Remove backslashes from task_id to avoid path issues
    task_id = task_id.replace("\\", "_")

    filename = f"{task_id}.py"
    file_path = os.path.join(output_base_path, model_name, dataset, technique, filename)

    # Create directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    except Exception as e:
        logger.error(
            f"Failed to create directory for {task_id} ({technique}): {str(e)}"
        )
        return False

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Successfully saved code to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {filename} ({technique}): {str(e)}")
        return False


def process_jsonl(jsonl_path, output_base_path):
    """Process the JSONL file and extract code into organized folders."""

    # Parse filename to get model, dataset, and technique
    filename = os.path.basename(jsonl_path)
    model_name, dataset, technique = parse_filename_info(filename)

    logger.info(
        f"Processing {filename} -> Model: {model_name}, Dataset: {dataset}, Technique: {technique}"
    )

    stats = {"success": 0, "failed": 0}
    failed_extractions = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                task_id = row["task_id"].split("/")[-1]

                # Extract code from solution field
                solution = row.get("solution", "")
                if not solution:
                    logger.warning(f"Line {line_num}: No solution field found")
                    continue

                code = solution
                if code:
                    if save_code_to_file(
                        code, output_base_path, model_name, dataset, technique, task_id
                    ):
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                        failed_extractions.append(
                            {
                                "line": line_num,
                                "task_id": task_id,
                                "reason": "Failed to save code",
                            }
                        )
                else:
                    stats["failed"] += 1
                    failed_extractions.append(
                        {
                            "line": line_num,
                            "task_id": task_id,
                            "reason": "No code extracted from solution",
                        }
                    )

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                stats["failed"] += 1
                failed_extractions.append(
                    {
                        "line": line_num,
                        "task_id": "unknown",
                        "reason": f"JSON decode error: {e}",
                    }
                )
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error: {e}")
                stats["failed"] += 1
                failed_extractions.append(
                    {
                        "line": line_num,
                        "task_id": "unknown",
                        "reason": f"Unexpected error: {e}",
                    }
                )

    # Print statistics
    logger.info(f"\nExtraction Statistics for {filename}:")
    logger.info(f"  Successful extractions: {stats['success']}")
    logger.info(f"  Failed extractions: {stats['failed']}")

    return failed_extractions


def extract_passing_solutions(eval_results_path, output_base_path):
    """Extract passing solutions from evaluation results and save them to organized folders.

    Args:
        eval_results_path (str): Path to the evaluation results JSON file
        output_base_path (str): Base path for output

    Returns:
        dict: Statistics about the extraction
    """
    # Parse filename to get model, dataset, and technique
    filename = os.path.basename(eval_results_path)
    model_name, dataset, technique = parse_filename_info(filename)

    logger.info(
        f"Processing evaluation results {filename} -> Model: {model_name}, Dataset: {dataset}, Technique: {technique}"
    )

    stats = {
        "total_tasks": 0,
        "passing_tasks": 0,
        "timeout_tasks": 0,
        "fail_tasks": 0,
        "saved": 0,
        "failed": 0,
    }

    try:
        with open(eval_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Navigate to the eval section
        if "eval" not in data:
            logger.error(f"No 'eval' section found in {filename}")
            return stats

        eval_data = data["eval"]
        stats["total_tasks"] = len(eval_data)

        for task_id, task_results in eval_data.items():
            if not isinstance(task_results, list) or len(task_results) == 0:
                continue

            # Get the first (and usually only) result for the task
            task_result = task_results[0]

            # Count different statuses
            status = task_result.get("status", "unknown")
            if status == "pass":
                stats["passing_tasks"] += 1

                # Extract the solution
                solution = task_result.get("solution", "")
                if not solution:
                    logger.warning(
                        f"Task {task_id}: No Python code extracted from solution"
                    )
                    stats["failed"] += 1
                    continue

                # Extract Python code from the solution
                code = solution
                if not code:
                    logger.warning(
                        f"Task {task_id}: No Python code extracted from solution"
                    )
                    stats["failed"] += 1
                    continue

                # Clean up the task_id (remove dataset prefix if present)
                clean_task_id = task_id.split("/")[-1] if "/" in task_id else task_id

                # Save the passing solution
                if save_code_to_file(
                    code,
                    output_base_path,
                    model_name,
                    dataset,
                    technique,
                    clean_task_id,
                ):
                    stats["saved"] += 1
                    logger.info(f"Saved passing solution for task {clean_task_id}")
                else:
                    stats["failed"] += 1
                    logger.error(f"Failed to save solution for task {clean_task_id}")
            elif status == "timeout":
                stats["timeout_tasks"] += 1
            elif status == "fail":
                stats["fail_tasks"] += 1
            # Note: We don't count "unknown" statuses separately, they're just not counted

        logger.info(f"\nExtraction Statistics for {filename}:")
        logger.info(f"  Total tasks: {stats['total_tasks']}")
        logger.info(f"  Passing tasks: {stats['passing_tasks']}")
        logger.info(f"  Timeout tasks: {stats['timeout_tasks']}")
        logger.info(f"  Failed tasks: {stats['fail_tasks']}")
        logger.info(f"  Successfully saved: {stats['saved']}")
        logger.info(f"  Failed to save: {stats['failed']}")

        # Calculate percentages
        if stats["total_tasks"] > 0:
            pass_rate = (stats["passing_tasks"] / stats["total_tasks"]) * 100
            timeout_rate = (stats["timeout_tasks"] / stats["total_tasks"]) * 100
            fail_rate = (stats["fail_tasks"] / stats["total_tasks"]) * 100
            logger.info(f"  Pass rate: {pass_rate:.1f}%")
            logger.info(f"  Timeout rate: {timeout_rate:.1f}%")
            logger.info(f"  Fail rate: {fail_rate:.1f}%")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filename}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing {filename}: {e}")

    return stats


def process_evaluation_results(eval_results_dir, output_base_path):
    """Process all evaluation results files and extract passing solutions.

    Args:
        eval_results_dir (str): Directory containing evaluation results
        output_base_path (str): Base path for output
    """
    if not os.path.exists(eval_results_dir):
        logger.error(f"Evaluation results directory not found: {eval_results_dir}")
        return

    logger.info(f"Processing evaluation results from: {eval_results_dir}")

    all_stats = []
    processed_files = 0

    # Process each JSON file in the evaluation results directory
    for fname in sorted(os.listdir(eval_results_dir)):
        if fname.endswith("_eval_results.json"):
            fpath = os.path.join(eval_results_dir, fname)
            logger.info(f"Processing evaluation results: {fname}")

            stats = extract_passing_solutions(fpath, output_base_path)
            all_stats.append(stats)
            processed_files += 1

    # Print summary statistics
    if all_stats:
        total_tasks = sum(stats["total_tasks"] for stats in all_stats)
        total_passing = sum(stats["passing_tasks"] for stats in all_stats)
        total_timeout = sum(stats["timeout_tasks"] for stats in all_stats)
        total_fail = sum(stats["fail_tasks"] for stats in all_stats)
        total_saved = sum(stats["saved"] for stats in all_stats)
        total_failed = sum(stats["failed"] for stats in all_stats)

        logger.info(f"\n=== EVALUATION RESULTS EXTRACTION SUMMARY ===")
        logger.info(f"Processed {processed_files} evaluation result files")
        logger.info(f"Total tasks across all files: {total_tasks}")
        logger.info(f"Total passing tasks: {total_passing}")
        logger.info(f"Total timeout tasks: {total_timeout}")
        logger.info(f"Total failed tasks: {total_fail}")
        logger.info(f"Successfully saved: {total_saved}")
        logger.info(f"Failed to save: {total_failed}")

        # Calculate overall rates
        if total_tasks > 0:
            pass_rate = (total_passing / total_tasks) * 100
            timeout_rate = (total_timeout / total_tasks) * 100
            fail_rate = (total_fail / total_tasks) * 100
            logger.info(f"Overall pass rate: {pass_rate:.1f}%")
            logger.info(f"Overall timeout rate: {timeout_rate:.1f}%")
            logger.info(f"Overall fail rate: {fail_rate:.1f}%")
        else:
            logger.info("Overall rates: N/A (no tasks found)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract code from sanitized JSONL files or evaluation results"
    )
    parser.add_argument(
        "--input",
        help="Input JSONL file containing the generated code",
    )
    parser.add_argument(
        "--input_dir",
        help="Directory containing JSONL files to extract code from",
    )
    parser.add_argument(
        "--output",
        default="extracted_code_passed",
        help="Output directory for extracted code (default: extracted_code)",
    )
    parser.add_argument(
        "--eval_results",
        action="store_true",
        help="Process evaluation results instead of sanitized results",
    )
    parser.add_argument(
        "--eval_results_dir",
        default="evaluation_results",
        help="Directory containing evaluation results (default: evaluation_results)",
    )
    args = parser.parse_args()

    output_base_path = args.output
    all_failed_extractions = []
    processed_files = set()

    # Process evaluation results if requested
    if args.eval_results:
        logger.info("Processing evaluation results to extract passing solutions...")
        process_evaluation_results(args.eval_results_dir, output_base_path)
        return

    # Process single input file if provided
    if args.input:
        logger.info(f"Processing {args.input}...")
        failed_extractions = process_jsonl(args.input, output_base_path)
        all_failed_extractions.extend(failed_extractions)
        processed_files.add(os.path.abspath(args.input))

    # Process all JSONL files in input_dir if provided
    if args.input_dir:
        logger.info(f"Processing all JSONL files in directory: {args.input_dir}")
        for fname in sorted(os.listdir(args.input_dir)):
            if fname.endswith(".jsonl"):
                fpath = os.path.abspath(os.path.join(args.input_dir, fname))
                if fpath in processed_files:
                    continue
                logger.info(f"Processing {fpath}...")
                failed_extractions = process_jsonl(fpath, output_base_path)
                all_failed_extractions.extend(failed_extractions)
                processed_files.add(fpath)

    # Auto-process sanitized results if no other input is specified
    if not args.input and not args.input_dir:
        sanitized_dir = "extracted_results_sanitized"
        if os.path.exists(sanitized_dir):
            logger.info(
                f"Automatically processing sanitized results from: {sanitized_dir}"
            )

            # Process each dataset directory
            for dataset_name in os.listdir(sanitized_dir):
                dataset_path = os.path.join(sanitized_dir, dataset_name)
                if os.path.isdir(dataset_path):
                    logger.info(f"Processing dataset: {dataset_name}")

                    # Process each JSONL file in the dataset directory
                    for fname in sorted(os.listdir(dataset_path)):
                        if fname.endswith(".jsonl"):
                            fpath = os.path.abspath(os.path.join(dataset_path, fname))
                            if fpath in processed_files:
                                continue

                            logger.info(f"Processing {fpath}...")
                            failed_extractions = process_jsonl(fpath, output_base_path)
                            all_failed_extractions.extend(failed_extractions)
                            processed_files.add(fpath)
                else:
                    # Handle case where dataset_name is a file (e.g., codereval files in root)
                    if dataset_name.endswith(".jsonl"):
                        fpath = os.path.abspath(
                            os.path.join(sanitized_dir, dataset_name)
                        )
                        if fpath in processed_files:
                            continue

                        logger.info(f"Processing {fpath}...")
                        failed_extractions = process_jsonl(fpath, output_base_path)
                        all_failed_extractions.extend(failed_extractions)
                        processed_files.add(fpath)
        else:
            logger.error(f"Sanitized directory not found: {sanitized_dir}")
            return

    # Save failed extractions to a JSON file for further analysis
    if all_failed_extractions:
        failed_extractions_file = os.path.join(
            output_base_path, "failed_extractions.json"
        )
        try:
            with open(failed_extractions_file, "w", encoding="utf-8") as f:
                json.dump(all_failed_extractions, f, indent=2, ensure_ascii=False)
            logger.info(f"Failed extractions saved to: {failed_extractions_file}")
        except Exception as e:
            logger.error(f"Failed to save failed extractions report: {str(e)}")

    logger.info("\nDone! Code has been extracted and organized into folders.")
    logger.info(f"Output location: {os.path.abspath(output_base_path)}")
    if all_failed_extractions:
        logger.warning(
            f"⚠️  {len(all_failed_extractions)} extractions failed. Check the log and failed_extractions.json for details."
        )


if __name__ == "__main__":
    main()
