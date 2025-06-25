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


def extract_python_code(text):
    """Extract Python code from text that may contain markdown code blocks or start directly with code."""
    if not isinstance(text, str):
        return ""

    # Try to find code between ```python and ``` markers (case insensitive)
    pattern = r"```(?:python)?(.*?)(?:```|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    # Only consider non-empty, non-blank matches
    if matches:
        # Filter out empty or blank matches
        non_blank_matches = [m for m in matches if m and m.strip()]
        if non_blank_matches:
            # Take the longest match if multiple are found
            return max(non_blank_matches, key=len).strip()

    # If no ```python blocks found, try just ``` blocks
    pattern = r"```\\n([\s\S]*?)"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Filter out empty or blank matches
        non_blank_matches = [m for m in matches if m and m.strip()]
        if non_blank_matches:
            # Take the longest match if multiple are found
            code = max(non_blank_matches, key=len).strip()
            # Verify it looks like Python code
            if any(
                line.strip().startswith(
                    ("def ", "class ", "import ", "from ", "#", "if ", "for ", "while ")
                )
                for line in code.split("\n")
            ):
                return code

    # If no code blocks found, try to extract any Python-like code
    lines = text.split("\n")
    code_lines = []
    in_code = False
    python_indicators = (
        "def ",
        "self.",
        "class ",
        "import ",
        "from ",
        "#",
        "if ",
        "for ",
        "while ",
        "return",
        "print(",
        "with ",
        "try:",
        "except:",
        "finally:",
        "@",
    )

    # Find the first line that looks like Python code
    first_code_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if any(stripped.startswith(indicator) for indicator in python_indicators) or (
            "=" in stripped and not stripped.startswith("=")
        ):
            first_code_idx = idx
            break

    if first_code_idx is not None:
        # Extract from the first code-like line to the end
        code_lines = lines[first_code_idx:]
        # Clean up trailing empty lines
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        # Remove the last line if it is exactly '```'
        if code_lines and code_lines[-1].strip() == "```":
            code_lines.pop()
        return "\n".join(code_lines).strip()

    return None


def save_code_to_file(code, model_path, dataset, technique, task_id):
    """Save extracted code to a Python file.

    Args:
        code (str): The code to save
        model_path (str): Base path for the model
        dataset (str): Name of the dataset (e.g., 'bigcodebench' or 'codereval')
        technique (str): Name of the technique used
        task_id (str): ID of the task
    """
    if not code.strip():
        logger.warning(f"Empty code extracted for {task_id} ({technique}) - skipping")
        return False

    # Remove backslashes from task_id to avoid path issues
    task_id = task_id.replace("\\", "_")

    filename = f"{task_id}.py"
    file_path = os.path.join(model_path, dataset, technique, filename)

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

    stats = defaultdict(lambda: defaultdict(lambda: {"success": 0, "failed": 0}))
    failed_extractions = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            task_id = row["task_id"].split("/")[-1]
            dataset = row["dataset"].split("/")[-1]
            logger.info(f"Processing task {task_id} from {dataset}")

            # Process each model's generations
            for model_name, model_generations in row["generations"].items():
                model_name = model_name.split("/")[-1]
                logger.info(f"Processing generations for model {model_name}")
                model_path = os.path.join(output_base_path, model_name)

                # Process each generation type
                for gen_type, content in model_generations.items():
                    if gen_type == "error":
                        error_msg = (
                            f"Error in task {task_id} for model {model_name}: {content}"
                        )
                        logger.error(error_msg)
                        failed_extractions.append(
                            {
                                "task_id": task_id,
                                "model": model_name,
                                "technique": gen_type,
                                "reason": "Generation error",
                                "details": content,
                            }
                        )
                        continue

                    if gen_type == "cot":
                        # Handle CoT special case
                        code = extract_python_code(content["final_code"])
                        if save_code_to_file(code, model_path, dataset, "cot", task_id):
                            stats[model_name]["cot"]["success"] += 1
                        else:
                            stats[model_name]["cot"]["failed"] += 1
                            failed_extractions.append(
                                {
                                    "task_id": task_id,
                                    "model": model_name,
                                    "technique": "cot",
                                    "reason": "Failed to save CoT code",
                                    "details": f"Code length: {len(code)} chars",
                                }
                            )
                    elif gen_type == "rci":
                        # Handle RCI special case
                        code = extract_python_code(content.get("improved_code", ""))
                        # If no code found in improved_code, try to extract from review
                        if not code:
                            code = extract_python_code(content.get("review", ""))
                        if save_code_to_file(code, model_path, dataset, "rci", task_id):
                            stats[model_name]["rci"]["success"] += 1
                        else:
                            stats[model_name]["rci"]["failed"] += 1
                            failed_extractions.append(
                                {
                                    "task_id": task_id,
                                    "model": model_name,
                                    "technique": "rci",
                                    "reason": "Failed to save RCI code",
                                    "details": f"Code length: {len(code)} chars",
                                }
                            )
                    else:
                        # Handle regular generation types
                        code = extract_python_code(content)
                        if save_code_to_file(
                            code, model_path, dataset, gen_type, task_id
                        ):
                            stats[model_name][gen_type]["success"] += 1
                        else:
                            stats[model_name][gen_type]["failed"] += 1
                            failed_extractions.append(
                                {
                                    "task_id": task_id,
                                    "model": model_name,
                                    "technique": gen_type,
                                    "reason": "Failed to save code",
                                    "details": f"Code length: {len(code)} chars",
                                }
                            )

    # Print statistics
    logger.info("\nExtraction Statistics:")
    for model_name, model_stats in stats.items():
        logger.info(f"\n{model_name}:")
        for technique, counts in model_stats.items():
            logger.info(f"  {technique}:")
            logger.info(f"    Successful extractions: {counts['success']}")
            logger.info(f"    Failed extractions: {counts['failed']}")

    # Print detailed failure report
    if failed_extractions:
        logger.error(f"\n=== FAILED EXTRACTIONS REPORT ===")
        logger.error(f"Total failed extractions: {len(failed_extractions)}")

        # Group failures by reason
        failures_by_reason = defaultdict(list)
        for failure in failed_extractions:
            failures_by_reason[failure["reason"]].append(failure)

        for reason, failures in failures_by_reason.items():
            logger.error(f"\n{reason} ({len(failures)} failures):")
            for failure in failures:
                logger.error(
                    f"  - Task {failure['task_id']} | Model: {failure['model']} | Technique: {failure['technique']}"
                )
                if failure.get("details"):
                    logger.error(f"    Details: {failure['details']}")
    else:
        logger.info("\n=== ALL EXTRACTIONS SUCCESSFUL ===")

    return failed_extractions


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description="Extract code from JSONL file(s)")
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
        default="extracted_code",
        help="Output directory for extracted code (default: extracted_code)",
    )
    args = parser.parse_args()

    # Configure paths
    output_base_path = args.output

    all_failed_extractions = []
    processed_files = set()

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
                    continue  # Avoid double-processing
                logger.info(f"Processing {fpath}...")
                failed_extractions = process_jsonl(fpath, output_base_path)
                all_failed_extractions.extend(failed_extractions)
                processed_files.add(fpath)

    if not args.input and not args.input_dir:
        logger.error("You must provide either --input or --input_dir.")
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
