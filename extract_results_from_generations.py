"""
Batch extract and merge results from generated jsonl files.
Automatically processes all files in results directory, groups by technique and model,
and merges them into single files.
"""

import os
import json
import argparse
import logging
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Set up logging
logger = logging.getLogger("batch_extract_results")
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler("batch_extract_results.log")
stream_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_format)
stream_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# Known model names and their mapping to actual model keys in JSON
MODELS = ["qwen", "phi-3", "phi-4"]
MODEL_MAPPING = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "phi-3": "microsoft/Phi-3-mini-128k-instruct",
    "phi-4": "microsoft/phi-4",
}


def extract_python_code(text: str) -> str:
    """Simple extraction method.

    This method looks for Python code patterns and stops when it finds clear explanatory text.
    """

    lines = text.split("\n")

    # Find the first line that contains Python code
    start_index = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and any(
            stripped.startswith(indicator)
            for indicator in [
                "def ",
                "class ",
                "import ",
                "from ",
                "if ",
                "for ",
                "while ",
                "try:",
                "else:",
                "elif ",
                "except:",
                "with ",
                "finally:",
                "return",
                "yield",
                "raise",
                "assert",
                "pass",
                "break",
                "continue",
            ]
        ):
            start_index = i
            break

    if start_index == -1:
        return text

    # Find the end of code by looking for clear explanatory text markers
    end_index = start_index

    for i in range(start_index, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            continue

        # Keep going if this line looks like code
        if any(
            keyword in stripped
            for keyword in [
                "def ",
                "class ",
                "import ",
                "from ",
                "if ",
                "for ",
                "while ",
                "try:",
                "except:",
                "with ",
                "return",
                "yield",
                "raise",
                "assert",
                "pass",
                "break",
                "continue",
                "elif ",
                "else:",
                "finally:",
                "=",
                "(",
                ")",
                "[",
                "]",
                "{",
                "}",
                ":",
                "#",
            ]
        ):
            end_index = i
            continue

        # Keep going if this line is indented (part of a code block)
        if line.startswith(" ") or line.startswith("\t"):
            end_index = i
            continue

        # Keep going if this line has code-like characters
        if any(char in stripped for char in "{}[]()"):
            end_index = i
            continue

        # Stop if we find clear explanatory text
        if any(
            pattern in stripped.lower()
            for pattern in [
                "this code will output",
                "example usage",
                "output:",
                "returns:",
                "raises:",
                "note:",
                "warning:",
                "caution:",
                "version:",
                "added:",
                "changed:",
                "```",
                "the output will be",
                "this will produce",
                "running this code",
                "when you run this",
                "if you execute this",
                "this function will",
                "this method will",
                "this class will",
                "this will return",
                "this will print",
                "this will output",
                "this will display",
                "this will show",
                "this will generate",
                "this will create",
            ]
        ):
            break

        # If we reach here, assume it's still part of the code
        end_index = i

    # Extract the code lines
    if start_index >= 0 and end_index >= start_index:
        code_lines = lines[start_index : end_index + 1]

        # Clean up trailing empty lines
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()

        if code_lines:
            result = "\n".join(code_lines).strip()
            logger.debug(
                f"Simple fallback extraction found {len(code_lines)} lines of code"
            )
            return result

    # If extraction failed, return original text
    logger.debug("Simple fallback extraction failed, returning original text")
    return text


def parse_filename(filename: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    """
    Parse filename to extract model, dataset, technique, and task range.

    Expected format: model-dataset-technique.jsonl or model-dataset-technique-taskrange.jsonl
    Handles model names with dashes like "phi-3", "phi-4", etc.

    Returns:
        Tuple of (model, dataset, technique, task_range) or None if parsing fails
    """
    # Remove .jsonl extension
    base_name = filename.replace(".jsonl", "")

    # Split by hyphens
    parts = base_name.split("-")

    if len(parts) < 3:
        return None

    # Find the model name by checking if it appears in the filename
    model: Optional[str] = None
    for model_name in MODELS:
        if model_name in base_name:
            model = model_name
            break

    # If we cannot infer the model, bail out (satisfies typing and avoids Nones)
    if model is None:
        return None

    # For dataset and technique, we know the structure is model-dataset-technique
    # Find the position after the model name and extract from there
    model_parts = model.split("-")
    if len(parts) >= len(model_parts) + 2:
        dataset = parts[len(model_parts)]
        technique = parts[len(model_parts) + 1]
    else:
        return None

    # Check if there's a task range in the remaining parts
    task_range: Optional[str] = None
    expected_parts = len(model.split("-")) + 2  # model parts + dataset + technique
    if len(parts) > expected_parts:
        # Join remaining parts and check if it looks like a task range
        remaining = "-".join(parts[expected_parts:])
        if re.match(r"\d+-\d+", remaining):
            task_range = remaining

    return model, dataset, technique, task_range


def extract_solution_from_result(
    result: Dict, dataset: str, technique: str, model: str
) -> Optional[str]:
    """
    Extract the solution code from a result based on the technique used.

    Args:
        result: The result dictionary from the generated file
        dataset: The dataset name
        technique: The technique used (baseline, quality_focused, persona, cot, rci)
        model: The model name (short name from filename)

    Returns:
        The extracted solution code or None if not found
    """
    try:
        # Map the short model name to the actual model key in JSON
        actual_model_key = MODEL_MAPPING.get(model, model)

        # Check if the model exists in generations
        if "generations" not in result or actual_model_key not in result["generations"]:
            logger.warning(
                f"No generations found for model {actual_model_key} (from {model}) in task {result.get('task_id', 'unknown')}"
            )
            return None

        model_generations = result["generations"][actual_model_key]

        if technique == "rci":
            # For RCI, extract the improved_code
            if (
                "rci" in model_generations
                and "improved_code" in model_generations["rci"]
            ):
                return model_generations["rci"]["improved_code"]
            else:
                logger.warning(
                    f"No improved_code found in RCI for task {result.get('task_id', 'unknown')}"
                )
                return None

        elif technique == "cot":
            # For CoT, extract the final_code
            if "cot" in model_generations and "final_code" in model_generations["cot"]:
                return model_generations["cot"]["final_code"]
            else:
                logger.warning(
                    f"No final_code found in CoT for task {result.get('task_id', 'unknown')}"
                )
                return None

        else:
            # For baseline, quality_focused, persona - extract the direct technique field
            if technique in model_generations:
                return model_generations[technique]
            else:
                logger.warning(
                    f"No {technique} found for task {result.get('task_id', 'unknown')}"
                )
                return None

    except Exception as e:
        logger.error(
            f"Error extracting solution for task {result.get('task_id', 'unknown')}: {str(e)}"
        )
        return None


def process_single_file(
    input_file: str, dataset: str, technique: str, model: str
) -> List[Dict]:
    """
    Process a single generated code file and extract results.

    Args:
        input_file: Path to the input generated code file
        dataset: The dataset name
        technique: The technique used
        model: The model name

    Returns:
        List of extracted results with task_id and solution
    """
    logger.info(f"Processing {input_file} for technique {technique} and model {model}")

    extracted_results = []
    processed_count = 0
    successful_count = 0

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    task_id = result.get("task_id", "")
                    solution = extract_solution_from_result(
                        result, dataset, technique, model
                    )
                    solution = extract_python_code(solution)

                    if solution:
                        extracted_results.append(
                            {"task_id": task_id, "solution": solution}
                        )
                        successful_count += 1
                    else:
                        logger.warning(f"No solution extracted for task {task_id}")

                    processed_count += 1

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue

        logger.info(f"Successfully processed {processed_count} tasks")
        logger.info(f"Successfully extracted {successful_count} solutions")

        return extracted_results

    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        return []


def merge_results_by_group(results_dir: str, output_dir: str):
    """
    Process all files in results directory and merge by technique and model.

    Args:
        results_dir: Directory containing generated code files
        output_dir: Directory to save merged results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group files by (model, technique)
    file_groups = defaultdict(list)

    # Find all JSONL files in results directory
    for filename in os.listdir(results_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(results_dir, filename)

            # Parse filename to get model, technique, etc.
            parsed = parse_filename(filename)
            if parsed is None:
                logger.warning(f"Could not parse filename: {filename}")
                continue

            model, dataset, technique, task_range = parsed
            logger.info(
                f"Parsed {filename}: model={model}, dataset={dataset}, technique={technique}, task_range={task_range}"
            )

            group_key = (model, dataset, technique)
            file_groups[group_key].append((file_path, task_range))

    logger.info(f"Found {len(file_groups)} groups of files to process")

    # Process each group
    for (model, dataset, technique), files in file_groups.items():
        logger.info(f"\nProcessing group: {model} - {dataset} - {technique}")
        logger.info(f"Found {len(files)} files for this group")

        all_results = []

        # Sort files by task range if available
        files.sort(key=lambda x: x[1] if x[1] else "")

        # Process each file in the group
        for file_path, task_range in files:
            logger.info(f"Processing file: {os.path.basename(file_path)}")
            if task_range:
                logger.info(f"Task range: {task_range}")

            results = process_single_file(file_path, dataset, technique, model)
            all_results.extend(results)

        # Remove duplicates based on task_id (keep the last occurrence)
        unique_results = {}
        for result in all_results:
            unique_results[result["task_id"]] = result

        # Sort by task_id for consistent output
        final_results = sorted(unique_results.values(), key=lambda x: x["task_id"])

        # Write merged results
        output_filename = f"{model}-{dataset}-{technique}-merged.jsonl"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in final_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"Merged {len(final_results)} unique results to {output_path}")

    logger.info(f"\nCompleted processing all groups. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract and merge results from generated code files"
    )
    parser.add_argument("results_dir", help="Directory containing generated code files")
    parser.add_argument(
        "--output_dir",
        default="extracted_results",
        help="Output directory for merged results (default: extracted_results)",
    )

    args = parser.parse_args()

    # Validate input directory exists
    if not os.path.exists(args.results_dir):
        logger.error(f"Results directory does not exist: {args.results_dir}")
        return

    # Process and merge results
    merge_results_by_group(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
