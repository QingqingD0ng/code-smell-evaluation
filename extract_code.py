import os
import csv
import re
import argparse
from generate_code import PRODUCTION_MODELS, DEBUG_MODELS, PROMPT_TEMPLATES
from collections import defaultdict


def extract_python_code(text):
    """Extract Python code from text that may contain markdown code blocks."""
    if not isinstance(text, str):
        return ""

    # Try to find code between ```python and ``` markers (case insensitive)
    pattern = r"```(?:python)?(.*?)(?:```|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        # Take the longest match if multiple are found
        return max(matches, key=len).strip()

    # If no ```python blocks found, try just ``` blocks
    pattern = r"```\\n([\s\S]*?)"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Take the longest match if multiple are found
        code = max(matches, key=len).strip()
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
        "class ",
        "import ",
        "from ",
        "#",
        "if ",
        "for ",
        "while ",
        "return ",
        "print(",
        "with ",
        "try:",
        "except:",
        "finally:",
        "@",
    )

    for line in lines:
        stripped = line.strip()
        # Start collecting code when we see Python-like patterns
        if any(stripped.startswith(indicator) for indicator in python_indicators) or (
            "=" in stripped and not stripped.startswith("=")
        ):
            in_code = True
        # Stop collecting when we hit markdown or non-code text
        elif in_code and (stripped.startswith("#") or stripped.startswith(">")):
            in_code = False

        if in_code:
            code_lines.append(line)
        elif code_lines and not stripped:
            # Keep empty lines between code blocks
            code_lines.append(line)

    if code_lines:
        # Clean up trailing empty lines
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        return "\n".join(code_lines).strip()

    return ""


def create_model_folders(base_path, models):
    """Create folders for each model and their prompting techniques."""
    # Create base directory
    os.makedirs(base_path, exist_ok=True)

    # Create folders for each model
    for model_name, _ in models:
        model_path = os.path.join(base_path, model_name)
        os.makedirs(model_path, exist_ok=True)


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
        return False

    # Remove backslashes from task_id to avoid path issues
    task_id = task_id.replace("\\", "_")

    filename = f"{task_id}.py"
    file_path = os.path.join(model_path, dataset, technique, filename)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Saved code to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}")
        return False


def process_csv(csv_path, output_base_path, models):
    """Process the CSV file and extract code into organized folders."""
    # Create all necessary folders
    create_model_folders(output_base_path, models)

    # Keep track of statistics
    stats = defaultdict(lambda: defaultdict(lambda: {"success": 0, "failed": 0}))

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            task_id = row["task_id"]
            dataset = row["dataset"]
            print(f"\nProcessing task {task_id}")

            # Process each model's outputs
            for model_name, _ in models:
                model_path = os.path.join(output_base_path, model_name)

                # Process each prompting technique
                for technique in PROMPT_TEMPLATES.keys():
                    # Handle special cases for cot and rci techniques
                    if technique in ["cot", "rci"]:
                        column_name = (
                            f"{model_name}_{technique}_final"
                            if technique == "cot"
                            else f"{model_name}_{technique}_improved"
                        )
                    else:
                        column_name = f"{model_name}_{technique}"

                    if column_name in row:
                        print(f"Extracting code for {model_name}/{technique}")
                        code = extract_python_code(row[column_name])

                        if save_code_to_file(
                            code, model_path, dataset, technique, task_id
                        ):
                            stats[model_name][technique]["success"] += 1
                        else:
                            stats[model_name][technique]["failed"] += 1

    # Print statistics
    print("\nExtraction Statistics:")
    for model_name, model_stats in stats.items():
        print(f"\n{model_name}:")
        for technique, counts in model_stats.items():
            print(f"  {technique}:")
            print(f"    Successful extractions: {counts['success']}")
            print(f"    Failed extractions: {counts['failed']}")


def main():
    # Add argument parser for debug mode
    parser = argparse.ArgumentParser(description="Extract code from CSV file")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with smaller models"
    )
    args = parser.parse_args()

    # Configure paths
    csv_path = "debug_results.csv" if args.debug else "results.csv"
    output_base_path = "extracted_code"

    # Select models based on debug flag
    models = DEBUG_MODELS if args.debug else PRODUCTION_MODELS

    # Process the CSV file
    print(f"Processing {csv_path}...")
    process_csv(csv_path, output_base_path, models)
    print("\nDone! Code has been extracted and organized into folders.")
    print(f"Output location: {os.path.abspath(output_base_path)}")


if __name__ == "__main__":
    main()
