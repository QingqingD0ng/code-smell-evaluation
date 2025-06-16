import os
import json
import re
import argparse
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


def process_jsonl(jsonl_path, output_base_path):
    """Process the JSONL file and extract code into organized folders."""

    stats = defaultdict(lambda: defaultdict(lambda: {"success": 0, "failed": 0}))

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            task_id = row["task_id"].split("/")[-1]
            dataset = row["dataset"].split("/")[-1]
            print(f"\nProcessing task {task_id} from {dataset}")

            # Process each model's generations
            for model_name, model_generations in row["generations"].items():
                model_name = model_name.split("/")[-1]
                print(f"Processing generations for model {model_name}")
                model_path = os.path.join(output_base_path, model_name)

                # Process each generation type
                for gen_type, content in model_generations.items():
                    if gen_type == "error":
                        print(
                            f"Error in task {task_id} for model {model_name}: {content}"
                        )
                        continue

                    if gen_type == "cot":
                        # Handle CoT special case
                        code = extract_python_code(content["final_code"])
                        if save_code_to_file(code, model_path, dataset, "cot", task_id):
                            stats[model_name]["cot"]["success"] += 1
                        else:
                            stats[model_name]["cot"]["failed"] += 1
                    elif gen_type == "rci":
                        # Handle RCI special case
                        code = extract_python_code(content["improved_code"])
                        if save_code_to_file(code, model_path, dataset, "rci", task_id):
                            stats[model_name]["rci"]["success"] += 1
                        else:
                            stats[model_name]["rci"]["failed"] += 1
                    else:
                        # Handle regular generation types
                        code = extract_python_code(content)
                        if save_code_to_file(
                            code, model_path, dataset, gen_type, task_id
                        ):
                            stats[model_name][gen_type]["success"] += 1
                        else:
                            stats[model_name][gen_type]["failed"] += 1

    # Print statistics
    print("\nExtraction Statistics:")
    for model_name, model_stats in stats.items():
        print(f"\n{model_name}:")
        for technique, counts in model_stats.items():
            print(f"  {technique}:")
            print(f"    Successful extractions: {counts['success']}")
            print(f"    Failed extractions: {counts['failed']}")


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description="Extract code from JSONL file")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file containing the generated code",
    )
    parser.add_argument(
        "--output",
        default="extracted_code",
        help="Output directory for extracted code (default: extracted_code)",
    )
    args = parser.parse_args()

    # Configure paths
    output_base_path = args.output

    # Process the JSONL file
    print(f"Processing {args.input}...")
    process_jsonl(args.input, output_base_path)
    print("\nDone! Code has been extracted and organized into folders.")
    print(f"Output location: {os.path.abspath(output_base_path)}")


if __name__ == "__main__":
    main()
