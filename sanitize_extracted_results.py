"""
Sanitize extracted results from batch_extract_results.py output.
Works with simple JSONL format: {"task_id": "...", "solution": "..."}
"""

import os
import json
import argparse
import logging
from typing import Optional

# Try to import tree-sitter
try:
    import tree_sitter_python
    from tree_sitter import Language, Node, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print(
        "Warning: tree-sitter not available. Install with: pip install tree-sitter tree-sitter-python"
    )

# Set up logging
logger = logging.getLogger("sanitize_extracted_results")
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler("sanitize_extracted_results.log")
stream_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_format)
stream_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Tree-sitter constants
CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def syntax_check(code: str) -> bool:
    """Check if code is syntactically valid using tree-sitter."""
    if not TREE_SITTER_AVAILABLE:
        return True  # Fallback to always valid if tree-sitter not available

    try:
        parser = Parser(Language(tree_sitter_python.language()))
        tree = parser.parse(bytes(code, "utf8"))
        return len(tree.root_node.children) > 0
    except Exception:
        return False


def code_extract(text: str) -> str:
    """Find the longest syntactically valid code block in the text."""
    if not TREE_SITTER_AVAILABLE:
        return text

    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_deps(nodes: list) -> dict:
    """Get dependencies for nodes."""

    def dfs_get_deps(node: Node, deps: set) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                text = child.text
                if text:
                    deps.add(text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps: set[str] = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: dict) -> set:
    """Get all functions reachable from entrypoint."""
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def get_definition_name(node: Node) -> str:
    """Extract the name from a definition node."""
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            text = child.text
            if text:
                return text.decode("utf8")
    return ""


def traverse_tree(node: Node):
    """Traverse tree nodes."""
    cursor = node.walk()
    depth = 0
    visited_children = False

    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    """Extract target code using tree-sitter analysis."""
    if not TREE_SITTER_AVAILABLE:
        return code

    code = code_extract(code.strip())
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)

    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if (
                name
                and name not in class_names
                and name not in variable_names
                and name not in function_names
            ):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if (
                name
                and name not in function_names
                and name not in variable_names
                and name not in class_names
            ):
                definition_nodes.append((name, child))
                function_names.add(name)
        elif (
            child.type == EXPRESSION_TYPE
            and child.children
            and child.children[0].type == ASSIGNMENT_TYPE
        ):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if (
                name
                and name not in variable_names
                and name not in function_names
                and name not in class_names
            ):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    reachable = set()
    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = ""

    # Add imports
    for node in import_nodes:
        sanitized_output += (
            code_bytes[node.start_byte : node.end_byte].decode("utf8") + "\n"
        )

    # Add reachable definitions
    for pair in definition_nodes:
        name, node = pair
        if entrypoint and name not in reachable:
            continue
        sanitized_output += (
            code_bytes[node.start_byte : node.end_byte].decode("utf8") + "\n"
        )

    # Remove unnecessary lines
    lines = sanitized_output.splitlines()
    outer_lines = []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(" "):
            break
        if not lines[i].startswith(" ") and entrypoint and entrypoint in lines[i]:
            outer_lines.append(i)
    if outer_lines:
        sanitized_output = "\n".join(lines[: outer_lines[-1]])

    return sanitized_output.strip()


def clean_code_blocks(code: str) -> str:
    """Extract Python code from markdown text, removing explanatory text."""
    import re

    # First, try to find complete Python code blocks
    python_blocks = re.findall(r"```python\s*\n(.*?)\n```", code, re.DOTALL)
    if python_blocks:
        return python_blocks[0].strip()

    # If no complete blocks found, look for incomplete blocks (starting with ```python but no closing ```)
    incomplete_block_match = re.search(r"```python\s*\n(.*)", code, re.DOTALL)
    if incomplete_block_match:
        return incomplete_block_match.group(1).strip()

    # If no markdown blocks found, try to find Python code without markdown
    # Look for lines that start with Python keywords or indentation
    lines = code.split("\n")
    python_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        # Check if line looks like Python code
        if (
            stripped.startswith(
                (
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
                )
            )
            or stripped.startswith(("return ", "yield ", "raise ", "assert "))
            or stripped.startswith(("elif ", "else:", "finally:"))
            or stripped.startswith(("pass", "break", "continue"))
            or stripped.startswith(("True", "False", "None"))
            or stripped.startswith(("print(", "len(", "range(", "str(", "int("))
            or line.startswith(" ")  # Indented lines
            or stripped.endswith(":")  # Lines ending with colon
            or "=" in stripped  # Assignment statements
            or stripped.startswith("#")  # Comments
        ):
            python_lines.append(line)
            in_code_block = True
        elif in_code_block and stripped:  # Continue code block if we're in one
            python_lines.append(line)
        elif in_code_block and not stripped:  # Empty line in code block
            python_lines.append(line)

    if python_lines:
        return "\n".join(python_lines).strip()

    # If still no code found, return original but cleaned
    return code.strip()


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    """Main sanitization function."""
    if not TREE_SITTER_AVAILABLE:
        logger.warning("Tree-sitter not available, returning original code")
        return code

    # First clean any markdown code blocks
    code = clean_code_blocks(code)

    try:
        # If no entrypoint specified, do basic code extraction without removing unreachable code
        if entrypoint is None:
            return code_extract(code).strip() or code

        sanitized_code = extract_target_code_or_empty(code, entrypoint).strip()

        # If sanitization resulted in empty or very short code, return original
        if not sanitized_code or len(sanitized_code) < len(code) * 0.3:
            logger.warning(
                f"Sanitization removed too much code (entrypoint: {entrypoint}), keeping original"
            )
            return code_extract(code).strip() or code

        return sanitized_code
    except Exception as e:
        logger.error(f"Sanitization failed: {str(e)}")
        return code


def load_entrypoints_from_dataset():
    """Load entry points from the CoderEval dataset file."""
    entrypoints = {}
    dataset_file = "dataset/CEPythonHumanLabel.jsonl"

    if not os.path.exists(dataset_file):
        logger.warning(f"Dataset file not found: {dataset_file}")
        return entrypoints

    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                question_id = data.get("question_id", "")
                signature = data.get("signature", "")

                if signature:
                    # Extract function name from signature (e.g., "def function_name(...)" -> "function_name")
                    import re

                    match = re.search(r"def\s+(\w+)\s*\(", signature)
                    if match:
                        function_name = match.group(1)
                        entrypoints[question_id] = function_name
                        logger.debug(
                            f"Loaded entrypoint for {question_id}: {function_name}"
                        )

    except Exception as e:
        logger.error(f"Error loading entrypoints: {str(e)}")

    logger.info(f"Loaded {len(entrypoints)} entrypoints from dataset")
    return entrypoints


def load_problem_prompts_from_dataset():
    """Load problem prompts from the CoderEval dataset file for calibration."""
    problem_prompts = {}
    dataset_file = "dataset/CEPythonHumanLabel.jsonl"

    if not os.path.exists(dataset_file):
        logger.warning(f"Dataset file not found: {dataset_file}")
        return problem_prompts

    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                question_id = data.get("question_id", "")
                input_prompt = data.get("input", "")
                if question_id and input_prompt:
                    problem_prompts[question_id] = input_prompt
    except Exception as e:
        logger.error(f"Error loading problem prompts: {str(e)}")

    logger.info(f"Loaded {len(problem_prompts)} problem prompts from dataset")
    return problem_prompts


def calibrate_code(code: str, problem_prompt: str) -> str:
    """
    Calibrate code by reconstructing the complete program from problem prompt and solution.

    This function takes the solution code and reconstructs a complete, runnable program
    by combining it with the original problem prompt (function signature + docstring).
    """
    if not problem_prompt:
        return code

    # Clean the solution code first
    cleaned_code = clean_code_blocks(code)

    # If the solution already contains the function signature, return as is
    if any(line.strip().startswith("def ") for line in cleaned_code.split("\n")):
        return cleaned_code

    # Reconstruct the complete program
    # Format: problem_prompt + "\n" + solution_body
    complete_program = problem_prompt.rstrip() + "\n"

    # Find the first non-empty line in the solution that's not a comment
    lines = cleaned_code.split("\n")
    solution_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            solution_start = i
            break

    # Add the solution body (indented to match the function body)
    solution_body = "\n".join(lines[solution_start:])

    # If solution body is empty, just return the problem prompt
    if not solution_body.strip():
        return problem_prompt

    # Add the solution body with proper indentation
    complete_program += solution_body

    return complete_program


def process_jsonl_file(
    input_file: str, output_file: str, enable_calibration: bool = True
):
    """Process a JSONL file containing extracted results and sanitize them."""
    logger.info(f"Processing {input_file}...")

    # Load entrypoints for CoderEval tasks
    entrypoints = load_entrypoints_from_dataset()

    # Load problem prompts for CoderEval tasks (for calibration) if enabled
    problem_prompts = {}
    if enable_calibration:
        problem_prompts = load_problem_prompts_from_dataset()
        logger.info(
            f"Calibration enabled - loaded {len(problem_prompts)} problem prompts"
        )
    else:
        logger.info("Calibration disabled")

    sanitized_data = []
    processed_count = 0
    sanitized_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                task_id = item.get("task_id", "")
                solution = item.get("solution", "")

                if solution:
                    original_solution = solution

                    # Determine the entry point and apply calibration for CoderEval
                    entrypoint = None
                    question_id = None

                    if "BigCodeBench" in task_id:
                        entrypoint = "task_func"
                    else:
                        # For CoderEval, use the entrypoint from the dataset
                        # Extract question_id from task_id (e.g., "62b43425903eeb48555d3ea1" from "62b43425903eeb48555d3ea1")
                        question_id = (
                            task_id.split("/")[-1] if "/" in task_id else task_id
                        )
                        entrypoint = entrypoints.get(question_id)

                        if entrypoint:
                            logger.debug(
                                f"Using entrypoint '{entrypoint}' for task {task_id}"
                            )
                        else:
                            logger.warning(f"No entrypoint found for task {task_id}")

                    # Apply calibration for CoderEval tasks (if enabled)
                    calibrated_solution = solution
                    if (
                        enable_calibration
                        and question_id
                        and question_id in problem_prompts
                    ):
                        problem_prompt = problem_prompts[question_id]
                        calibrated_solution = calibrate_code(solution, problem_prompt)
                        if calibrated_solution != solution:
                            logger.debug(f"Calibrated solution for {task_id}")
                            logger.debug(f"Original: {solution[:100]}...")
                            logger.debug(f"Calibrated: {calibrated_solution[:100]}...")

                    # Sanitize the solution (use calibrated version if available)
                    logger.debug(f"Input solution length: {len(calibrated_solution)}")
                    sanitized_solution = sanitize(calibrated_solution, entrypoint)
                    logger.debug(
                        f"Sanitized solution length: {len(sanitized_solution)}"
                    )

                    if sanitized_solution != original_solution:
                        sanitized_count += 1
                        logger.info(f"Sanitized: {task_id} (entrypoint: {entrypoint})")
                        logger.debug(f"Original: {original_solution[:100]}...")
                        logger.debug(f"Sanitized: {sanitized_solution[:100]}...")

                    item["solution"] = sanitized_solution

                sanitized_data.append(item)
                processed_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                continue

    # Write sanitized data
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sanitized_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"Processed {processed_count} items, sanitized {sanitized_count} solutions"
    )
    logger.info(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Sanitize extracted results")
    parser.add_argument(
        "input", help="Input JSONL file or directory containing extracted results"
    )
    parser.add_argument(
        "--output", help="Output file or directory (auto-generated if not provided)"
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable code calibration (reconstruction of complete programs)",
    )

    args = parser.parse_args()

    if os.path.isfile(args.input):
        # Single file
        if args.output is None:
            args.output = args.input.replace(".jsonl", "-sanitized.jsonl")
        process_jsonl_file(
            args.input, args.output, enable_calibration=not args.no_calibration
        )
    elif os.path.isdir(args.input):
        # Directory
        output_dir = args.output if args.output else args.input + "-sanitized"
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(args.input):
            if filename.endswith(".jsonl"):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(
                    output_dir, filename.replace(".jsonl", "-sanitized.jsonl")
                )
                process_jsonl_file(
                    input_path, output_path, enable_calibration=not args.no_calibration
                )
    else:
        logger.error(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()
