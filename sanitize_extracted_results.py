"""
Post-processing LLM-generated Python code implemented using tree-sitter.
"""

import os
import json
import argparse
import logging
import ast
import traceback
from typing import Dict, Generator, List, Optional, Set, Tuple

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
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def syntax_check(code: str, verbose: bool = False) -> bool:
    """Check if code is syntactically valid using ast.parse()."""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str, verbose: bool = False) -> str:
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines, verbose):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:

    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
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


def get_function_dependency(
    entrypoint: str, call_graph: Dict[str, Set[str]]
) -> Set[str]:
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
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            text = child.text
            if text:
                return text.decode("utf8")
    return ""


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            current_node = cursor.node
            if current_node:
                yield current_node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False


def extract_target_code_or_empty(
    code: str, entrypoint: Optional[str] = None, verbose: bool = False
) -> str:
    code = code_extract(code.strip(), verbose)
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
            if not (
                name in class_names or name in variable_names or name in function_names
            ):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (
                name in function_names or name in variable_names or name in class_names
            ):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif (
            child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE
        ):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (
                name in variable_names or name in function_names or name in class_names
            ):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    reachable = set()
    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = ""

    for node in import_nodes:
        sanitized_output += (
            code_bytes[node.start_byte : node.end_byte].decode("utf8") + "\n"
        )

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


def sanitize(code: str, entrypoint: Optional[str] = None, verbose: bool = False) -> str:
    sanitized_code = extract_target_code_or_empty(code, entrypoint, verbose).strip()
    if not sanitized_code:
        return code_extract(code, verbose)
    return sanitized_code


def is_bigcodebench_task(task_id: str) -> bool:
    """Check if the task is from BigCodeBench dataset."""
    return "BigCodeBench" in task_id or "bigcodebench" in task_id.lower()


def process_jsonl_file(input_file: str, output_file: str, verbose: bool = False):
    """Process a JSONL file containing extracted results and sanitize only BigCodeBench tasks."""
    logger.info(f"Processing {input_file}...")

    sanitized_data = []
    processed_count = 0
    sanitized_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                task_id = item.get("task_id", "")
                solution = item.get("solution", "")

                if solution and is_bigcodebench_task(task_id):
                    original_solution = solution

                    # For BigCodeBench, use "task_func" as the entrypoint
                    entrypoint = "task_func"

                    # Sanitize the solution
                    logger.debug(f"Input solution length: {len(solution)}")
                    sanitized_solution = sanitize(solution, entrypoint, verbose=verbose)
                    logger.debug(
                        f"Sanitized solution length: {len(sanitized_solution)}"
                    )

                    if sanitized_solution != original_solution:
                        sanitized_count += 1
                        logger.info(
                            f"Sanitized BigCodeBench task: {task_id} (entrypoint: {entrypoint})"
                        )
                        logger.debug(f"Original: {original_solution[:100]}...")
                        logger.debug(f"Sanitized: {sanitized_solution[:100]}...")

                    item["solution"] = sanitized_solution
                else:
                    # For non-BigCodeBench tasks, keep the solution as-is
                    if solution and not is_bigcodebench_task(task_id):
                        logger.debug(
                            f"Skipping sanitization for non-BigCodeBench task: {task_id}"
                        )

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
        f"Processed {processed_count} items, sanitized {sanitized_count} BigCodeBench solutions"
    )
    logger.info(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize extracted results (BigCodeBench only)"
    )
    parser.add_argument(
        "input", help="Input JSONL file or directory containing extracted results"
    )
    parser.add_argument(
        "--output", help="Output file or directory (auto-generated if not provided)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose syntax error reporting"
    )

    args = parser.parse_args()

    if os.path.isfile(args.input):
        # Single file
        if args.output is None:
            args.output = args.input.replace(".jsonl", "_sanitized.jsonl")
        process_jsonl_file(args.input, args.output, args.verbose)
    elif os.path.isdir(args.input):
        # Directory
        output_dir = args.output if args.output else args.input + "_sanitized"
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(args.input):
            if filename.endswith(".jsonl"):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(
                    output_dir, filename.replace(".jsonl", "_sanitized.jsonl")
                )
                process_jsonl_file(input_path, output_path, args.verbose)
    else:
        logger.error(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()
