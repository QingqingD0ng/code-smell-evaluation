import os
import json
from datasets import load_dataset, Dataset
import torch
from huggingface_hub import login
import argparse
import platform
import transformers
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import logging
import time
import traceback

# Set up logging for code generation
logger = logging.getLogger("code_generation")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("code_generation.log")
stream_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_format)
stream_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Define the models to use
MODELS = {
    "qwen": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "phi-4": "microsoft/phi-4",
    "phi-3": "microsoft/Phi-3-mini-128k-instruct",
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen0.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen0.5-coder": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
}

# Define prompt templates
PROMPT_TEMPLATES = {
    "baseline": "Generate Python code for the following task. Avoid docstrings and comments. {task}",
    "quality_focused": "Generate Python code for the following task, ensuring it is clean and avoids code smells. Avoid docstrings and comments. {task}",
    "cot": {
        "initial": "Generate Python code for the following task, ensuring it is clean and avoids code smells. Avoid docstrings and comments. Let's think step by step. {task}",
        "final": "Therefore, final Python implementation without docstrings or comments is:",
    },
    "rci": {
        "initial": "Generate Python code for the following task. Avoid docstrings and comments. {task}",
        "review": "Review your previous answer and find code smells with it",
        "improve": "Improve your code based on the code smells you found. Therefore, final Python implementation without docstrings or comments is:",
    },
    "persona": "Act as a software quality expert. Provide outputs that a quality expert would give. Generate Python code for the following task. Avoid docstrings and comments. {task}",
}


def get_device():
    """Get the appropriate device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class CodeGenerator:
    def __init__(self, model_name, max_new_tokens=512):
        self.device = get_device()
        self.model_name = model_name
        logger.info(f"Using device: {self.device}")

        # Use pipeline for all models
        self.model = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="float16",
            device_map="auto",
            max_new_tokens=max_new_tokens,
        )

    def cleanup(self):
        """Clean up model resources and free memory"""
        if hasattr(self, "model"):
            del self.model

        # Force garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Cleaned up resources for model: {self.model_name}")


def load_jsonl_dataset(file_path):
    """Load and process the JSONL dataset"""
    tasks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tasks.append(
                {"task_id": data["question_id"], "complete_prompt": data["input"]}
            )
    return tasks


def process_technique_batch(generator, tasks, technique):
    """Process all tasks with a specific technique using Dataset for batching"""
    logger.info(f"Starting {technique} processing for {len(tasks)} tasks")
    start_time = time.time()

    # Convert tasks to Dataset
    task_dataset = Dataset.from_dict(
        {
            "task_id": [task.get("task_id") for task in tasks],
            "prompt": [task.get("complete_prompt") for task in tasks],
        }
    )

    if technique not in PROMPT_TEMPLATES:
        logger.error(f"Unknown technique: {technique}")
        raise ValueError(f"Unknown technique: {technique}")

    # Create dataset with formatted prompts
    prompt_dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATES[technique].format(task=prompt),
                    }
                ]
                for prompt in task_dataset["prompt"]
            ]
        }
    )

    # Generate responses
    try:
        responses = list(
            tqdm(
                generator.model(KeyDataset(prompt_dataset, "messages")),
                desc=f"Processing {technique}",
                total=len(prompt_dataset),
            )
        )
        logger.info(
            f"Successfully generated {len(responses)} responses for {technique}"
        )
    except Exception as e:
        logger.error(f"Error generating responses for {technique}: {str(e)}")
        raise

    # Return formatted results
    results = [resp[0]["generated_text"][-1]["content"] for resp in responses]
    end_time = time.time()
    logger.info(
        f"Completed {technique} processing in {end_time - start_time:.2f} seconds"
    )
    return results


def main():
    # Set up logging
    logger.info("Starting code generation process")

    # Add argument parser for debug mode
    parser = argparse.ArgumentParser(description="Generate code using various models")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug models instead of production models",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (qwen, phi-4, phi-3, llama, qwen0.5, qwen0.5-coder)",
        required=False,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="bigcodebench",
        help="Dataset to use (codereval or bigcodebench)",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--task_range",
        type=str,
        help="Range of tasks to process (e.g., '100-200')",
        default=None,
    )
    parser.add_argument(
        "--technique",
        type=str,
        choices=["baseline", "quality_focused", "persona", "cot", "rci"],
        help="Technique to use for code generation",
        default="baseline",
    )
    parser.add_argument(
        "--baseline_file",
        type=str,
        help="JSONL file containing baseline code (required for rci technique)",
        default=None,
    )
    args = parser.parse_args()

    # Validate arguments
    if args.technique == "rci" and not args.baseline_file:
        parser.error("--baseline_file is required for rci techniques")

    # Process model selection
    selected_model = MODELS[args.model]
    logger.info(f"Using model: {selected_model}")

    # Set default output filename
    if args.task_range:
        args.output = (
            f"{args.model}-{args.dataset}-{args.technique}-{args.task_range}.jsonl"
        )
    else:
        args.output = f"{args.model}-{args.dataset}-{args.technique}.jsonl"
    logger.info(f"Output will be saved to: {args.output}")

    # Process task range if specified
    task_range = None
    if args.task_range:
        try:
            start, end = map(int, args.task_range.split("-"))
            task_range = (start, end)
            logger.info(f"Processing tasks from {start} to {end}")
        except ValueError:
            parser.error("Task range must be in format 'start-end' (e.g., '100-200')")

    # Set dataset name based on argument
    DATASET_NAME = (
        "bigcode/bigcodebench-hard" if args.dataset == "bigcodebench" else "coderEval"
    )
    if not args.debug:
        with open("/scratch/qido00001/hf_key.txt", "r") as file:
            key = file.read().strip()
        login(key)

    # Print system information
    logger.info("\nSystem Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}\n")

    # Load datasets based on argument
    if args.dataset == "bigcodebench":
        dataset = load_dataset(DATASET_NAME, split="v0.1.4")
        if task_range:
            items = dataset.select(range(task_range[0], task_range[1]))
        else:
            items = dataset

    if args.dataset == "codereval":
        dataset = load_jsonl_dataset("dataset/CEPythonHumanLabel.jsonl")
        if task_range:
            items = dataset[task_range[0] : task_range[1]]
        else:
            items = dataset

    # Create or load existing results
    all_results = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                all_results[result["task_id"]] = result

        # Process each model

    generator = CodeGenerator(selected_model, max_new_tokens=args.max_new_tokens)

    try:
        # Process each dataset
        logger.info(f"\nProcessing {len(items)} tasks from {args.dataset}")

        # Process each technique for all tasks
        technique = args.technique  # Only process the specified technique

        # First generate baseline responses for all tasks if needed
        if args.technique == "rci":
            logger.info(f"\nLoading baseline code from {args.baseline_file}")
            baseline_codes = {}
            with open(args.baseline_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    task_id = data["task_id"]
                    if "generations" in data and selected_model in data["generations"]:
                        if "baseline" in data["generations"][selected_model]:
                            baseline_codes[task_id] = data["generations"][
                                selected_model
                            ]["baseline"]

            # Filter items to only those that have baseline code
            items = [item for item in items if item.get("task_id") in baseline_codes]
            if not items:
                raise ValueError(
                    f"No matching baseline code found in {args.baseline_file}"
                )
            logger.info(f"Found baseline code for {len(items)} tasks")

            # Extract baseline codes in the same order as items
            baseline_codes = [baseline_codes[item.get("task_id")] for item in items]
            # For CoT and RCI, use the baseline codes we already generated
            task_dataset = Dataset.from_dict(
                {
                    "task_id": [task.get("task_id") for task in items],
                    "prompt": [task.get("complete_prompt") for task in items],
                    "baseline_code": baseline_codes,
                }
            )

            # Generate review
            review_dataset = Dataset.from_dict(
                {
                    "messages": [
                        [
                            {
                                "role": "user",
                                "content": PROMPT_TEMPLATES["rci"]["initial"].format(
                                    task=task
                                ),
                            },
                            {"role": "assistant", "content": code},
                            {
                                "role": "user",
                                "content": PROMPT_TEMPLATES["rci"]["review"],
                            },
                        ]
                        for task, code in zip(
                            task_dataset["prompt"],
                            task_dataset["baseline_code"],
                        )
                    ]
                }
            )
            review_responses = list(
                tqdm(
                    generator.model(KeyDataset(review_dataset, "messages")),
                    desc="Generating review for RCI",
                    total=len(review_dataset),
                )
            )

            # Generate improved code
            improve_dataset = Dataset.from_dict(
                {
                    "messages": [
                        review[0]["generated_text"]
                        + [
                            {
                                "role": "user",
                                "content": PROMPT_TEMPLATES["rci"]["improve"],
                            }
                        ]
                        for review in review_responses
                    ]
                }
            )
            improve_responses = list(
                tqdm(
                    generator.model(KeyDataset(improve_dataset, "messages")),
                    desc="Generating improved code for RCI",
                    total=len(improve_dataset),
                )
            )

            results = [
                {
                    "rci": {
                        "initial_code": baseline,
                        "review": review[0]["generated_text"][-1]["content"],
                        "improved_code": improved[0]["generated_text"][-1]["content"],
                    }
                }
                for baseline, review, improved in zip(
                    task_dataset["baseline_code"],
                    review_responses,
                    improve_responses,
                )
            ]
        elif technique == "cot":
            # Generate reasoning using CoT initial prompt
            task_dataset = Dataset.from_dict(
                {
                    "task_id": [task.get("task_id") for task in items],
                    "prompt": [task.get("complete_prompt") for task in items],
                }
            )
            reasoning_dataset = Dataset.from_dict(
                {
                    "messages": [
                        [
                            {
                                "role": "user",
                                "content": PROMPT_TEMPLATES["cot"]["initial"].format(
                                    task=task
                                ),
                            }
                        ]
                        for task in task_dataset["prompt"]
                    ]
                }
            )
            reasoning_responses = list(
                tqdm(
                    generator.model(KeyDataset(reasoning_dataset, "messages")),
                    desc="Generating reasoning for CoT",
                    total=len(reasoning_dataset),
                )
            )
            # Generate final implementation
            final_dataset = Dataset.from_dict(
                {
                    "messages": [
                        messages[0]["generated_text"]
                        + [
                            {
                                "role": "user",
                                "content": PROMPT_TEMPLATES["cot"]["final"],
                            }
                        ]
                        for messages in reasoning_responses
                    ]
                }
            )
            final_responses = list(
                tqdm(
                    generator.model(KeyDataset(final_dataset, "messages")),
                    desc="Generating final implementation for CoT",
                    total=len(final_dataset),
                )
            )

            results = [
                {
                    "cot": {
                        "reasoning": reasoning[0]["generated_text"][-1]["content"],
                        "final_code": final[0]["generated_text"][-1]["content"],
                    }
                }
                for reasoning, final in zip(reasoning_responses, final_responses)
            ]
        else:
            results = [
                {technique: r}
                for r in process_technique_batch(generator, items, technique)
            ]

        # Update results for each task
        for task, result in zip(items, results):
            task_id = task.get("task_id")

            # Initialize result dictionary for this task if not exists
            if task_id not in all_results:
                all_results[task_id] = {
                    "dataset": DATASET_NAME,
                    "task_id": task_id,
                    "original_prompt": task.get("complete_prompt"),
                    "generations": {},
                }

            # Initialize model's generations if not exists
            if selected_model not in all_results[task_id]["generations"]:
                all_results[task_id]["generations"][selected_model] = {}

            # Update with the new results
            all_results[task_id]["generations"][selected_model].update(result)

            # Write results after each technique

            with open(args.output, "w", encoding="utf-8") as f:
                for tid, result in all_results.items():
                    f.write(json.dumps(result) + "\n")
                f.flush()

    except Exception as e:
        logger.error(f"Error processing model {selected_model}: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise
    finally:
        # Clean up resources after processing all tasks for this model
        logger.info(f"\nCleaning up resources for model: {selected_model}")
        generator.cleanup()
        del generator

    logger.info(f"Done. All results saved in {args.output}")


if __name__ == "__main__":
    main()
