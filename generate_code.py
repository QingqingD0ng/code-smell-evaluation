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
import datasets

# Define the models to use
PRODUCTION_MODELS = [
    ("qwen", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    ("phi-4", "microsoft/phi-4"),
    ("llama", "meta-llama/Llama-3.3-70B-Instruct"),
]

DEBUG_MODELS = [
    ("qwen3", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
]

# Define prompt templates
PROMPT_TEMPLATES = {
    "baseline": "Generate Python code for the following: {task}",
    "quality_focused": "Generate Python code for the following, ensuring it is clean and avoids code smells: {task}",
    "cot": {
        "initial": """Generate Python code for the following: {task}
Ensuring it is clean and avoids code smells. Let's think step by step.""",
        "final": "Therefore, final clean Python implementation is:",
    },
    "rci": {
        "initial": "Generate Python code for the following: {task}",
        "review": "Review your previous answer and find code smells with it",
        "improve": "Based on the code smells you found, improve your answer",
    },
    "persona": """Act as a software quality expert. Provide outputs that a quality expert would give. Ensuring it is clean and avoids code smells.
Generate clean Python code for the following: {task}""",
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
    def __init__(self, model_name):
        self.device = get_device()
        self.model_name = model_name
        print(f"Using device: {self.device}")

        # Use pipeline for all models
        self.model = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="float16",
            device_map="auto",
            max_new_tokens=512,
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

        print(f"Cleaned up resources for model: {self.model_name}")


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
    # Convert tasks to Dataset
    task_dataset = Dataset.from_dict(
        {
            "task_id": [task.get("task_id") for task in tasks],
            "prompt": [task.get("complete_prompt") for task in tasks],
        }
    )

    if technique not in PROMPT_TEMPLATES:
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
    responses = list(
        tqdm(
            generator.model(KeyDataset(prompt_dataset, "messages")),
            desc=f"Processing {technique}",
            total=len(prompt_dataset),
        )
    )

    # Return formatted results
    return [resp[0]["generated_text"][-1]["content"] for resp in responses]


def main():
    # Add argument parser for debug mode
    parser = argparse.ArgumentParser(
        description="Generate code using different prompting strategies"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with smaller model"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: use full dataset)",
    )
    parser.add_argument(
        "--dataset",
        choices=["bigcodebench", "codereval", "both"],
        default="bigcodebench",
        help="Dataset to use (BigCodeBench, CoderEval, or both)",
    )
    parser.add_argument(
        "--output",
        default="results.jsonl",
        help="Output JSONL file name",
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "phi-4", "llama", "all"],
        help="Model(s) to use for generation (required when not in debug mode).",
    )
    args = parser.parse_args()

    # Validate model argument
    if not args.debug and not args.model:
        parser.error("--model is required when not in debug mode")

    # Process model selection
    if args.debug:
        MODEL_LIST = DEBUG_MODELS
    else:
        if args.model == "all":
            selected_models = list(PRODUCTION_MODELS.keys())
        else:
            selected_models = [args.model.strip()]
        MODEL_LIST = [(model, PRODUCTION_MODELS[model]) for model in selected_models]

    # Set dataset name based on argument
    DATASET_NAME = (
        "bigcode/bigcodebench-hard"
        if args.dataset in ["bigcodebench", "both"]
        else None
    )
    if not args.debug:
        with open("/scratch/qido00001/hf_key.txt", "r") as file:
            key = file.read().strip()
        login(key)

    # Print system information
    print("\nSystem Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}\n")

    # Load datasets based on argument
    datasets = []
    if args.dataset in ["bigcodebench", "both"]:
        bigcodebench_dataset = load_dataset(DATASET_NAME, split="v0.1.4")
        datasets.append(("bigcodebench", bigcodebench_dataset))

    if args.dataset in ["codereval", "both"]:
        jsonl_dataset = load_jsonl_dataset("dataset/CEPythonHumanLabel.jsonl")
        datasets.append(("codereval", jsonl_dataset))

    # Create or load existing results
    all_results = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                all_results[result["task_id"]] = result

    # Process each model
    for model_display, model_internal in MODEL_LIST:
        print(f"\nProcessing model: {model_display}")

        # Create generator for this model
        generator = CodeGenerator(model_internal)

        try:
            # Process each dataset
            for dataset_name, dataset in datasets:
                # Handle different dataset types
                if dataset_name == "bigcodebench":
                    items = (
                        dataset.select(range(args.num_samples))
                        if args.num_samples
                        else dataset
                    )
                else:  # codereval dataset (list)
                    items = dataset[: args.num_samples] if args.num_samples else dataset

                print(f"\nProcessing {len(items)} tasks from {dataset_name}")

                # Process each technique for all tasks
                techniques = ["baseline", "quality_focused", "persona", "cot", "rci"]

                # First generate baseline responses for all tasks
                print("\nGenerating baseline responses for all tasks...")
                baseline_codes = process_technique_batch(generator, items, "baseline")

                for technique in techniques:
                    print(f"\nProcessing technique: {technique}")

                    if technique == "baseline":
                        results = [{"baseline": r} for r in baseline_codes]
                    elif technique in ["cot", "rci"]:
                        # For CoT and RCI, use the baseline codes we already generated
                        task_dataset = Dataset.from_dict(
                            {
                                "task_id": [task.get("task_id") for task in items],
                                "prompt": [
                                    task.get("complete_prompt") for task in items
                                ],
                                "baseline_code": baseline_codes,
                            }
                        )

                        if technique == "cot":
                            # Generate reasoning using CoT initial prompt
                            reasoning_dataset = Dataset.from_dict(
                                {
                                    "messages": [
                                        [
                                            {
                                                "role": "user",
                                                "content": PROMPT_TEMPLATES["cot"][
                                                    "initial"
                                                ].format(task=task),
                                            }
                                        ]
                                        for task in task_dataset["prompt"]
                                    ]
                                }
                            )
                            reasoning_responses = list(
                                tqdm(
                                    generator.model(
                                        KeyDataset(reasoning_dataset, "messages")
                                    ),
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
                                                "content": PROMPT_TEMPLATES["cot"][
                                                    "final"
                                                ],
                                            }
                                        ]
                                        for messages in reasoning_responses
                                    ]
                                }
                            )
                            final_responses = list(
                                tqdm(
                                    generator.model(
                                        KeyDataset(final_dataset, "messages")
                                    ),
                                    desc="Generating final implementation for CoT",
                                    total=len(final_dataset),
                                )
                            )

                            results = [
                                {
                                    "cot": {
                                        "reasoning": reasoning[0]["generated_text"][-1][
                                            "content"
                                        ],
                                        "final_code": final[0]["generated_text"][-1][
                                            "content"
                                        ],
                                    }
                                }
                                for reasoning, final in zip(
                                    reasoning_responses, final_responses
                                )
                            ]

                        elif technique == "rci":
                            # Generate review
                            review_dataset = Dataset.from_dict(
                                {
                                    "messages": [
                                        [
                                            {
                                                "role": "user",
                                                "content": PROMPT_TEMPLATES["rci"][
                                                    "initial"
                                                ].format(task=task),
                                            },
                                            {"role": "assistant", "content": code},
                                            {
                                                "role": "user",
                                                "content": PROMPT_TEMPLATES["rci"][
                                                    "review"
                                                ],
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
                                    generator.model(
                                        KeyDataset(review_dataset, "messages")
                                    ),
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
                                                "content": PROMPT_TEMPLATES["rci"][
                                                    "improve"
                                                ],
                                            }
                                        ]
                                        for review in review_responses
                                    ]
                                }
                            )
                            improve_responses = list(
                                tqdm(
                                    generator.model(
                                        KeyDataset(improve_dataset, "messages")
                                    ),
                                    desc="Generating improved code for RCI",
                                    total=len(improve_dataset),
                                )
                            )

                            results = [
                                {
                                    "rci": {
                                        "initial_code": baseline,
                                        "review": review[0]["generated_text"][-1][
                                            "content"
                                        ],
                                        "improved_code": improved[0]["generated_text"][
                                            -1
                                        ]["content"],
                                    }
                                }
                                for baseline, review, improved in zip(
                                    task_dataset["baseline_code"],
                                    review_responses,
                                    improve_responses,
                                )
                            ]
                    else:
                        results = [
                            {technique: r}
                            for r in process_technique_batch(
                                generator, items, technique
                            )
                        ]

                    # Update results for each task
                    for task, result in zip(items, results):
                        task_id = task.get("task_id")

                        # Initialize result dictionary for this task if not exists
                        if task_id not in all_results:
                            all_results[task_id] = {
                                "dataset": dataset_name,
                                "task_id": task_id,
                                "original_prompt": task.get("complete_prompt"),
                                "generations": {},
                            }

                        # Initialize model's generations if not exists
                        if model_display not in all_results[task_id]["generations"]:
                            all_results[task_id]["generations"][model_display] = {}

                        # Update with the new results
                        all_results[task_id]["generations"][model_display].update(
                            result
                        )

                        # Write results after each technique
                        with open(args.output, "w", encoding="utf-8") as f:
                            for tid, result in all_results.items():
                                f.write(json.dumps(result) + "\n")
                            f.flush()

        finally:
            # Clean up resources after processing all tasks for this model
            print(f"\nCleaning up resources for model: {model_display}")
            generator.cleanup()
            del generator

    print(f"Done. All results saved in {args.output}")


if __name__ == "__main__":
    main()
