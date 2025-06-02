import os
import csv
import json
from datasets import load_dataset
import torch
from huggingface_hub import login
import argparse
import platform
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Set up cache location for cluster
cache_location = "/scratch/USER/tuning/.cache"
os.environ["HF_HOME"] = cache_location
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_location

# Define the models to use
PRODUCTION_MODELS = [
    ("qwen", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    ("phi-4", "microsoft/phi-4"),
    ("llama", "meta-llama/Llama-3.3-70B-Instruct"),
]

DEBUG_MODELS = [("qwen3", "Qwen/Qwen3-0.6B"), ("qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct")]

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

        # Use pipeline approach for phi-4 and llama models
        if "phi-4" in model_name or "llama" in model_name:
            self.model = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": "auto"},
                device_map="auto",
            )
            self.is_pipeline = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=self.device, trust_remote_code=True
            ).eval()
            self.is_pipeline = False
        self.history = []

    def cleanup(self):
        """Clean up model resources and free memory"""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        self.history = []

        # Force garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Cleaned up resources for model: {self.model_name}")

    def generate_response(self, prompt, max_new_tokens=1024):
        if self.is_pipeline:
            # For phi-4 and llama, use pipeline approach
            messages = self.history + [{"role": "user", "content": prompt}]
            outputs = self.model(messages, max_new_tokens=max_new_tokens)
            response = outputs[0]["generated_text"][-1]["content"]
        else:
            # Original approach for other models
            messages = self.history + [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        # Update history
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        return response

    def reset_history(self):
        self.history = []


def generate_cot_solution(task, model_id, generator):
    """Generate solution using Chain of Thought prompting"""
    # Step 1: Generate reasoning
    initial_prompt = PROMPT_TEMPLATES["cot"]["initial"].format(task=task)
    reasoning = generator.generate_response(initial_prompt)

    # Step 2: Generate final implementation
    final_prompt = PROMPT_TEMPLATES["cot"]["final"]
    final_code = generator.generate_response(final_prompt)

    return {"reasoning": reasoning, "final_code": final_code}


def generate_rci_solution(task, model_id, generator):
    """Generate solution using Review-Critique-Improve process"""
    # Step 1: Initial code generation
    initial_prompt = PROMPT_TEMPLATES["rci"]["initial"].format(task=task)
    initial_code = generator.generate_response(initial_prompt)

    # Step 2: Code review
    review_prompt = PROMPT_TEMPLATES["rci"]["review"]
    review = generator.generate_response(review_prompt)

    # Step 3: Code improvement
    improve_prompt = PROMPT_TEMPLATES["rci"]["improve"]
    improved_code = generator.generate_response(improve_prompt)

    return {
        "initial_code": initial_code,
        "review": review,
        "improved_code": improved_code,
    }


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
        help="Model(s) to use for generation (required when not in debug mode). Can specify multiple models separated by commas (e.g., 'qwen,phi-4')",
    )
    args = parser.parse_args()

    # Validate model argument
    if not args.debug and not args.model:
        parser.error("--model is required when not in debug mode")

    # Define model mapping
    MODEL_MAP = {
        "qwen": ("qwen", "Qwen/Qwen2.5-Coder-32B-Instruct"),
        "phi-4": ("phi-4", "microsoft/phi-4"),
        "llama": ("llama", "meta-llama/Llama-3.3-70B-Instruct"),
    }

    # Process model selection
    if args.debug:
        MODEL_LIST = DEBUG_MODELS
    else:
        if args.model == "all":
            selected_models = list(MODEL_MAP.keys())
        else:
            selected_models = [args.model.strip()]
        MODEL_LIST = [(model, MODEL_MAP[model][1]) for model in selected_models]

    # Set dataset name based on argument
    DATASET_NAME = (
        "bigcode/bigcodebench-hard"
        if args.dataset in ["bigcodebench", "both"]
        else None
    )

    with open("/scratch/qido00001/hf_key.txt", "r") as file:
        key = file.readline()
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

                # Process each task
                for item in items:
                    task_id = item.get("task_id")
                    original_prompt = item.get("complete_prompt")

                    # Skip if task is already completed
                    if (
                        task_id in all_results
                        and "generations" in all_results[task_id]
                        and all_results[task_id]["generations"]
                    ):
                        print(f"Skipping completed task {task_id}")
                        continue

                    # Initialize result dictionary for this task if not exists
                    if task_id not in all_results:
                        all_results[task_id] = {
                            "dataset": dataset_name,
                            "task_id": task_id,
                            "original_prompt": original_prompt,
                            "model": model_display,
                            "generations": {},
                        }
                        # Write initial task entry
                        with open(args.output, "a", encoding="utf-8") as f:
                            f.write(json.dumps(all_results[task_id]) + "\n")
                            f.flush()

                    print(f"\nProcessing task {task_id} from {dataset_name}")

                    try:
                        # Handle regular templates
                        print("Generating baseline solution...")
                        baseline_code = generator.generate_response(
                            PROMPT_TEMPLATES["baseline"].format(task=original_prompt)
                        )
                        all_results[task_id]["generations"]["baseline"] = baseline_code
                        generator.reset_history()
                        # Write after baseline
                        with open(args.output, "w", encoding="utf-8") as f:
                            for tid, result in all_results.items():
                                f.write(json.dumps(result) + "\n")
                            f.flush()

                        print("Generating quality-focused solution...")
                        quality_code = generator.generate_response(
                            PROMPT_TEMPLATES["quality_focused"].format(
                                task=original_prompt
                            )
                        )
                        all_results[task_id]["generations"][
                            "quality_focused"
                        ] = quality_code
                        generator.reset_history()
                        # Write after quality-focused
                        with open(args.output, "w", encoding="utf-8") as f:
                            for tid, result in all_results.items():
                                f.write(json.dumps(result) + "\n")
                            f.flush()

                        print("Generating persona solution...")
                        persona_code = generator.generate_response(
                            PROMPT_TEMPLATES["persona"].format(task=original_prompt)
                        )
                        all_results[task_id]["generations"]["persona"] = persona_code
                        generator.reset_history()
                        # Write after persona
                        with open(args.output, "w", encoding="utf-8") as f:
                            for tid, result in all_results.items():
                                f.write(json.dumps(result) + "\n")
                            f.flush()

                        # Handle CoT
                        print("Generating CoT solution...")
                        cot_result = generate_cot_solution(
                            original_prompt, model_internal, generator
                        )
                        all_results[task_id]["generations"]["cot"] = {
                            "reasoning": cot_result["reasoning"],
                            "final_code": cot_result["final_code"],
                        }
                        generator.reset_history()
                        # Write after CoT
                        with open(args.output, "w", encoding="utf-8") as f:
                            for tid, result in all_results.items():
                                f.write(json.dumps(result) + "\n")
                            f.flush()

                        # Handle RCI
                        print("Generating RCI solution...")
                        rci_result = generate_rci_solution(
                            original_prompt, model_internal, generator
                        )
                        all_results[task_id]["generations"]["rci"] = {
                            "initial_code": rci_result["initial_code"],
                            "review": rci_result["review"],
                            "improved_code": rci_result["improved_code"],
                        }
                        generator.reset_history()
                        # Write after RCI
                        with open(args.output, "w", encoding="utf-8") as f:
                            for tid, result in all_results.items():
                                f.write(json.dumps(result) + "\n")
                            f.flush()

                    except Exception as e:
                        print(f"Error processing task {task_id}: {str(e)}")
                        all_results[task_id]["generations"] = {"error": str(e)}
                        # Write error state
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
