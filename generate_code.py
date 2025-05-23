import os
import csv
from datasets import load_dataset
import torch
from huggingface_hub import login
import argparse
import platform
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up cache location for cluster
cache_location = '/scratch/USER/tuning/.cache'
os.environ['HF_HOME'] = cache_location
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_location

# Download BigCodeBench dataset from Hugging Face
DATASET_NAME = "bigcode/bigcodebench-hard"

# Define the models to use
PRODUCTION_MODELS = [
    ("qwen", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    ("phi-4", "microsoft/phi-4"),
    ("llama", "meta-llama/Llama-3.3-70B-Instruct")
]

DEBUG_MODELS = [
    ("qwen3", "Qwen/Qwen3-0.6B"),
    ("qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct")
]

# Define prompt templates
PROMPT_TEMPLATES = {
    "baseline": "Generate Python code for the following: {task}",
    "quality_focused": "Generate Python code for the following, ensuring it is clean and avoids code smells: {task}",
    "cot": {
        "initial": """Generate Python code for the following: {task}
Ensuring it is clean and avoids code smells. Let's think step by step.""",
        "final": "Therefore, final clean Python implementation is:"
    },
    "rci": {
        "initial": "Generate Python code for the following: {task}",
        "review": "Review your previous answer and find code smells with it",
        "improve": "Based on the code smells you found, improve your answer"
    },
    "persona": """Act as a software quality expert. Provide outputs that a quality expert would give. Ensuring it is clean and avoids code smells.
Generate clean Python code for the following: {task}"""
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
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True
        ).eval()
        self.history = []

    def generate_response(self, prompt, max_new_tokens=1024):
        messages = self.history + [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )[0][len(inputs.input_ids[0]):].tolist()
        
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Update history
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        
        return response

    def reset_history(self):
        self.history = []

def generate_solution(prompt, model_id):
    """Generate solution using the model"""
    generator = CodeGenerator(model_id)
    return generator.generate_response(prompt)

def generate_cot_solution(task, model_id):
    """Generate solution using Chain of Thought prompting"""
    generator = CodeGenerator(model_id)
    
    # Step 1: Generate reasoning
    initial_prompt = PROMPT_TEMPLATES["cot"]["initial"].format(task=task)
    reasoning = generator.generate_response(initial_prompt)
    
    # Step 2: Generate final implementation
    final_prompt = PROMPT_TEMPLATES["cot"]["final"]
    final_code = generator.generate_response(final_prompt)
    
    return {
        "reasoning": reasoning,
        "final_code": final_code
    }

def generate_rci_solution(task, model_id):
    """Generate solution using Review-Critique-Improve process"""
    generator = CodeGenerator(model_id)
    
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
        "improved_code": improved_code
    }

def main():
    # Add argument parser for debug mode
    parser = argparse.ArgumentParser(description='Generate code using different prompting strategies')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with smaller model')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to generate')
    args = parser.parse_args()
    
    # Print system information
    print("\nSystem Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}\n")
    
    # Select model list based on debug flag
    MODEL_LIST = DEBUG_MODELS if args.debug else PRODUCTION_MODELS
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="v0.1.4")
    
    # Generate code for tasks and save results to CSV
    output_csv = "debug_results.csv" if args.debug else "bigcodebench_prompt_comparison.csv"
    
    # Create fieldnames for CSV
    fieldnames = ["task_id", "original_prompt"]
    for model_name, _ in MODEL_LIST:
        fieldnames.extend([
            f"{model_name}_baseline",
            f"{model_name}_quality_focused",
            f"{model_name}_persona"
        ])
        fieldnames.extend([
            f"{model_name}_cot_reasoning",
            f"{model_name}_cot_final"
        ])
        fieldnames.extend([
            f"{model_name}_rci_initial",
            f"{model_name}_rci_review",
            f"{model_name}_rci_improved"
        ])
    
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, item in enumerate(dataset.select(range(args.num_samples))):
            task_id = item.get("task_id")
            original_prompt = item.get("complete_prompt")
            row = {"task_id": task_id, "original_prompt": original_prompt}
            
            for model_display, model_internal in MODEL_LIST:
                print(f"\nProcessing task {task_id} with model {model_display}")
                
                try:
                    # Handle regular templates
                    print("Generating baseline solution...")
                    baseline_code = generate_solution(PROMPT_TEMPLATES["baseline"].format(task=original_prompt), model_internal)
                    row[f"{model_display}_baseline"] = baseline_code
                    
                    print("Generating quality-focused solution...")
                    quality_code = generate_solution(PROMPT_TEMPLATES["quality_focused"].format(task=original_prompt), model_internal)
                    row[f"{model_display}_quality_focused"] = quality_code
                    
                    print("Generating persona-based solution...")
                    persona_code = generate_solution(PROMPT_TEMPLATES["persona"].format(task=original_prompt), model_internal)
                    row[f"{model_display}_persona"] = persona_code
                    
                    # Handle CoT template
                    print("Generating CoT solution...")
                    cot_results = generate_cot_solution(original_prompt, model_internal)
                    row[f"{model_display}_cot_reasoning"] = cot_results["reasoning"]
                    row[f"{model_display}_cot_final"] = cot_results["final_code"]
                    
                    # Handle RCI template
                    print("Generating RCI solution...")
                    rci_results = generate_rci_solution(original_prompt, model_internal)
                    row[f"{model_display}_rci_initial"] = rci_results["initial_code"]
                    row[f"{model_display}_rci_review"] = rci_results["review"]
                    row[f"{model_display}_rci_improved"] = rci_results["improved_code"]
                
                except Exception as e:
                    print(f"Error processing task {task_id} with model {model_display}: {str(e)}")
                    for field in fieldnames:
                        if field.startswith(model_display):
                            row[field] = f"ERROR: {str(e)}"
            
            writer.writerow(row)
            print(f"Completed task {task_id}")
            
            if args.debug:
                print("\nDebug mode: Showing first few characters of generated solutions:")
                for key, value in row.items():
                    if key not in ["task_id", "original_prompt"]:
                        print(f"{key}: {value[:100]}...")
                print("\n")
    
    print(f"Done. All results saved in {output_csv}")

if __name__ == "__main__":
    main()
