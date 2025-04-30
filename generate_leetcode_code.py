import os
import requests
import csv
from datasets import load_dataset

# Download LeetCode dataset from Hugging Face
DATASET_NAME = "greengerong/leetcode"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_LIST = [
    ("qwen", "qwen2.5-coder:32b"),
    ("opencoder", "opencoder"),
    ("deepseek-coder", "deepseek-coder:33b")
]

# Load dataset (change split as needed)
dataset = load_dataset(DATASET_NAME, split="train")

def generate_solution(problem_description, model_name, language="Python"):
    prompt = f"""### LeetCode Problem\n{problem_description}\n### Solution in {language}:\n"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 256}
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "")
    else:
        print(f"Ollama API error: {response.status_code} - {response.text}")
        return "# ERROR: Could not get response from Ollama."

# Generate code for the first 5 problems and save results to CSV
output_csv = "leetcode_model_comparison.csv"
fieldnames = ["problem_id", "problem_description"] + [name for name, _ in MODEL_LIST]

with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, item in enumerate(dataset.select(range(5))):
        description = item.get("content")
        row = {"problem_id": i+1, "problem_description": description}
        for model_display, model_internal in MODEL_LIST:
            code = generate_solution(description, model_internal)
            row[model_display] = code
        writer.writerow(row)
        print(f"Generated solutions for problem {i+1}")

print(f"Done. All results saved in {output_csv}")
