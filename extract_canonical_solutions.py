#!/usr/bin/env python3
"""
Script to extract code values from CoderEval4Python.json and save them as individual Python files.
Each file will be named with the _id as the filename.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset


def extract_bigcodebench_code():

    # Define paths
    output_dir = "extracted_code/canonical/bigcodebench-hard/canonical"

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the dataset from Hugging Face
    print("Loading BigCodeBench dataset from Hugging Face...")
    dataset = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")

    print(f"Found {len(dataset)} items in the dataset")

    # Process each item
    extracted_count = 0
    for item in dataset:
        try:
            # Extract taskid and canonical_solution
            taskid = item.get("task_id")
            canonical_solution = item.get("canonical_solution")

            if taskid and canonical_solution:
                # Create filename with taskid.split('/')[-1]
                filename = f"{taskid.split('/')[-1]}.py"
                file_path = os.path.join(output_dir, filename)

                # Write canonical_solution to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(canonical_solution)

                extracted_count += 1

                if extracted_count % 100 == 0:
                    print(f"Processed {extracted_count} files...")

        except Exception as e:
            print(f"Error processing item: {e}")
            continue

    print(f"Successfully extracted {extracted_count} Python files to {output_dir}/")


def extract_codereval_code():
    # Define paths
    json_file_path = "dataset/CoderEval4Python.json"
    output_dir = "extracted_code/canonical/coderEval/canonical"

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read the JSON file
    print(f"Reading {json_file_path}...")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} items in the JSON file")

    # Process each item
    extracted_count = 0
    for item in data["RECORDS"]:
        try:
            # Extract _id and code
            item_id = item.get("_id")
            code = item.get("code")

            if item_id and code:
                # Create filename with _id
                filename = f"{item_id}.py"
                file_path = os.path.join(output_dir, filename)

                # Write code to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)

                extracted_count += 1

                if extracted_count % 100 == 0:
                    print(f"Processed {extracted_count} files...")

        except Exception as e:
            print(f"Error processing item: {e}")
            continue

    print(f"Successfully extracted {extracted_count} Python files to {output_dir}/")


if __name__ == "__main__":
    extract_bigcodebench_code()
    extract_codereval_code()
