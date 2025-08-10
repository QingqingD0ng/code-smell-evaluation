#!/usr/bin/env python3
"""
Script to extract passing files directly from existing evaluation results JSON file.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("extract_passing_files_from_results.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PassingFilesExtractorFromResults:
    def __init__(self):
        """Initialize the passing files extractor from results."""
        pass

    def load_evaluation_results(self, results_file: Path) -> Dict[str, Any]:
        """
        Load evaluation results from JSON file.

        Args:
            results_file: Path to the evaluation results JSON file

        Returns:
            Dictionary containing evaluation results
        """
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Successfully loaded results from {results_file}")
            return results
        except Exception as e:
            logger.error(f"Failed to load results from {results_file}: {str(e)}")
            return {}

    def extract_passing_files_from_results(
        self, results: Dict[str, Any], extracted_code_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Extract passing files based on evaluation results.

        Args:
            results: Evaluation results dictionary
            extracted_code_dir: Path to the original extracted_code directory
            output_dir: Path to save passing files

        Returns:
            Dictionary containing extraction results
        """
        logger.info(f"Starting extraction of passing files from results")

        extraction_results = {
            "total_files_evaluated": 0,
            "passing_files": 0,
            "failing_files": 0,
            "timeout_files": 0,
            "error_files": 0,
            "passing_files_list": [],
            "model_results": {},
        }

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each model in the results
        for model_name, model_results in results.get("models", {}).items():
            logger.info(f"Processing model: {model_name}")

            model_extraction_results = {
                "total_files": 0,
                "passing_files": 0,
                "failing_files": 0,
                "timeout_files": 0,
                "error_files": 0,
                "passing_files_details": [],
            }

            # Process detailed results for this model
            for result in model_results.get("detailed_results", []):
                extraction_results["total_files_evaluated"] += 1
                model_extraction_results["total_files"] += 1

                file_path = result.get("file_path", "")
                success = result.get("success", False)
                result_type = result.get("result", "UNKNOWN")
                problem_id = result.get("problem_id", "")

                if success:
                    extraction_results["passing_files"] += 1
                    model_extraction_results["passing_files"] += 1

                    # Extract model and technique from file path
                    # Expected path format: extracted_code/model/dataset/technique/filename.py
                    path_parts = Path(file_path).parts
                    if len(path_parts) >= 4:
                        model_from_path = path_parts[1]  # extracted_code/model
                        technique_from_path = path_parts[
                            3
                        ]  # extracted_code/model/dataset/technique
                        filename = path_parts[-1]  # filename.py

                        # Create output directory for this model/technique
                        technique_output_dir = (
                            output_dir / model_from_path / technique_from_path
                        )
                        technique_output_dir.mkdir(parents=True, exist_ok=True)

                        # Copy the passing file to output directory
                        source_file = Path(file_path)
                        if source_file.exists():
                            output_file = technique_output_dir / filename
                            shutil.copy2(source_file, output_file)

                            # Record passing file details
                            passing_file_info = {
                                "model": model_from_path,
                                "technique": technique_from_path,
                                "problem_id": problem_id,
                                "original_path": str(source_file),
                                "copied_path": str(output_file),
                                "execution_time": result.get("execution_time", 0),
                            }
                            extraction_results["passing_files_list"].append(
                                passing_file_info
                            )
                            model_extraction_results["passing_files_details"].append(
                                passing_file_info
                            )

                            logger.info(f"✓ PASS: {filename} -> {output_file}")
                        else:
                            logger.warning(f"Source file not found: {source_file}")
                    else:
                        logger.warning(f"Unexpected file path format: {file_path}")
                else:
                    if result_type == "TIMEOUT":
                        extraction_results["timeout_files"] += 1
                        model_extraction_results["timeout_files"] += 1
                    else:
                        extraction_results["failing_files"] += 1
                        model_extraction_results["failing_files"] += 1

                    # Add error count for any evaluation errors
                    if result.get("error"):
                        extraction_results["error_files"] += 1
                        model_extraction_results["error_files"] += 1

            extraction_results["model_results"][model_name] = model_extraction_results

        # Calculate pass rate
        if extraction_results["total_files_evaluated"] > 0:
            extraction_results["pass_rate"] = (
                extraction_results["passing_files"]
                / extraction_results["total_files_evaluated"]
            )
        else:
            extraction_results["pass_rate"] = 0

        logger.info(
            f"Extraction complete: {extraction_results['passing_files']}/{extraction_results['total_files_evaluated']} files passed ({extraction_results['pass_rate']:.2%})"
        )

        return extraction_results

    def save_extraction_results(self, results: Dict[str, Any], output_dir: Path):
        """
        Save extraction results to files.

        Args:
            results: Extraction results
            output_dir: Output directory for results
        """
        # Save JSON results
        with open(output_dir / "extraction_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save CSV with passing files list
        if results["passing_files_list"]:
            df = pd.DataFrame(results["passing_files_list"])
            df.to_csv(output_dir / "passing_files_list.csv", index=False)

        # Save summary CSV
        summary_data = []
        for model_name, model_results in results["model_results"].items():
            summary_data.append(
                {
                    "model": model_name,
                    "total_files": model_results["total_files"],
                    "passing_files": model_results["passing_files"],
                    "failing_files": model_results["failing_files"],
                    "timeout_files": model_results["timeout_files"],
                    "error_files": model_results["error_files"],
                    "pass_rate": (
                        model_results["passing_files"] / model_results["total_files"]
                        if model_results["total_files"] > 0
                        else 0
                    ),
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "extraction_summary.csv", index=False)

        logger.info(f"Extraction results saved to {output_dir}")


def main():
    """Main extraction function."""
    print("Extracting passing files from existing evaluation results...")
    print("=" * 70)

    # Initialize extractor
    extractor = PassingFilesExtractorFromResults()

    # Load evaluation results
    results_file = Path(
        "generated_code_evaluation_results/generated_code_evaluation_results.json"
    )
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        print(f"Please make sure the results file exists: {results_file}")
        return False

    results = extractor.load_evaluation_results(results_file)
    if not results:
        logger.error("Failed to load evaluation results")
        return False

    # Extract passing files
    extracted_code_dir = Path("extracted_code")
    output_dir = Path("passing_files_for_analysis")
    extraction_results = extractor.extract_passing_files_from_results(
        results, extracted_code_dir, output_dir
    )

    # Save results
    extractor.save_extraction_results(extraction_results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("PASSING FILES EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total files evaluated: {extraction_results['total_files_evaluated']}")
    print(f"Passing files: {extraction_results['passing_files']}")
    print(f"Failing files: {extraction_results['failing_files']}")
    print(f"Timeout files: {extraction_results['timeout_files']}")
    print(f"Error files: {extraction_results['error_files']}")
    print(f"Pass rate: {extraction_results['pass_rate']:.2%}")
    print(f"Passing files saved to: {output_dir}")
    print("=" * 70)

    for model_name, model_results in extraction_results["model_results"].items():
        print(f"\n{model_name}:")
        print(
            f"  Pass rate: {model_results['passing_files']/model_results['total_files']:.2%}"
        )
        print(f"  Passing files: {model_results['passing_files']}")
        print(f"  Failing files: {model_results['failing_files']}")
        print(f"  Timeout files: {model_results['timeout_files']}")
        print(f"  Error files: {model_results['error_files']}")

    print(f"\nPassing files have been extracted to: {output_dir}")
    print("You can now run code smell analysis on these files using:")
    print(f"python analyze_code_smells.py --input-dir {output_dir}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
