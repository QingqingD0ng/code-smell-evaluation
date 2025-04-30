import os
import subprocess
import json
import csv
from collections import defaultdict

def analyze_with_pylint(file_path):
    # Run pylint with JSON output
    result = subprocess.run([
        "pylint", file_path, "--output-format=json", "--score=n"
    ], capture_output=True, text=True)
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        output = []
    return output

def categorize_smells(pylint_output):
    categories = {}
    for item in pylint_output:
        msg_id = item.get('message-id', 'UNKNOWN')
        symbol = item.get('symbol', 'unknown')
        msg = item.get('message', '')
        categories.setdefault(symbol, []).append({
            'line': item.get('line'),
            'message': msg,
            'msg_id': msg_id
        })
    return categories

# Filters for basic/syntax-related code smells
FILTERS = ['C0303', 'C0304', 'C0305', 'C0103', 'C0112', 'C0114', 'C0115', 'C0116', 'W0611', 'C0415', 'W0404',
           'C0413', 'C0411', 'C0412', 'W0401', 'W0614', 'C0410', 'C0413', 'C0414', 'R0402', 'C2403', 'E0401', 'E0001']

def is_basic_smell(item):
    msg_id = item.get("message-id", "")
    if msg_id in FILTERS or msg_id.startswith('F'):
        return True
    return False

def analyze_with_bandit(files, output_json="bandit_results.json"):
    # Run Bandit on a list of files and save output to JSON
    result = subprocess.run([
        "bandit", "-r", *files, "-f", "json", "-o", output_json
    ], capture_output=True, text=True)
    # Load the results
    if os.path.exists(output_json):
        with open(output_json) as f:
            data = json.load(f)
        return data
    return None

def aggregate_bandit_results(bandit_json, output_csv="bandit_code_smells.csv"):
    smellDict = defaultdict(int)
    if not bandit_json or "results" not in bandit_json:
        return
    for item in bandit_json["results"]:
        key = item["test_id"] + "_" + item["test_name"]
        smellDict[key] += 1
    # Write to CSV
    header = ['Message', 'Count']
    data = [[key, smellDict[key]] for key in smellDict]
    with open(output_csv, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"Bandit analysis complete. Results saved to {output_csv}.")

def main():
    solutions_dir = "leetcode_solutions"
    report = {}
    smellDictName = defaultdict(int)
    py_files = [os.path.join(solutions_dir, fname) for fname in os.listdir(solutions_dir) if fname.endswith('.py')]
    for file_path in py_files:
        pylint_output = analyze_with_pylint(file_path)
        # Filter out basic smells
        filtered = [item for item in pylint_output if not is_basic_smell(item)]
        # Aggregate by message-id and symbol
        for item in filtered:
            key = f"{item['message-id']}-{item['symbol']}"
            smellDictName[key] += 1
        # Optionally, keep categorized report as before
        categorized = categorize_smells(filtered)
        report[os.path.basename(file_path)] = categorized
    # Save the filtered/categorized report as JSON (optional)
    with open("filtered_code_smell_report.json", "w") as f:
        json.dump(report, f, indent=2)
    # Save the aggregated data to CSV
    header = ['Message', 'Count']
    data = [[key, smellDictName[key]] for key in smellDictName]
    with open('filtered_code_smells.csv', 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print("Analysis complete. Filtered results saved to filtered_code_smell_report.json and filtered_code_smells.csv.")
    # --- Bandit analysis ---
    bandit_json = analyze_with_bandit(py_files)
    aggregate_bandit_results(bandit_json)

if __name__ == "__main__":
    main()
