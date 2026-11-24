"""Phase 3: Baseline Evaluation Runner.

Runs evaluation on the un-finetuned base model across:
- test_unseen (500)
- test_seen (200)
- spider dev (if available)

Outputs:
- evaluation/results/base.json
- evaluation/results/failures_base.jsonl
Prints markdown summary table.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import run_evaluation

def print_summary_table(results: dict):
    m = results["metrics"]
    unseen = m["test_unseen"]
    seen = m["test_seen"]
    spider = m["spider_dev"]

    print("\n" + "=" * 70)
    print("SQL-FORGE v2 BASELINE EVALUATION SUMMARY")
    print(f"Model: {results['model_path']}")
    print(f"Commit: {results['git_commit']} | Timestamp: {results['timestamp']}")
    print("=" * 70)
    print(f"{'Split':<15} | {'Count':<7} | {'Exact Match':<13} | {'SQL Validity':<13} | {'Exec Accuracy':<13}")
    print("-" * 70)
    print(f"{'test_unseen':<15} | {unseen['count']:<7} | {unseen['exact_match_ratio']*100:>6.2f}%       | {unseen['sql_validity_ratio']*100:>6.2f}%       | {'N/A (no data)':<13}")
    print(f"{'test_seen':<15} | {seen['count']:<7} | {seen['exact_match_ratio']*100:>6.2f}%       | {seen['sql_validity_ratio']*100:>6.2f}%       | {'N/A (no data)':<13}")
    if spider.get("status") == "evaluated":
        print(f"{'spider_dev':<15} | {spider['count']:<7} | {spider['exact_match_ratio']*100:>6.2f}%       | {spider['sql_validity_ratio']*100:>6.2f}%       | {spider['execution_accuracy']*100:>6.2f}%")
    else:
        print(f"{'spider_dev':<15} | {'N/A':<7} | {'N/A':<13} | {'N/A':<13} | {'MISSING (needs download)':<13}")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation on base model")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--spider-dir", type=str, default="data/spider")
    parser.add_argument("--results-dir", type=str, default="evaluation/results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mock", action="store_true", help="Run mock evaluation offline")
    args = parser.parse_args()

    results = run_evaluation(
        model_path=args.model_path,
        adapter_path=None,
        output_run_name="base",
        data_dir=args.data_dir,
        spider_dir=args.spider_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        mock=args.mock
    )

    print_summary_table(results)

if __name__ == "__main__":
    main()
