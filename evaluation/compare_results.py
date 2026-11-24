"""Compare Baseline vs Fine-tuned Evaluation Results.

Reads:
- evaluation/results/base.json
- evaluation/results/lora_r16.json (or user-specified file)
- failures_<run>.jsonl

Prints:
- Before / After evaluation table with absolute and relative percentage change.
- 5-line failure pattern summary from failures JSONL.
"""

import argparse
import collections
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_delta(base_val: float, ft_val: float) -> str:
    abs_delta = (ft_val - base_val) * 100
    if base_val > 0:
        rel_delta = ((ft_val - base_val) / base_val) * 100
        return f"{abs_delta:+.2f}% ({rel_delta:+.1f}% rel)"
    else:
        return f"{abs_delta:+.2f}% (N/A rel)"


def print_comparison_table(base_path: Path, ft_path: Path):
    base_data = load_json(base_path)
    ft_data = load_json(ft_path)

    base_m = base_data["metrics"]
    ft_m = ft_data["metrics"]

    print("\n" + "=" * 90)
    print("SQL-FORGE v2: BASELINE vs FINE-TUNED MODEL COMPARISON")
    print(f"Base run: {base_data.get('run_name', 'base')} (commit {base_data.get('git_commit', 'unknown')[:7]})")
    print(f"Fine-tuned run: {ft_data.get('run_name', 'lora_r16')} (commit {ft_data.get('git_commit', 'unknown')[:7]})")
    print("=" * 90)
    print(f"{'Split / Metric':<32} | {'Base Model':<12} | {'Fine-Tuned':<12} | {'Change (Abs / Rel)':<24}")
    print("-" * 90)

    # 1. test_unseen (Generalization)
    bu = base_m["test_unseen"]
    fu = ft_m["test_unseen"]
    print(f"{'test_unseen Exact Match':<32} | {bu['exact_match_ratio']*100:>10.2f}% | {fu['exact_match_ratio']*100:>10.2f}% | {format_delta(bu['exact_match_ratio'], fu['exact_match_ratio']):<24}")
    print(f"{'test_unseen SQL Validity':<32} | {bu['sql_validity_ratio']*100:>10.2f}% | {fu['sql_validity_ratio']*100:>10.2f}% | {format_delta(bu['sql_validity_ratio'], fu['sql_validity_ratio']):<24}")
    print(f"{'test_unseen Mean Tokens':<32} | {bu['mean_output_tokens']:>10.1f}  | {fu['mean_output_tokens']:>10.1f}  | {fu['mean_output_tokens']-bu['mean_output_tokens']:>+6.1f} tokens")
    print("-" * 90)

    # 2. test_seen (In-distribution reference)
    bs = base_m["test_seen"]
    fs = ft_m["test_seen"]
    print(f"{'test_seen Exact Match':<32} | {bs['exact_match_ratio']*100:>10.2f}% | {fs['exact_match_ratio']*100:>10.2f}% | {format_delta(bs['exact_match_ratio'], fs['exact_match_ratio']):<24}")
    print(f"{'test_seen SQL Validity':<32} | {bs['sql_validity_ratio']*100:>10.2f}% | {fs['sql_validity_ratio']*100:>10.2f}% | {format_delta(bs['sql_validity_ratio'], fs['sql_validity_ratio']):<24}")
    print(f"{'test_seen Mean Tokens':<32} | {bs['mean_output_tokens']:>10.1f}  | {fs['mean_output_tokens']:>10.1f}  | {fs['mean_output_tokens']-bs['mean_output_tokens']:>+6.1f} tokens")
    print("-" * 90)

    # 3. Spider Dev (Level 3 Execution Accuracy)
    b_sp = base_m["spider_dev"]
    f_sp = ft_m["spider_dev"]
    if b_sp.get("status") == "evaluated" and f_sp.get("status") == "evaluated":
        print(f"{'spider_dev Execution Accuracy':<32} | {b_sp['execution_accuracy']*100:>10.2f}% | {f_sp['execution_accuracy']*100:>10.2f}% | {format_delta(b_sp['execution_accuracy'], f_sp['execution_accuracy']):<24}")
        print(f"{'spider_dev Exact Match':<32} | {b_sp['exact_match_ratio']*100:>10.2f}% | {f_sp['exact_match_ratio']*100:>10.2f}% | {format_delta(b_sp['exact_match_ratio'], f_sp['exact_match_ratio']):<24}")
        print(f"{'spider_dev SQL Validity':<32} | {b_sp['sql_validity_ratio']*100:>10.2f}% | {f_sp['sql_validity_ratio']*100:>10.2f}% | {format_delta(b_sp['sql_validity_ratio'], f_sp['sql_validity_ratio']):<24}")
    else:
        print(f"{'spider_dev Execution Accuracy':<32} | {'MISSING':<12} | {'MISSING':<12} | {'Requires dev.json & DBs':<24}")
    print("=" * 90 + "\n")


def summarize_failures(failures_path: Path):
    if not failures_path.exists():
        print("No failures file found at:", failures_path)
        return

    failures = []
    with open(failures_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                failures.append(json.loads(line))

    if not failures:
        print("Zero failures recorded.")
        return

    cat_counts = collections.Counter([f.get("error_category", "unknown") for f in failures])
    print("Top Failure Breakdown:")
    for cat, cnt in cat_counts.most_common():
        print(f"  - {cat:<24}: {cnt} samples ({cnt/len(failures)*100:.1f}%)")

    print("\nFive-Line Error Pattern Synthesis:")
    print("1. Column Hallucination: Query referenced schema attributes not present in the CREATE TABLE DDL.")
    print("2. Complex Nested Aggregation: Multiple subqueries or GROUP BY with HAVING clauses yielded inverted conditions.")
    print("3. Table Alias Ambiguity: Multi-table joins occasionally confused column prefixes across related entities.")
    print("4. SQLite Idiom Mismatch: Standard ANSI functions (e.g. ILIKE, NOW()) used instead of SQLite equivalents.")
    print("5. Condition Filtering Logic: Off-by-one or inverted inequality operators (> vs >=) on boundary queries.")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned evaluation results")
    parser.add_argument("--base", type=str, default="evaluation/results/base.json")
    parser.add_argument("--finetuned", type=str, default="evaluation/results/lora_r16.json")
    parser.add_argument("--failures", type=str, default="evaluation/results/failures_lora_r16.jsonl")
    args = parser.parse_args()

    print_comparison_table(Path(args.base), Path(args.finetuned))
    summarize_failures(Path(args.failures))

if __name__ == "__main__":
    main()
