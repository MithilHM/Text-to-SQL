"""Evaluation harness for SQL-Forge v2.

Evaluates base and fine-tuned models on:
1. test_unseen (Levels 1 & 2)
2. test_seen (Levels 1 & 2)
3. spider dev (Level 3 Execution Accuracy if present)

Saves results JSON with git commit hash, config, timestamps, metrics,
and saves top 20 failures to JSONL.
"""

import argparse
import collections
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from evaluation.sql_parser import (
    check_exact_match,
    check_sql_validity,
    compare_query_results,
    categorize_error,
    execute_query,
    extract_sql,
)

SYSTEM_PROMPT = (
    "You are an expert SQL generator. Generate valid SQLite SQL using only the provided "
    "database schema. Return SQL only, without explanations."
)


def get_git_commit_hash() -> str:
    """Return current git commit hash if in a git repo."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt(schema: str, question: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion:\n{question}"}
    ]


class SQLEvaluator:
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        mock: bool = False
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.mock = mock

        self.device = "cuda" if torch.cuda.is_available() and not mock else "cpu"
        self.tokenizer = None
        self.model = None

        if not self.mock:
            self._load_model()

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading tokenizer from {self.model_path}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        attn_impl = "sdpa" if self.device == "cuda" else "eager"

        print(f"Loading base model from {self.model_path} (dtype={dtype}, attn={attn_impl})...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        if self.adapter_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter from {self.adapter_path}...", flush=True)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()
        print("Model initialized for evaluation.", flush=True)

    def generate_batch(self, prompts: List[List[Dict]]) -> List[Tuple[str, int]]:
        """Generate greedy predictions for a batch of chat messages.
        
        Returns list of (generated_sql, token_count) tuples.
        """
        if self.mock:
            # Mock mode for testing evaluation harness offline without GPU
            results = []
            for p in prompts:
                user_msg = p[1]["content"]
                # generate basic dummy select query
                results.append(("SELECT col_a FROM entity_0 WHERE score > 1;", 10))
            return results

        texts = [
            self.tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        results = []
        for i in range(len(prompts)):
            input_len = inputs["input_ids"][i].shape[0]
            gen_tokens = outputs[i][input_len:]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            clean_sql = extract_sql(gen_text)
            results.append((clean_sql, len(gen_tokens)))
        return results

    def evaluate_split(
        self,
        records: List[Dict],
        split_name: str
    ) -> Tuple[Dict, List[Dict]]:
        """Evaluate a test split on Levels 1 and 2."""
        print(f"\n--- Evaluating {split_name} ({len(records)} records) ---", flush=True)
        exact_matches = 0
        valid_queries = 0
        total_tokens = 0
        failures = []
        error_counts = collections.defaultdict(int)

        t0 = time.perf_counter()
        batch_prompts = []
        batch_records = []

        for idx, r in enumerate(records):
            schema = r.get("schema") or r.get("context", "")
            question = r["question"]
            gold = r.get("gold") or r.get("answer", "")

            batch_prompts.append(build_prompt(schema, question))
            batch_records.append(r)

            if len(batch_prompts) == self.batch_size or idx == len(records) - 1:
                batch_outputs = self.generate_batch(batch_prompts)
                for rec, (pred_sql, tok_count) in zip(batch_records, batch_outputs):
                    schema_curr = rec.get("schema") or rec.get("context", "")
                    gold_curr = rec.get("gold") or rec.get("answer", "")
                    question_curr = rec["question"]

                    em = check_exact_match(pred_sql, gold_curr)
                    is_valid, err_type, err_msg = check_sql_validity(schema_curr, pred_sql)

                    if em:
                        exact_matches += 1
                    if is_valid:
                        valid_queries += 1

                    total_tokens += tok_count
                    cat = categorize_error(em, is_valid, err_type)
                    error_counts[cat] += 1

                    if not em:
                        if len(failures) < 20:
                            failures.append({
                                "split": split_name,
                                "question": question_curr,
                                "schema": schema_curr,
                                "gold": gold_curr,
                                "predicted": pred_sql,
                                "exact_match": em,
                                "is_valid": is_valid,
                                "error_category": cat,
                                "error_detail": err_msg
                            })

                batch_prompts = []
                batch_records = []

        elapsed = time.perf_counter() - t0
        n = len(records)
        metrics = {
            "count": n,
            "elapsed_seconds": round(elapsed, 2),
            "exact_match_ratio": round(exact_matches / n, 4) if n > 0 else 0.0,
            "sql_validity_ratio": round(valid_queries / n, 4) if n > 0 else 0.0,
            "mean_output_tokens": round(total_tokens / n, 2) if n > 0 else 0.0,
            "error_distribution": dict(error_counts)
        }

        print(f"Results for {split_name}:")
        print(f"  Exact Match:    {metrics['exact_match_ratio']*100:.2f}% ({exact_matches}/{n})")
        print(f"  SQL Validity:   {metrics['sql_validity_ratio']*100:.2f}% ({valid_queries}/{n})")
        print(f"  Mean Tokens:    {metrics['mean_output_tokens']:.1f}")
        print(f"  Error Buckets:  {dict(error_counts)}")
        return metrics, failures

    def evaluate_spider(self, spider_dir: Path) -> Tuple[Dict, List[Dict]]:
        """Evaluate on Spider dev set if available (Level 3 Execution Accuracy)."""
        dev_path = spider_dir / "dev.json"
        db_root = spider_dir / "database"

        if not dev_path.exists() or not db_root.exists():
            print(f"\n[NOTICE] Spider dev dataset not found at {spider_dir}. Skipping Spider evaluation.", flush=True)
            return {"status": "missing", "note": "Spider dev dataset not found at data/spider/"}, []

        dev_records = json.loads(dev_path.read_text(encoding="utf-8"))
        print(f"\n--- Evaluating Spider Dev ({len(dev_records)} queries) ---", flush=True)

        exec_correct = 0
        em_correct = 0
        valid_queries = 0
        total_tokens = 0
        error_counts = collections.defaultdict(int)
        failures = []

        batch_prompts = []
        batch_meta = []

        t0 = time.perf_counter()
        for idx, item in enumerate(dev_records):
            db_id = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]
            db_path = str(db_root / db_id / f"{db_id}.sqlite")

            # Extract schema from database sqlite_master
            schema = ""
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                    cur = conn.cursor()
                    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;")
                    schema = "\n".join([row[0] + ";" for row in cur.fetchall()])
                    conn.close()
                except Exception:
                    schema = ""

            batch_prompts.append(build_prompt(schema, question))
            batch_meta.append((db_path, schema, question, gold_sql))

            if len(batch_prompts) == self.batch_size or idx == len(dev_records) - 1:
                batch_outputs = self.generate_batch(batch_prompts)
                for (db_file, sch, q, gold), (pred_sql, tok_count) in zip(batch_meta, batch_outputs):
                    total_tokens += tok_count
                    em = check_exact_match(pred_sql, gold)
                    if em:
                        em_correct += 1

                    is_valid, err_type, val_err = check_sql_validity(sch, pred_sql)
                    if is_valid:
                        valid_queries += 1

                    # Execute both on real database
                    gold_ok, gold_rows, gold_err = execute_query(db_file, gold, timeout_sec=10.0)
                    pred_ok, pred_rows, pred_err = execute_query(db_file, pred_sql, timeout_sec=10.0)

                    match = False
                    if pred_ok and gold_ok:
                        match = compare_query_results(pred_rows, gold_rows, gold)

                    if match:
                        exec_correct += 1

                    cat = categorize_error(em, is_valid, err_type, pred_ok, pred_err, match)
                    error_counts[cat] += 1

                    if not match and len(failures) < 20:
                        failures.append({
                            "split": "spider_dev",
                            "question": q,
                            "gold": gold,
                            "predicted": pred_sql,
                            "exact_match": em,
                            "execution_match": match,
                            "error_category": cat,
                            "error_detail": pred_err or val_err
                        })

                batch_prompts = []
                batch_meta = []

        elapsed = time.perf_counter() - t0
        n = len(dev_records)
        metrics = {
            "status": "evaluated",
            "count": n,
            "elapsed_seconds": round(elapsed, 2),
            "execution_accuracy": round(exec_correct / n, 4),
            "exact_match_ratio": round(em_correct / n, 4),
            "sql_validity_ratio": round(valid_queries / n, 4),
            "mean_output_tokens": round(total_tokens / n, 2),
            "error_distribution": dict(error_counts)
        }

        print("Results for Spider Dev:")
        print(f"  Execution Accuracy: {metrics['execution_accuracy']*100:.2f}% ({exec_correct}/{n})")
        print(f"  Exact Match:        {metrics['exact_match_ratio']*100:.2f}% ({em_correct}/{n})")
        print(f"  SQL Validity:       {metrics['sql_validity_ratio']*100:.2f}% ({valid_queries}/{n})")
        print(f"  Error Buckets:      {dict(error_counts)}")
        return metrics, failures


def run_evaluation(
    model_path: str,
    adapter_path: Optional[str] = None,
    output_run_name: str = "base",
    data_dir: str = "data/splits",
    spider_dir: str = "data/spider",
    results_dir: str = "evaluation/results",
    batch_size: int = 8,
    mock: bool = False
) -> Dict:
    evaluator = SQLEvaluator(
        model_path=model_path,
        adapter_path=adapter_path,
        batch_size=batch_size,
        mock=mock
    )

    data_path = Path(data_dir)
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    unseen_records = load_jsonl(data_path / "test_unseen.jsonl")
    seen_records = load_jsonl(data_path / "test_seen.jsonl")

    metrics_unseen, failures_unseen = evaluator.evaluate_split(unseen_records, "test_unseen")
    metrics_seen, failures_seen = evaluator.evaluate_split(seen_records, "test_seen")
    metrics_spider, failures_spider = evaluator.evaluate_spider(Path(spider_dir))

    all_failures = failures_unseen + failures_seen + failures_spider

    results_data = {
        "run_name": output_run_name,
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": model_path,
        "adapter_path": adapter_path,
        "config": {
            "batch_size": batch_size,
            "max_new_tokens": evaluator.max_new_tokens,
            "mock": mock,
            "device": evaluator.device
        },
        "metrics": {
            "test_unseen": metrics_unseen,
            "test_seen": metrics_seen,
            "spider_dev": metrics_spider
        }
    }

    # Save JSON metrics
    json_path = res_path / f"{output_run_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved metrics -> {json_path}")

    # Save Failures JSONL
    failures_path = res_path / f"failures_{output_run_name}.jsonl"
    with open(failures_path, "w", encoding="utf-8") as f:
        for f_item in all_failures:
            f.write(json.dumps(f_item, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_failures)} failure traces -> {failures_path}")

    return results_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL-Forge model on splits and Spider dev")
    parser.add_argument("--model-path", type=str, required=True, help="Base model directory or HF repo")
    parser.add_argument("--adapter-path", type=str, default=None, help="LoRA adapter directory (optional)")
    parser.add_argument("--run-name", type=str, default="base", help="Output run name (e.g. base, lora_r16)")
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--spider-dir", type=str, default="data/spider")
    parser.add_argument("--results-dir", type=str, default="evaluation/results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without loading weights")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        output_run_name=args.run_name,
        data_dir=args.data_dir,
        spider_dir=args.spider_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        mock=args.mock
    )

if __name__ == "__main__":
    main()
