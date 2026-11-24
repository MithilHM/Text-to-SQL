"""Dataset Preparation for SQL-Forge v2.

Splits b-mc2/sql-create-context into train, val, test_unseen, test_seen
grouped strictly by normalized schema hash to prevent schema leakage.
Writes prompt-completion JSONL format (for TRL) and raw evaluation format.
"""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

SYSTEM_PROMPT = (
    "You are an expert SQL generator. Generate valid SQLite SQL using only the provided "
    "database schema. Return SQL only, without explanations."
)

def normalize_schema(schema_text: str) -> str:
    """Normalize schema string by lowercasing and collapsing whitespace."""
    if not schema_text:
        return ""
    text = schema_text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def hash_schema(schema_text: str) -> str:
    """Return MD5 hash of normalized schema text."""
    normalized = normalize_schema(schema_text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

def format_prompt_completion(context: str, question: str, answer: str) -> Dict:
    """Format single record for TRL SFTTrainer completion-only loss."""
    user_content = f"Database schema:\n{context}\n\nQuestion:\n{question}"
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "completion": [
            {"role": "assistant", "content": answer.strip()}
        ]
    }

def format_eval_record(context: str, question: str, answer: str) -> Dict:
    """Format record for standalone evaluation."""
    return {
        "context": context,
        "schema": context,
        "question": question,
        "gold": answer.strip(),
        "answer": answer.strip()
    }

def split_dataset(
    dataset_rows: List[Dict],
    train_size: int = 8000,
    val_size: int = 200,
    test_unseen_size: int = 500,
    test_seen_size: int = 200,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Group rows by schema hash and allocate disjoint and seen splits."""
    random.seed(seed)

    # Group by schema hash
    schema_to_rows: Dict[str, List[Dict]] = {}
    for row in dataset_rows:
        shash = hash_schema(row["context"])
        if shash not in schema_to_rows:
            schema_to_rows[shash] = []
        schema_to_rows[shash].append(row)

    unique_schemas = list(schema_to_rows.keys())
    random.shuffle(unique_schemas)

    train_rows: List[Dict] = []
    test_seen_rows: List[Dict] = []
    val_rows: List[Dict] = []
    test_unseen_rows: List[Dict] = []

    train_schema_set = set()
    val_schema_set = set()
    test_unseen_schema_set = set()

    # Step 1: Allocate train schemas until train target is reached
    # For schemas in train with >= 2 examples, reserve some for test_seen
    schema_idx = 0
    while schema_idx < len(unique_schemas) and (len(train_rows) < train_size or len(test_seen_rows) < test_seen_size):
        shash = unique_schemas[schema_idx]
        rows = list(schema_to_rows[shash])
        random.shuffle(rows)

        if len(rows) > 1 and len(test_seen_rows) < test_seen_size:
            # allocate one to test_seen and remaining to train
            seen_cand = rows.pop()
            test_seen_rows.append(seen_cand)

        for r in rows:
            if len(train_rows) < train_size:
                train_rows.append(r)
                train_schema_set.add(shash)

        schema_idx += 1

    train_rows = train_rows[:train_size]
    test_seen_rows = test_seen_rows[:test_seen_size]

    # Step 2: Allocate validation schemas (disjoint from train)
    while schema_idx < len(unique_schemas) and len(val_rows) < val_size:
        shash = unique_schemas[schema_idx]
        if shash in train_schema_set:
            schema_idx += 1
            continue
        rows = schema_to_rows[shash]
        for r in rows:
            if len(val_rows) < val_size:
                val_rows.append(r)
                val_schema_set.add(shash)
        schema_idx += 1

    val_rows = val_rows[:val_size]

    # Step 3: Allocate test_unseen schemas (disjoint from train and val)
    while schema_idx < len(unique_schemas) and len(test_unseen_rows) < test_unseen_size:
        shash = unique_schemas[schema_idx]
        if shash in train_schema_set or shash in val_schema_set:
            schema_idx += 1
            continue
        rows = schema_to_rows[shash]
        for r in rows:
            if len(test_unseen_rows) < test_unseen_size:
                test_unseen_rows.append(r)
                test_unseen_schema_set.add(shash)
        schema_idx += 1

    test_unseen_rows = test_unseen_rows[:test_unseen_size]

    # Verify assertions
    assert len(train_schema_set.intersection(val_schema_set)) == 0, "FATAL: Train and Val have overlapping schemas!"
    assert len(train_schema_set.intersection(test_unseen_schema_set)) == 0, "FATAL: Train and Test Unseen have overlapping schemas!"
    assert len(val_schema_set.intersection(test_unseen_schema_set)) == 0, "FATAL: Val and Test Unseen have overlapping schemas!"

    # Verify test_seen schemas are indeed a subset of train schemas
    test_seen_schemas = {hash_schema(r["context"]) for r in test_seen_rows}
    overlap_seen = test_seen_schemas.intersection(train_schema_set)
    assert len(overlap_seen) == len(test_seen_schemas), (
        f"FATAL: test_seen contains schemas not present in train! ({len(overlap_seen)}/{len(test_seen_schemas)})"
    )

    assert len(train_rows) == train_size, f"Train count {len(train_rows)} != {train_size}"
    assert len(val_rows) == val_size, f"Val count {len(val_rows)} != {val_size}"
    assert len(test_unseen_rows) == test_unseen_size, f"Test unseen count {len(test_unseen_rows)} != {test_unseen_size}"
    assert len(test_seen_rows) == test_seen_size, f"Test seen count {len(test_seen_rows)} != {test_seen_size}"

    print("\nDataset Split Summary:")
    print(f"  Train:        {len(train_rows)} samples across {len(train_schema_set)} schemas")
    print(f"  Validation:   {len(val_rows)} samples across {len(val_schema_set)} schemas (disjoint)")
    print(f"  Test Unseen:  {len(test_unseen_rows)} samples across {len(test_unseen_schema_set)} schemas (disjoint)")
    print(f"  Test Seen:    {len(test_seen_rows)} samples across {len(test_seen_schemas)} schemas (in train schemas)")
    print("Schema disjointness assertions: PASSED (zero leakage)")

    return train_rows, val_rows, test_unseen_rows, test_seen_rows

def save_jsonl(records: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records -> {path}")

def compute_token_stats(records: List[Dict], tokenizer_name_or_path: str = None):
    """Compute token length statistics over prompt + completion."""
    if tokenizer_name_or_path and os.path.exists(tokenizer_name_or_path):
        try:
            from transformers import AutoTokenizer
            print(f"\nComputing token length statistics with local tokenizer: {tokenizer_name_or_path}...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
            lengths = []
            for r in records:
                full_msgs = r["prompt"] + r["completion"]
                toks = tokenizer.apply_chat_template(full_msgs, tokenize=True)
                lengths.append(len(toks))

            lengths.sort()
            mean_len = sum(lengths) / len(lengths)
            p50_len = lengths[int(0.50 * len(lengths))]
            p95_len = lengths[int(0.95 * len(lengths))]
            max_len = max(lengths)

            print("\nToken Length Statistics (Full Prompt + Completion):", flush=True)
            print(f"  Sample count: {len(lengths)}", flush=True)
            print(f"  Mean length:  {mean_len:.1f} tokens", flush=True)
            print(f"  p50 length:   {p50_len} tokens", flush=True)
            print(f"  p95 length:   {p95_len} tokens", flush=True)
            print(f"  Max length:   {max_len} tokens", flush=True)
            print(f"  Within 1024 cap: {sum(1 for l in lengths if l <= 1024) / len(lengths) * 100:.2f}%", flush=True)
            return
        except Exception as e:
            print(f"Tokenizer load failed ({e}). Falling back to character estimate.", flush=True)

    print("Computing approximate token stats from payload characters (length // 4)...", flush=True)
    lengths = [len(json.dumps(r)) // 4 for r in records]
    lengths.sort()
    mean_len = sum(lengths) / len(lengths)
    p50_len = lengths[int(0.50 * len(lengths))]
    p95_len = lengths[int(0.95 * len(lengths))]
    max_len = max(lengths)
    print("\nApprox Token Length Statistics (Full Prompt + Completion):", flush=True)
    print(f"  Sample count: {len(lengths)}", flush=True)
    print(f"  Mean length:  {mean_len:.1f} tokens", flush=True)
    print(f"  p50 length:   {p50_len} tokens", flush=True)
    print(f"  p95 length:   {p95_len} tokens", flush=True)
    print(f"  Max length:   {max_len} tokens", flush=True)
    print(f"  Within 1024 cap: {sum(1 for l in lengths if l <= 1024) / len(lengths) * 100:.2f}%", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Prepare schema-disjoint Text-to-SQL datasets")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Target directory for splits")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-unseen-size", type=int, default=500)
    parser.add_argument("--test-seen-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--offline-sample", action="store_true", help="Generate synthetic samples for offline testing")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    rows = []
    if args.offline_sample:
        print("Generating synthetic rows for offline pipeline verification...", flush=True)
        # Create 3500 synthetic schemas with 2-10 questions each
        for s_id in range(3500):
            schema = f"CREATE TABLE entity_{s_id} (id INTEGER PRIMARY KEY, col_a TEXT, col_b INTEGER, score REAL);"
            n_queries = random.randint(2, 8)
            for q_id in range(n_queries):
                rows.append({
                    "context": schema,
                    "question": f"Find col_a for entity_{s_id} where score > {q_id}",
                    "answer": f"SELECT col_a FROM entity_{s_id} WHERE score > {q_id};"
                })
    else:
        print("Loading b-mc2/sql-create-context from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset("b-mc2/sql-create-context", split="train")
            rows = [{"context": r["context"], "question": r["question"], "answer": r["answer"]} for r in ds]
        except Exception as e:
            print(f"Could not load HuggingFace dataset directly ({e}).")
            print("Use --offline-sample to verify data pipeline offline or connect to network.")
            return

    train_r, val_r, test_u_r, test_s_r = split_dataset(
        rows,
        train_size=args.train_size,
        val_size=args.val_size,
        test_unseen_size=args.test_unseen_size,
        test_seen_size=args.test_seen_size,
        seed=args.seed
    )

    # Save TRL training format (prompt + completion)
    train_sft = [format_prompt_completion(r["context"], r["question"], r["answer"]) for r in train_r]
    val_sft = [format_prompt_completion(r["context"], r["question"], r["answer"]) for r in val_r]

    save_jsonl(train_sft, out_dir / "train.jsonl")
    save_jsonl(val_sft, out_dir / "validation.jsonl")

    # Save evaluation format (schema, question, gold)
    save_jsonl([format_eval_record(r["context"], r["question"], r["answer"]) for r in test_u_r], out_dir / "test_unseen.jsonl")
    save_jsonl([format_eval_record(r["context"], r["question"], r["answer"]) for r in test_s_r], out_dir / "test_seen.jsonl")
    save_jsonl([format_eval_record(r["context"], r["question"], r["answer"]) for r in val_r], out_dir / "validation_eval.jsonl")

    # Save split manifest summary metadata
    manifest = {
        "dataset": "b-mc2/sql-create-context",
        "seed": args.seed,
        "counts": {
            "train": len(train_r),
            "validation": len(val_r),
            "test_unseen": len(test_u_r),
            "test_seen": len(test_s_r)
        },
        "assertions": {
            "train_val_schema_disjoint": True,
            "train_test_unseen_schema_disjoint": True,
            "test_seen_in_train_schemas": True
        }
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {out_dir / 'manifest.json'}")

    # Token stats
    compute_token_stats(train_sft, tokenizer_name_or_path=args.tokenizer)

if __name__ == "__main__":
    main()
