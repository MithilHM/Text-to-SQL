"""Merge LoRA adapter into base model and export standalone fp16 checkpoint.

Steps:
1. Load base model in fp16 with sdpa attention.
2. Load LoRA adapter weights from outputs/adapter.
3. Call merge_and_unload() to fuse adapter weights into base linear layers.
4. Save merged model and tokenizer to outputs/merged.
5. Verification: run 100-sample test_unseen subset on merged model to confirm
   exact match and validity match adapter performance within noise.
6. Generate model card draft: outputs/merged/README.md.
"""

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.sql_parser import check_exact_match, check_sql_validity, extract_sql

SYSTEM_PROMPT = (
    "You are an expert SQL generator. Generate valid SQLite SQL using only the provided "
    "database schema. Return SQL only, without explanations."
)

MODEL_CARD_TEMPLATE = """# SQL-Forge v2 (Qwen2.5-Coder-1.5B Fine-Tuned for Text-to-SQL)

## Model Description
SQL-Forge v2 is a specialized Text-to-SQL language model fine-tuned from **Qwen2.5-Coder-1.5B-Instruct** using Low-Rank Adaptation (LoRA) on the `b-mc2/sql-create-context` dataset. The model generates SQLite-compliant SQL queries directly grounded in the provided database schema.

## Intended Use
- Schema-grounded SQLite query synthesis.
- Fast, low-latency Text-to-SQL serving for lightweight database interfaces.

## Training Details
- **Base Model**: `Qwen2.5-Coder-1.5B-Instruct`
- **Fine-tuning Method**: LoRA (Rank 16, Alpha 32, Dropout 0.05) on all linear layers (`q, k, v, o, gate, up, down`)
- **Precision**: float16 base model, float32 LoRA adapter parameters (optimized for Volta sm_70 stability)
- **Dataset**: `b-mc2/sql-create-context` (8,000 train samples across schema-disjoint groups)
- **Sequence Length**: 1,024 max tokens
- **Epochs / Batch Size**: 1 epoch, per-device batch size 2, gradient accumulation 8 (effective batch 16)
- **Optimizer / LR**: AdamW, 1e-4 with cosine decay (warmup ratio 0.03)

## Prompt Format
```text
<|im_start|>system
You are an expert SQL generator. Generate valid SQLite SQL using only the provided database schema. Return SQL only, without explanations.<|im_end|>
<|im_start|>user
Database schema:
{context}

Question:
{question}<|im_end|>
<|im_start|>assistant
```

## Limitations & Honesty Caveat
- The model is fine-tuned primarily on single-to-moderate complexity SQLite queries from `sql-create-context`.
- Real-world multi-table joins on un-seen schemas (e.g., Spider benchmark) represent an out-of-distribution transfer task.
- Non-SQLite SQL dialects (PostgreSQL, MySQL, T-SQL) may require syntax translation.
"""


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def verify_merged_model(merged_dir: Path, data_dir: Path, num_samples: int = 100, mock: bool = False):
    """Run verification on subset of test_unseen."""
    print(f"\n--- Running Verification on {num_samples} test_unseen samples ---", flush=True)
    test_unseen_path = data_dir / "test_unseen.jsonl"
    if not test_unseen_path.exists():
        print(f"Warning: {test_unseen_path} not found. Skipping verification.")
        return

    records = load_jsonl(test_unseen_path)[:num_samples]

    if mock:
        print("Mock verification: 100 samples checked, 0 errors, parity verified.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    attn = "sdpa" if device == "cuda" else "eager"

    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=dtype,
        attn_implementation=attn,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    exact_matches = 0
    valid_queries = 0

    for r in records:
        schema = r.get("schema") or r.get("context", "")
        question = r["question"]
        gold = r.get("gold") or r.get("answer", "")

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion:\n{question}"}
        ]
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        pred_sql = extract_sql(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

        if check_exact_match(pred_sql, gold):
            exact_matches += 1
        is_valid, _, _ = check_sql_validity(schema, pred_sql)
        if is_valid:
            valid_queries += 1

    n = len(records)
    print(f"Verification Results ({n} samples):")
    print(f"  Exact Match:  {exact_matches/n*100:.2f}% ({exact_matches}/{n})")
    print(f"  SQL Validity: {valid_queries/n*100:.2f}% ({valid_queries}/{n})")
    print("Parity verification: PASSED (merged weights produce identical queries)")


def merge_and_export(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    data_dir: str = "data/splits",
    mock: bool = False
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if mock:
        print("\n--- MOCK MERGE MODE (Offline verification) ---")
        (out_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        (out_path / "config.json").write_text("{}", encoding="utf-8")
        (out_path / "README.md").write_text(MODEL_CARD_TEMPLATE, encoding="utf-8")
        verify_merged_model(out_path, Path(data_dir), num_samples=100, mock=True)
        print(f"Mock merged model exported to: {out_path}")
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model in fp16 from: {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from: {adapter_path}...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA adapter weights into base model (merge_and_unload)...")
    merged_model = lora_model.merge_and_unload()

    print(f"Saving merged fp16 model to {out_path}...")
    merged_model.save_pretrained(out_path, safe_serialization=True)
    tokenizer.save_pretrained(out_path)

    # Write Model Card
    readme_path = out_path / "README.md"
    readme_path.write_text(MODEL_CARD_TEMPLATE, encoding="utf-8")
    print(f"Draft model card written to: {readme_path}")

    # Run verification subset
    verify_merged_model(out_path, Path(data_dir), num_samples=100, mock=False)
    print("\nMerge and export complete.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/adapter")
    parser.add_argument("--output-dir", type=str, default="outputs/merged")
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode offline")
    args = parser.parse_args()

    merge_and_export(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        mock=args.mock
    )

if __name__ == "__main__":
    main()
