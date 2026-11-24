"""Phase 1 Smoke Test: Load Qwen2.5-Coder-1.5B-Instruct in fp16 with sdpa, test generation."""

import argparse
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are an expert SQL generator. Generate valid SQLite SQL using only the provided "
    "database schema. Return SQL only, without explanations."
)

SAMPLE_SCHEMA = "CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
SAMPLE_QUESTION = "List all students whose age is greater than 20"

def run_smoke_test(model_path: str):
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    attn_impl = "sdpa" if torch.cuda.is_available() else "eager"

    print(f"Device: {device}, Dtype: {dtype}, Attention: {attn_impl}")
    print(f"Loading model from: {model_path}...")
    start_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to(device)
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Database schema:\n{SAMPLE_SCHEMA}\n\nQuestion:\n{SAMPLE_QUESTION}"}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    print("Running inference...")
    start_gen = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id
        )
    gen_time = time.perf_counter() - start_gen
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print("\n" + "=" * 50)
    print("SMOKE TEST RESULT:")
    print(f"Prompt Schema:   {SAMPLE_SCHEMA}")
    print(f"Prompt Question: {SAMPLE_QUESTION}")
    print(f"Generated SQL:   {sql}")
    print(f"Generation Time: {gen_time:.3f}s ({len(generated_tokens)} tokens, {len(generated_tokens)/gen_time:.1f} tok/s)")
    print("=" * 50)
    return sql

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 smoke test for Qwen2.5-Coder-1.5B")
    parser.add_argument("--model-path", type=str, required=True, help="Path or HuggingFace ID of model")
    args = parser.parse_args()
    run_smoke_test(args.model_path)
