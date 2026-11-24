"""Model loader and inference engine for SQL-Forge serving layer.

Supports:
- BACKEND=hf (Transformers with asyncio.Lock for thread-safe concurrent GPU queueing)
- BACKEND=vllm (Proxy to OpenAI-compatible vLLM server)
- MOCK_MODEL=1 (Mock inference for offline testing and benchmarking harnesses)
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.sql_parser import extract_sql

SYSTEM_PROMPT = (
    "You are an expert SQL generator. Generate valid SQLite SQL using only the provided "
    "database schema. Return SQL only, without explanations."
)


class ModelEngine:
    def __init__(self):
        self.backend = os.getenv("BACKEND", "hf").lower()
        self.model_path = os.getenv("MODEL_PATH", "outputs/merged")
        self.mock = os.getenv("MOCK_MODEL", "0").lower() in ("1", "true", "yes")
        self.vllm_api_base = os.getenv("VLLM_API_BASE", "http://localhost:8001/v1")
        self.device = "cpu"
        self.model_name = "sql-forge-1.5b"

        self.tokenizer = None
        self.model = None
        self.lock = asyncio.Lock()
        self.initialized = False

    def load(self):
        if self.initialized:
            return

        if self.mock:
            print("[INFO] ModelEngine running in MOCK mode (no GPU memory consumed).", flush=True)
            self.device = "mock"
            self.initialized = True
            return

        if self.backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            attn = "sdpa" if self.device == "cuda" else "eager"

            # Check if model path exists, else fallback to HF base
            target_path = self.model_path
            if not Path(target_path).exists():
                print(f"[WARN] Local merged model path '{target_path}' not found. Falling back to base model.", flush=True)
                target_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

            print(f"Loading HF model from {target_path} (dtype={dtype}, attn={attn}, device={self.device})...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                target_path,
                torch_dtype=dtype,
                attn_implementation=attn,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()
            print("HF Model successfully loaded into memory.", flush=True)
        elif self.backend == "vllm":
            print(f"Configured for vLLM proxy at {self.vllm_api_base}", flush=True)
            self.device = "vllm_service"

        self.initialized = True

    async def generate(self, schema: str, question: str) -> Tuple[str, float, int]:
        """Generate SQL query given schema and question.
        
        Returns:
            (sql_query, latency_ms, output_tokens)
        """
        if not self.initialized:
            self.load()

        t_start = time.perf_counter()

        if self.mock:
            await asyncio.sleep(0.015)  # simulate fast 15ms inference
            simulated_sql = "SELECT name FROM students WHERE age > 20;"
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return simulated_sql, latency_ms, 9

        if self.backend == "hf":
            import torch
            # Lock ensures sequential GPU access for concurrent request safety
            async with self.lock:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion:\n{question}"}
                ]
                prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                sql = extract_sql(raw_text)
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                return sql, latency_ms, len(gen_tokens)

        elif self.backend == "vllm":
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "model": "sql-forge-1.5b",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion:\n{question}"}
                    ],
                    "max_tokens": 256,
                    "temperature": 0.0
                }
                resp = await client.post(f"{self.vllm_api_base}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                sql = extract_sql(raw_text)
                output_tokens = data.get("usage", {}).get("completion_tokens", 10)
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                return sql, latency_ms, output_tokens

        raise ValueError(f"Unsupported backend: {self.backend}")


engine = ModelEngine()
