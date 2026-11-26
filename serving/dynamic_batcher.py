"""Dynamic Batching Inference Engine for High-Concurrency Serving.

Replaces sequential lock with an async micro-batching queue:
- Aggregates incoming requests within a configurable time window (e.g. 50ms).
- When batch reaches max_batch_size (e.g. 8) or timer expires, executes batched generate().
- Resolves each individual caller's asyncio.Future with its prediction.
- Eliminates sequential lock bottleneck and prevents request timeouts under concurrency 32.
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.sql_parser import extract_sql

SYSTEM_PROMPT = (
    "You are an expert SQL generator. Generate valid SQLite SQL using only the provided "
    "database schema. Return SQL only, without explanations."
)


@dataclass
class GenerationTask:
    schema: str
    question: str
    future: asyncio.Future
    created_at: float


class DynamicBatchingEngine:
    def __init__(
        self,
        batch_window_ms: float = 50.0,
        max_batch_size: int = 8,
        max_new_tokens: int = 256
    ):
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens

        self.model_path = os.getenv("MODEL_PATH", "outputs/merged")
        self.mock = os.getenv("MOCK_MODEL", "0").lower() in ("1", "true", "yes")
        self.model_name = "sql-forge-1.5b"
        self.device = "cpu"

        self.queue: asyncio.Queue = asyncio.Queue()
        self.tokenizer = None
        self.model = None
        self.worker_task: Optional[asyncio.Task] = None
        self.initialized = False

    def load(self):
        if self.initialized:
            return

        if self.mock:
            self.device = "mock"
            self.initialized = True
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        attn = "sdpa" if self.device == "cuda" else "eager"

        target_path = self.model_path
        if not Path(target_path).exists():
            target_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

        print(f"Loading Dynamic Batching Engine from {target_path}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
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
        self.initialized = True
        print("Dynamic Batching Engine initialized.", flush=True)

    def start_worker(self):
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._batching_loop())

    async def stop_worker(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _batching_loop(self):
        """Continuous micro-batching consumer loop."""
        while True:
            # Wait for at least one item
            first_task: GenerationTask = await self.queue.get()
            batch: List[GenerationTask] = [first_task]

            deadline = time.perf_counter() + (self.batch_window_ms / 1000.0)

            # Drain queue up to max_batch_size until window deadline
            while len(batch) < self.max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    task = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(task)
                except asyncio.TimeoutError:
                    break

            # Execute the accumulated micro-batch
            try:
                await self._process_batch(batch)
            except Exception as e:
                for t in batch:
                    if not t.future.done():
                        t.future.set_exception(e)

    async def _process_batch(self, batch: List[GenerationTask]):
        if not batch:
            return

        if self.mock:
            await asyncio.sleep(0.020)  # simulate 20ms batched inference
            for t in batch:
                lat = (time.perf_counter() - t.created_at) * 1000.0
                t.future.set_result(("SELECT * FROM table;", lat, 10))
            return

        import torch

        # Build batched prompts with left padding
        prompts = []
        for t in batch:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Database schema:\n{t.schema}\n\nQuestion:\n{t.question}"}
            ]
            prompts.append(self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Run batched forward pass on GPU
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        )

        for i, t in enumerate(batch):
            inp_len = inputs["input_ids"][i].shape[0]
            gen_tokens = outputs[i][inp_len:]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            sql = extract_sql(gen_text)
            lat_ms = (time.perf_counter() - t.created_at) * 1000.0
            if not t.future.done():
                t.future.set_result((sql, lat_ms, len(gen_tokens)))

    async def generate(self, schema: str, question: str) -> Tuple[str, float, int]:
        if not self.initialized:
            self.load()
        self.start_worker()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        task = GenerationTask(
            schema=schema,
            question=question,
            future=future,
            created_at=time.perf_counter()
        )
        await self.queue.put(task)
        return await future


dynamic_engine = DynamicBatchingEngine()
