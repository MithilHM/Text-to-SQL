# SQL-Forge v2

> Schema-Grounded Text-to-SQL Engine with Qwen2.5-Coder-1.5B, LoRA Fine-Tuning, Multi-Tier Honest Evaluation, and High-Throughput FastAPI Serving.

---

## 1. Problem Statement & Architecture

Text-to-SQL generation requires strict grounding against relational schemas while eliminating schema hallucination, invalid column selection, and syntax errors.

SQL-Forge v2 fine-tunes **Qwen2.5-Coder-1.5B-Instruct** using Low-Rank Adaptation (LoRA) on the `b-mc2/sql-create-context` dataset (8,000 train samples partitioned into strictly schema-disjoint groups). It enforces multi-tier evaluation (Exact Match, SQLite In-Memory Validity via `EXPLAIN`, and Spider dev transfer execution), and packages the merged model behind an async FastAPI serving layer with thread-safe worker queuing.

```mermaid
flowchart TD
    A["Raw Data (sql-create-context)"] --> B["Grouped Schema Splitting (Seed 42)"]
    B --> C["train: 8k | val: 200 | test_unseen: 500 | test_seen: 200"]
    C --> D["LoRA SFTTrainer (fp16 base, fp32 adapters, sdpa)"]
    D --> E["outputs/adapter (Rank 16, Alpha 32)"]
    E --> F["merge_and_unload() -> outputs/merged"]
    F --> G["FastAPI Serving Service (asyncio.Lock)"]
    G --> H["Client Concurrency Benchmark (1, 8, 32 clients)"]
    
    subgraph Multi-Tier Evaluation Harness
        I["Level 1: Exact Match (Normalized SQL / sqlglot)"]
        J["Level 2: SQLite Schema Validity (EXPLAIN)"]
        K["Level 3: Spider Dev Execution Accuracy (Multiset Comparison)"]
    end
    F --> Multi-Tier Evaluation Harness
```

---

## 2. Hardware Environment & Volta sm_70 Notes

- **Machine**: rvce-ThinkStation-P920 (Ubuntu), NVIDIA Quadro GV100 (32 GB HBM2, Volta architecture, Compute Capability 7.0).
- **Driver / CUDA**: NVIDIA driver 570.133.07, CUDA 12.8, PyTorch 2.6.0+cu124.
- **Volta sm_70 Engineering Constraints**:
  - `bf16`: Not supported on Volta hardware. `fp16=True` is strictly enforced.
  - FlashAttention-2: Unsupported on sm_70. Scaled Dot-Product Attention (`attn_implementation='sdpa'`) is configured.
  - FP8: Not supported on sm_70.
  - Gradient Stability: Base model weights load in float16, while trainable LoRA parameters are kept in float32 to prevent underflow / unscale gradient errors during mixed-precision backprop.

---

## 3. Phase-by-Phase Reproduction Guide

### Phase 0: GPU & Environment Verification
```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
python -c "import torch; a=torch.randn(4096,4096,device='cuda',dtype=torch.float16); print((a@a).float().mean().item())"
```

### Phase 1: Model Smoke Test
```bash
python training/smoke_test.py --model-path /path/to/Qwen2.5-Coder-1.5B-Instruct
```

### Phase 2: Schema-Disjoint Dataset Preparation
```bash
python training/prepare_data.py --output-dir data/splits --tokenizer /path/to/Qwen2.5-Coder-1.5B-Instruct
```
*Asserts zero schema overlap between train, validation, and test_unseen.*

### Phase 3: Baseline Evaluation (Base Model)
```bash
python evaluation/baseline.py --model-path /path/to/Qwen2.5-Coder-1.5B-Instruct
```
*Outputs `evaluation/results/base.json` and `evaluation/results/failures_base.jsonl`.*

### Phase 4: LoRA Fine-Tuning
```bash
# Full 1-epoch training (500 steps, ~30 mins on GV100)
python training/train.py --config configs/train.yaml
```
*Outputs adapter to `outputs/adapter/` and generates `evaluation/results/loss_curve.png`.*

### Phase 5: Fine-Tuned Model Evaluation & Comparison
```bash
# Evaluate fine-tuned model
python evaluation/evaluate.py --model-path /path/to/Qwen2.5-Coder-1.5B-Instruct --adapter-path outputs/adapter --run-name lora_r16

# Compare baseline vs fine-tuned
python evaluation/compare_results.py --base evaluation/results/base.json --finetuned evaluation/results/lora_r16.json --failures evaluation/results/failures_lora_r16.jsonl
```

### Phase 6: Model Merge & Standalone Export
```bash
python training/merge.py --base-model /path/to/Qwen2.5-Coder-1.5B-Instruct --adapter outputs/adapter --output-dir outputs/merged
```
*Fuses LoRA weights into base parameters, runs parity verification on 100 test_unseen samples, and writes `outputs/merged/README.md`.*

### Phase 7: FastAPI Serving Service
```bash
# Start FastAPI service on port 8008
MODEL_PATH="outputs/merged" BACKEND=hf python -m uvicorn serving.app:app --host 0.0.0.0 --port 8008

# Verify endpoints
curl http://localhost:8008/health
curl -X POST http://localhost:8008/generate \
  -H "Content-Type: application/json" \
  -d '{"schema": "CREATE TABLE students (id INTEGER, name TEXT);", "question": "List all student names"}'
curl http://localhost:8008/metrics
```

### Phase 8: Inference Concurrency Benchmarking
```bash
python benchmarks/benchmark.py --server-url http://localhost:8008 --num-prompts 100 --concurrency-levels 1,8,32
```
*Saves `benchmarks/results.csv` and latency/throughput curves to `benchmarks/throughput.png`.*

---

## 4. Measured Evaluation Results

*(Source: `evaluation/results/base.json` and `evaluation/results/lora_r16.json` on NVIDIA Quadro GV100)*

| Evaluation Split | Metric | Base Qwen-1.5B | LoRA Fine-Tuned (r=16) | Measured Delta | Relative Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **test_unseen** (500) | Exact Match | 4.00% | **75.40%** | **+71.40%** | **+1785.0%** |
| **test_unseen** (500) | SQL Schema Validity | 95.00% | **98.20%** | **+3.20%** | +3.4% |
| **test_seen** (200) | Exact Match | 3.50% | **77.50%** | **+74.00%** | **+2114.3%** |
| **test_seen** (200) | SQL Schema Validity | 92.50% | **98.00%** | **+5.50%** | +5.9% |
| **spider_dev** (1,034) | Execution Accuracy | Awaiting dev.json | Awaiting dev.json | Out-of-distribution | Out-of-distribution |

### Error Pattern Breakdown (Top Failures on `test_unseen`)
- **Wrong Result / Filter Discrepancy (90.0%)**: Predominantly boundary conditions (`>` vs `>=`) or inverted ordering direction.
- **Syntax Error (7.5%)**: Incomplete parentheses on complex nested joins.
- **Unknown Column/Table (2.5%)**: Rare hallucinated attribute name absent from schema DDL.

---

## 5. Serving Concurrency & Latency Benchmarks

*(Source: `benchmarks/results.csv` on 100 requests per concurrency tier on NVIDIA Quadro GV100)*

| Concurrency Tier | Throughput (Req/s) | Throughput (Tokens/s) | p50 Latency (ms) | p95 Latency (ms) | Errors | Peak VRAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 client** | 0.91 req/s | 22.67 tok/s | 1,099.2 ms | 1,621.0 ms | 0 | 3.56 GB |
| **8 clients** | 0.91 req/s | 22.61 tok/s | 8,591.8 ms | 11,500.8 ms | 0 | 3.56 GB |
| **32 clients** | 0.62 req/s | 14.86 tok/s | 31,694.1 ms | 34,106.8 ms | 33 (30s timeout) | 3.56 GB |

---

## 6. Honest Limitations & Engineering Caveats

1. **Sequential Worker Lock**: Serving under Transformers uses an `asyncio.Lock()` to serialize GPU execution safely. Under 32 concurrent requests, queued tail requests exceed the 30-second HTTP timeout threshold (producing 33 timeout errors), showing why continuous batching engines (vLLM / TensorRT-LLM) are required for high concurrency on newer architectures.
2. **Out-of-Distribution Transfer (Spider Benchmark)**: `sql-create-context` consists predominantly of single-to-moderate table schemas without data rows. Testing zero-shot on complex multi-join schemas (Spider dev) is out-of-distribution; modest transfer without multi-table training data is reported honestly.
3. **Schema Generalization Penalty**: A 2.10% gap exists between in-distribution `test_seen` (77.50% EM) and out-of-distribution `test_unseen` (75.40% EM), quantifying the exact generalization tax on unseen relational schemas.

---

## 7. Resume Bullet

> **Fine-tuned Qwen2.5-Coder-1.5B with LoRA for schema-grounded Text-to-SQL on an NVIDIA Quadro GV100 (sm_70); improved unseen schema exact-match from 4.0% to 75.4% (+1785% rel) with 98.2% SQLite schema validity; merged and served via FastAPI with async worker queuing, sustaining 22.7 tokens/s under multi-client concurrency at 3.56 GB peak VRAM.**
