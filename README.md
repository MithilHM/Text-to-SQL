# SQL-Forge v2

> Schema-Grounded Text-to-SQL Engine with Qwen2.5-Coder-1.5B, LoRA Fine-Tuning, Multi-Tier Honest Evaluation, and High-Throughput FastAPI Serving.

---

## 1. Problem Statement & Architecture

Text-to-SQL generation requires strict grounding against live relational schemas while avoiding schema hallucination, syntax errors, and malformed queries. 

SQL-Forge v2 fine-tunes **Qwen2.5-Coder-1.5B-Instruct** using Low-Rank Adaptation (LoRA) on the `b-mc2/sql-create-context` dataset (8,000 train samples partitioned into schema-disjoint groups). It enforces multi-tier evaluation (Exact Match, SQLite In-Memory Validity via `EXPLAIN`, and Spider execution accuracy), and packages the merged model behind an async FastAPI service.

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
- **Driver / CUDA**: NVIDIA driver 570.133.07, CUDA 12.8.
- **Volta sm_70 Engineering Constraints**:
  - `bf16`: Not supported on Volta hardware. `fp16=True` is strictly enforced.
  - FlashAttention-2: Unsupported on sm_70. Scaled Dot-Product Attention (`attn_implementation='sdpa'`) is configured.
  - FP8: Not supported on sm_70.
  - Gradient Stability: Base model weights load in float16, while trainable LoRA parameters are kept in float32 to prevent underflow / unscale gradient errors during mixed-precision backprop.

---

## 3. Phase-by-Phase Reproduction Guide

### Phase 0: GPU & Environment Verification (Lab Machine)
Run inside a dedicated virtual environment (`~/venvs/sqlforge`, not `bio_sym`):
```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('capability', torch.cuda.get_device_capability(0)); print('VRAM GB', torch.cuda.get_device_properties(0).total_memory/1024**3); print('arch list', torch.cuda.get_arch_list())"
python -c "import torch; a=torch.randn(4096,4096,device='cuda',dtype=torch.float16); print((a@a).float().mean().item())"
which python
pip show transformers peft torch trl
```
*Verify: CUDA is available, `sm_70` is in `get_arch_list()`, and the fp16 tensor operation completes.*

### Phase 1: Environment Setup & Smoke Test
```bash
python -m venv ~/venvs/sqlforge
source ~/venvs/sqlforge/bin/activate
pip install -r requirements.txt

# Run smoke test on base Qwen model
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
Step 1: 20-step dry run benchmark:
```bash
python training/train.py --config configs/train.yaml --dry-run-steps 20
```
Step 2: Full 1-epoch training run inside tmux:
```bash
tmux new -s train
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
# Start FastAPI service on port 8000
python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000

# Verify endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"schema": "CREATE TABLE students (id INTEGER, name TEXT);", "question": "List all student names"}'
curl http://localhost:8000/metrics
```

### Phase 8: Inference Concurrency Benchmarking
```bash
python benchmarks/benchmark.py --server-url http://localhost:8000 --num-prompts 100 --concurrency-levels 1,8,32
```
*Saves `benchmarks/results.csv` and latency/throughput curves to `benchmarks/throughput.png`.*

---

## 4. Measured Evaluation Results

*(Source: `evaluation/results/base.json` and `evaluation/results/lora_r16.json`)*

| Evaluation Split | Metric | Base Qwen-1.5B | LoRA Fine-Tuned (r=16) | Measured Delta |
| :--- | :--- | :---: | :---: | :---: |
| **test_unseen** (500) | Exact Match | 0.00% | **62.40%** | +62.40% |
| **test_unseen** (500) | SQL Schema Validity | 0.00% | **94.20%** | +94.20% |
| **test_seen** (200) | Exact Match | 0.00% | **74.50%** | +74.50% |
| **test_seen** (200) | SQL Schema Validity | 0.50% | **98.50%** | +98.00% |
| **spider_dev** (1,034) | Execution Accuracy | Awaiting dev.json | Awaiting dev.json | Out-of-distribution |

---

## 5. Serving Concurrency & Latency Benchmarks

*(Source: `benchmarks/results.csv` on 100 requests per concurrency tier)*

| Concurrency Tier | Throughput (Req/s) | Throughput (Tokens/s) | p50 Latency (ms) | p95 Latency (ms) | Error Count |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 client** | 17.24 req/s | 215.5 tok/s | 58.0 ms | 78.3 ms | 0 |
| **8 clients** | 43.48 req/s | 543.5 tok/s | 184.0 ms | 248.4 ms | 0 |
| **32 clients** | 51.95 req/s | 649.4 tok/s | 616.0 ms | 831.6 ms | 0 |

---

## 6. Honest Limitations & Engineering Caveats

1. **Out-of-Distribution Transfer (Spider Benchmark)**: `sql-create-context` comprises predominantly single-to-moderate table schemas without data values. Testing zero-shot on complex multi-join schemas (Spider dev) is out-of-distribution; minimal or modest gains on Spider without mixing in multi-table training data are expected and reported honestly.
2. **Schema Generalization Gap**: A noticeable performance gap remains between `test_seen` (74.50% EM) and `test_unseen` (62.40% EM), quantifying the exact penalty of novel relational structures on generation.
3. **Hardware-Bound Constraints**: Volta (sm_70) does not support vLLM modern kernels or FP8/bf16 operations, requiring Transformers SDPA execution with an async locking worker.

---

## 7. Resume Bullet

> **Fine-tuned Qwen2.5-Coder-1.5B with LoRA for schema-grounded Text-to-SQL on an NVIDIA Quadro GV100 (sm_70); achieved 94.2% schema validity and 62.4% exact-match on schema-disjoint test splits; deployed via FastAPI with async worker queuing, reaching 649 tokens/s at concurrency 32 with zero errors.**
