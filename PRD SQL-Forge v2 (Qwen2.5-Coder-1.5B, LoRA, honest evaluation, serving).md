# PRD: SQL-Forge v2 (Qwen2.5-Coder-1.5B, LoRA, honest evaluation, serving)

Owner: Mithil | Hardware: NVIDIA Quadro GV100 32 GB (Volta, sm\_70) | Date: 24 Nov 2026 | Paste this whole document into Antigravity as the project brief.

## 0. Agent operating rules (read first)

1. Work one phase at a time (section 9). At the end of each phase, print what you ran, the measured results, and the files changed, then STOP and wait for my go-ahead.
2. Never invent a number. Every metric in the README must come from a results JSON file produced by a script in this repo. Each results file stores the git commit hash, the config used, and a timestamp.
3. This is a shared lab machine. Do not touch the existing `bio_sym` environment, do not kill processes you did not start, and do not use `sudo`. Run anything longer than 2 minutes inside `tmux`.
4. Ask before installing or upgrading packages. Pin the versions that work into `requirements.txt` once verified.
5. Before using any library argument that varies across versions (TRL, PEFT, vLLM), check the installed version with `pip show` and adapt the argument names to it.
6. If something fails twice, stop, show the full error, and propose two options instead of looping.
7. Commit after every phase with a message like `phase-3: baseline eval on spider dev`.

## 1. Objective

Fine-tune Qwen2.5-Coder-1.5B-Instruct for schema-grounded Text-to-SQL, prove whether it helps with a defensible evaluation, serve it behind a FastAPI service, and measure GPU memory, throughput and latency under concurrency.

Four engineering components: model training, model evaluation, model deployment, performance engineering.

Project rule: every claimed improvement must be backed by an experiment. A negative or small result, reported honestly with error analysis, is acceptable.

Why this project: the resume is strong on distributed systems and classical ML but has no LLM fine-tuning, evaluation or inference-serving work. This fills that gap and also reads as backend engineering because it ships as an API.

## 2. Environment and constraints

| Item | Value |
| --- | --- |
| Machine | rvce-ThinkStation-P920 (Ubuntu), single GPU |
| GPU | Quadro GV100, 32 GB HBM2, compute capability 7.0, NVIDIA driver 570.133.07, CUDA 12.8 |
| Known warning | `infoROM is corrupted` in nvidia-smi: usually harmless for compute; do not rely on persistence mode or ECC |
| Base model | Qwen2.5-Coder-1.5B-Instruct (reuse the existing local directory, do not download another copy) |
| Time budget | about 4 hours of GPU access in total |

Volta constraints:

- No bf16: use fp16 (`fp16=True`).
- No FlashAttention-2: use `attn_implementation='sdpa'`.
- No FP8.
- Confirm the installed PyTorch build contains sm\_70 kernels: `torch.cuda.get_arch_list()` must list `sm_70`. If it does not, install a PyTorch build that does and tell me before proceeding.
- Mixed-precision note: load the base model in float16 but keep LoRA adapter weights in float32. If training fails with `Attempting to unscale FP16 gradients`, cast the trainable parameters to float32.

## 3. Model and data

### 3.1 Base model

Qwen2.5-Coder-1.5B-Instruct: about 1.54B parameters, 28 layers, 32,768-token advertised context. Always build prompts with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`; never hand-write the special tokens.

### 3.2 Training dataset

`b-mc2/sql-create-context` (about 78.6k rows). Fields: `question`, `context` (CREATE TABLE statements), `answer` (gold SQL). It has schemas and queries but NO table rows.

### 3.3 Splits (grouped by schema, not random)

Group rows by a normalized hash of `context` (lowercase, collapse whitespace) and split on groups with a fixed seed (42), so no schema in test or validation appears in train.

| Split | Size | Rule |
| --- | --- | --- |
| train | 8,000 | from train-group schemas only |
| validation | 200 | schema-disjoint from train and test |
| test\_unseen | 500 | schema-disjoint from train (schema-generalization test) |
| test\_seen | 200 | new questions whose schemas DO appear in train (in-distribution reference) |

Write the split manifests to `data/splits/*.jsonl` and assert zero schema overlap in `prepare_data.py`.

### 3.4 Prompt format

System message:

```text
You are an expert SQL generator. Generate valid SQLite SQL using only the provided database schema. Return SQL only, without explanations.
```

User message:

```text
Database schema:
{context}

Question:
{question}
```

Expected assistant output: the SQL only, for example `SELECT name FROM students WHERE age > 20;`

Training records use the TRL prompt-completion conversational format so loss is computed only on the SQL: `{"prompt": [system, user], "completion": [{"role": "assistant", "content": sql}]}`.

## 4. Evaluation design (corrected)

Problem with the earlier plan: sql-create-context has no data rows. Executing predicted and gold SQL on an empty SQLite database makes a wrong query and a right query both return empty results, so execution accuracy would be meaningless. Do not report it for this dataset.

| Level | Metric | Dataset | What it proves |
| --- | --- | --- | --- |
| 1 | Exact-match after normalization | test\_unseen, test\_seen | Output matches the reference SQL (lowercase, strip trailing semicolon, collapse whitespace; optionally normalize through `sqlglot`) |
| 2 | SQL validity | test\_unseen, test\_seen | Query is valid against its schema: build an in-memory SQLite DB from `context` and run `EXPLAIN <sql>`; failure = invalid (catches syntax errors and unknown tables/columns) |
| 3 | Execution accuracy | Spider dev (1,034 questions, 20 databases with real data) | Predicted and gold queries return equivalent result sets on populated databases |

Execution accuracy rules (Spider dev):

- Execute both queries with a 10-second timeout per query. A prediction that errors or times out counts as incorrect.
- Compare result sets as multisets of rows; compare as ordered lists only when the gold query contains `ORDER BY`.
- Build the prompt schema from the database's own `sqlite_master` CREATE statements, using the same prompt format as section 3.4.
- Data location: `data/spider/dev.json` and `data/spider/database/<db_id>/<db_id>.sqlite`. These come from the official Spider release. If they are not on the machine, ask me to download them elsewhere and `scp` them over; do not fabricate or substitute data.

Honesty caveat to put in the README: training on sql-create-context (mostly simple, single-table queries) and testing on Spider (multi-table joins) is out of distribution. The fine-tune may improve Spider only slightly or not at all. Report whatever is measured. Optional Experiment B (stretch): also train on Spider train (about 7,000 examples) and compare.

Also record for every run: mean output tokens, error categories (syntax error, unknown column/table, wrong result, timeout), and 20 hand-picked failure examples saved to `evaluation/results/failures_<run>.jsonl`.

Generation settings for all evals: greedy decoding, `max_new_tokens=256`, same prompt, same batch size, fp16, sdpa.

## 5. Training configuration

Method: supervised fine-tuning with LoRA on the fp16 model (no quantization first). Use TRL `SFTTrainer` with the prompt-completion dataset and completion-only loss.

| Parameter | Value |
| --- | --- |
| Precision | fp16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | all linear layers (q, k, v, o, gate, up, down projections) |
| Learning rate | 1e-4 |
| Epochs | 1 |
| Per-device batch size | 2 |
| Gradient accumulation | 8 (effective batch 16) |
| Max sequence length | 1,024 |
| Optimizer | AdamW |
| Scheduler | cosine, warmup ratio 0.03 |
| Gradient clipping | 1.0 |
| Checkpoints | every 100 steps, keep last 3 |
| Logging | local JSONL log plus W&B if logged in |

All values live in `configs/train.yaml`; `train.py` reads them and writes the final merged config next to the adapter.

### Ablation (after the main run works)

Run LoRA rank 8, 16, 32 with identical data and seed. Record for each: final train loss, validation loss, exact-match, validity, Spider execution accuracy, training time, peak GPU memory.

Optional follow-up: 4-bit QLoRA at rank 16 to compare training speed, memory and quality against fp16 LoRA.

## 6. Repository structure

```text
sql-forge/
  configs/train.yaml
  data/processed/
  data/splits/
  data/spider/                 (dev.json + database/, not committed)
  training/prepare_data.py
  training/train.py
  training/merge.py
  evaluation/baseline.py
  evaluation/evaluate.py
  evaluation/sql_parser.py     (normalization, EXPLAIN validity, result comparison)
  evaluation/results/
  serving/app.py
  serving/model_loader.py
  benchmarks/benchmark.py
  benchmarks/results.csv
  outputs/checkpoints/
  outputs/adapter/
  outputs/merged/
  requirements.txt
  README.md
  .gitignore                   (outputs/, data/spider/, *.safetensors, wandb/)
```

## 7. Serving

### 7.1 Backends

- Primary: Transformers + FastAPI, merged fp16 model, `generate` under a lock (or a bounded worker) so concurrent requests queue safely.
- Experimental: vLLM. Current vLLM releases require compute capability 7.5 or higher, which excludes the GV100 (7.0); older vLLM documentation (for example 0.8.1) lists V100 / compute 7.0 as supported. Try a compatible older build in a SEPARATE virtual environment, verify against that version's docs, and do not make the project depend on it.
- `serving/app.py` selects the backend with the `BACKEND=hf|vllm` environment variable. For vLLM, call its OpenAI-compatible server from FastAPI.

### 7.2 API

| Endpoint | Function |
| --- | --- |
| `POST /generate` | Accept schema and question, return SQL |
| `GET /health` | Model loaded, device, backend, uptime |
| `GET /metrics` | Request count, error count, p50/p95 latency, mean tokens/s, current GPU memory |

Request:

```json
{"schema": "CREATE TABLE students (id INTEGER, name TEXT)", "question": "List all student names"}
```

Response:

```json
{"sql": "SELECT name FROM students;", "model": "sql-forge-1.5b", "latency_ms": 412, "output_tokens": 9}
```

Requirements: Pydantic validation (non-empty fields, schema length cap), 30-second request timeout, structured error responses, SQL-only output (strip code fences).

## 8. Measurements to record (no preset targets)

| Experiment | Measurement |
| --- | --- |
| Base model | Exact-match, validity (test\_unseen, test\_seen), Spider execution accuracy |
| Fine-tuned model | Same metrics, plus absolute and relative change vs base |
| Training | Wall-clock time, peak GPU memory, loss curve |
| Generalization | test\_unseen vs test\_seen gap |
| Ablation | Rank 8 / 16 / 32 comparison |
| Serving | Tokens/s, p50 and p95 latency, peak GPU memory |
| Concurrency | 1, 8, 32 concurrent clients, 100 requests each |
| Backends | HF vs vLLM (only if vLLM works) |

## 9. Phases (execute one at a time)

### Phase 0: GPU and environment verification

Agent task: run the commands below and show me the raw output.

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('capability', torch.cuda.get_device_capability(0)); print('VRAM GB', torch.cuda.get_device_properties(0).total_memory/1024**3); print('arch list', torch.cuda.get_arch_list())"
python -c "import torch; a=torch.randn(4096,4096,device='cuda',dtype=torch.float16); print((a@a).float().mean().item())"
which python
pip show transformers peft torch trl
```

Done when: CUDA is available, `sm_70` is in the arch list, the fp16 matmul runs, and I have seen the installed versions.

### Phase 1: Locate model, create repo and environment

Agent task:

1. Find the model: `find ~ -maxdepth 5 -type d -name '*Qwen2.5-Coder-1.5B*' 2>/dev/null`. Show me the paths and wait for me to pick one.
2. Create `~/sql-forge` with the structure in section 6, run `git init`, add `.gitignore`.
3. Create a fresh venv `python -m venv ~/venvs/sqlforge` (not `bio_sym`). Ask before installing: torch (matching sm\_70), transformers, peft, trl, datasets, accelerate, sqlglot, fastapi, uvicorn, httpx, pydantic, pyyaml, pandas, matplotlib, wandb (optional).
4. Smoke test: load the model in fp16 with sdpa attention, apply the chat template to the example in section 3.4, generate once, print the SQL and the time taken.

Done when: the smoke test prints a SQL query and `requirements.txt` is committed.

### Phase 2: Dataset preparation

Agent task: implement `training/prepare_data.py`:

1. Load `b-mc2/sql-create-context` and group by normalized schema hash (section 3.3).
2. Produce train 8,000 / validation 200 / test\_unseen 500 / test\_seen 200 with seed 42; assert no schema overlap between train and test\_unseen or validation.
3. Write the prompt-completion JSONL files (section 3.4) and the plain eval files (schema, question, gold).
4. Print token-length statistics (mean, p95, max) using the Qwen tokenizer to confirm 1,024 is enough.

Done when: split files exist, overlap assertions pass, and the stats are printed.

### Phase 3: Baseline evaluation

Agent task: implement `evaluation/sql_parser.py`, `evaluation/evaluate.py` and `evaluation/baseline.py` as specified in section 4, and run the BASE model on test\_unseen, test\_seen and Spider dev.

1. Batched greedy generation with left padding, SQL extraction (strip fences, take the first statement).
2. Save `evaluation/results/base.json` with all metrics and `failures_base.jsonl`.
3. Print a summary table.

Done when: `base.json` exists and Spider execution accuracy is recorded (or Spider is flagged as missing and I have been asked to provide it).

### Phase 4: LoRA fine-tuning

Agent task: implement `configs/train.yaml` and `training/train.py` with the section 5 settings.

1. First run a 20-step dry run, then print: loss trend, seconds per step, peak memory. Estimate total time and ask me to approve the full run.
2. Run the full 1-epoch training in `tmux new -s train`, log every 10 steps, checkpoint every 100.
3. Save the final adapter to `outputs/adapter` and a `training_summary.json` (time, peak memory, final losses).

Done when: the adapter and `training_summary.json` exist and the loss curve is saved as `evaluation/results/loss_curve.png`.

### Phase 5: Evaluate the fine-tuned model

Agent task: run the same evaluation code with the adapter loaded.

1. Save `evaluation/results/lora_r16.json` and `failures_lora_r16.jsonl`.
2. Print a before/after table: base vs fine-tuned for every metric and split, with absolute and relative change.
3. Summarize the most common failure patterns in 5 lines of plain text.

Done when: the before/after table is printed from the two JSON files only.

### Phase 5b: Rank ablation (only if time allows)

Repeat Phases 4 and 5 for rank 8 and 32 (alpha = 2 x rank). Save `lora_r8.json` and `lora_r32.json`, and produce one comparison table.

### Phase 6: Merge and export

Agent task: implement `training/merge.py`: load the base model in fp16, load the adapter, `merge_and_unload()`, save to `outputs/merged` with the tokenizer. Re-run a 100-example subset of test\_unseen on the merged model and confirm the metrics match the adapter run within noise. Write a model card draft (dataset, hyperparameters, results, limitations). Ask me before pushing anything to Hugging Face.

Done when: the merged model reproduces the adapter results.

### Phase 7: FastAPI serving layer

Agent task: implement `serving/model_loader.py` and `serving/app.py` per section 7.

1. Run `uvicorn serving.app:app --port 8000` in tmux.
2. Verify with `curl` on `/health`, `/generate` (use the example request) and `/metrics`; show the outputs.
3. Add a test that sends 5 sequential requests and checks the SQL is non-empty.
4. If I ask for it, set up the vLLM backend in a separate venv and report whether it starts on this GPU.

Done when: all three endpoints respond correctly.

### Phase 8: Benchmark inference

Agent task: implement `benchmarks/benchmark.py` (asyncio + httpx).

1. Use 100 prompts sampled from test\_unseen. Run at concurrency 1, 8, 32.
2. Record per level: requests/s, output tokens/s, p50 and p95 latency, error count, peak GPU memory (sample `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` every 0.5 s).
3. Write `benchmarks/results.csv` and `benchmarks/throughput.png`. Repeat for vLLM only if it works.

Done when: the CSV and chart exist and the numbers are printed.

### Phase 9: Documentation and packaging

Agent task: write `README.md` with: problem statement, architecture diagram (text or Mermaid), setup and reproduction commands per phase, the before/after table, ablation table, benchmark chart, honest limitations (out-of-distribution evaluation, small training set, single run per config), and the Volta notes. Pull every number from the results files. Then print the final resume bullet with real numbers filled in.

Done when: a fresh clone can reproduce each stage by following the README.

## 10. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| PyTorch build lacks sm\_70 kernels | Check `get_arch_list()` in Phase 0; install a build that includes sm\_70 |
| fp16 gradient errors or loss spikes | Keep LoRA weights in fp32, clip at 1.0, drop lr to 5e-5 |
| Spider files unavailable offline | Download elsewhere and `scp`; until then report only levels 1 and 2 and label execution accuracy as pending |
| No Spider improvement | Report honestly; run Experiment B with Spider train mixed in |
| vLLM will not run on compute 7.0 | Ship the HF backend as primary; document the vLLM attempt and the exact error |
| GPU access ends early | Push adapter checkpoints and results JSON to GitHub after each phase |
| Schema leakage inflates scores | Grouped split with overlap assertions (section 3.3) |

## 11. Deliverables

- GitHub repo `sql-forge` with code, configs, results JSON files and README
- Merged model and adapter (Hugging Face upload only after I approve)
- Before/after evaluation table and rank ablation table
- Benchmark CSV and chart
- Resume bullet using only measured numbers, in this shape: Fine-tuned Qwen2.5-Coder-1.5B with LoRA for schema-grounded Text-to-SQL on a Quadro GV100; evaluated on schema-disjoint splits and Spider dev (execution accuracy X% to Y%); served via FastAPI at Z tokens/s, p95 latency N ms at concurrency C.

## 12. Stretch goals

- Experiment B: add Spider train to the training mix and compare against the sql-create-context-only model
- 4-bit QLoRA comparison against fp16 LoRA
- Simple dynamic batching in the HF server and its throughput effect
- Dockerfile and a GitHub Actions workflow that runs the unit tests for `sql_parser.py`
