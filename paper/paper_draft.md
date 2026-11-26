# SQL-Forge: Empirical Evaluation of Low-Rank Adaptation and Dynamic Serving for Schema-Grounded Text-to-SQL Under Hardware Constraints

**Author**: Mithil H M  
**Affiliation**: RV College of Engineering  
**Target Venue**: IEEE Access / Systems for Machine Learning (SysML) Workshop Track  

---

## Abstract
Deploying compact Large Language Models (LLMs) for schema-grounded Text-to-SQL generation is essential for privacy-preserving, enterprise-grade database interfaces. However, fine-tuning and serving sub-2B parameter models under legacy hardware constraints (e.g., NVIDIA Volta architecture without BF16 or FlashAttention-2 support) poses critical challenges in numerical stability, inference throughput, and generalization across unseen relational schemas. In this paper, we present **SQL-Forge**, an end-to-end framework fine-tuning `Qwen2.5-Coder-1.5B-Instruct` using Low-Rank Adaptation (LoRA) on schema-disjoint partitions. We introduce a multi-tier evaluation protocol combining AST-normalized exact match, in-memory SQLite AST verification (`EXPLAIN`), and zero-shot transfer execution. Furthermore, we investigate inference dynamics under multi-client concurrency, contrasting sequential lock queueing against an asynchronous dynamic micro-batching engine. Our empirical evaluation on an NVIDIA Quadro GV100 (32 GB, `sm_70`) demonstrates that LoRA fine-tuning raises unseen schema exact match from **4.00% to 75.40%** (+1785.0% relative increase) with **98.20%** schema validity. We establish a comprehensive 5-point failure taxonomy across schema complexities and show that dynamic micro-batching scales throughput from 14.86 tok/s to **136.90 tok/s** at concurrency 32 while eliminating 100% of request timeouts.

---

## 1. Introduction
Relational database querying via natural language (Text-to-SQL) bridges business intelligence users and complex enterprise databases. While proprietary frontier models (e.g., GPT-4o, Claude 3.5 Sonnet) achieve high execution accuracy, enterprise compliance and data governance strictly necessitate local on-premise deployment. Compact code-specialized models, specifically `Qwen2.5-Coder-1.5B-Instruct`, provide a viable low-latency substrate, but require domain fine-tuning to reliably synthesize valid schema-grounded SQLite syntax without hallucinating columns or tables.

Prior Text-to-SQL benchmarks frequently suffer from two methodological deficiencies:
1. **Schema Leakage**: Random sample splitting inadvertently places identical relational schemas into both training and test sets, inflating apparent generalization.
2. **Impractical Hardware Assumptions**: Modern training and serving stacks routinely assume Ada Lovelace or Hopper GPUs with BF16, FP8, and FlashAttention-2, leaving legacy enterprise workstations (e.g., Volta `sm_70`) vulnerable to numerical instability and runtime failure.

### Contributions
- **Rigorous Disjoint Splitting**: We enforce grouped MD5 schema hashing, producing strictly disjoint partitions (`train`: 8,000, `val`: 200, `test_unseen`: 500, `test_seen`: 200) to isolate genuine generalization from schema memorization.
- **Volta Precision Stability**: We engineer a stable training pipeline on compute capability 7.0 (Quadro GV100), loading base weights in FP16, maintaining trainable LoRA parameters in FP32, and executing scaled dot-product attention (`sdpa`).
- **LoRA Rank & Precision Ablation**: We evaluate ranks 8, 16, and 32 against 4-bit QLoRA, quantifying the tradeoffs between parameter volume, peak VRAM, and unseen schema accuracy.
- **Concurrency & Dynamic Micro-Batching Study**: We analyze tail latency and throughput under concurrent client load, showing how dynamic micro-batching eliminates head-of-line blocking.
- **Formal 5-Point Error Taxonomy**: We systematically classify 123 residual failure modes across small, medium, and large schema topologies.

---

## 2. System Architecture & Methodology

### 2.1 Schema-Disjoint Partitioning
We utilize the `b-mc2/sql-create-context` corpus (78,577 records). To prevent schema leakage, each database schema context $S$ is normalized (whitespace collapsed, lowercased) and mapped to an MD5 hash $H(S)$. Dataset splits are formed over disjoint schema clusters:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test\_unseen}} = \emptyset$$
In addition, we sample a reference split `test_seen` where schemas overlap with train but questions and queries are held out, directly measuring the generalization tax.

### 2.2 LoRA Configuration & Loss Masking
Supervised fine-tuning is performed using completion-only cross-entropy loss over assistant-generated SQL tokens. We inject low-rank decomposition matrices $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{k \times r}$ across all linear projections (`q, k, v, o, gate, up, down`):
$$W_{\text{effective}} = W_0 + \frac{\alpha}{r} B A$$
To circumvent unscaled FP16 gradient underflow on sm_70 architectures, adapter parameters $A, B$ are held in `float32`, while frozen base parameters $W_0$ operate in `float16`.

### 2.3 Multi-Tier Evaluation Protocol
1. **Level 1 (Exact Match)**: Predictions are stripped of Markdown code blocks, normalized using `sqlglot`, and compared against reference queries.
2. **Level 2 (In-Memory SQLite Schema Validity)**: An in-memory SQLite instance is constructed per sample by executing the schema DDL. We execute `EXPLAIN <predicted_sql>;`. If the SQLite engine parses the statement and resolves all table/column references, it is marked valid.
3. **Level 3 (Spider Dev Execution Transfer)**: Cross-domain transfer on Spider dev (1,034 queries across 20 multi-table relational databases) evaluated with multiset equivalence.

### 2.4 Dynamic Micro-Batching Inference Serving
In standard sequential lock serving (`asyncio.Lock`), concurrent client requests queue serially. Under high concurrency ($C=32$), cumulative queuing delay exceeds HTTP timeouts. Our dynamic batching engine introduces a background consumer that aggregates incoming requests within a time window $\Delta t = 50\text{ ms}$ up to a maximum batch size $B=8$, executing batched tensor generation with left-padding on the GPU.

---

## 3. Experimental Results & Analysis

### 3.1 Baseline vs Fine-Tuned Performance
Experiments were conducted on an NVIDIA Quadro GV100 (31.72 GB VRAM, driver 570.133.07, CUDA 12.8, PyTorch 2.6.0+cu124). Training executed for 500 steps (1 epoch) in 29m 54s.

| Metric | Base Qwen-1.5B | LoRA r=16 (SQL-Forge) | Absolute $\Delta$ | Relative $\Delta$ |
| :--- | :---: | :---: | :---: | :---: |
| **test_unseen Exact Match** | 4.00% | **75.40%** | **+71.40%** | **+1785.0%** |
| **test_unseen SQL Validity** | 95.00% | **98.20%** | **+3.20%** | +3.4% |
| **test_seen Exact Match** | 3.50% | **77.50%** | **+74.00%** | **+2114.3%** |
| **test_seen SQL Validity** | 92.50% | **98.00%** | **+5.50%** | +5.9% |

**Key Finding**: The base pre-trained model understands relational semantics (95% validity) but lacks template grounding, producing verbose reasoning rather than executable SQL. LoRA fine-tuning aligns the model strictly to schema-grounded generation, achieving a minimal generalization gap of only 2.10% between seen and unseen schemas.

---

### 3.2 LoRA Rank & Precision Ablation Study

| Configuration | Rank / Alpha | Trainable Params | VRAM | Train Duration | Unseen EM | Unseen Validity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LoRA r=8** | 8 / 16 | 9.23 M (0.59%) | 3.12 GB | 28.5 min | 72.80% | 97.40% |
| **LoRA r=16** | 16 / 32 | 18.46 M (1.18%) | 3.41 GB | 29.9 min | 75.40% | 98.20% |
| **LoRA r=32** | 32 / 64 | 36.93 M (2.36%) | 4.08 GB | 32.6 min | **76.20%** | **98.40%** |
| **QLoRA r=16 (4-bit)**| 16 / 32 | 18.46 M (1.18%) | **1.84 GB** | 42.7 min | 74.20% | 97.80% |

**Ablation Takeaways**:
- Rank 16 yields the optimal Pareto efficiency: quadrupling the rank from 8 to 32 only yields a +0.8% gain over rank 16 while doubling parameters.
- 4-bit QLoRA reduces peak VRAM by 46% (1.84 GB vs 3.41 GB), making deployment feasible on 4 GB edge GPUs at the cost of a 1.2% accuracy drop and 42% slower dequantization backpropagation.

---

### 3.3 Serving Concurrency & Queue Dynamics

| Concurrency Tier | Sequential Lock Throughput | Dynamic Batching Throughput | Sequential p95 Latency | Dynamic Batching p95 | Timeouts (>30s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **C = 1** | 22.67 tok/s | 22.50 tok/s | 1,621.0 ms | 1,650.0 ms | 0 (Both) |
| **C = 8** | 22.61 tok/s | **102.80 tok/s** | 11,500.8 ms | **2,420.1 ms** | 0 (Both) |
| **C = 32** | 14.86 tok/s | **136.90 tok/s** | 34,106.8 ms | **6,840.5 ms** | **33 (Seq) vs 0 (Dyn)** |

Under sequential locking, execution time scales linearly with concurrent queue depth ($O(C \cdot t_{\text{gen}})$), resulting in 33% request timeouts at $C=32$. Dynamic micro-batching leverages tensor core concurrency, delivering a **9.2x throughput increase** (136.9 tok/s) and dropping p95 latency by 79.9%.

---

## 4. Formal 5-Point Error Taxonomy

Statistical analysis of residual errors ($N=123$) on `test_unseen`:

```text
+-------------------------------------------------------------------------------+
| Category                                | Share (%) | Dominant Driver         |
+-------------------------------------------------------------------------------+
| 1. Boundary / Operator Mismatch        | 50.4%     | Polarity (> vs >=, !=)  |
| 2. Nested Aggregation / Group-By       | 25.2%     | Inverted HAVING filters |
| 3. Cross-Table Attribute Ambiguity     | 12.2%     | Shared column naming    |
| 4. SQLite Dialect Syntax Discrepancy   | 8.9%      | ANSI function leakage   |
| 5. Schema Attribute Hallucination      | 3.3%      | Non-existent columns    |
+-------------------------------------------------------------------------------+
```

**Cross-Tabulation Analysis**:
- Schema Attribute Hallucination is exceptionally rare (3.3%), proving that Qwen2.5-Coder possesses strong in-context grounding.
- 75.6% of errors stem from boundary operators and aggregate logic rather than structural syntax failure.

---

## 5. Limitations & Future Work
1. **Multi-Table Join Transfer**: Training on single-to-moderate table datasets (`sql-create-context`) creates an out-of-distribution distribution shift when evaluating on complex relational joins (Spider benchmark). Integrating Spider training data into the LoRA fine-tuning mix is the immediate next step.
2. **Dialect Specialization**: Queries are evaluated strictly for SQLite. Extending the AST parser to PostgreSQL and BigQuery represents valuable enterprise future work.

---

## 6. Conclusion
This work presents a comprehensive engineering and empirical study of lightweight Text-to-SQL synthesis using Qwen2.5-Coder-1.5B under legacy hardware constraints. By combining schema-disjoint data splitting, Volta-stable LoRA fine-tuning, multi-tier AST validation, dynamic micro-batching, and formal error taxonomy analysis, SQL-Forge establishes that small 1.5B parameter models can achieve high accuracy (75.4% exact match, 98.2% validity) and robust throughput (136.9 tok/s) within tight enterprise VRAM footprints.
