"""FastAPI Serving Application for SQL-Forge v2.

Endpoints:
- POST /generate : Schema-grounded SQL query generation
- GET  /health   : Readiness, backend, device and uptime
- GET  /metrics  : Request count, errors, p50/p95 latency, throughput, VRAM
"""

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from serving.model_loader import engine
from serving.dynamic_batcher import dynamic_engine

# Select serving engine mode: "sequential" vs "dynamic"
ENGINE_MODE = os.getenv("ENGINE_MODE", "dynamic").lower()
active_engine = dynamic_engine if ENGINE_MODE == "dynamic" else engine

# Server state metrics
START_TIME = time.time()
REQUEST_COUNT = 0
ERROR_COUNT = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_GENERATION_TIME_SEC = 0.0
LATENCY_HISTORY_MS: List[float] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    print(f"Initializing SQL-Forge inference engine on startup (mode={ENGINE_MODE})...", flush=True)
    active_engine.load()
    if hasattr(active_engine, "start_worker"):
        active_engine.start_worker()
    yield
    print("Shutting down SQL-Forge serving engine...", flush=True)
    if hasattr(active_engine, "stop_worker"):
        await active_engine.stop_worker()


app = FastAPI(
    title="SQL-Forge v2 Serving Service",
    description="Schema-grounded Text-to-SQL API powered by fine-tuned Qwen2.5-Coder-1.5B",
    version="2.0.0",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_text: str = Field(..., alias="schema", min_length=5, max_length=10000, description="Database CREATE TABLE DDL")
    question: str = Field(..., min_length=3, max_length=1000, description="Natural language question to answer in SQL")


class GenerateResponse(BaseModel):
    sql: str
    model: str
    latency_ms: float
    output_tokens: int


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str
    device: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    request_count: int
    error_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    mean_tokens_per_sec: float
    current_gpu_memory_gb: float


def calculate_percentile(data: List[float], percentile: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int((percentile / 100.0) * len(sorted_data))
    idx = min(idx, len(sorted_data) - 1)
    return float(sorted_data[idx])


@app.post("/generate", response_model=GenerateResponse)
async def generate_sql(request: GenerateRequest):
    """Generate SQL from database schema and user question."""
    global REQUEST_COUNT, ERROR_COUNT, TOTAL_OUTPUT_TOKENS, TOTAL_GENERATION_TIME_SEC, LATENCY_HISTORY_MS
    REQUEST_COUNT += 1

    try:
        # Enforce 30s hard request timeout
        sql, latency_ms, tok_count = await asyncio.wait_for(
            active_engine.generate(schema=request.schema_text, question=request.question),
            timeout=30.0
        )

        LATENCY_HISTORY_MS.append(latency_ms)
        if len(LATENCY_HISTORY_MS) > 1000:
            LATENCY_HISTORY_MS.pop(0)

        TOTAL_OUTPUT_TOKENS += tok_count
        TOTAL_GENERATION_TIME_SEC += (latency_ms / 1000.0)

        return GenerateResponse(
            sql=sql,
            model=active_engine.model_name,
            latency_ms=round(latency_ms, 2),
            output_tokens=tok_count
        )

    except asyncio.TimeoutError:
        ERROR_COUNT += 1
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request processing exceeded 30-second timeout limit."
        )
    except Exception as e:
        ERROR_COUNT += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health and backend details."""
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="healthy" if active_engine.initialized else "initializing",
        model=active_engine.model_name,
        backend=getattr(active_engine, "backend", ENGINE_MODE),
        device=active_engine.device,
        uptime_seconds=round(uptime, 2)
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Retrieve operational latency, throughput, and GPU memory metrics."""
    p50 = calculate_percentile(LATENCY_HISTORY_MS, 50)
    p95 = calculate_percentile(LATENCY_HISTORY_MS, 95)

    mean_tok_s = (
        round(TOTAL_OUTPUT_TOKENS / TOTAL_GENERATION_TIME_SEC, 2)
        if TOTAL_GENERATION_TIME_SEC > 0
        else 0.0
    )

    gpu_mem_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
    except Exception:
        gpu_mem_gb = 0.0

    return MetricsResponse(
        request_count=REQUEST_COUNT,
        error_count=ERROR_COUNT,
        latency_p50_ms=round(p50, 2),
        latency_p95_ms=round(p95, 2),
        mean_tokens_per_sec=mean_tok_s,
        current_gpu_memory_gb=gpu_mem_gb
    )
