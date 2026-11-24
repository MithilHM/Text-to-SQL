"""Concurrency and Latency Benchmarking Suite for SQL-Forge v2.

Tests serving layer performance across concurrency levels 1, 8, 32:
- Measures: Requests/sec, Tokens/sec, p50 and p95 latency (ms), error count.
- Monitors peak GPU memory during load test.
- Outputs:
  - benchmarks/results.csv
  - benchmarks/throughput.png
"""

import argparse
import asyncio
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


class GPUMemoryMonitor:
    """Background monitor that samples nvidia-smi memory usage every 0.5s."""
    def __init__(self, interval_sec: float = 0.5):
        self.interval = interval_sec
        self.stop_event = threading.Event()
        self.peak_memory_mb = 0.0
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self.thread = None

    def _sample_loop(self):
        while not self.stop_event.is_set():
            if self.has_nvidia_smi:
                try:
                    res = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    val = float(res.stdout.strip().split("\n")[0])
                    if val > self.peak_memory_mb:
                        self.peak_memory_mb = val
                except Exception:
                    pass
            time.sleep(self.interval)

    def start(self):
        self.stop_event.clear()
        self.peak_memory_mb = 0.0
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        return self.peak_memory_mb


def load_prompts(prompts_file: Path, limit: int = 100) -> List[Dict]:
    prompts = []
    if prompts_file.exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompts.append({
                        "schema": item.get("schema") or item.get("context", ""),
                        "question": item.get("question", "")
                    })
                if len(prompts) >= limit:
                    break
    if not prompts:
        # Fallback prompts if dataset file not yet populated
        for i in range(limit):
            prompts.append({
                "schema": f"CREATE TABLE entity_{i} (id INTEGER PRIMARY KEY, name TEXT, score REAL);",
                "question": f"Find all records in entity_{i} with score > 10"
            })
    return prompts[:limit]


async def send_single_request(
    client,
    server_url: str,
    payload: dict,
    semaphore: asyncio.Semaphore
) -> Tuple[bool, float, int]:
    """Send request to /generate under concurrency semaphore limit."""
    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.post(f"{server_url}/generate", json=payload, timeout=35.0)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                data = resp.json()
                tokens = data.get("output_tokens", 10)
                return True, lat_ms, tokens
            else:
                return False, lat_ms, 0
        except Exception:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return False, lat_ms, 0


async def benchmark_concurrency(
    server_url: str,
    prompts: List[Dict],
    concurrency: int,
    gpu_monitor: GPUMemoryMonitor,
    mock: bool = False
) -> Dict:
    """Run benchmark batch for a given concurrency level."""
    print(f"\n--- Benchmarking Concurrency = {concurrency} ({len(prompts)} requests) ---", flush=True)

    if mock:
        # Generate realistic simulated figures for offline verification
        base_lat = 40.0 + (concurrency * 18.0)
        req_s = 1000.0 / (base_lat / concurrency)
        tok_s = req_s * 12.5
        return {
            "concurrency": concurrency,
            "total_requests": len(prompts),
            "requests_per_sec": round(req_s, 2),
            "tokens_per_sec": round(tok_s, 2),
            "p50_latency_ms": round(base_lat, 2),
            "p95_latency_ms": round(base_lat * 1.35, 2),
            "errors": 0,
            "peak_gpu_memory_mb": round(14500.0 + (concurrency * 25.0), 1)
        }

    import httpx
    gpu_monitor.start()
    semaphore = asyncio.Semaphore(concurrency)

    limits = httpx.Limits(max_keepalive_connections=concurrency + 5, max_connections=concurrency + 10)
    async with httpx.AsyncClient(limits=limits, timeout=40.0) as client:
        t_start = time.perf_counter()
        tasks = [
            send_single_request(client, server_url, p, semaphore)
            for p in prompts
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t_start

    peak_gpu_mem = gpu_monitor.stop()

    successes = [r for r in results if r[0]]
    errors = len(results) - len(successes)
    latencies = [r[1] for r in successes]
    tokens = sum(r[2] for r in successes)

    latencies.sort()
    p50 = latencies[int(0.50 * len(latencies))] if latencies else 0.0
    p95 = latencies[int(0.95 * len(latencies))] if latencies else 0.0
    req_s = len(successes) / total_time if total_time > 0 else 0.0
    tok_s = tokens / total_time if total_time > 0 else 0.0

    metrics = {
        "concurrency": concurrency,
        "total_requests": len(prompts),
        "requests_per_sec": round(req_s, 2),
        "tokens_per_sec": round(tok_s, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "errors": errors,
        "peak_gpu_memory_mb": round(peak_gpu_mem, 1)
    }

    print(f"Results (Concurrency {concurrency}):")
    print(f"  Requests/sec:  {metrics['requests_per_sec']}")
    print(f"  Tokens/sec:    {metrics['tokens_per_sec']}")
    print(f"  p50 Latency:   {metrics['p50_latency_ms']} ms")
    print(f"  p95 Latency:   {metrics['p95_latency_ms']} ms")
    print(f"  Errors:        {metrics['errors']}")
    print(f"  Peak VRAM:     {metrics['peak_gpu_memory_mb']} MB")
    return metrics


def plot_benchmark_charts(results: List[Dict], output_image: Path):
    """Generate throughput and latency scaling curves."""
    output_image.parent.mkdir(parents=True, exist_ok=True)
    conc = [r["concurrency"] for r in results]
    tok_s = [r["tokens_per_sec"] for r in results]
    p50 = [r["p50_latency_ms"] for r in results]
    p95 = [r["p95_latency_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Plot 1: Throughput
    ax1.plot(conc, tok_s, marker="o", color="#2ca02c", linewidth=2.5, markersize=8)
    ax1.set_title("Inference Throughput vs Concurrency", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Concurrent Clients", fontsize=11)
    ax1.set_ylabel("Throughput (Tokens / sec)", fontsize=11)
    ax1.set_xticks(conc)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Latency
    ax2.plot(conc, p50, marker="s", label="p50 Latency (ms)", color="#1f77b4", linewidth=2)
    ax2.plot(conc, p95, marker="^", label="p95 Latency (ms)", color="#d62728", linewidth=2)
    ax2.set_title("Request Latency vs Concurrency", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Concurrent Clients", fontsize=11)
    ax2.set_ylabel("Latency (ms)", fontsize=11)
    ax2.set_xticks(conc)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()
    print(f"Benchmark chart saved -> {output_image}")


def save_results_csv(results: List[Dict], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "concurrency", "total_requests", "requests_per_sec",
        "tokens_per_sec", "p50_latency_ms", "p95_latency_ms",
        "errors", "peak_gpu_memory_mb"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Results CSV saved -> {output_csv}")


def print_summary_table(results: List[Dict]):
    print("\n" + "=" * 88)
    print("SQL-FORGE v2 INFERENCE BENCHMARK SUMMARY")
    print("=" * 88)
    print(f"{'Concurrency':<12} | {'Req / sec':<10} | {'Tokens / sec':<13} | {'p50 Latency':<13} | {'p95 Latency':<13} | {'Errors':<8}")
    print("-" * 88)
    for r in results:
        print(f"{r['concurrency']:<12} | {r['requests_per_sec']:<10.2f} | {r['tokens_per_sec']:<13.2f} | {r['p50_latency_ms']:>8.1f} ms   | {r['p95_latency_ms']:>8.1f} ms   | {r['errors']:<8}")
    print("=" * 88 + "\n")


async def async_main():
    parser = argparse.ArgumentParser(description="Benchmark SQL-Forge FastAPI serving performance")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument("--prompts-file", type=str, default="data/splits/test_unseen.jsonl")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--concurrency-levels", type=str, default="1,8,32")
    parser.add_argument("--output-csv", type=str, default="benchmarks/results.csv")
    parser.add_argument("--output-plot", type=str, default="benchmarks/throughput.png")
    parser.add_argument("--mock", action="store_true", help="Run simulated benchmark without live server")
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts_file), limit=args.num_prompts)
    levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
    gpu_monitor = GPUMemoryMonitor()

    all_results = []
    for c in levels:
        res = await benchmark_concurrency(args.server_url, prompts, c, gpu_monitor, mock=args.mock)
        all_results.append(res)

    save_results_csv(all_results, Path(args.output_csv))
    plot_benchmark_charts(all_results, Path(args.output_plot))
    print_summary_table(all_results)


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
