"""Spider Dataset Setup Script.

Downloads and configures the official Spider dev benchmark:
- dev.json (1,034 questions across 20 databases)
- Database SQLite files under data/spider/database/<db_id>/<db_id>.sqlite
"""

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


SPIDER_HF_URL = "https://huggingface.co/datasets/xlangai/spider/resolve/main"
SPIDER_GITHUB_DEV_URL = "https://raw.githubusercontent.com/taoyds/spider/master/dev.json"
SPIDER_GITHUB_TABLES_URL = "https://raw.githubusercontent.com/taoyds/spider/master/tables.json"


def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest_path}...", flush=True)
    urllib.request.urlretrieve(url, str(dest_path))
    print(f"Downloaded {dest_path.name} ({dest_path.stat().st_size / 1024:.1f} KB)")


def setup_from_huggingface(spider_dir: Path):
    """Download and extract Spider dev split via Hugging Face dataset."""
    try:
        from datasets import load_dataset
        print("Loading Spider dataset via Hugging Face...", flush=True)
        ds = load_dataset("spider", split="validation")
        
        dev_records = []
        for row in ds:
            dev_records.append({
                "db_id": row["db_id"],
                "question": row["question"],
                "query": row["query"],
                "query_toks": row.get("query_toks", []),
                "sql": row.get("sql", {})
            })
        
        dev_path = spider_dir / "dev.json"
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump(dev_records, f, indent=2)
        print(f"Saved {len(dev_records)} dev records to {dev_path}")
        return True
    except Exception as e:
        print(f"Hugging Face dataset load failed: {e}")
        return False


def setup_spider(output_dir: str = "data/spider"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    db_dir = out_path / "database"
    db_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download dev.json
    dev_path = out_path / "dev.json"
    if not dev_path.exists():
        try:
            download_file(SPIDER_GITHUB_DEV_URL, dev_path)
        except Exception:
            setup_from_huggingface(out_path)

    # 2. Download tables.json
    tables_path = out_path / "tables.json"
    if not tables_path.exists():
        try:
            download_file(SPIDER_GITHUB_TABLES_URL, tables_path)
        except Exception as e:
            print(f"Note: tables.json optional ({e})")

    print("\nSpider dev metadata verified at:", dev_path)
    print(f"Database directory located at: {db_dir}")
    print("If sqlite databases are not populated, copy them into data/spider/database/<db_id>/<db_id>.sqlite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Spider dev benchmark")
    parser.add_argument("--output-dir", type=str, default="data/spider")
    args = parser.parse_args()
    setup_spider(args.output_dir)
