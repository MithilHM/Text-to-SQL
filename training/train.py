"""LoRA Fine-tuning pipeline for Qwen2.5-Coder-1.5B on Volta (sm_70).

Features:
- SFTTrainer with prompt-completion dataset and completion-only loss.
- Volta fp16 stability: float16 base model, float32 LoRA trainable adapters.
- 20-step dry run benchmarking mode (loss trend, timing, peak memory, ETA).
- Full 1-epoch training with checkpointing every 100 steps.
- Plots loss curve to evaluation/results/loss_curve.png.
- Exports training_summary.json with measured metrics.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import yaml


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def plot_loss_curve(log_entries: list, output_path: Path):
    """Generate and save publication-quality loss curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = [e["step"] for e in log_entries if "loss" in e]
    train_losses = [e["loss"] for e in log_entries if "loss" in e]
    eval_steps = [e["step"] for e in log_entries if "eval_loss" in e]
    eval_losses = [e["eval_loss"] for e in log_entries if "eval_loss" in e]

    if not steps:
        print("No training loss entries recorded to plot.")
        return

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(steps, train_losses, label="Training Loss", color="#1f77b4", linewidth=2)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="#ff7f0e", marker="o", linewidth=2)

    plt.title("SQL-Forge LoRA Fine-Tuning Loss Curve (Qwen2.5-Coder-1.5B)", fontsize=13, fontweight="bold")
    plt.xlabel("Global Step", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Loss curve saved -> {output_path}")


def train(
    cfg: dict,
    dry_run_steps: int = 0,
    rank_override: int = None,
    qlora: bool = False,
    adapter_dir_override: str = None,
    mock: bool = False
):
    if mock:
        print("\n--- MOCK TRAINING MODE (Offline verification) ---")
        summary = {
            "git_commit": get_git_commit(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": cfg["model"]["model_name_or_path"],
            "lora_rank": rank_override or cfg["lora"]["r"],
            "dry_run": bool(dry_run_steps > 0),
            "total_steps": dry_run_steps if dry_run_steps > 0 else 500,
            "total_time_seconds": 12.5,
            "peak_gpu_memory_gb": 14.2,
            "initial_loss": 2.45,
            "final_loss": 0.38,
            "final_validation_loss": 0.42
        }
        out_adapter = Path(cfg["training"]["adapter_dir"])
        out_adapter.mkdir(parents=True, exist_ok=True)
        summary_path = Path(cfg["training"]["summary_file"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Create mock plot
        mock_logs = [
            {"step": i, "loss": round(2.5 * (0.95 ** i), 4)} for i in range(1, 21)
        ]
        plot_loss_curve(mock_logs, Path(cfg["training"]["plot_file"]))
        print(f"Mock training summary saved -> {summary_path}")
        return summary

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    class MetricsTrackingCallback(TrainerCallback):
        """Callback to record step losses, step timestamps and peak memory."""
        def __init__(self, log_path: Path):
            self.log_path = log_path
            self.step_history = []
            self.eval_history = []
            self.last_time = time.perf_counter()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            now = time.perf_counter()
            elapsed = now - self.last_time
            self.last_time = now

            mem_gb = 0.0
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "mem_gb": round(mem_gb, 2),
                **logs
            }
            self.step_history.append(entry)

            if "eval_loss" in logs:
                self.eval_history.append({"step": state.global_step, "eval_loss": logs["eval_loss"]})

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    # Verify device and Volta compatibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        arch_list = torch.cuda.get_arch_list()
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch Arch List: {arch_list}")
        if "sm_70" not in arch_list and "7.0" not in str(arch_list):
            print("[WARNING] sm_70 not found in PyTorch CUDA arch list! Volta kernel issues may occur.")

    # 1. Load Tokenizer
    model_name = cfg["model"]["model_name_or_path"]
    print(f"\nLoading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in fp16 (or 4-bit QLoRA)
    print(f"Loading base model ({'4-bit QLoRA' if qlora else 'fp16'}) with sdpa attention: {model_name}...")
    torch_dtype = torch.float16 if cfg["model"]["torch_dtype"] == "float16" else torch.float32
    attn_impl = cfg["model"]["attn_implementation"]

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_impl,
        "device_map": "auto" if device == "cuda" else None,
        "trust_remote_code": True,
    }
    if qlora:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if qlora:
        model = prepare_model_for_kbit_training(model)
        # Ensure no bfloat16 tensors remain on Volta sm_70
        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)
    model.config.use_cache = False

    # 3. LoRA Configuration
    lora_cfg_dict = dict(cfg["lora"])
    if rank_override:
        lora_cfg_dict["r"] = rank_override
        lora_cfg_dict["lora_alpha"] = rank_override * 2
        print(f"Overriding LoRA rank={rank_override}, alpha={lora_cfg_dict['lora_alpha']}")

    peft_config = LoraConfig(
        r=lora_cfg_dict["r"],
        lora_alpha=lora_cfg_dict["lora_alpha"],
        lora_dropout=lora_cfg_dict["lora_dropout"],
        bias=lora_cfg_dict["bias"],
        task_type=lora_cfg_dict["task_type"],
        target_modules=lora_cfg_dict["target_modules"]
    )

    model = get_peft_model(model, peft_config)

    # Cast trainable adapter weights to float32 for Volta numerical stability in fp16
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()

    # 4. Load Datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset("json", data_files=cfg["data"]["train_file"], split="train")
    val_dataset = load_dataset("json", data_files=cfg["data"]["validation_file"], split="train")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 5. Training Arguments
    t_cfg = cfg["training"]
    total_steps = dry_run_steps if dry_run_steps > 0 else None
    epochs = 1 if dry_run_steps == 0 else 1

    import inspect
    sft_sig = inspect.signature(SFTConfig.__init__).parameters

    sft_kwargs = {
        "output_dir": t_cfg["output_dir"],
        "num_train_epochs": epochs,
        "max_steps": total_steps if total_steps else -1,
        "per_device_train_batch_size": t_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": t_cfg["gradient_accumulation_steps"],
        "learning_rate": float(t_cfg["learning_rate"]),
        "optim": t_cfg["optim"],
        "lr_scheduler_type": t_cfg["lr_scheduler_type"],
        "max_grad_norm": t_cfg["max_grad_norm"],
        "fp16": t_cfg["fp16"] and (device == "cuda"),
        "bf16": False,
        "logging_steps": t_cfg["logging_steps"] if not dry_run_steps else 1,
        "save_steps": t_cfg["save_steps"] if not dry_run_steps else 9999,
        "save_total_limit": t_cfg["save_total_limit"],
        "eval_steps": 100 if not dry_run_steps else None,
        "seed": t_cfg["seed"],
        "report_to": "none"
    }

    if "warmup_ratio" in sft_sig:
        sft_kwargs["warmup_ratio"] = t_cfg["warmup_ratio"]

    # Version-adaptive parameter mapping (TRL 0.x vs 1.x)
    if "max_length" in sft_sig:
        sft_kwargs["max_length"] = cfg["data"]["max_seq_length"]
    elif "max_seq_length" in sft_sig:
        sft_kwargs["max_seq_length"] = cfg["data"]["max_seq_length"]

    if "eval_strategy" in sft_sig:
        sft_kwargs["eval_strategy"] = "steps" if not dry_run_steps else "no"
    elif "evaluation_strategy" in sft_sig:
        sft_kwargs["evaluation_strategy"] = "steps" if not dry_run_steps else "no"

    if "dataset_text_field" in sft_sig:
        sft_kwargs["dataset_text_field"] = None
    if "packing" in sft_sig:
        sft_kwargs["packing"] = False

    training_args = SFTConfig(**sft_kwargs)

    log_file = Path(t_cfg["output_dir"]) / "train_logs.jsonl"
    metrics_cb = MetricsTrackingCallback(log_file)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not dry_run_steps else None,
        callbacks=[metrics_cb]
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    if dry_run_steps > 0:
        print(f"\n=======================================================")
        print(f"STARTING 20-STEP DRY RUN BENCHMARK...")
        print(f"=======================================================")
        train_res = trainer.train()
        t_elapsed = time.perf_counter() - t_start

        peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        sec_per_step = t_elapsed / dry_run_steps
        effective_batch = t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"]
        total_epoch_steps = len(train_dataset) // effective_batch
        eta_seconds = sec_per_step * total_epoch_steps

        print("\n" + "=" * 55)
        print("20-STEP DRY RUN RESULTS:")
        print(f"  Steps completed:     {dry_run_steps}")
        print(f"  Elapsed time:        {t_elapsed:.2f}s ({sec_per_step:.2f} s/step)")
        print(f"  Peak GPU Memory:     {peak_mem:.2f} GB")
        print(f"  Total epoch steps:   {total_epoch_steps}")
        print(f"  Estimated Full Run:  {eta_seconds/60:.1f} minutes ({eta_seconds/3600:.2f} hours)")
        print("=" * 55)
        print("Please review dry-run numbers and approve the full run.\n")
        return {
            "dry_run": True,
            "sec_per_step": sec_per_step,
            "peak_gpu_memory_gb": peak_mem,
            "eta_minutes": eta_seconds / 60
        }

    print(f"\nStarting full training (1 epoch)...")
    train_res = trainer.train()
    t_elapsed = time.perf_counter() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    # Save adapter
    adapter_dir = Path(adapter_dir_override or t_cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving final LoRA adapter to {adapter_dir}...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save config copy next to adapter
    with open(adapter_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Save loss curve
    plot_loss_curve(metrics_cb.step_history, Path(t_cfg["plot_file"]))

    summary = {
        "git_commit": get_git_commit(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": cfg["model"]["model_name_or_path"],
        "lora_rank": lora_cfg_dict["r"],
        "lora_alpha": lora_cfg_dict["lora_alpha"],
        "qlora_4bit": qlora,
        "total_steps": trainer.state.global_step,
        "total_time_seconds": round(t_elapsed, 2),
        "peak_gpu_memory_gb": round(peak_mem, 2),
        "train_loss": round(train_res.training_loss, 4),
        "history": metrics_cb.step_history
    }

    summary_path = adapter_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved -> {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for SQL-Forge")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML training config")
    parser.add_argument("--dry-run-steps", type=int, default=0, help="Run N steps benchmark before full training")
    parser.add_argument("--rank", type=int, default=None, help="Override LoRA rank (alpha set to 2*rank)")
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit NF4 QLoRA quantization")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Override output adapter directory")
    parser.add_argument("--mock", action="store_true", help="Run mock training offline to test pipeline")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(
        cfg=cfg,
        dry_run_steps=args.dry_run_steps,
        rank_override=args.rank,
        qlora=args.qlora,
        adapter_dir_override=args.adapter_dir,
        mock=args.mock
    )

if __name__ == "__main__":
    main()
