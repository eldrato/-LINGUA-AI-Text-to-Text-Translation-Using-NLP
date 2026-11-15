"""
Fine-tuning pipeline for MarianMT Sequence-to-Sequence models on parallel corpora.
Implements data loading, tokenization, training and validation loops, AdamW optimizer,
checkpoint management, loss tracking, and learning curve visualization.
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add project root to path if running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.dataset_loader import DatasetLoader
from src.preprocessing import preprocess_dataframe, split_dataset, TranslationDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_and_save_loss_curve(history: Dict[str, List[float]], output_path: Path, lang_pair: str):
    """Generates and saves a publication-quality training and validation loss plot."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5), dpi=150)
    
    plt.plot(epochs, history["train_loss"], marker="o", linewidth=2, color="#4f46e5", label="Training Loss")
    plt.plot(epochs, history["val_loss"], marker="s", linewidth=2, color="#06b6d4", label="Validation Loss")
    
    plt.title(f"Fine-Tuning Loss Curve ({lang_pair.upper()})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="medium")
    plt.ylabel("Cross-Entropy Loss", fontsize=11, fontweight="medium")
    plt.xticks(list(epochs))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Loss curve visualization saved to: {output_path}")

def train_model(
    lang_pair: str = "en-hi",
    data_path: Optional[str] = None,
    epochs: int = 3,
    batch_size: Optional[int] = None,
    learning_rate: float = 5e-5,
    max_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Executes actual fine-tuning of MarianMT on the parallel dataset.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    if batch_size is None:
        batch_size = config.TRAIN_CONFIG["batch_size_gpu"] if device.type == "cuda" else config.TRAIN_CONFIG["batch_size_cpu"]

    # Normalize pair key
    pair_key = "en-hi" if "en" in lang_pair and "hi" in lang_pair and lang_pair.startswith("en") else "hi-en"
    src_col, tgt_col = ("en", "hi") if pair_key == "en-hi" else ("hi", "en")
    
    base_model_name = config.MODEL_CHECKPOINTS[pair_key]
    save_dir = Path(output_dir or config.LOCAL_MODEL_PATHS[pair_key])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Dataset preparation
    logger.info(f"Preparing parallel dataset for {pair_key}...")
    loader = DatasetLoader()
    raw_df = loader.get_or_create_dataset(custom_path=data_path)
    clean_df = preprocess_dataframe(raw_df, source_col="en", target_col="hi")

    if max_samples and len(clean_df) > max_samples:
        logger.info(f"Limiting dataset to {max_samples} samples for rapid training / CPU budget.")
        clean_df = clean_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    train_df, val_df, test_df = split_dataset(
        clean_df,
        train_ratio=config.DATASET_CONFIG["train_ratio"],
        val_ratio=config.DATASET_CONFIG["val_ratio"],
        test_ratio=config.DATASET_CONFIG["test_ratio"],
        random_seed=config.DATASET_CONFIG["random_seed"],
        prefix=f"opus_{pair_key.replace('-', '_')}"
    )

    # 3. Load Tokenizer & Model
    start_checkpoint = resume_from_checkpoint or base_model_name
    logger.info(f"Loading base tokenizer from '{start_checkpoint}'...")
    tokenizer = AutoTokenizer.from_pretrained(start_checkpoint)
    
    logger.info(f"Loading MarianMT model from '{start_checkpoint}'...")
    model = AutoModelForSeq2SeqLM.from_pretrained(start_checkpoint)
    model.to(device)

    # 4. Create PyTorch Datasets and DataLoaders
    train_dataset = TranslationDataset(
        df=train_df,
        tokenizer=tokenizer,
        source_col=src_col,
        target_col=tgt_col,
        max_length=max_length
    )
    val_dataset = TranslationDataset(
        df=val_df,
        tokenizer=tokenizer,
        source_col=src_col,
        target_col=tgt_col,
        max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.TRAIN_CONFIG["weight_decay"]
    )
    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(total_training_steps * config.TRAIN_CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    logger.info(
        f"Starting fine-tuning: Epochs={epochs}, BatchSize={batch_size}, "
        f"TrainSamples={len(train_df)}, ValSamples={len(val_df)}, TotalSteps={total_training_steps}"
    )

    # 6. Training & Validation Loop
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    best_val_loss = float("inf")

    overall_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN_CONFIG["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | "
                    f"Current Batch Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epoch_times"].append(epoch_time)

        logger.info(
            f"=== Epoch {epoch}/{epochs} Completed in {epoch_time:.1f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ==="
        )

        # Checkpoint if best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    total_time = time.time() - overall_start

    # 7. Save outputs, logs and visualization
    loss_plot_path = config.TRAINING_LOGS_DIR / f"loss_curve_{pair_key.replace('-', '_')}.png"
    plot_and_save_loss_curve(history, loss_plot_path, pair_key)

    training_summary = {
        "lang_pair": pair_key,
        "base_model": base_model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "total_training_time_sec": round(total_time, 2),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "best_val_loss": best_val_loss,
        "history": history,
        "saved_model_path": str(save_dir),
        "device": str(device)
    }

    metrics_path = config.TRAINING_LOGS_DIR / f"training_history_{pair_key.replace('-', '_')}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    logger.info(f"Training metrics saved to: {metrics_path}")
    logger.info("Fine-tuning completed successfully!")
    return training_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MarianMT on English-Hindi parallel corpus.")
    parser.add_argument("--lang_pair", type=str, default="en-hi", choices=["en-hi", "hi-en"], help="Language pair")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (defaults to auto)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for training (useful for quick local runs)")
    parser.add_argument("--data_path", type=str, default=None, help="Custom dataset CSV path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for saved model")

    args = parser.parse_args()
    train_model(
        lang_pair=args.lang_pair,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )
