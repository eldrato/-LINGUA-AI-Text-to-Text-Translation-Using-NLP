"""
Fine-tuning pipeline for Sequence-to-Sequence Translation Models (MarianMT / NLLB).
Implements Hugging Face Seq2SeqTrainer with:
- Target text tokenization using text_target= and label padding with -100
- AdamW optimizer with linear warmup, weight decay, and label smoothing (0.1)
- predict_with_generate=True with beam search
- Validation evaluation using SacreBLEU and chrF2++
- Best checkpoint selection by validation BLEU and EarlyStoppingCallback
- Configurable learning rate sweeps, loss tracking, and visualization
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.preprocessing import postprocess_translation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_and_save_loss_curve(history: dict[str, list[float]], output_path: Path, lang_pair: str):
    """Generates and saves a publication-quality training and validation loss plot."""
    try:
        import matplotlib.pyplot as plt

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        max_len = max(len(train_loss), len(val_loss), 1)
        epochs = list(range(1, max_len + 1))

        plt.figure(figsize=(8, 5), dpi=150)
        has_labels = False
        if train_loss:
            x_train = list(range(1, len(train_loss) + 1))
            plt.plot(x_train, train_loss, marker="o", linewidth=2, color="#4f46e5", label="Training Loss")
            has_labels = True
        if val_loss:
            x_val = list(range(1, len(val_loss) + 1))
            plt.plot(x_val, val_loss, marker="s", linewidth=2, color="#06b6d4", label="Validation Loss")
            has_labels = True

        plt.title(f"Fine-Tuning Loss Curve ({lang_pair.upper()})", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=11, fontweight="medium")
        plt.ylabel("Cross-Entropy Loss (Label Smoothed)", fontsize=11, fontweight="medium")
        plt.xticks(epochs)
        plt.grid(True, linestyle="--", alpha=0.5)
        if has_labels:
            plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info(f"Loss curve visualization saved to: {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate loss curve: {e}")

def load_clean_data(pair_key: str, data_path: str | None = None, max_samples: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads clean train, val, and test splits for the language pair."""
    src_col, tgt_col = ("en", "hi") if pair_key == "en-hi" else ("hi", "en")
    
    if data_path and Path(data_path).exists():
        logger.info(f"Loading custom dataset from {data_path}...")
        df = pd.read_csv(data_path, encoding="utf-8")
        from src.preprocessing import preprocess_dataframe, split_dataset
        clean_df = preprocess_dataframe(df, source_col=src_col, target_col=tgt_col)
        train_df, val_df, test_df = split_dataset(clean_df, prefix=f"custom_{pair_key.replace('-', '_')}")
    else:
        prefix = f"opus_{pair_key.replace('-', '_')}"
        train_file = config.PROCESSED_DATASET_DIR / f"{prefix}_train.csv"
        val_file = config.PROCESSED_DATASET_DIR / f"{prefix}_val.csv"
        test_file = config.PROCESSED_DATASET_DIR / f"{prefix}_test.csv"

        if not (train_file.exists() and val_file.exists()):
            # Fallback to standard en_hi processed files
            train_file = config.PROCESSED_DATASET_DIR / "opus_en_hi_train.csv"
            val_file = config.PROCESSED_DATASET_DIR / "opus_en_hi_val.csv"
            test_file = config.PROCESSED_DATASET_DIR / "opus_en_hi_test.csv"

        train_df = pd.read_csv(train_file, encoding="utf-8")
        val_df = pd.read_csv(val_file, encoding="utf-8")
        test_df = pd.read_csv(test_file, encoding="utf-8")

    # Invert columns if pair is hi-en and dataset has en first
    for d in [train_df, val_df, test_df]:
        if src_col not in d.columns and "en" in d.columns:
            d[src_col] = d["hi"]
            d[tgt_col] = d["en"]

    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(10, int(max_samples * 0.15)))

    logger.info(f"Dataset loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def train_model(
    lang_pair: str = "en-hi",
    base_model: str | None = None,
    data_path: str | None = None,
    epochs: int = 3,
    batch_size: int | None = None,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    label_smoothing: float = 0.1,
    num_beams: int = 4,
    max_length: int = 128,
    max_samples: int | None = None,
    output_dir: str | None = None,
    gradient_accumulation_steps: int | None = None,
    early_stopping_patience: int = 2
) -> dict[str, Any]:
    """
    Fine-tunes a Sequence-to-Sequence translation model using Hugging Face Seq2SeqTrainer.
    """
    import sacrebleu
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    pair_key = "en-hi" if "en" in lang_pair and "hi" in lang_pair and lang_pair.startswith("en") else "hi-en"
    src_col, tgt_col = ("en", "hi") if pair_key == "en-hi" else ("hi", "en")
    
    model_identifier = base_model or config.MODEL_CHECKPOINTS[pair_key]
    final_output_dir = Path(output_dir or config.LOCAL_MODEL_PATHS[pair_key])
    final_output_dir.mkdir(parents=True, exist_ok=True)

    is_cuda = torch.cuda.is_available()
    device_name = "cuda" if is_cuda else "cpu"
    logger.info(f"Training on device: {device_name.upper()} | Base Model: {model_identifier}")

    if batch_size is None:
        batch_size = config.TRAIN_CONFIG["batch_size_gpu"] if is_cuda else 8

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 1 if is_cuda else 2

    # 1. Load Data
    train_df, val_df, _test_df = load_clean_data(pair_key, data_path=data_path, max_samples=max_samples)

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier)

    # 3. Tokenize Datasets using text_target=
    def preprocess_function(examples):
        inputs = [str(x) for x in examples[src_col]]
        targets = [str(x) for x in examples[tgt_col]]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_pandas(train_df[[src_col, tgt_col]]).map(
        preprocess_function, batched=True, remove_columns=[src_col, tgt_col]
    )
    val_ds = Dataset.from_pandas(val_df[[src_col, tgt_col]]).map(
        preprocess_function, batched=True, remove_columns=[src_col, tgt_col]
    )

    # 4. Data Collator (Pads labels with -100 so loss ignores them)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if is_cuda else None
    )

    # 5. Compute Metrics for Seq2SeqTrainer
    tok_method = "intl" if tgt_col == "hi" else "13a"
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100 in labels so tokenizer can decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Apply postprocessing
        clean_preds = [postprocess_translation(p.strip(), target_lang=tgt_col) for p in decoded_preds]
        clean_labels = [postprocess_translation(l.strip(), target_lang=tgt_col) for l in decoded_labels]

        try:
            bleu = sacrebleu.corpus_bleu(clean_preds, [[l] for l in clean_labels], tokenize=tok_method)
            bleu_score = round(float(bleu.score), 2)
        except Exception:
            bleu_score = 0.0

        try:
            chrf = sacrebleu.corpus_chrf(clean_preds, [[l] for l in clean_labels])
            chrf_score = round(float(chrf.score), 2)
        except Exception:
            chrf_score = 0.0

        return {"bleu": bleu_score, "chrf": chrf_score}

    # 6. Training Arguments
    temp_checkpoint_dir = final_output_dir / "checkpoints"
    steps_per_epoch = max(1, len(train_ds) // (batch_size * max(1, gradient_accumulation_steps)))
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_training_steps * warmup_ratio))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(temp_checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=2,
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=max_length,
        generation_num_beams=num_beams,
        label_smoothing_factor=label_smoothing,
        warmup_steps=warmup_steps,
        fp16=is_cuda,
        gradient_accumulation_steps=gradient_accumulation_steps,
        metric_for_best_model="bleu",
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        seed=42
    )

    # 7. Trainer Initialization
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    }
    import inspect
    sig_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in sig_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    logger.info(
        f"Starting Seq2Seq fine-tuning: Epochs={epochs}, LR={learning_rate}, "
        f"BatchSize={batch_size}, GradAccum={gradient_accumulation_steps}"
    )

    train_start = time.time()
    train_result = trainer.train()
    total_train_time = round(time.time() - train_start, 2)

    # 8. Save Best Model & Tokenizer to final destination
    logger.info(f"Saving best fine-tuned model and tokenizer to: {final_output_dir}")
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    # 9. Extract and Plot Loss Curve
    log_history = trainer.state.log_history
    train_losses = []
    val_losses = []
    val_bleus = []
    val_chrfs = []

    for entry in log_history:
        if "loss" in entry:
            train_losses.append(round(float(entry["loss"]), 4))
        if "eval_loss" in entry:
            val_losses.append(round(float(entry["eval_loss"]), 4))
        if "eval_bleu" in entry:
            val_bleus.append(round(float(entry["eval_bleu"]), 2))
        if "eval_chrf" in entry:
            val_chrfs.append(round(float(entry["eval_chrf"]), 2))

    if not train_losses and hasattr(train_result, "training_loss"):
        train_losses.append(round(float(train_result.training_loss), 4))

    history_summary = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "eval_bleu": val_bleus,
        "eval_chrf": val_chrfs,
    }

    loss_plot_path = config.TRAINING_LOGS_DIR / f"loss_curve_{pair_key.replace('-', '_')}.png"
    plot_and_save_loss_curve(history_summary, loss_plot_path, pair_key)

    best_bleu = max(val_bleus) if val_bleus else None
    best_chrf = max(val_chrfs) if val_chrfs else None

    # 10. Save Model Metadata
    metadata = {
        "lang_pair": pair_key,
        "base_model": model_identifier,
        "is_finetuned": True,
        "best_val_bleu": best_bleu,
        "best_val_chrf": best_chrf,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_train_samples": len(train_df),
        "total_train_time_sec": total_train_time,
        "train_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device_name
    }
    with open(final_output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save training log JSON
    summary_path = config.TRAINING_LOGS_DIR / f"training_history_{pair_key.replace('-', '_')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({**metadata, "history": history_summary}, f, indent=2)

    logger.info(f"Fine-tuning complete for {pair_key}! Best Val BLEU: {best_bleu} | Time: {total_train_time}s")
    return metadata

def run_sweep(lang_pair: str = "en-hi", max_samples: int | None = 200):
    """
    Runs a hyperparameter sweep over learning rate × epochs
    and saves the best-performing model.
    """
    logger.info(f"Starting hyperparameter sweep for {lang_pair}...")
    sweep_configs = [
        {"lr": 3e-5, "epochs": 2},
        {"lr": 5e-5, "epochs": 3},
    ]

    best_score = -1.0
    best_cfg = None

    for cfg in sweep_configs:
        logger.info(f"\n--- Testing config: LR={cfg['lr']}, Epochs={cfg['epochs']} ---")
        meta = train_model(
            lang_pair=lang_pair,
            epochs=cfg["epochs"],
            learning_rate=cfg["lr"],
            max_samples=max_samples
        )
        score = meta.get("best_val_bleu") or 0.0
        if score > best_score:
            best_score = score
            best_cfg = cfg

    logger.info(f"\n=== SWEEP COMPLETE for {lang_pair} ===")
    logger.info(f"Best Configuration: {best_cfg} with Val BLEU: {best_score}")
    return best_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Translation Model with Hugging Face Seq2SeqTrainer.")
    parser.add_argument("--lang_pair", type=str, default="en-hi", choices=["en-hi", "hi-en"])
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")

    args = parser.parse_args()
    if args.sweep:
        run_sweep(lang_pair=args.lang_pair, max_samples=args.max_samples or 300)
    else:
        train_model(
            lang_pair=args.lang_pair,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
