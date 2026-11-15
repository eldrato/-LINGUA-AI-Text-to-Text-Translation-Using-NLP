"""
Evaluation module for Lingua AI using SacreBLEU.
Generates translations on held-out test data, compares against ground truth references,
computes corpus BLEU and sentence BLEU metrics, and distinguishes between
pretrained and fine-tuned models without hardcoding or fabricating results.
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path if running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.dataset_loader import DatasetLoader
from src.preprocessing import preprocess_dataframe, split_dataset
from src.model_manager import get_model_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(
    lang_pair: str = "en-hi",
    test_path: Optional[str] = None,
    model_source: str = "auto",
    max_eval_samples: Optional[int] = None,
    num_beams: int = 4,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Evaluates translation quality on held-out test dataset using SacreBLEU.
    
    Parameters:
    - lang_pair: 'en-hi' or 'hi-en'.
    - test_path: Path to test dataset CSV.
    - model_source: 'auto' (prefers fine-tuned if available), 'finetuned', or 'pretrained'.
    - max_eval_samples: Number of test samples to evaluate.
    
    Returns:
    - Evaluation metrics dictionary.
    """
    import torch
    import sacrebleu
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    pair_key = "en-hi" if "en" in lang_pair and "hi" in lang_pair and lang_pair.startswith("en") else "hi-en"
    src_col, tgt_col = ("en", "hi") if pair_key == "en-hi" else ("hi", "en")

    # 1. Determine model checkpoint based on model_source
    manager = get_model_manager()
    is_finetuned_avail = manager.is_local_finetuned_available(pair_key)
    
    if model_source == "finetuned":
        if not is_finetuned_avail:
            raise FileNotFoundError(f"No fine-tuned model checkpoint found in {config.LOCAL_MODEL_PATHS[pair_key]}")
        model_path = str(config.LOCAL_MODEL_PATHS[pair_key])
        is_finetuned = True
    elif model_source == "pretrained":
        model_path = config.MODEL_CHECKPOINTS[pair_key]
        is_finetuned = False
    else:  # auto
        if is_finetuned_avail:
            model_path = str(config.LOCAL_MODEL_PATHS[pair_key])
            is_finetuned = True
        else:
            model_path = config.MODEL_CHECKPOINTS[pair_key]
            is_finetuned = False

    logger.info(f"Evaluating {pair_key} using {'FINE-TUNED' if is_finetuned else 'PRETRAINED'} model from: {model_path}")

    # 2. Load test dataset
    test_file = Path(test_path) if test_path else (config.PROCESSED_DATASET_DIR / f"opus_{pair_key.replace('-', '_')}_test.csv")
    
    if not test_file.exists():
        logger.info(f"Test split not found at {test_file}. Creating dataset splits...")
        loader = DatasetLoader()
        raw_df = loader.get_or_create_dataset()
        clean_df = preprocess_dataframe(raw_df, source_col="en", target_col="hi")
        _, _, test_df = split_dataset(clean_df, prefix=f"opus_{pair_key.replace('-', '_')}")
    else:
        logger.info(f"Loading test split from {test_file}...")
        test_df = pd.read_csv(test_file, encoding="utf-8")

    if max_eval_samples and len(test_df) > max_eval_samples:
        test_df = test_df.head(max_eval_samples)

    logger.info(f"Test dataset has {len(test_df)} sentence pairs for evaluation.")

    # 3. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(manager.device)
    model.eval()

    # 4. Generate Predictions
    sources = test_df[src_col].astype(str).tolist()
    references = test_df[tgt_col].astype(str).tolist()
    hypotheses = []
    sample_records = []

    logger.info(f"Generating translations with beam size {num_beams}...")
    start_eval = time.time()
    
    for i, (src_text, ref_text) in enumerate(zip(sources, references)):
        inputs = tokenizer(
            src_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids = inputs["input_ids"].to(manager.device)
        attention_mask = inputs["attention_mask"].to(manager.device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True
            )

        pred_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        hypotheses.append(pred_text)

        # Calculate individual sentence BLEU
        try:
            sent_bleu = sacrebleu.sentence_bleu(pred_text, [ref_text]).score
        except Exception:
            sent_bleu = 0.0

        sample_records.append({
            "index": i + 1,
            "source": src_text,
            "reference": ref_text,
            "prediction": pred_text,
            "sentence_bleu": round(sent_bleu, 2)
        })

    eval_time = time.time() - start_eval

    # 5. Compute Corpus BLEU score via SacreBLEU
    # SacreBLEU standard corpus BLEU
    # For Hindi target, we use tokenize='intl' or 'char' to properly handle Devanagari words
    tok_method = "intl" if tgt_col == "hi" else "13a"
    bleu_result = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize=tok_method
    )
    corpus_bleu = round(bleu_result.score, 2)

    # Calculate average sentence BLEU
    avg_sent_bleu = round(float(np.mean([r["sentence_bleu"] for r in sample_records])), 2)

    # 6. Read training history if available
    train_history_path = config.TRAINING_LOGS_DIR / f"training_history_{pair_key.replace('-', '_')}.json"
    train_loss = None
    val_loss = None
    n_train = None
    n_val = None

    if train_history_path.exists():
        try:
            with open(train_history_path, "r", encoding="utf-8") as f:
                th = json.load(f)
                train_loss = th.get("final_train_loss")
                val_loss = th.get("final_val_loss")
                n_train = th.get("train_samples")
                n_val = th.get("val_samples")
        except Exception as e:
            logger.warning(f"Could not read training history: {e}")

    # 7. Compile Evaluation Results
    results = {
        "lang_pair": pair_key,
        "source_lang": src_col,
        "target_lang": tgt_col,
        "model_identifier": model_path,
        "is_finetuned": is_finetuned,
        "sacrebleu_corpus_score": corpus_bleu,
        "average_sentence_bleu": avg_sent_bleu,
        "sacrebleu_signature": str(bleu_result),
        "tokenizer_method": tok_method,
        "num_test_examples": len(test_df),
        "num_train_examples": n_train,
        "num_val_examples": n_val,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "total_eval_time_sec": round(eval_time, 2),
        "avg_inference_time_per_sentence_sec": round(eval_time / max(1, len(test_df)), 4),
        "device": str(manager.device),
    }

    # Save evaluation results JSON
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    eval_json_path = config.EVALUATION_DIR / f"eval_results_{pair_key.replace('-', '_')}.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save translation sample comparisons
    samples_df = pd.DataFrame(sample_records)
    samples_csv_path = config.EVALUATION_DIR / f"sample_translations_{pair_key.replace('-', '_')}.csv"
    samples_df.to_csv(samples_csv_path, index=False, encoding="utf-8")

    logger.info(f"Evaluation complete for {pair_key}!")
    logger.info(f"SacreBLEU Corpus Score: {corpus_bleu} (Tokenizer: {tok_method})")
    logger.info(f"Results saved to: {eval_json_path}")
    logger.info(f"Sample comparisons saved to: {samples_csv_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MarianMT on translation test set using SacreBLEU.")
    parser.add_argument("--lang_pair", type=str, default="en-hi", choices=["en-hi", "hi-en"], help="Language pair")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test CSV")
    parser.add_argument("--model_source", type=str, default="auto", choices=["auto", "finetuned", "pretrained"])
    parser.add_argument("--max_samples", type=int, default=50, help="Max test samples to evaluate")

    args = parser.parse_args()
    res = evaluate_model(
        lang_pair=args.lang_pair,
        test_path=args.test_path,
        model_source=args.model_source,
        max_eval_samples=args.max_samples
    )
    print("\n--- EVALUATION SUMMARY ---")
    for k, v in res.items():
        print(f"{k}: {v}")
