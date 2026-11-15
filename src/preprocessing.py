"""
Preprocessing and tokenization pipeline for Lingua AI.
Performs text cleaning, Unicode normalization (preserving Hindi Devanagari script),
whitespace formatting, length filtering, dataset splitting, and PyTorch Dataset creation.
"""
import re
import sys
import unicodedata
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np

# Add project root to path if running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logger = logging.getLogger(__name__)

# Regular expressions for cleaning
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

def clean_text(text: str, is_hindi: bool = False) -> str:
    """
    Cleans and normalizes text for NLP translation.
    
    Parameters:
    - text: Input sentence or paragraph.
    - is_hindi: If True, preserves all Devanagari characters, matras, nuktas,
      and punctuation (danda '।', double danda '॥').
      
    Returns:
    - Cleaned and normalized string.
    """
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)

    # 1. Unicode normalization (NFC ensures combined glyphs are properly maintained)
    # Using NFC maintains Hindi character + matra combinations without decomposing them.
    text = unicodedata.normalize("NFC", text)

    # 2. Strip non-printable / control characters
    text = CONTROL_CHARS_RE.sub("", text)

    # 3. Handle zero-width characters (ZWJ, ZWNJ often used in Indic scripts - keep only if needed, clean others)
    # \u200b is zero-width space, remove it
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")

    # 4. Collapse consecutive whitespace, tabs, and newlines into single spaces
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text

def preprocess_dataframe(
    df: pd.DataFrame,
    source_col: str = "en",
    target_col: str = "hi",
    min_chars: int = 1,
    max_chars: int = 500
) -> pd.DataFrame:
    """
    Cleans a parallel DataFrame by:
    - Dropping nulls/NaNs
    - Cleaning source and target texts
    - Removing empty sentences
    - Removing duplicate sentence pairs
    - Filtering sentences outside [min_chars, max_chars]
    
    Returns:
    - Cleaned DataFrame.
    """
    initial_count = len(df)
    
    # 1. Drop missing rows
    df_clean = df.dropna(subset=[source_col, target_col]).copy()
    null_dropped = initial_count - len(df_clean)

    # 2. Apply text cleaning
    df_clean[source_col] = df_clean[source_col].apply(lambda x: clean_text(x, is_hindi=False))
    df_clean[target_col] = df_clean[target_col].apply(lambda x: clean_text(x, is_hindi=True))

    # 3. Drop empty sentences
    df_clean = df_clean[(df_clean[source_col].str.len() > 0) & (df_clean[target_col].str.len() > 0)]

    # 4. Drop duplicate pairs
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[source_col, target_col])
    duplicates_dropped = before_dedup - len(df_clean)

    # 5. Length filtering
    df_clean = df_clean[
        (df_clean[source_col].str.len() >= min_chars) &
        (df_clean[source_col].str.len() <= max_chars) &
        (df_clean[target_col].str.len() >= min_chars) &
        (df_clean[target_col].str.len() <= max_chars)
    ]
    
    df_clean = df_clean.reset_index(drop=True)
    logger.info(
        f"Preprocessing complete. Initial: {initial_count}, "
        f"Nulls dropped: {null_dropped}, Duplicates dropped: {duplicates_dropped}, "
        f"Final clean pairs: {len(df_clean)}"
    )
    return df_clean

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
    save_splits: bool = True,
    processed_dir: Optional[Path] = None,
    prefix: str = "opus_en_hi"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the cleaned dataset into Train (80%), Validation (10%), and Test (10%) splits
    using a reproducible random seed.
    
    Optionally saves the splits as CSV files to the processed dataset directory.
    """
    from sklearn.model_selection import train_test_split

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # Second split: val vs test (proportional)
    val_prop_in_val_test = val_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=(1.0 - val_prop_in_val_test),
        random_state=random_seed,
        shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        f"Dataset split: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )

    if save_splits:
        target_dir = Path(processed_dir or config.PROCESSED_DATASET_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        train_path = target_dir / f"{prefix}_train.csv"
        val_path = target_dir / f"{prefix}_val.csv"
        test_path = target_dir / f"{prefix}_test.csv"

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")
        logger.info(f"Saved processed splits to:\n - {train_path}\n - {val_path}\n - {test_path}")

    return train_df, val_df, test_df

# PyTorch Dataset implementation for Sequence-to-Sequence MarianMT tokenization
class TranslationDataset:
    """
    PyTorch Dataset wrapper for sequence-to-sequence translation.
    Handles tokenization of source sentences and target labels,
    replacing target padding tokens with -100 for CrossEntropyLoss ignore_index.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        source_col: str = "en",
        target_col: str = "hi",
        max_length: int = 128
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.source_col = source_col
        self.target_col = target_col
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch

        row = self.df.iloc[idx]
        src_text = str(row[self.source_col])
        tgt_text = str(row[self.target_col])

        # Tokenize source
        model_inputs = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target
        # For MarianTokenizer, target tokenization is done via text_target
        labels = self.tokenizer(
            text_target=tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        # Replace padding token id with -100 so it is ignored by CrossEntropyLoss
        labels = labels.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

if __name__ == "__main__":
    from dataset_loader import DatasetLoader
    loader = DatasetLoader()
    raw_df = loader.get_or_create_dataset()
    clean_df = preprocess_dataframe(raw_df)
    train_df, val_df, test_df = split_dataset(clean_df)
    print("Preprocessing verification successful!")
