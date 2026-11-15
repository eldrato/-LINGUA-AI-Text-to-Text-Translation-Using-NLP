"""
Preprocessing and tokenization pipeline for Lingua AI.
Performs text cleaning, Unicode normalization (preserving Hindi Devanagari script),
whitespace formatting, length filtering, dataset splitting, and PyTorch Dataset creation.
"""
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path if running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logger = logging.getLogger(__name__)

# Regular expressions for cleaning and validation
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
LATIN_RE = re.compile(r"[a-zA-Z]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
SOFTWARE_NOISE_RE = re.compile(
    r"(%[0-9]*[sd]|<[^>]+>|http[s]?://|^\s*_[A-Za-z]|\bCtrl\b|\bAlt\b|\bF[0-9]{1,2}\b|\.desktop|\.png|\.xml|\.exe)"
)

def clean_text(text: str, is_hindi: bool = False) -> str:
    """
    Cleans and normalizes text for NLP translation using NFKC normalization.
    
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

    # 1. Unicode normalization (NFKC normalizes compatibility characters and composite glyphs)
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip non-printable / control characters
    text = CONTROL_CHARS_RE.sub("", text)

    # 3. Handle zero-width characters (ZWJ, ZWNJ often used in Indic scripts - keep only if needed, clean others)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")

    # 4. Collapse consecutive whitespace, tabs, and newlines into single spaces
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text

def preprocess_dataframe(
    df: pd.DataFrame,
    source_col: str = "en",
    target_col: str = "hi",
    min_chars: int = 1,
    max_chars: int = 500,
    min_words: int = 1,
    min_ratio: float = 0.20,
    max_ratio: float = 4.5,
) -> pd.DataFrame:
    """
    Thoroughly cleans and validates parallel translation corpora:
    1. NFKC normalization and null removal.
    2. Deduplication of identical parallel pairs.
    3. Removal of untranslated copies (source == target).
    4. Script validation (Latin letters in English, Devanagari in Hindi).
    5. Removal of software boilerplate, code formatting (%s, <tags>, URLs).
    6. Length constraints and extreme length ratio filtering.
    7. Disjoint deduplication to guarantee 0% train/val/test data leakage.
    
    Logs counts before and after every filter step.
    """
    initial_count = len(df)
    logger.info(f"[Filter 0/7] Initial raw input rows: {initial_count}")
    
    # 1. Drop missing rows
    df_clean = df.dropna(subset=[source_col, target_col]).copy()
    logger.info(f"[Filter 1/7] After dropping nulls: {len(df_clean)} (-{initial_count - len(df_clean)})")

    # 2. Apply NFKC Unicode cleaning
    is_hi_src = (source_col.lower() in ["hi", "hindi"])
    is_hi_tgt = (target_col.lower() in ["hi", "hindi"])
    df_clean[source_col] = df_clean[source_col].astype(str).apply(lambda x: clean_text(x, is_hindi=is_hi_src))
    df_clean[target_col] = df_clean[target_col].astype(str).apply(lambda x: clean_text(x, is_hindi=is_hi_tgt))

    # 3. Drop exact duplicate parallel pairs
    prev_cnt = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[source_col, target_col]).reset_index(drop=True)
    logger.info(f"[Filter 2/7] After pair deduplication: {len(df_clean)} (-{prev_cnt - len(df_clean)} duplicates)")

    # 4. Drop untranslated copies where source == target
    prev_cnt = len(df_clean)
    df_clean = df_clean[
        df_clean[source_col].str.lower().str.strip() != df_clean[target_col].str.lower().str.strip()
    ].reset_index(drop=True)
    logger.info(f"[Filter 3/7] After removing source==target copies: {len(df_clean)} (-{prev_cnt - len(df_clean)})")

    # 5. Script validation: verify Devanagari script in Hindi and Latin in English
    prev_cnt = len(df_clean)
    en_col = source_col if not is_hi_src else target_col
    hi_col = target_col if not is_hi_src else source_col

    has_latin = df_clean[en_col].apply(lambda x: bool(LATIN_RE.search(x)))
    has_deva = df_clean[hi_col].apply(lambda x: bool(DEVANAGARI_RE.search(x)))
    df_clean = df_clean[has_latin & has_deva].reset_index(drop=True)
    logger.info(f"[Filter 4/7] After script validation: {len(df_clean)} (-{prev_cnt - len(df_clean)})")

    # 6. Software formatting / markup filter
    prev_cnt = len(df_clean)
    is_software_src = df_clean[source_col].apply(lambda x: bool(SOFTWARE_NOISE_RE.search(x)))
    is_software_tgt = df_clean[target_col].apply(lambda x: bool(SOFTWARE_NOISE_RE.search(x)))
    df_clean = df_clean[~(is_software_src | is_software_tgt)].reset_index(drop=True)
    logger.info(f"[Filter 5/7] After software/markup filtering: {len(df_clean)} (-{prev_cnt - len(df_clean)})")

    # 7. Sentence length, word count, and length ratio filters
    prev_cnt = len(df_clean)
    src_len = df_clean[source_col].str.len()
    tgt_len = df_clean[target_col].str.len()
    ratio = src_len / (tgt_len + 1e-5)

    valid_len = (src_len >= min_chars) & (src_len <= max_chars) & (tgt_len >= min_chars) & (tgt_len <= max_chars)
    valid_words = (
        df_clean[source_col].apply(lambda x: len(x.split()) >= min_words) &
        df_clean[target_col].apply(lambda x: len(x.split()) >= min_words)
    )
    valid_ratio = (ratio >= min_ratio) & (ratio <= max_ratio)

    df_clean = df_clean[valid_len & valid_words & valid_ratio].reset_index(drop=True)
    logger.info(f"[Filter 6/7] After length & ratio filtering: {len(df_clean)} (-{prev_cnt - len(df_clean)})")

    # 8. Strict disjoint deduplication on normalized source and target to eliminate all data leakage
    prev_cnt = len(df_clean)
    df_clean["_norm_src"] = df_clean[source_col].astype(str).str.strip().str.lower()
    df_clean["_norm_tgt"] = df_clean[target_col].astype(str).str.strip().str.lower()
    df_clean = df_clean.drop_duplicates(subset=["_norm_src"]).reset_index(drop=True)
    df_clean = df_clean.drop_duplicates(subset=["_norm_tgt"]).reset_index(drop=True)
    df_clean = df_clean.drop(columns=["_norm_src", "_norm_tgt"])
    logger.info(f"[Filter 7/7] After disjoint normalized sentence deduplication: {len(df_clean)} (-{prev_cnt - len(df_clean)})")

    logger.info(f"Preprocessing complete. Retained {len(df_clean)}/{initial_count} high-fidelity pairs ({len(df_clean)/initial_count*100:.1f}%).")
    return df_clean

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
    save_splits: bool = True,
    processed_dir: Path | None = None,
    prefix: str = "opus_en_hi"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # Verify zero leakage across splits
    cols_to_check = [c for c in ["en", "hi", "source", "target"] if c in df.columns]
    for c in cols_to_check:
        train_set = set(train_df[c].astype(str).str.strip().str.lower())
        val_set = set(val_df[c].astype(str).str.strip().str.lower())
        test_set = set(test_df[c].astype(str).str.strip().str.lower())
        assert len(train_set.intersection(val_set)) == 0, f"Data leakage between train and val on column {c}"
        assert len(train_set.intersection(test_set)) == 0, f"Data leakage between train and test on column {c}"
        assert len(val_set.intersection(test_set)) == 0, f"Data leakage between val and test on column {c}"
    logger.info("Leakage verification passed: 0.0% overlap across train, val, and test splits.")

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

        # Also save corresponding reverse direction splits
        if prefix == "opus_en_hi" and "en" in df.columns and "hi" in df.columns:
            train_rev = train_df[["hi", "en"]]
            val_rev = val_df[["hi", "en"]]
            test_rev = test_df[["hi", "en"]]
            train_rev.to_csv(target_dir / "opus_hi_en_train.csv", index=False, encoding="utf-8")
            val_rev.to_csv(target_dir / "opus_hi_en_val.csv", index=False, encoding="utf-8")
            test_rev.to_csv(target_dir / "opus_hi_en_test.csv", index=False, encoding="utf-8")
            logger.info("Saved corresponding reverse direction splits (opus_hi_en_*.csv)")

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

    def __getitem__(self, idx: int) -> dict[str, Any]:

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

def postprocess_translation(text: str, target_lang: str = "hi") -> str:
    """
    Applies production-grade linguistic post-processing and detokenization:
    - Removes Moses tokenization artifacts (detached punctuation, subwords).
    - Cleans punctuation spacing for English and Hindi (danda '।', double danda '॥', commas, quotes).
    - Repairs English contractions (e.g. isn't, don't, I'm, we'll, it's).
    - Capitalizes initial sentence letters in English outputs.
    - Preserves Devanagari nuktas, matras, and compound ligatures.
    """
    if not text or not isinstance(text, str):
        return ""

    tgt = str(target_lang).lower().strip()
    is_hindi = (tgt in ["hi", "hindi"])

    # 1. Strip special token remnants and sentencepiece subword glues
    text = re.sub(r"<unk>|<pad>|<s>|</s>", "", text)
    text = text.replace(" ", " ")

    # 2. Collapse consecutive whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()

    # 3. Detokenize closing punctuation (remove whitespace before , . ! ? : ; % ) ] } । ॥)
    text = re.sub(r"\s+([,.:;!?%)\]}।॥])", r"\1", text)

    # 4. Detokenize opening punctuation (remove whitespace after ( [ {)
    text = re.sub(r"([(\[{])\s+", r"\1", text)

    # 5. Fix double quotes spacing: " word " -> "word"
    text = re.sub(r'\"\s+([^"]+?)\s+\"', r'"\1"', text)
    text = re.sub(r'“\s+([^”]+?)\s+”', r'“\1”', text)

    # 6. Language-specific enhancements
    if is_hindi:
        # Ensure single space after Danda / double danda if followed by another word
        text = re.sub(r"([।॥])([^\s\d\"'”’।॥])", r"\1 \2", text)
        # Ensure single space after comma
        text = re.sub(r",([^\s\d])", r", \1", text)
        # Fix hyphens in reduplications: 'धीरे - धीरे' -> 'धीरे-धीरे'
        text = re.sub(r"([\u0900-\u097F]+)\s*-\s*([\u0900-\u097F]+)", r"\1-\2", text)
    else:
        # English: repair contractions: don ' t -> don't, it ' s -> it's
        text = re.sub(r"\b(\w+)\s*'\s*(s|m|d|ll|re|ve)\b", r"\1'\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w+)\s+n\s*'\s*t\b", r"\1n't", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w+)\s*'\s*t\b", r"\1't", text, flags=re.IGNORECASE)

        # Capitalize the first letter of each sentence
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        cap_sentences = []
        for s in sentences:
            if s and len(s) > 0:
                s = s[0].upper() + s[1:]
            cap_sentences.append(s)
        text = " ".join(cap_sentences)

        # Capitalize standalone pronoun 'i' in English
        text = re.sub(r"\b(i)\b", "I", text)

    return text.strip()

if __name__ == "__main__":
    try:
        from src.dataset_loader import DatasetLoader
    except ImportError:
        from dataset_loader import DatasetLoader
    loader = DatasetLoader()
    raw_df = loader.get_or_create_dataset()
    clean_df = preprocess_dataframe(raw_df)
    train_df, val_df, test_df = split_dataset(clean_df)
    print("Preprocessing verification successful!")
