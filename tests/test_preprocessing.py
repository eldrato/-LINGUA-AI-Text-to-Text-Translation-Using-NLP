"""
Unit tests for preprocessing and Unicode normalization.
"""
import sys
import site
from pathlib import Path

# Ensure User Site-Packages and Project Root are in sys.path
_user_site = site.getusersitepackages()
if _user_site and Path(_user_site).exists() and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from src.preprocessing import clean_text, preprocess_dataframe, split_dataset

def test_clean_text_english():
    raw = "   Hello   world!  \t This is a   test. \n  "
    cleaned = clean_text(raw, is_hindi=False)
    assert cleaned == "Hello world! This is a test."

def test_clean_text_devanagari_preservation():
    # Hindi text with matras, halant, and danda
    hindi_text = "नमस्ते,   आप कैसे हैं?  मैं ठीक हूँ।   "
    cleaned = clean_text(hindi_text, is_hindi=True)
    # Ensure Devanagari danda and characters are fully intact
    assert "नमस्ते" in cleaned
    assert "हूँ।" in cleaned
    assert "  " not in cleaned

def test_clean_text_control_characters():
    text_with_control = "Valid text\x00\x08with null bytes"
    cleaned = clean_text(text_with_control)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Valid textwith null bytes" == cleaned

def test_preprocess_dataframe():
    raw_data = {
        "en": ["Hello", "World", None, "Hello", "   ", "A" * 600],
        "hi": ["नमस्ते", "संसार", "कुछ", "नमस्ते", "   ", "ब" * 600]
    }
    df = pd.DataFrame(raw_data)
    clean_df = preprocess_dataframe(df, source_col="en", target_col="hi", min_chars=2, max_chars=500)
    
    # "Hello"/"नमस्ते" was duplicated -> only 1 kept
    # None dropped
    # "   " dropped (empty)
    # 600 chars dropped (exceeds max_chars)
    assert len(clean_df) == 2
    assert "Hello" in clean_df["en"].values
    assert "World" in clean_df["en"].values

def test_split_dataset():
    # Create 100 sample rows
    rows = [{"en": f"Sentence {i}", "hi": f"वाक्य {i}"} for i in range(100)]
    df = pd.DataFrame(rows)
    
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        save_splits=False
    )
    
    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10
    # Ensure no overlap
    train_indices = set(train_df["en"])
    val_indices = set(val_df["en"])
    test_indices = set(test_df["en"])
    assert len(train_indices.intersection(val_indices)) == 0
    assert len(train_indices.intersection(test_indices)) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
