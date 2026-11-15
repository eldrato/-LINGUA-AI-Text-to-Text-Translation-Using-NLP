"""
Unit tests for dataset loading, CSV parsing, and validation.
"""
import sys
import site
import tempfile
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

from src.dataset_loader import DatasetLoader

def test_validate_dataframe_valid():
    loader = DatasetLoader()
    df = pd.DataFrame({"en": ["Hello"], "hi": ["नमस्ते"]})
    is_valid, msg = loader.validate_dataframe(df, "en", "hi")
    assert is_valid is True

def test_validate_dataframe_missing_col():
    loader = DatasetLoader()
    df = pd.DataFrame({"english": ["Hello"], "hindi": ["नमस्ते"]})
    is_valid, msg = loader.validate_dataframe(df, "en", "hi")
    assert is_valid is False
    assert "Missing required columns" in msg

def test_validate_dataframe_empty():
    loader = DatasetLoader()
    df = pd.DataFrame({"en": [], "hi": []})
    is_valid, msg = loader.validate_dataframe(df, "en", "hi")
    assert is_valid is False
    assert "empty" in msg.lower()

def test_load_from_csv_custom_columns():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8") as f:
        f.write("source_sentence,target_sentence\n")
        f.write("Good morning,सुप्रभात\n")
        f.write("Thank you,धन्यवाद\n")
        temp_path = f.name

    try:
        loader = DatasetLoader()
        loaded_df = loader.load_from_csv(temp_path)
        assert len(loaded_df) == 2
        assert list(loaded_df.columns) == ["en", "hi"]
        assert loaded_df.iloc[0]["en"] == "Good morning"
        assert loaded_df.iloc[0]["hi"] == "सुप्रभात"
    finally:
        Path(temp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
