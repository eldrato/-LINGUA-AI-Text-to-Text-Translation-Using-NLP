"""
Unit tests for translation inference, validation, and language selection.
"""
import site
import sys
from pathlib import Path

# Ensure User Site-Packages and Project Root are in sys.path
_user_site = site.getusersitepackages()
if _user_site and Path(_user_site).exists() and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest  # type: ignore

from src.model_manager import get_model_manager
from src.translation import Translator


def test_normalize_lang_pair():
    mgr = get_model_manager()
    assert mgr.normalize_lang_pair("en-hi") == "en-hi"
    assert mgr.normalize_lang_pair("English → Hindi") == "en-hi"
    assert mgr.normalize_lang_pair("Hindi to English") == "hi-en"
    assert mgr.normalize_lang_pair(("en", "hi")) == "en-hi"

def test_unsupported_language_pair():
    mgr = get_model_manager()
    with pytest.raises(ValueError):
        mgr.normalize_lang_pair("French to Spanish")

def test_empty_input_validation():
    translator = Translator()
    with pytest.raises(ValueError, match="Input text is empty"):
        translator.translate("", "English", "Hindi")

    with pytest.raises(ValueError, match="Input text is empty"):
        translator.translate("   \t \n  ", "English", "Hindi")

def test_identical_source_target_translation():
    translator = Translator()
    sample = "Hello world"
    res = translator.translate(sample, "English", "English")
    assert res["translated_text"] == sample
    assert res["source_lang"] == res["target_lang"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
