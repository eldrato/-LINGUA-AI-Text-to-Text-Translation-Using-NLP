"""
Unit tests for evaluation and SacreBLEU metric computation.
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
import sacrebleu  # type: ignore


def test_sacrebleu_perfect_match():
    hypotheses = ["नमस्ते, आप कैसे हैं?"]
    references = ["नमस्ते, आप कैसे हैं?"]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl")
    assert bleu.score > 99.0

def test_sacrebleu_partial_match():
    hypotheses = ["नमस्ते, आप कैसे हैं?"]
    references = ["नमस्ते, तुम कैसे हो?"]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl")
    assert 0.0 <= bleu.score <= 100.0

def test_sacrebleu_mismatch():
    hypotheses = ["यह एक किताब है।"]
    references = ["सूर्य पूर्व में उगता है।"]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl")
    assert bleu.score < 20.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
