"""
Inference engine and translation pipeline for Lingua AI.
Handles single sentences, multi-sentence paragraphs, model caching,
and input validation for English-Hindi and Hindi-English translation.
"""
import time
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path if running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.model_manager import get_model_manager
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)

# Sentence splitting regex for chunking long inputs
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥\n])\s+")

class Translator:
    """
    Production-grade translation service using MarianMT Sequence-to-Sequence models.
    Supports English <-> Hindi translation with beam search decoding and paragraph chunking.
    """

    def __init__(self):
        self.model_manager = get_model_manager()

    def _split_into_chunks(self, text: str, max_chunk_len: int = 250) -> List[str]:
        """
        Splits large text into manageable chunks respecting paragraphs and sentence boundaries.
        """
        paragraphs = text.split("\n")
        chunks = []

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(p) <= max_chunk_len:
                chunks.append(p)
            else:
                sentences = SENTENCE_SPLIT_RE.split(p)
                current_chunk = []
                current_len = 0
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if current_len + len(s) > max_chunk_len and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [s]
                        current_len = len(s)
                    else:
                        current_chunk.append(s)
                        current_len += len(s)
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    def translate_chunk(
        self,
        text: str,
        model: Any,
        tokenizer: Any,
        num_beams: int = 4,
        max_length: int = 128,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """Translates an individual chunk of text using MarianMT model.generate()."""
        import torch

        # Tokenize chunk
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        input_ids = inputs["input_ids"].to(self.model_manager.device)
        attention_mask = inputs["attention_mask"].to(self.model_manager.device)

        # Generate translation with beam search
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True
            )

        # Decode tokens to string
        translated_text = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return translated_text.strip()

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None,
        length_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Translates text from source_lang to target_lang.

        Parameters:
        - text: Input string to translate.
        - source_lang: Source language name or code ('en', 'hi', 'English', 'Hindi').
        - target_lang: Target language name or code ('en', 'hi', 'English', 'Hindi').
        - num_beams: Beam search width (default from config: 4).
        - max_length: Maximum generation length (default from config: 128).
        - length_penalty: Generation length penalty (default from config: 1.0).

        Returns:
        - Result dictionary containing translation, metadata, and execution time.
        """
        start_time = time.perf_counter()

        # 1. Validation: Empty or whitespace input
        if not text or not text.strip():
            raise ValueError("Input text is empty. Please enter text to translate.")

        # 2. Normalize language codes
        src_code = config.SUPPORTED_LANGUAGES.get(source_lang, source_lang.lower())
        tgt_code = config.SUPPORTED_LANGUAGES.get(target_lang, target_lang.lower())

        if src_code == tgt_code:
            return {
                "source_text": text,
                "translated_text": text,
                "source_lang": src_code,
                "target_lang": tgt_code,
                "lang_pair": f"{src_code}-{tgt_code}",
                "model_id": "identity",
                "is_finetuned": False,
                "latency_sec": 0.0,
                "device": self.model_manager.device,
                "note": "Source and target languages are identical."
            }

        lang_pair = f"{src_code}-{tgt_code}"
        if lang_pair not in config.MODEL_CHECKPOINTS:
            raise ValueError(
                f"Unsupported language pair: '{source_lang} → {target_lang}'. "
                f"Currently supported: English ↔ Hindi."
            )

        # 3. Clean input text
        is_hindi_input = (src_code == "hi")
        cleaned_input = clean_text(text, is_hindi=is_hindi_input)
        if not cleaned_input:
            raise ValueError("Input text contained only unprintable or invalid characters.")

        # 4. Load or fetch cached model & tokenizer
        model, tokenizer, metadata = self.model_manager.load_model_and_tokenizer(lang_pair)

        # 5. Extract hyperparameters with defaults from config
        beams = num_beams or config.GENERATION_CONFIG["num_beams"]
        max_len = max_length or config.GENERATION_CONFIG["max_length"]
        len_pen = length_penalty or config.GENERATION_CONFIG["length_penalty"]

        # 6. Execute translation (with chunking support for multi-paragraph text)
        is_multiline = "\n" in text or len(cleaned_input) > 250
        if is_multiline:
            chunks = self._split_into_chunks(cleaned_input)
            translated_chunks = []
            for chunk in chunks:
                if chunk.strip():
                    chunk_trans = self.translate_chunk(
                        chunk, model, tokenizer,
                        num_beams=beams, max_length=max_len, length_penalty=len_pen
                    )
                    translated_chunks.append(chunk_trans)
            translated_text = "\n".join(translated_chunks)
        else:
            translated_text = self.translate_chunk(
                cleaned_input, model, tokenizer,
                num_beams=beams, max_length=max_len, length_penalty=len_pen
            )

        elapsed = time.perf_counter() - start_time

        return {
            "source_text": text,
            "translated_text": translated_text,
            "source_lang": src_code,
            "target_lang": tgt_code,
            "lang_pair": lang_pair,
            "model_id": metadata["model_id"],
            "is_finetuned": metadata["is_finetuned"],
            "latency_sec": round(elapsed, 3),
            "device": metadata["device"],
        }

# Module-level convenience function
_global_translator = None

def translate(
    text: str,
    source_lang: str,
    target_lang: str,
    num_beams: Optional[int] = None,
    max_length: Optional[int] = None
) -> Dict[str, Any]:
    global _global_translator
    if _global_translator is None:
        _global_translator = Translator()
    return _global_translator.translate(
        text, source_lang, target_lang, num_beams=num_beams, max_length=max_length
    )

if __name__ == "__main__":
    t = Translator()
    print("Testing translation module...")
    sample = "Hello, welcome to our modern translation application!"
    try:
        res = t.translate(sample, "en", "hi")
        print(f"Source: {sample}")
        print(f"Translation: {res['translated_text']}")
        print(f"Time: {res['latency_sec']}s, Device: {res['device']}")
    except Exception as e:
        print(f"Error during test: {e}")
