"""
Inference engine and translation pipeline for Lingua AI.
Handles single sentences, multi-sentence paragraphs, model caching,
and input validation for English-Hindi and Hindi-English translation.
"""
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path if running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.model_manager import get_model_manager
from src.preprocessing import clean_text, postprocess_translation

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

    def _split_into_chunks(self, text: str, max_chunk_len: int = 250) -> list[str]:
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
        target_lang: str = "hi",
        num_beams: int = 5,
        max_length: int = 160,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.12,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """Translates an individual sentence or chunk using MarianMT model.generate()."""
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

        # For Hindi target, avoid harsh 3-gram restriction which breaks valid reduplications like धीरे-धीरे
        is_hindi = (target_lang in ["hi", "hindi"])
        ngram_size = 0 if is_hindi else no_repeat_ngram_size

        # Generate translation with refined beam search
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=ngram_size,
                early_stopping=True
            )

        # Decode tokens to string
        raw_text = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return raw_text.strip()

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        num_beams: int | None = None,
        max_length: int | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        prefer_finetuned: bool = False,
    ) -> dict[str, Any]:
        """
        Translates text from source_lang to target_lang with intelligent sentence segmentation,
        refined beam search, and production-grade detokenization.

        Parameters:
        - text: Input string to translate.
        - source_lang: Source language name or code ('en', 'hi', 'English', 'Hindi').
        - target_lang: Target language name or code ('en', 'hi', 'English', 'Hindi').
        - num_beams: Beam search width (default: 5).
        - max_length: Maximum generation length (default: 160).
        - length_penalty: Generation length penalty (default: 1.0).
        - repetition_penalty: Repetition penalty (default: 1.12).
        - prefer_finetuned: If True, uses the local fine-tuned checkpoint if available.

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

        # 3. Clean and validate full input text
        is_hindi_input = (src_code == "hi")
        cleaned_input = clean_text(text, is_hindi=is_hindi_input)
        if not cleaned_input:
            raise ValueError("Input text contained only unprintable or invalid characters.")

        # 4. Load or fetch cached model & tokenizer (pretrained vs fine-tuned)
        model, tokenizer, metadata = self.model_manager.load_model_and_tokenizer(
            lang_pair, prefer_finetuned=prefer_finetuned
        )

        # 5. Extract hyperparameters with defaults
        beams = num_beams or 5
        max_len = max_length or 160
        len_pen = length_penalty if length_penalty is not None else 1.0
        rep_pen = repetition_penalty if repetition_penalty is not None else 1.12

        # 6. Intelligent Paragraph and Sentence Segmentation
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        translated_paragraphs = []

        for p in paragraphs:
            # Segment paragraph into individual sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?।॥])\s+", p) if s.strip()]
            if not sentences:
                continue

            translated_sentences = []
            for sent in sentences:
                cleaned_sent = clean_text(sent, is_hindi=is_hindi_input)
                if not cleaned_sent:
                    continue

                raw_trans = self.translate_chunk(
                    cleaned_sent,
                    model=model,
                    tokenizer=tokenizer,
                    target_lang=tgt_code,
                    num_beams=beams,
                    max_length=max_len,
                    length_penalty=len_pen,
                    repetition_penalty=rep_pen
                )
                clean_trans = postprocess_translation(raw_trans, target_lang=tgt_code)
                translated_sentences.append(clean_trans)

            if translated_sentences:
                translated_paragraphs.append(" ".join(translated_sentences))

        final_translated_text = "\n\n".join(translated_paragraphs)
        elapsed = time.perf_counter() - start_time

        return {
            "source_text": text,
            "translated_text": final_translated_text,
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
    num_beams: int | None = None,
    max_length: int | None = None,
    prefer_finetuned: bool = False
) -> dict[str, Any]:
    global _global_translator
    if _global_translator is None:
        _global_translator = Translator()
    return _global_translator.translate(
        text, source_lang, target_lang, num_beams=num_beams, max_length=max_length, prefer_finetuned=prefer_finetuned
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
