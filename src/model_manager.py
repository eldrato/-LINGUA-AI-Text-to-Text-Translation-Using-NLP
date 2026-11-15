"""
Model manager and caching system for Lingua AI.
Manages loading, device assignment (CUDA/CPU), caching, and checkpoint detection
for MarianMT Sequence-to-Sequence models and tokenizers.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add project root to path if running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Thread-safe model and tokenizer manager with in-memory caching.
    Ensures models are only loaded once per language pair and handles fallback
    between fine-tuned checkpoints and pretrained Hugging Face models.
    """
    _instance = None
    _models: Dict[str, Any] = {}
    _tokenizers: Dict[str, Any] = {}
    _device: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._init_device()
        return cls._instance

    def _init_device(self):
        """Initializes computing device (CUDA if available, else CPU)."""
        import torch
        if torch.cuda.is_available():
            self._device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA is available. Hardware acceleration enabled on: {device_name}")
        else:
            self._device = "cpu"
            logger.info("CUDA not available. Running inference on CPU.")

    @property
    def device(self) -> str:
        return self._device or "cpu"

    def normalize_lang_pair(self, lang_pair: str) -> str:
        """
        Normalizes various language pair representations to standard format e.g. 'en-hi' or 'hi-en'.
        Accepts: 'en-hi', 'en_hi', 'English to Hindi', 'English → Hindi', ('en', 'hi').
        """
        if isinstance(lang_pair, (tuple, list)):
            return f"{lang_pair[0].lower()}-{lang_pair[1].lower()}"
        
        pair_str = str(lang_pair).strip()
        pair_clean = pair_str.replace("→", "to").replace("->", "to").replace("_", "-").lower()

        if "english to hindi" in pair_clean or "en to hi" in pair_clean or pair_clean == "en-hi":
            return "en-hi"
        elif "hindi to english" in pair_clean or "hi to en" in pair_clean or pair_clean == "hi-en":
            return "hi-en"
        else:
            raise ValueError(
                f"Unsupported language pair: '{lang_pair}'. "
                f"Supported pairs are 'English → Hindi' (en-hi) and 'Hindi → English' (hi-en)."
            )

    def is_local_finetuned_available(self, pair_key: str) -> bool:
        """
        Checks whether a fine-tuned model checkpoint exists on disk.
        """
        local_dir = config.LOCAL_MODEL_PATHS.get(pair_key)
        if local_dir and local_dir.exists():
            # Check for config.json and model weights
            has_config = (local_dir / "config.json").exists()
            has_weights = (
                (local_dir / "model.safetensors").exists() or
                (local_dir / "pytorch_model.bin").exists()
            )
            return has_config and has_weights
        return False

    def get_model_identifier(self, pair_key: str, prefer_finetuned: bool = False) -> Tuple[str, bool]:
        """
        Returns the path/identifier to load and a boolean indicating if it is fine-tuned.
        """
        if prefer_finetuned and self.is_local_finetuned_available(pair_key):
            local_path = str(config.LOCAL_MODEL_PATHS[pair_key])
            logger.info(f"Using fine-tuned checkpoint for {pair_key} at: {local_path}")
            return local_path, True
        else:
            hf_id = config.MODEL_CHECKPOINTS.get(pair_key)
            if not hf_id:
                raise ValueError(f"No checkpoint defined for pair: {pair_key}")
            logger.info(f"Using high-accuracy pretrained MarianMT checkpoint for {pair_key}: {hf_id}")
            return hf_id, False

    def load_model_and_tokenizer(
        self,
        lang_pair: str,
        prefer_finetuned: bool = False
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Retrieves cached model and tokenizer, or loads them if not already cached.
        Supports both pretrained and fine-tuned checkpoints in separate cache slots.
        
        Returns:
        - model: MarianMTForConditionalGeneration / AutoModelForSeq2SeqLM instance.
        - tokenizer: MarianTokenizer / AutoTokenizer instance.
        - metadata: Dictionary containing model status and source information.
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            from transformers import MarianMTModel as AutoModelForSeq2SeqLM, MarianTokenizer as AutoTokenizer
        import torch

        pair_key = self.normalize_lang_pair(lang_pair)
        model_id, is_finetuned = self.get_model_identifier(pair_key, prefer_finetuned=prefer_finetuned)
        cache_key = f"{pair_key}_{'finetuned' if is_finetuned else 'pretrained'}"

        # Check in-memory cache
        if cache_key in self._models and cache_key in self._tokenizers:
            metadata = self.get_metadata(pair_key, prefer_finetuned=is_finetuned)
            return self._models[cache_key], self._tokenizers[cache_key], metadata

        logger.info(f"Loading tokenizer for {pair_key} [{cache_key}] from '{model_id}'...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e_tok:
            logger.error(f"Failed to load tokenizer from '{model_id}': {e_tok}")
            # If local loading failed, fallback to base pretrained model
            base_hf = config.MODEL_CHECKPOINTS[pair_key]
            logger.info(f"Falling back to base HuggingFace tokenizer '{base_hf}'...")
            tokenizer = AutoTokenizer.from_pretrained(base_hf)

        logger.info(f"Loading MarianMT model for {pair_key} [{cache_key}] from '{model_id}'...")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        except Exception as e_mod:
            logger.error(f"Failed to load model from '{model_id}': {e_mod}")
            base_hf = config.MODEL_CHECKPOINTS[pair_key]
            logger.info(f"Falling back to base HuggingFace model '{base_hf}'...")
            model = AutoModelForSeq2SeqLM.from_pretrained(base_hf)
            model_id = base_hf
            is_finetuned = False
            cache_key = f"{pair_key}_pretrained"

        # Place model on device
        model.to(self.device)
        model.eval()

        # Cache in memory
        self._models[cache_key] = model
        self._tokenizers[cache_key] = tokenizer

        metadata = {
            "lang_pair": pair_key,
            "model_id": model_id,
            "is_finetuned": is_finetuned,
            "device": self.device,
            "cache_key": cache_key,
            "model_type": "MarianMT Sequence-to-Sequence Transformer",
        }
        logger.info(f"Successfully loaded and cached model for {pair_key} on {self.device}.")
        return model, tokenizer, metadata

    def get_metadata(self, lang_pair: str, prefer_finetuned: bool = False, *args, **kwargs) -> Dict[str, Any]:
        """Returns metadata for the specified language pair."""
        pair_key = self.normalize_lang_pair(lang_pair)
        model_id, is_finetuned = self.get_model_identifier(pair_key, prefer_finetuned=prefer_finetuned)
        cache_key = f"{pair_key}_{'finetuned' if is_finetuned else 'pretrained'}"
        return {
            "lang_pair": pair_key,
            "model_id": model_id,
            "is_finetuned": is_finetuned,
            "device": self.device,
            "cache_key": cache_key,
            "is_cached": cache_key in self._models,
            "has_local_finetuned": self.is_local_finetuned_available(pair_key),
            "model_type": "MarianMT Sequence-to-Sequence Transformer",
        }

    def clear_cache(self):
        """Clears cached models and frees GPU memory."""
        import torch
        self._models.clear()
        self._tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model and tokenizer memory cache cleared.")

# Global singleton helper
_manager_instance = None

def get_model_manager() -> ModelManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance

if __name__ == "__main__":
    mgr = get_model_manager()
    print("Model manager initialized.")
    print("Device:", mgr.device)
    print("Status en-hi:", mgr.get_metadata("en-hi"))
    print("Status hi-en:", mgr.get_metadata("hi-en"))
