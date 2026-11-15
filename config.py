"""
Configuration settings for Lingua AI Translation Application.
Centralized path definitions, model parameters, and training hyperparameters.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
EN_HI_MODEL_DIR = MODELS_DIR / "en_hi"
HI_EN_MODEL_DIR = MODELS_DIR / "hi_en"

OUTPUTS_DIR = BASE_DIR / "outputs"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
TRAINING_LOGS_DIR = OUTPUTS_DIR / "training_logs"

ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "custom.css"

# Pretrained Model Identifiers (Hugging Face)
MODEL_CHECKPOINTS = {
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en",
}

# Fine-tuned Local Paths
LOCAL_MODEL_PATHS = {
    "en-hi": EN_HI_MODEL_DIR,
    "hi-en": HI_EN_MODEL_DIR,
}

# Supported Languages & Display Names
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
}

LANGUAGE_PAIRS = {
    "English → Hindi": ("en", "hi"),
    "Hindi → English": ("hi", "en"),
}

# Dataset Settings
DATASET_CONFIG = {
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
    "random_seed": 42,
    "min_seq_length": 1,
    "max_seq_length_chars": 500,
    "max_token_length": 128,
}

# Training Hyperparameters
TRAIN_CONFIG = {
    "learning_rate": 5e-5,
    "batch_size_gpu": 8,
    "batch_size_cpu": 2,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 2,
    "logging_steps": 10,
    "save_total_limit": 2,
}

# Inference / Generation Settings (Optimized for High Accuracy & Fluency)
GENERATION_CONFIG = {
    "num_beams": 5,
    "max_length": 160,
    "early_stopping": True,
    "length_penalty": 1.0,
    "repetition_penalty": 1.12,
    "no_repeat_ngram_size": 3,
}

# UI Settings
APP_METADATA = {
    "title": "LINGUA AI",
    "subtitle": "Break Language Barriers with Intelligent Translation",
    "version": "1.0.0",
    "authors": "Senior ML / NLP Research Team",
}

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        RAW_DATASET_DIR,
        PROCESSED_DATASET_DIR,
        EN_HI_MODEL_DIR,
        HI_EN_MODEL_DIR,
        EVALUATION_DIR,
        TRAINING_LOGS_DIR,
        ASSETS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Run on import to ensure directories exist
ensure_directories()
