"""Wrapper around the Hugging Face TimesFM model."""
from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/timesfm-1.0-200m"

def load_model(model_name: str = MODEL_NAME):
    """Load the pretrained TimesFM model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
