"""
This module contains the models.py file, which is used to load the model and tokenizer.
"""

from unsloth import FastLanguageModel

from utils.config import settings

loaded_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=settings.base_model,
    max_seq_length=settings.max_seq_length,
    load_in_4bit=settings.load_in_4bit,
)
