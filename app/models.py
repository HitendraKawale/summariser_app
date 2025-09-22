import os
from functools import lru_cache
from transformers import pipeline

DEFAULT_MODEL = os.getenv("SUMMARISER_MODEL", "facebook/bart-large-cnn")


@lru_cache(maxsize=1)
def get_summariser(model_name: str = DEFAULT_MODEL):
    """
    Lazy-load a HF summarization pipeline once.
    """
    return pipeline("summarization", model=model_name)
