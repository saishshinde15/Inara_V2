"""
Shared Embeddings Utility - Singleton pattern for expensive model loading.

This module provides a cached embeddings instance to avoid re-loading
sentence-transformers models multiple times, saving ~4 seconds per call.
"""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Singleton embeddings instance
_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


def get_shared_embeddings() -> HuggingFaceEmbeddings:
    """
    Get or create singleton HuggingFace embeddings instance.
    
    This avoids re-loading the sentence-transformers model on every tool call,
    which saves approximately 4 seconds per query.
    
    Returns:
        HuggingFaceEmbeddings instance (cached after first call)
    """
    global _embeddings_instance
    
    if _embeddings_instance is None:
        logger.info("🔄 Loading shared sentence-transformers embeddings (one-time)...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✅ Shared embeddings loaded and cached")
    
    return _embeddings_instance


def preload_embeddings() -> None:
    """
    Pre-load embeddings at application startup.
    
    Call this during app initialization to avoid latency on first query.
    """
    get_shared_embeddings()


__all__ = ["get_shared_embeddings", "preload_embeddings"]
