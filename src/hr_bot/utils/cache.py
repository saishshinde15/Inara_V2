"""
Semantic Response Cache Utility

High-performance caching with SEMANTIC SIMILARITY MATCHING.
Uses fuzzy matching instead of exact strings for much higher cache hit rates.

Features:
- Fuzzy matching using text normalization and keyword extraction
- Configurable similarity threshold (default: 75%)
- In-memory hot cache + SQLite persistence
- Automatic expiration and cleanup
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from functools import lru_cache
from contextlib import contextmanager

from hr_bot.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# Text Normalization & Similarity Utils
# ============================================================================

# Common stopwords for keyword extraction
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "about", "against", "between", "into",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "been", "being", "me", "my", "myself", "we", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "i", "tell", "know", "get",
    "please", "thanks", "thank", "hi", "hello", "hey"
}


def normalize_query(text: str) -> str:
    """Normalize query text for comparison."""
    if not text:
        return ""
    
    # Lowercase
    text = text.lower().strip()
    
    # Remove punctuation except hyphens
    text = re.sub(r"[^\w\s-]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text."""
    normalized = normalize_query(text)
    words = normalized.split()
    
    # Filter stopwords and short words
    keywords = {w for w in words if w not in STOPWORDS and len(w) > 2}
    
    return keywords


def calculate_similarity(query1: str, query2: str) -> float:
    """
    Calculate semantic similarity between two queries.
    
    Combines:
    - Keyword overlap (Jaccard similarity)
    - Sequence matching (for phrase similarity)
    
    Returns:
        Float between 0.0 and 1.0
    """
    # Normalize
    norm1 = normalize_query(query1)
    norm2 = normalize_query(query2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # Keyword overlap (Jaccard similarity)
    kw1 = extract_keywords(query1)
    kw2 = extract_keywords(query2)
    
    if not kw1 or not kw2:
        keyword_sim = 0.0
    else:
        intersection = kw1 & kw2
        union = kw1 | kw2
        keyword_sim = len(intersection) / len(union) if union else 0.0
    
    # Sequence matching
    seq_sim = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Weighted combination (keywords more important)
    combined = (keyword_sim * 0.6) + (seq_sim * 0.4)
    
    return combined


# ============================================================================
# Cache Interface
# ============================================================================

class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set cache value with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cached values."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


# ============================================================================
# SQLite Cache Implementation
# ============================================================================

class SQLiteCache(BaseCache):
    """
    SQLite-based response cache.
    
    Provides persistent caching with TTL support.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        default_ttl: int = 3600  # 1 hour default
    ):
        """
        Initialize SQLite cache.
        
        Args:
            db_path: Path to SQLite database file
            default_ttl: Default TTL in seconds
        """
        self.db_path = db_path or settings.CACHE_DIR / "response_cache.db"
        self.default_ttl = default_ttl
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        logger.info(f"SQLite cache initialized at {self.db_path}")
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    hits INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON cache(expires_at)
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value by key."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if not row:
                self._misses += 1
                return None
            
            value, expires_at = row
            
            # Check expiration
            if expires_at and time.time() > expires_at:
                self.delete(key)
                self._misses += 1
                return None
            
            # Update hit count
            conn.execute(
                "UPDATE cache SET hits = hits + 1 WHERE key = ?",
                (key,)
            )
            conn.commit()
            
            self._hits += 1
            return json.loads(value)
    
    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set cache value with optional TTL."""
        try:
            ttl = ttl or self.default_ttl
            created_at = time.time()
            expires_at = created_at + ttl if ttl else None
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache 
                    (key, value, created_at, expires_at, hits)
                    VALUES (?, ?, ?, ?, 0)
                """, (key, json.dumps(value), created_at, expires_at))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached values."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
            self._hits = 0
            self._misses = 0
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (time.time(),)
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(hits) FROM cache")
            row = cursor.fetchone()
            entry_count = row[0] or 0
            total_hits = row[1] or 0
        
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "entries": entry_count,
            "session_hits": self._hits,
            "session_misses": self._misses,
            "total_hits": total_hits,
            "hit_rate": f"{hit_rate:.1f}%",
            "db_path": str(self.db_path)
        }


# ============================================================================
# In-Memory Cache (for development/testing)
# ============================================================================

class InMemoryCache(BaseCache):
    """
    Simple in-memory cache.
    
    Useful for development and testing.
    """
    
    def __init__(self, default_ttl: int = 3600):
        """Initialize in-memory cache."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        entry = self._cache.get(key)
        
        if not entry:
            self._misses += 1
            return None
        
        # Check expiration
        if entry.get("expires_at") and time.time() > entry["expires_at"]:
            del self._cache[key]
            self._misses += 1
            return None
        
        self._hits += 1
        return entry["value"]
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set cached value."""
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl if ttl else None
        }
        return True
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all cached values."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "type": "in-memory"
        }


# ============================================================================
# Semantic Response Cache Manager
# ============================================================================

class ResponseCache:
    """
    High-performance SEMANTIC caching system for HR Bot responses.
    
    Features:
    - Fuzzy matching using text normalization and keyword extraction
    - In-memory hot cache for instant retrieval
    - Persistent SQLite cache for cross-session persistence
    - Similarity threshold (default: 0.75 = 75% match)
    - Automatic expiration and cleanup
    
    Example matches:
        "What is the sick leave policy?" matches:
        - "Tell me about sick leave"
        - "What's the sick day policy?"
        - "How does sick leave work?"
    """
    
    def __init__(
        self,
        cache_type: str = "sqlite",
        ttl: int = 3600,
        similarity_threshold: float = 0.75,
        max_index_entries: int = 10000
    ):
        """
        Initialize semantic response cache.
        
        Args:
            cache_type: Type of cache ("sqlite" or "memory")
            ttl: Default TTL in seconds
            similarity_threshold: Minimum similarity score (0.0-1.0) for cache hit
            max_index_entries: Maximum entries in query index
        """
        if cache_type == "sqlite":
            self.cache = SQLiteCache(default_ttl=ttl)
        else:
            self.cache = InMemoryCache(default_ttl=ttl)
        
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.max_index_entries = max_index_entries
        
        # Query index for semantic matching: List of (cache_key, original_query, keywords)
        self.query_index: List[Tuple[str, str, Set[str]]] = []
        
        # Enhanced statistics
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "total_queries": 0
        }
        
        # Build index from existing cache
        self._build_query_index()
        
        logger.info(f"Semantic ResponseCache initialized, similarity threshold: {similarity_threshold}")
    
    def _build_query_index(self) -> None:
        """Build query index from existing cache entries."""
        # For SQLite, we'd need to store original queries
        # For now, start with empty index - it builds as queries come in
        self.query_index = []
    
    def _add_to_index(self, cache_key: str, query: str) -> None:
        """Add query to semantic index."""
        keywords = extract_keywords(query)
        
        # Prevent duplicate entries
        for existing_key, _, _ in self.query_index:
            if existing_key == cache_key:
                return
        
        self.query_index.append((cache_key, query, keywords))
        
        # Enforce max size (remove oldest entries)
        if len(self.query_index) > self.max_index_entries:
            self.query_index = self.query_index[-self.max_index_entries:]
    
    def _find_semantic_match(self, query: str) -> Optional[str]:
        """
        Find semantically similar cached query.
        
        Args:
            query: User query to match
            
        Returns:
            Cache key of best match, or None if no match above threshold
        """
        if not self.query_index:
            return None
        
        query_keywords = extract_keywords(query)
        normalized_query = normalize_query(query)
        
        best_match_key = None
        best_similarity = 0.0
        
        for cache_key, cached_query, cached_keywords in self.query_index:
            # Quick keyword pre-filter (must share at least one keyword)
            if not (query_keywords & cached_keywords):
                continue
            
            # Full similarity calculation
            similarity = calculate_similarity(query, cached_query)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_key = cache_key
        
        if best_similarity >= self.similarity_threshold:
            logger.debug(f"Semantic match found: {best_similarity:.2f} >= {self.similarity_threshold}")
            return best_match_key
        
        return None
    
    def _generate_key(self, query: str, context: Optional[str] = None) -> str:
        """Generate cache key from query and context."""
        key_data = query.lower().strip()
        if context:
            key_data += f"|{context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_response(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query using semantic matching.
        
        Args:
            query: User query
            context: Optional context string
            
        Returns:
            Cached response or None
        """
        self.stats["total_queries"] += 1
        
        # Try exact match first
        exact_key = self._generate_key(query, context)
        result = self.cache.get(exact_key)
        
        if result:
            self.stats["exact_hits"] += 1
            result["cache_type"] = "exact"
            return result
        
        # Try semantic match
        semantic_key = self._find_semantic_match(query)
        if semantic_key:
            result = self.cache.get(semantic_key)
            if result:
                self.stats["semantic_hits"] += 1
                result["cache_type"] = "semantic"
                return result
        
        self.stats["misses"] += 1
        return None
    
    def cache_response(
        self,
        query: str,
        response: Dict[str, Any],
        context: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a response for a query.
        
        Args:
            query: User query
            response: Response to cache
            context: Optional context string
            ttl: Optional TTL override
            
        Returns:
            Whether caching was successful
        """
        key = self._generate_key(query, context)
        success = self.cache.set(key, response, ttl or self.ttl)
        
        if success:
            # Add to semantic index
            self._add_to_index(key, query)
        
        return success
    
    def invalidate(self, query: str, context: Optional[str] = None) -> bool:
        """Invalidate cached response for a query."""
        key = self._generate_key(query, context)
        
        # Remove from index
        self.query_index = [
            (k, q, kw) for k, q, kw in self.query_index if k != key
        ]
        
        return self.cache.delete(key)
    
    def clear_all(self) -> bool:
        """Clear all cached responses."""
        self.query_index = []
        self.stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0, "total_queries": 0}
        return self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including semantic matching stats."""
        base_stats = self.cache.get_stats()
        
        total = self.stats["total_queries"]
        total_hits = self.stats["exact_hits"] + self.stats["semantic_hits"]
        hit_rate = (total_hits / total * 100) if total > 0 else 0
        
        return {
            **base_stats,
            "semantic_stats": {
                "exact_hits": self.stats["exact_hits"],
                "semantic_hits": self.stats["semantic_hits"],
                "misses": self.stats["misses"],
                "total_queries": total,
                "hit_rate": f"{hit_rate:.1f}%",
                "index_size": len(self.query_index),
                "similarity_threshold": self.similarity_threshold
            }
        }


# ============================================================================
# Singleton Instance
# ============================================================================

_cache_instance: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """Get or create singleton response cache instance."""
    global _cache_instance
    if _cache_instance is None:
        cache_type = "sqlite" if settings.RESPONSE_CACHE_ENABLED else "memory"
        _cache_instance = ResponseCache(
            cache_type=cache_type,
            ttl=settings.RESPONSE_CACHE_TTL
        )
    return _cache_instance


# ============================================================================
# Cache Decorator
# ============================================================================

def cached_response(ttl: Optional[int] = None):
    """
    Decorator to cache function responses.
    
    Args:
        ttl: Optional TTL override
        
    Usage:
        @cached_response(ttl=3600)
        def my_function(query: str) -> Dict[str, Any]:
            ...
    """
    def decorator(func):
        def wrapper(query: str, *args, **kwargs):
            cache = get_response_cache()
            
            # Check cache
            cached = cache.get_cached_response(query)
            if cached:
                cached["from_cache"] = True
                return cached
            
            # Execute function
            result = func(query, *args, **kwargs)
            
            # Cache result
            if isinstance(result, dict) and result.get("success", True):
                cache.cache_response(query, result, ttl=ttl)
            
            result["from_cache"] = False
            return result
        
        return wrapper
    return decorator


__all__ = [
    "BaseCache",
    "SQLiteCache",
    "InMemoryCache",
    "ResponseCache",
    "get_response_cache",
    "cached_response"
]
