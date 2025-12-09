"""
Conversation Memory Module for HR Bot V2

Provides persistent conversation memory using:
1. SQLite Checkpointer - Thread-based conversation persistence
2. Summary Memory - Automatic summarization for long conversations
3. Semantic Memory - Vector-based context retrieval (using Titan embeddings)

Features:
- Thread-based conversations (each user session has its own thread)
- Automatic conversation summarization to handle long conversations
- Persistent storage across restarts
- Semantic search over past conversations for context
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache

import numpy as np
from langgraph.checkpoint.sqlite import SqliteSaver

from hr_bot.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# SQLite Checkpointer for LangGraph
# ============================================================================

_checkpointer_instance: Optional[SqliteSaver] = None
_db_connection = None


def get_checkpointer() -> SqliteSaver:
    """
    Get or create SQLite checkpointer for conversation persistence.
    
    Returns:
        SqliteSaver instance for LangGraph
    """
    global _checkpointer_instance, _db_connection
    
    if _checkpointer_instance is None:
        import sqlite3
        
        # Create memory directory if needed
        memory_dir = settings.memory_dir
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = memory_dir / "conversation_memory.db"
        
        # Create connection and checkpointer
        # Note: SqliteSaver needs the connection object directly
        _db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
        _checkpointer_instance = SqliteSaver(_db_connection)
        
        logger.info(f"SQLite checkpointer initialized at: {db_path}")
    
    return _checkpointer_instance


# ============================================================================
# Conversation Summary Store
# ============================================================================

class ConversationSummaryStore:
    """
    Stores conversation summaries for long-running conversations.
    
    When conversations exceed a threshold, older messages are summarized
    to maintain context while reducing token usage.
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_messages_before_summary: int = 10,
    ):
        """
        Initialize summary store.
        
        Args:
            db_path: Path to SQLite database
            max_messages_before_summary: Trigger summarization after this many messages
        """
        self.db_path = db_path or (settings.memory_dir / "summaries.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_messages = max_messages_before_summary
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    thread_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_context TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def get_summary(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get existing summary for a thread."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT summary, message_count, user_context FROM conversation_summaries WHERE thread_id = ?",
                (thread_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    "summary": row[0],
                    "message_count": row[1],
                    "user_context": json.loads(row[2]) if row[2] else {}
                }
        return None
    
    def update_summary(
        self,
        thread_id: str,
        summary: str,
        message_count: int,
        user_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update or create summary for a thread."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversation_summaries (thread_id, summary, message_count, user_context, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                    summary = excluded.summary,
                    message_count = excluded.message_count,
                    user_context = excluded.user_context,
                    last_updated = CURRENT_TIMESTAMP
            """, (thread_id, summary, message_count, json.dumps(user_context or {})))
            conn.commit()
    
    def should_summarize(self, thread_id: str, current_message_count: int) -> bool:
        """Check if conversation should be summarized."""
        existing = self.get_summary(thread_id)
        if existing:
            # Summarize if we've added enough new messages
            return (current_message_count - existing["message_count"]) >= self.max_messages
        return current_message_count >= self.max_messages
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT profile_data FROM user_profiles WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Update user profile data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_profiles (user_id, profile_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_data = excluded.profile_data,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, json.dumps(profile_data)))
            conn.commit()


# ============================================================================
# Semantic Memory (using Titan Embeddings)
# ============================================================================

class SemanticMemory:
    """
    Semantic memory using Amazon Titan embeddings for context retrieval.
    
    Stores important facts and context from conversations that can be
    retrieved based on semantic similarity.
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedding_dim: int = 1024,  # Titan embed text v2 dimension
        similarity_threshold: float = 0.4,  # Lower threshold for better recall
    ):
        """
        Initialize semantic memory.
        
        Args:
            db_path: Path to SQLite database
            embedding_dim: Dimension of embeddings (Titan v2 = 1024)
            similarity_threshold: Minimum similarity for retrieval
        """
        self.db_path = db_path or (settings.memory_dir / "semantic_memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Lazy-load embeddings model
        self._embeddings = None
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    user_id TEXT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    importance REAL DEFAULT 0.5
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON memories(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)")
            conn.commit()
    
    def _get_embeddings(self):
        """Lazy-load Titan embeddings model."""
        if self._embeddings is None:
            try:
                import boto3
                from langchain_aws import BedrockEmbeddings
                
                # Use Titan embeddings via Bedrock
                self._embeddings = BedrockEmbeddings(
                    client=boto3.client(
                        "bedrock-runtime",
                        aws_access_key_id=settings.aws_access_key_id,
                        aws_secret_access_key=settings.aws_secret_access_key,
                        region_name=settings.aws_region,
                    ),
                    model_id="amazon.titan-embed-text-v2:0",
                )
                logger.info("Titan embeddings initialized for semantic memory")
            except Exception as e:
                logger.warning(f"Could not initialize Titan embeddings: {e}")
                # Fallback to HuggingFace embeddings
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                self.embedding_dim = 384  # MiniLM dimension
                logger.info("Using HuggingFace embeddings fallback")
        
        return self._embeddings
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        embeddings = self._get_embeddings()
        vector = embeddings.embed_query(text)
        return np.array(vector, dtype=np.float32)
    
    def store_memory(
        self,
        thread_id: str,
        content: str,
        memory_type: str = "fact",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> int:
        """
        Store a memory with embedding.
        
        Args:
            thread_id: Conversation thread ID
            content: Memory content
            memory_type: Type of memory (fact, preference, context, etc.)
            user_id: Optional user identifier
            metadata: Additional metadata
            importance: Importance score (0-1)
            
        Returns:
            Memory ID
        """
        embedding = self._embed_text(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO memories (thread_id, user_id, memory_type, content, embedding, metadata, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                thread_id,
                user_id,
                memory_type,
                content,
                embedding.tobytes(),
                json.dumps(metadata or {}),
                importance
            ))
            conn.commit()
            return cursor.lastrowid
    
    def retrieve_memories(
        self,
        query: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using semantic search.
        
        Args:
            query: Query text
            thread_id: Filter by thread
            user_id: Filter by user
            memory_types: Filter by memory types
            top_k: Number of results
            
        Returns:
            List of relevant memories with scores
        """
        query_embedding = self._embed_text(query)
        
        # Build SQL query with filters
        sql = "SELECT id, thread_id, user_id, memory_type, content, embedding, metadata, importance FROM memories WHERE 1=1"
        params = []
        
        if thread_id:
            sql += " AND thread_id = ?"
            params.append(thread_id)
        
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(memory_types)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
        
        # Calculate similarities
        results = []
        for row in rows:
            stored_embedding = np.frombuffer(row[5], dtype=np.float32)
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity >= self.similarity_threshold:
                results.append({
                    "id": row[0],
                    "thread_id": row[1],
                    "user_id": row[2],
                    "memory_type": row[3],
                    "content": row[4],
                    "metadata": json.loads(row[6]),
                    "importance": row[7],
                    "similarity": float(similarity)
                })
        
        # Sort by weighted score (similarity * importance)
        results.sort(key=lambda x: x["similarity"] * x["importance"], reverse=True)
        
        return results[:top_k]
    
    def get_thread_memories(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a thread."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, memory_type, content, metadata, importance, created_at FROM memories WHERE thread_id = ? ORDER BY created_at DESC",
                (thread_id,)
            )
            return [
                {
                    "id": row[0],
                    "memory_type": row[1],
                    "content": row[2],
                    "metadata": json.loads(row[3]),
                    "importance": row[4],
                    "created_at": row[5]
                }
                for row in cursor.fetchall()
            ]


# ============================================================================
# Memory Manager - Unified Interface
# ============================================================================

class MemoryManager:
    """
    Unified memory manager combining all memory types.
    
    Provides a single interface for:
    - Conversation checkpointing (LangGraph)
    - Conversation summaries
    - Semantic memory retrieval
    - User profiles
    """
    
    def __init__(self):
        """Initialize memory manager."""
        self.checkpointer = get_checkpointer()
        self.summary_store = ConversationSummaryStore()
        self.semantic_memory = SemanticMemory()
        
        logger.info("MemoryManager initialized with all memory types")
    
    def get_context_for_query(
        self,
        query: str,
        thread_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all relevant context for a query.
        
        Args:
            query: User query
            thread_id: Conversation thread
            user_id: Optional user ID
            
        Returns:
            Dict with summary, relevant memories, and user profile
        """
        context = {
            "summary": None,
            "relevant_memories": [],
            "user_profile": None
        }
        
        # Get conversation summary
        summary_data = self.summary_store.get_summary(thread_id)
        if summary_data:
            context["summary"] = summary_data["summary"]
            context["user_context"] = summary_data.get("user_context", {})
        
        # Get relevant memories
        memories = self.semantic_memory.retrieve_memories(
            query=query,
            thread_id=thread_id,
            user_id=user_id,
            top_k=5
        )
        context["relevant_memories"] = memories
        
        # Get user profile if available
        if user_id:
            profile = self.summary_store.get_user_profile(user_id)
            if profile:
                context["user_profile"] = profile
        
        return context
    
    def store_important_fact(
        self,
        thread_id: str,
        fact: str,
        fact_type: str = "user_stated",
        user_id: Optional[str] = None
    ) -> None:
        """
        Store an important fact from the conversation.
        
        Args:
            thread_id: Conversation thread
            fact: The fact to store
            fact_type: Type of fact (user_stated, inferred, preference)
            user_id: Optional user ID
        """
        self.semantic_memory.store_memory(
            thread_id=thread_id,
            content=fact,
            memory_type=fact_type,
            user_id=user_id,
            importance=0.8 if fact_type == "user_stated" else 0.5
        )
    
    def update_conversation_summary(
        self,
        thread_id: str,
        summary: str,
        message_count: int,
        user_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update conversation summary."""
        self.summary_store.update_summary(
            thread_id=thread_id,
            summary=summary,
            message_count=message_count,
            user_context=user_context
        )
    
    def should_summarize(self, thread_id: str, message_count: int) -> bool:
        """Check if conversation needs summarization."""
        return self.summary_store.should_summarize(thread_id, message_count)
    
    def add_memory(
        self,
        thread_id: str,
        content: str,
        memory_type: str = "conversation",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> int:
        """
        Add a memory to semantic storage.
        
        This is a convenience method that wraps semantic_memory.store_memory().
        
        Args:
            thread_id: Conversation thread ID
            content: Memory content
            memory_type: Type of memory (conversation, fact, preference, etc.)
            user_id: Optional user ID
            metadata: Additional metadata
            importance: Importance score (0-1)
            
        Returns:
            Memory ID
        """
        return self.semantic_memory.store_memory(
            thread_id=thread_id,
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            metadata=metadata,
            importance=importance
        )


# ============================================================================
# Singleton Access
# ============================================================================

_memory_manager_instance: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create singleton MemoryManager instance."""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "get_checkpointer",
    "ConversationSummaryStore",
    "SemanticMemory",
    "MemoryManager",
    "get_memory_manager",
]
