"""
HR RAG Tool - Hybrid RAG search for HR policy documents
Uses BM25 + Vector search with FAISS for accurate policy retrieval
Includes CrossEncoder reranking for improved precision
"""

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from langchain.tools import tool
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from hr_bot.config.settings import get_settings
from hr_bot.utils.s3_loader import S3DocumentLoader

# Optional CrossEncoder reranker (CPU-friendly)
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    RERANKER_AVAILABLE = False


@dataclass
class RAGSearchResult:
    """Result from RAG search."""
    content: str
    source: str
    score: float
    chunk_id: int


class HybridRAGRetriever:
    """
    Production-ready Hybrid RAG Retriever using BM25 + Vector search.
    Optimized for HR policy document search with semantic understanding.
    """
    
    def __init__(
        self,
        document_dir: Optional[Path] = None,
        user_role: str = "employee",
    ):
        """
        Initialize the Hybrid RAG Retriever.
        
        Args:
            document_dir: Directory containing HR documents
            user_role: User role for access control (employee/executive)
        """
        self.settings = get_settings()
        self.document_dir = document_dir or self.settings.data_dir
        self.user_role = user_role
        
        # Use shared embeddings singleton (avoids re-loading model)
        from hr_bot.utils.embeddings import get_shared_embeddings
        self.embeddings = get_shared_embeddings()
        
        # Core components
        self.vector_store: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self.documents: List[Document] = []
        
        # Index storage
        self.index_dir = self.settings.index_dir / "hr_rag"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_hash: Optional[str] = None
        
        # Search configuration
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
        self.top_k = self.settings.top_k_results
        self.bm25_weight = self.settings.bm25_weight
        self.vector_weight = self.settings.vector_weight
        
        # Reranker configuration - OPTIMIZED for speed
        self.rerank_enabled = os.getenv("RERANK_ENABLED", "false").lower() in ("1", "true")  # Disabled by default
        self.rerank_top_n = int(os.getenv("RERANK_TOP_N", "20"))  # Reduced from 50
        self.reranker_model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._reranker: Optional[Any] = None  # Lazy-initialized
        
        self._last_sources: List[str] = []
    
    def _get_reranker(self):
        """Lazy-initialize reranker if enabled and available."""
        if not self.rerank_enabled or not RERANKER_AVAILABLE:
            return None
        
        if self._reranker is None:
            try:
                print(f"🔄 Loading reranker: {self.reranker_model_name}")
                self._reranker = CrossEncoder(
                    self.reranker_model_name,
                    max_length=512,
                    device="cpu"
                )
                print("✓ Reranker loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load reranker: {e}")
                self.rerank_enabled = False
                return None
        
        return self._reranker
    
    def _rerank_results(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using CrossEncoder for improved precision.
        
        Args:
            query: Original search query
            documents: Documents to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        reranker = self._get_reranker()
        
        if not reranker or not documents:
            # Return documents with default scores
            return [(doc, 0.5) for doc in documents[:top_k]]
        
        try:
            # Prepare query-document pairs
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get reranker scores
            scores = reranker.predict(pairs)
            
            # Combine documents with scores and sort
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores[:top_k]
            
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
            return [(doc, 0.5) for doc in documents[:top_k]]
    
    def _compute_version_hash(self) -> str:
        """Compute hash for cache invalidation based on document directory."""
        hasher = hashlib.md5()
        
        # Hash document files
        if self.document_dir.exists():
            for doc_path in sorted(self.document_dir.glob("**/*.docx")):
                hasher.update(doc_path.name.encode())
                try:
                    mtime = doc_path.stat().st_mtime
                    hasher.update(str(mtime).encode())
                except Exception:
                    pass
        
        # Include config in hash
        hasher.update(f"chunk:{self.chunk_size}|overlap:{self.chunk_overlap}|role:{self.user_role}".encode())
        hasher.update(b"hr_rag_v2")
        
        return hasher.hexdigest()
    
    def _load_documents(self) -> List[Document]:
        """Load HR policy documents from S3 (primary) or local directory (fallback)."""
        documents = []
        
        # Try S3 first
        if self.settings.S3_BUCKET:
            print(f"📦 Loading documents from S3 bucket: {self.settings.S3_BUCKET}")
            try:
                s3_loader = S3DocumentLoader(
                    user_role=self.user_role,
                    bucket_name=self.settings.S3_BUCKET,
                    region=self.settings.AWS_REGION,
                )
                
                # Get documents from S3 using generator
                for doc_data in s3_loader.load_all_documents():
                    content = doc_data.get("content", "")
                    if not content:
                        continue
                        
                    doc = Document(
                        page_content=self._sanitize_content(content),
                        metadata={
                            "source": doc_data.get("name", "unknown"),
                            "file_path": doc_data.get("key", ""),
                            "folder": doc_data.get("category", ""),
                            "access_level": "executive" if doc_data.get("is_executive", False) else "employee",
                            "from_s3": True,
                        }
                    )
                    documents.append(doc)
                    print(f"   ✓ Loaded from S3: {doc_data.get('name', 'unknown')}")
                
                if documents:
                    print(f"📊 Total documents loaded from S3: {len(documents)}")
                    return documents
                    
            except Exception as e:
                print(f"⚠️ S3 loading failed: {e}, falling back to local files")
        
        # Fallback to local directory
        print(f"📂 Loading documents from local directory for role: {self.user_role}")
        
        if not self.document_dir.exists():
            print(f"⚠️ Document directory not found: {self.document_dir}")
            return documents
        
        # Determine which folders to include based on role
        include_folders = ["Regular-Employee-Documents", "Master-Document"]
        if self.user_role == "executive":
            include_folders.append("Executive-Only-Documents")
        
        for folder in include_folders:
            folder_path = self.document_dir / folder
            if not folder_path.exists():
                continue
            
            for doc_path in folder_path.glob("*.docx"):
                try:
                    loader = Docx2txtLoader(str(doc_path))
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata["source"] = doc_path.name
                        doc.metadata["file_path"] = str(doc_path)
                        doc.metadata["folder"] = folder
                        doc.metadata["access_level"] = "executive" if "Executive" in folder else "employee"
                        
                        # Clean content
                        doc.page_content = self._sanitize_content(doc.page_content)
                        documents.append(doc)
                    
                    print(f"   ✓ Loaded: {doc_path.name}")
                    
                except Exception as e:
                    print(f"   ✗ Error loading {doc_path.name}: {e}")
        
        print(f"📊 Total documents loaded: {len(documents)}")
        return documents
    
    def _sanitize_content(self, text: str) -> str:
        """Clean placeholder tokens and normalize text."""
        if not text:
            return text
        
        # Common placeholder replacements
        replacements = {
            r"\[insert name and job title\]": "HR Representative",
            r"\[insert job title\]": "HR Representative",
            r"\[the Company\]": "the company",
            r"\[Company Name\]": "the company",
            r"\[Employee\]": "employee",
            r"\[INSERT LOGO HERE\]": "",
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Generic placeholder cleanup
        text = re.sub(r"\[\s*insert[^\]]*\]", "the appropriate details", text, flags=re.IGNORECASE)
        
        return text
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents for optimal retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", " "],
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk IDs
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
        
        return chunks
    
    def _save_index(self, vector_path: Path, bm25_path: Path) -> None:
        """Save indexes to disk."""
        try:
            if self.vector_store:
                self.vector_store.save_local(str(vector_path))
            
            if self.bm25:
                clean_docs = []
                for doc in self.documents:
                    clean_doc = Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata.copy()
                    )
                    clean_docs.append(clean_doc)
                
                with open(bm25_path, "wb") as f:
                    pickle.dump({
                        "bm25": self.bm25,
                        "documents": clean_docs,
                        "index_hash": self.index_hash
                    }, f)
            
            print("✓ HR RAG index saved to disk")
        except Exception as e:
            print(f"⚠️ Could not save HR RAG index: {e}")
    
    def _load_index(self, vector_path: Path, bm25_path: Path, current_hash: str) -> bool:
        """Load indexes from disk if valid."""
        try:
            if not vector_path.exists() or not bm25_path.exists():
                return False
            
            # Check index age (24 hour TTL)
            import time
            index_age_hours = (time.time() - bm25_path.stat().st_mtime) / 3600
            if index_age_hours > 24:
                print(f"⏰ HR RAG index is {index_age_hours:.1f}h old - rebuilding...")
                return False
            
            # Load and validate hash
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            
            if data.get("index_hash") != current_hash:
                print("🔄 HR RAG index hash mismatch - rebuilding...")
                return False
            
            # Load FAISS
            self.vector_store = FAISS.load_local(
                str(vector_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load BM25 data
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self.index_hash = data["index_hash"]
            
            # Rebuild retrievers
            self._build_retrievers()
            
            print("✓ Loaded HR RAG index from disk")
            return True
            
        except Exception as e:
            print(f"Could not load HR RAG index: {e}")
            return False
    
    def _build_retrievers(self) -> None:
        """Build BM25 and FAISS retrievers."""
        if not self.documents:
            return
        
        base_k = max(self.top_k, 5)
        
        # FAISS retriever
        self.faiss_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": base_k * 2}
        )
        
        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = base_k * 2
        
        # Ensemble retriever (hybrid)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.vector_weight],
        )
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """Build or load hybrid search index."""
        current_hash = self._compute_version_hash()
        
        vector_path = self.index_dir / "faiss_index"
        bm25_path = self.index_dir / "bm25_index.pkl"
        
        # Try loading existing index
        if not force_rebuild and self._load_index(vector_path, bm25_path, current_hash):
            return
        
        print("🔨 Building HR RAG index...")
        
        # Load and chunk documents
        raw_docs = self._load_documents()
        if not raw_docs:
            print("⚠️ No HR documents found - tool will return NO_RELEVANT_DOCUMENTS")
            return
        
        self.documents = self._chunk_documents(raw_docs)
        print(f"📊 Created {len(self.documents)} chunks")
        
        if not self.documents:
            return
        
        # Build FAISS vector store
        texts = [doc.page_content for doc in self.documents]
        metadatas = [doc.metadata for doc in self.documents]
        
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Build BM25 index
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Build retrievers
        self._build_retrievers()
        
        self.index_hash = current_hash
        print(f"✅ HR RAG index built with {len(self.documents)} chunks")
        
        # Save index
        self._save_index(vector_path, bm25_path)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[RAGSearchResult]:
        """
        Search for relevant HR policy documents with optional reranking.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RAGSearchResult with content, source, and score
        """
        if not self.ensemble_retriever:
            self.build_index()
        
        if not self.ensemble_retriever:
            return []
        
        k = top_k or self.top_k
        
        try:
            # Get more candidates for reranking
            candidate_k = self.rerank_top_n if self.rerank_enabled else k
            docs = self.ensemble_retriever.invoke(query)[:candidate_k]
            
            # Apply reranking if enabled
            if self.rerank_enabled and len(docs) > k:
                reranked = self._rerank_results(query, docs, k)
                docs_with_scores = reranked
            else:
                docs_with_scores = [(doc, 0.5) for doc in docs[:k]]
            
            results = []
            seen_content = set()
            
            for doc, score in docs_with_scores:
                # Deduplicate by content hash
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                results.append(RAGSearchResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    score=float(score),
                    chunk_id=doc.metadata.get("chunk_id", 0),
                ))
            
            # Track sources for response formatting
            self._last_sources = list(set(r.source for r in results))[:2]
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_last_sources(self) -> List[str]:
        """Get source documents from last search."""
        return self._last_sources


# Global retriever instance (lazy initialization)
_hr_retriever: Optional[HybridRAGRetriever] = None


def create_hr_rag_retriever(user_role: str = "employee") -> HybridRAGRetriever:
    """Create or get HR RAG retriever instance."""
    global _hr_retriever
    
    if _hr_retriever is None or _hr_retriever.user_role != user_role:
        _hr_retriever = HybridRAGRetriever(user_role=user_role)
        _hr_retriever.build_index()
    
    return _hr_retriever


@tool
def hr_document_search(query: str, top_k: int = 5) -> str:
    """
    Search HR policy documents using hybrid RAG (BM25 + Vector search).
    
    Use this tool for questions about:
    - Leave policies (annual, sick, maternity, paternity)
    - Benefits (health insurance, retirement plans)
    - Harassment, whistleblower, ethics policies
    - Onboarding, probation, resignation procedures
    - Travel policy, expense reimbursement rules
    - Performance reviews, training programs
    
    Args:
        query: The search query - extract key terms from user's question
        top_k: Number of results to return (3-8 recommended)
    
    Returns:
        Relevant policy excerpts with source document names, or NO_RELEVANT_DOCUMENTS
    """
    retriever = create_hr_rag_retriever()
    
    results = retriever.search(query, top_k=top_k)
    
    if not results:
        return "NO_RELEVANT_DOCUMENTS: Could not find relevant HR policies for this query."
    
    # Format results
    output_parts = []
    for i, result in enumerate(results, 1):
        output_parts.append(f"**Result {i}** (Source: {result.source})\n{result.content}\n")
    
    sources = retriever.get_last_sources()
    if sources:
        output_parts.append(f"\n**Sources:** {' • '.join(sources)}")
    
    return "\n---\n".join(output_parts)
