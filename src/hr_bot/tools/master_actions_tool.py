"""
Master Actions Tool - Procedural guidance for HR system actions
Handles "How to" queries with step-by-step instructions and links
Loads Master Document from S3 for centralized document management
"""

import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
from hr_bot.utils.s3_loader import S3DocumentLoader, DocumentCategory


@dataclass
class ActionSearchResult:
    """Result from action search."""
    content: str
    source: str
    score: float
    chunk_id: int


class MasterActionsRetriever:
    """
    Retriever for Master Actions Document - procedural "How to" guidance.
    Optimized for action-oriented queries with links and step-by-step instructions.
    Loads documents from S3 for centralized management.
    """
    
    def __init__(self, document_dir: Optional[Path] = None, user_role: str = "employee"):
        """Initialize the Master Actions Retriever."""
        self.settings = get_settings()
        self.document_dir = document_dir or self.settings.data_dir / "Master-Document"
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
        self.index_dir = self.settings.index_dir / "master_actions"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_hash: Optional[str] = None
        
        # Action-optimized search settings
        self.chunk_size = 600  # Smaller chunks for action steps
        self.chunk_overlap = 150
        self.top_k = 5
        self.bm25_weight = 0.6  # Higher BM25 for keyword matching (action names)
        self.vector_weight = 0.4
        
        self._last_sources: List[str] = []
    
    def _load_master_document_from_s3(self) -> List[Document]:
        """Load Master Document from S3."""
        documents = []
        
        if not self.settings.S3_BUCKET:
            print("⚠️ S3_BUCKET not configured, falling back to local files")
            return self._find_local_master_document()
        
        print(f"📦 Loading Master Document from S3 bucket: {self.settings.S3_BUCKET}")
        
        try:
            s3_loader = S3DocumentLoader(
                user_role=self.user_role,
                bucket_name=self.settings.S3_BUCKET,
                region=self.settings.AWS_REGION,
            )
            
            # Load only Master Documents category
            for doc_data in s3_loader.load_all_documents(category=DocumentCategory.MASTER):
                content = doc_data.get("content", "")
                if not content:
                    continue
                
                # Look for keywords that indicate this is the Master Actions document
                name = doc_data.get("name", "").lower()
                keywords = ["guide", "portal", "action", "master", "knowledge"]
                
                if any(kw in name for kw in keywords) or doc_data.get("category") == "master":
                    doc = Document(
                        page_content=self._sanitize_content(content),
                        metadata={
                            "source": doc_data.get("name", "Master Actions Document"),
                            "file_path": doc_data.get("key", ""),
                            "type": "master_actions",
                            "from_s3": True,
                        }
                    )
                    documents.append(doc)
                    print(f"   ✓ Loaded Master Document from S3: {doc_data.get('name', 'unknown')}")
            
            if documents:
                print(f"📊 Total Master Documents loaded from S3: {len(documents)}")
            else:
                print("⚠️ No Master Documents found in S3, trying local fallback")
                return self._find_local_master_document()
            
            return documents
            
        except Exception as e:
            print(f"⚠️ S3 loading failed: {e}, falling back to local files")
            return self._find_local_master_document()
    
    def _find_local_master_document(self) -> List[Document]:
        """Find and load Master Document from local directory as fallback."""
        documents = []
        
        if not self.document_dir.exists():
            # Try alternate locations
            alternate_paths = [
                self.settings.data_dir / "Master-Document",
                self.settings.data_dir,
            ]
            for path in alternate_paths:
                if path.exists():
                    self.document_dir = path
                    break
        
        if not self.document_dir.exists():
            return documents
        
        # Search for master document
        keywords = ["knowledge", "action", "master", "guide", "portal"]
        master_path = None
        
        for doc_path in self.document_dir.glob("*.docx"):
            if any(kw in doc_path.name.lower() for kw in keywords):
                master_path = doc_path
                break
        
        # Return first docx if no keyword match
        if not master_path:
            docx_files = list(self.document_dir.glob("*.docx"))
            master_path = docx_files[0] if docx_files else None
        
        if not master_path:
            return documents
        
        # Load the document
        try:
            loader = Docx2txtLoader(str(master_path))
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source"] = master_path.name
                doc.metadata["file_path"] = str(master_path)
                doc.metadata["type"] = "master_actions"
                doc.page_content = self._sanitize_content(doc.page_content)
                documents.append(doc)
            
            print(f"✅ Loaded local Master Document: {master_path.name}")
            
        except Exception as e:
            print(f"❌ Error loading local Master Document: {e}")
        
        return documents
    
    def _compute_version_hash(self) -> str:
        """Compute hash for cache invalidation."""
        hasher = hashlib.md5()
        
        # Include user role and S3 bucket in hash
        hasher.update(f"role:{self.user_role}".encode())
        hasher.update(f"bucket:{self.settings.S3_BUCKET or 'local'}".encode())
        hasher.update(f"chunk:{self.chunk_size}|overlap:{self.chunk_overlap}".encode())
        hasher.update(b"master_actions_v3_s3")
        
        return hasher.hexdigest()
    
    def _load_master_document(self) -> List[Document]:
        """Load Master Document from S3 (primary) or local (fallback)."""
        # Try S3 first
        return self._load_master_document_from_s3()
    
    def _sanitize_content(self, text: str) -> str:
        """Clean placeholder tokens and normalize text."""
        if not text:
            return text
        
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
        
        text = re.sub(r"\[\s*insert[^\]]*\]", "the appropriate details", text, flags=re.IGNORECASE)
        
        return text
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents with action-aware splitting."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", "Action Name:", "\n", ". ", " "],
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
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
            
            print("✓ Master Actions index saved")
        except Exception as e:
            print(f"⚠️ Could not save Master Actions index: {e}")
    
    def _load_index(self, vector_path: Path, bm25_path: Path, current_hash: str) -> bool:
        """Load indexes from disk if valid."""
        try:
            if not vector_path.exists() or not bm25_path.exists():
                return False
            
            import time
            index_age_hours = (time.time() - bm25_path.stat().st_mtime) / 3600
            if index_age_hours > 24:
                print(f"⏰ Master Actions index is {index_age_hours:.1f}h old - rebuilding...")
                return False
            
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            
            if data.get("index_hash") != current_hash:
                print("🔄 Master Actions index hash mismatch - rebuilding...")
                return False
            
            self.vector_store = FAISS.load_local(
                str(vector_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self.index_hash = data["index_hash"]
            
            self._build_retrievers()
            
            print("✓ Loaded Master Actions index from disk")
            return True
            
        except Exception as e:
            print(f"Could not load Master Actions index: {e}")
            return False
    
    def _build_retrievers(self) -> None:
        """Build BM25 and FAISS retrievers."""
        if not self.documents:
            return
        
        base_k = max(self.top_k, 5)
        
        self.faiss_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": base_k * 2}
        )
        
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = base_k * 2
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.vector_weight],
        )
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """Build or load hybrid search index."""
        current_hash = self._compute_version_hash()
        
        vector_path = self.index_dir / "faiss_index"
        bm25_path = self.index_dir / "bm25_index.pkl"
        
        if not force_rebuild and self._load_index(vector_path, bm25_path, current_hash):
            return
        
        print("🔨 Building Master Actions index...")
        
        raw_docs = self._load_master_document()
        if not raw_docs:
            print("⚠️ No Master Document found - tool will return NO_ACTION_FOUND")
            return
        
        self.documents = self._chunk_documents(raw_docs)
        print(f"📊 Created {len(self.documents)} chunks")
        
        if not self.documents:
            return
        
        texts = [doc.page_content for doc in self.documents]
        metadatas = [doc.metadata for doc in self.documents]
        
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self._build_retrievers()
        
        self.index_hash = current_hash
        print(f"✅ Master Actions index built with {len(self.documents)} chunks")
        
        self._save_index(vector_path, bm25_path)
    
    def _expand_query(self, query: str) -> str:
        """Expand query with action-related synonyms."""
        q = query.strip().lower()
        expansions = []
        
        action_expansions = {
            "apply": ["request", "submit", "file"],
            "download": ["get", "access", "view", "fetch"],
            "leave": ["vacation", "time off", "absence", "pto"],
            "payslip": ["salary slip", "pay stub", "salary statement"],
            "profile": ["personal details", "employee info", "my details"],
            "training": ["learning", "course", "certification", "skill"],
            "expense": ["reimbursement", "claim", "travel claim"],
            "attendance": ["punch", "check in", "clock"],
        }
        
        for base_term, synonyms in action_expansions.items():
            if base_term in q:
                expansions.extend(synonyms)
        
        if expansions:
            return f"{query} {' '.join(expansions[:3])}"
        return query
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[ActionSearchResult]:
        """
        Search for action guidance in Master Document.
        
        Args:
            query: Action query (e.g., "How to apply for leave")
            top_k: Number of results
            
        Returns:
            List of ActionSearchResult
        """
        if not self.ensemble_retriever:
            self.build_index()
        
        if not self.ensemble_retriever:
            return []
        
        k = top_k or self.top_k
        expanded_query = self._expand_query(query)
        
        try:
            docs = self.ensemble_retriever.invoke(expanded_query)[:k]
            
            results = []
            seen_content = set()
            
            for doc in docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                results.append(ActionSearchResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Master Actions Document"),
                    score=doc.metadata.get("score", 0.5),
                    chunk_id=doc.metadata.get("chunk_id", 0),
                ))
            
            self._last_sources = list(set(r.source for r in results))[:2]
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_last_sources(self) -> List[str]:
        """Get source documents from last search."""
        return self._last_sources


# Global retriever instance
_master_retriever: Optional[MasterActionsRetriever] = None


def create_master_actions_retriever() -> MasterActionsRetriever:
    """Create or get Master Actions retriever instance."""
    global _master_retriever
    
    if _master_retriever is None:
        _master_retriever = MasterActionsRetriever()
        _master_retriever.build_index()
    
    return _master_retriever


@tool
def master_actions_guide(query: str) -> str:
    """
    Search Master Actions document for procedural "How to" guidance.
    
    Use this tool for questions like:
    - "How do I apply for leave?"
    - "How to download my payslip?"
    - "How to update my personal information?"
    - "How to submit expense claims?"
    - "How to enroll in training programs?"
    
    Args:
        query: The action query - what the user wants to do
    
    Returns:
        Step-by-step instructions and links, or NO_ACTION_FOUND
    """
    retriever = create_master_actions_retriever()
    
    results = retriever.search(query, top_k=5)
    
    if not results:
        return "NO_ACTION_FOUND: Could not find procedural guidance for this action."
    
    # Format results
    output_parts = []
    for i, result in enumerate(results, 1):
        output_parts.append(f"**Step Guide {i}**\n{result.content}\n")
    
    sources = retriever.get_last_sources()
    if sources:
        output_parts.append(f"\n**Source:** {' • '.join(sources)}")
    
    return "\n---\n".join(output_parts)
