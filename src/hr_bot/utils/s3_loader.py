"""
S3 Document Loader with Role-Based Access Control (RBAC)

Loads documents from AWS S3 buckets based on user role (executive vs employee).
Implements intelligent ETag-based caching for optimal performance and cost.

Features:
- Role-based document access (executive/employee)
- ETag-based change detection (no downloads needed for validation)
- Local file caching to avoid repeated S3 downloads
- Configurable cache TTL
- Multi-format support (txt, md, pdf, docx)
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator, Tuple, Set
from functools import lru_cache
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from hr_bot.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# Enums and Data Classes
# ============================================================================

class UserRole(Enum):
    """User role for access control."""
    EXECUTIVE = "executive"
    EMPLOYEE = "employee"
    ADMIN = "admin"


class DocumentCategory(Enum):
    """Document access categories in S3."""
    EXECUTIVE = "executive"
    EMPLOYEE = "employee"
    MASTER = "master"


@dataclass
class DocumentMetadata:
    """Metadata for an S3 document."""
    key: str
    name: str
    size: int
    last_modified: str
    etag: str
    extension: str
    category: DocumentCategory
    local_path: Optional[str] = None


# ============================================================================
# RBAC Document Loader
# ============================================================================

class S3DocumentLoader:
    """
    Load documents from AWS S3 with Role-Based Access Control (RBAC).
    
    Access Control:
    - Executive users: Executive + Employee + Master documents
    - Employee users: Employee + Master documents
    - Admin users: All documents
    
    Features:
    - ETag-based change detection (no downloads needed for validation)
    - Local file caching to avoid repeated S3 downloads
    - Configurable cache TTL (default: 24 hours)
    - Automatic S3 version tracking with metadata
    - Multi-format support (txt, md, pdf, docx)
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".doc", ".json"}
    DEFAULT_CACHE_TTL = 86400  # 24 hours
    
    def __init__(
        self,
        user_role: str = "employee",
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        domain: str = "hr"
    ):
        """
        Initialize S3 document loader with RBAC.
        
        Args:
            user_role: "executive", "employee", or "admin"
            bucket_name: S3 bucket name (defaults to settings)
            region: AWS region (defaults to settings)
            cache_ttl: Cache time-to-live in seconds (default: 24 hours)
            domain: Domain for multi-domain support (default: "hr")
        """
        self.user_role = UserRole(user_role.lower()) if isinstance(user_role, str) else user_role
        self.domain = domain.lower()
        self.cache_ttl = cache_ttl
        
        # S3 configuration
        self.bucket_name = bucket_name or settings.S3_BUCKET
        self.region = region or settings.AWS_REGION
        
        # Configure S3 prefixes based on role
        self._configure_prefixes()
        
        # Local cache directory
        cache_base = os.getenv("S3_CACHE_DIR", tempfile.gettempdir())
        self.cache_dir = Path(cache_base) / "hr_bot_v2_cache" / self.domain / self.user_role.value
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata files
        self.manifest_file = self.cache_dir / ".cache_manifest.json"
        self.version_file = self.cache_dir / ".s3_version"
        self.metadata_file = self.cache_dir / ".cache_metadata.json"
        
        # Initialize S3 client
        self._init_s3_client()
        
        # Load cached metadata
        self.file_metadata: Dict[str, Dict[str, Any]] = self._load_metadata()
        
        logger.info(
            f"S3 Loader initialized: role={self.user_role.value}, "
            f"bucket={self.bucket_name}, prefixes={self.s3_prefixes}"
        )
    
    def _configure_prefixes(self) -> None:
        """Configure S3 prefixes based on user role."""
        # Get prefix configurations from environment or use defaults
        # The S3 structure is: {role}/{category}/ where role is 'employee' or 'executive'
        executive_docs_prefix = os.getenv("S3_EXECUTIVE_DOCS_PREFIX", "Executive-Only-Documents/")
        employee_docs_prefix = os.getenv("S3_EMPLOYEE_DOCS_PREFIX", "Regular-Employee-Documents/")
        master_docs_prefix = os.getenv("S3_MASTER_DOCS_PREFIX", "Master-Document/")
        
        # Ensure prefixes end with /
        def normalize_prefix(prefix: str) -> str:
            prefix = prefix.strip()
            return prefix if prefix.endswith('/') else prefix + '/'
        
        executive_docs_prefix = normalize_prefix(executive_docs_prefix)
        employee_docs_prefix = normalize_prefix(employee_docs_prefix)
        master_docs_prefix = normalize_prefix(master_docs_prefix)
        
        # Build full prefixes with role-based top-level folder
        # S3 structure: employee/Regular-Employee-Documents/, executive/Executive-Only-Documents/, etc.
        role_folder = self.user_role.value  # 'employee' or 'executive'
        
        # Map prefixes to categories (using full paths)
        self.prefix_categories: Dict[str, DocumentCategory] = {}
        
        # Set accessible prefixes based on role
        if self.user_role == UserRole.ADMIN:
            # Admin sees all: executive folder + employee folder
            self.s3_prefixes = [
                f"executive/{executive_docs_prefix}",
                f"executive/{master_docs_prefix}",
                f"employee/{employee_docs_prefix}",
                f"employee/{master_docs_prefix}",
            ]
            self.prefix_categories = {
                f"executive/{executive_docs_prefix}": DocumentCategory.EXECUTIVE,
                f"executive/{master_docs_prefix}": DocumentCategory.MASTER,
                f"employee/{employee_docs_prefix}": DocumentCategory.EMPLOYEE,
                f"employee/{master_docs_prefix}": DocumentCategory.MASTER,
            }
        elif self.user_role == UserRole.EXECUTIVE:
            # Executive sees: executive folder (all docs)
            self.s3_prefixes = [
                f"executive/{executive_docs_prefix}",
                f"executive/{master_docs_prefix}",
                f"employee/{employee_docs_prefix}",  # Executive also sees employee docs
            ]
            self.prefix_categories = {
                f"executive/{executive_docs_prefix}": DocumentCategory.EXECUTIVE,
                f"executive/{master_docs_prefix}": DocumentCategory.MASTER,
                f"employee/{employee_docs_prefix}": DocumentCategory.EMPLOYEE,
            }
        else:  # EMPLOYEE
            # Employee sees: employee folder only
            self.s3_prefixes = [
                f"employee/{employee_docs_prefix}",
                f"employee/{master_docs_prefix}",
            ]
            self.prefix_categories = {
                f"employee/{employee_docs_prefix}": DocumentCategory.EMPLOYEE,
                f"employee/{master_docs_prefix}": DocumentCategory.MASTER,
            }
        
        logger.debug(f"Accessible prefixes for {self.user_role.value}: {self.s3_prefixes}")
    
    def _init_s3_client(self) -> None:
        """Initialize S3 client with credentials."""
        try:
            self.s3_client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            # Verify access
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 client connected to bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.warning("AWS credentials not configured. Using local files only.")
            self.s3_client = None
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.warning(f"S3 access error ({error_code}): {str(e)}")
            self.s3_client = None
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cached file metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata: {str(e)}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save file metadata to cache."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.file_metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
    
    def _get_s3_version_hash(self) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """
        Get version hash by comparing S3 object ETags (no downloads required).
        
        Returns:
            Tuple of (version_hash, metadata_dict)
        """
        if not self.s3_client:
            return "", {}
        
        try:
            etags = []
            metadata = {}
            
            for prefix in self.s3_prefixes:
                paginator = self.s3_client.get_paginator("list_objects_v2")
                
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        ext = Path(key).suffix.lower()
                        
                        if ext in self.SUPPORTED_EXTENSIONS:
                            etag = obj.get("ETag", "").strip('"')
                            etags.append(f"{key}:{etag}")
                            
                            # Determine category from prefix
                            category = DocumentCategory.EMPLOYEE
                            for pref, cat in self.prefix_categories.items():
                                if key.startswith(pref):
                                    category = cat
                                    break
                            
                            metadata[key] = {
                                "size": obj.get("Size"),
                                "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
                                "etag": etag,
                                "category": category.value
                            }
            
            if not etags:
                logger.warning(f"No documents found in S3 prefixes: {self.s3_prefixes}")
                return "", {}
            
            # Sort for deterministic hash
            etags.sort()
            combined = "|".join(etags)
            version_hash = hashlib.sha256(combined.encode()).hexdigest()
            
            logger.debug(f"S3 Version: {version_hash[:16]}... ({len(metadata)} docs)")
            return version_hash, metadata
            
        except Exception as e:
            logger.error(f"Error computing S3 version hash: {e}")
            return "", {}
    
    def _is_cache_valid(self) -> Tuple[bool, str]:
        """
        Check if local cache is still valid using ETag-based validation.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if cache files exist
        if not self.manifest_file.exists():
            return False, "manifest_missing"
        
        if not self.version_file.exists():
            return False, "version_file_missing"
        
        # Check TTL expiry
        manifest_age = time.time() - self.manifest_file.stat().st_mtime
        if manifest_age > self.cache_ttl:
            return False, f"ttl_expired ({manifest_age/3600:.1f}h old)"
        
        # Load manifest and verify files exist
        try:
            with open(self.manifest_file, "r") as f:
                manifest = json.load(f)
            
            cached_files = [Path(item["local_path"]) for item in manifest.get("files", [])]
            if not all(f.exists() for f in cached_files):
                return False, "cached_files_missing"
        except Exception:
            return False, "manifest_parse_error"
        
        # ETag-based validation
        with open(self.version_file, "r") as f:
            cached_version = f.read().strip()
        
        current_version, _ = self._get_s3_version_hash()
        
        if not current_version:
            return False, "s3_version_check_failed"
        
        if cached_version != current_version:
            logger.info(f"S3 change detected: cached={cached_version[:16]}..., current={current_version[:16]}...")
            return False, "s3_changed"
        
        logger.debug(f"Cache valid: {len(cached_files)} files, version={cached_version[:16]}...")
        return True, "valid"
    
    def list_documents(self, category: Optional[DocumentCategory] = None) -> List[DocumentMetadata]:
        """
        List all accessible documents in S3.
        
        Args:
            category: Optional filter by document category
            
        Returns:
            List of DocumentMetadata objects
        """
        if not self.s3_client:
            return self._list_local_documents(category)
        
        documents = []
        
        try:
            for prefix in self.s3_prefixes:
                # Filter by category if specified
                prefix_category = self.prefix_categories.get(prefix, DocumentCategory.EMPLOYEE)
                if category and prefix_category != category:
                    continue
                
                paginator = self.s3_client.get_paginator("list_objects_v2")
                
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        ext = Path(key).suffix.lower()
                        
                        if ext in self.SUPPORTED_EXTENSIONS:
                            documents.append(DocumentMetadata(
                                key=key,
                                name=Path(key).name,
                                size=obj["Size"],
                                last_modified=obj["LastModified"].isoformat() if obj.get("LastModified") else "",
                                etag=obj["ETag"].strip('"'),
                                extension=ext,
                                category=prefix_category
                            ))
            
            logger.info(f"Found {len(documents)} documents for role {self.user_role.value}")
            return documents
            
        except ClientError as e:
            logger.error(f"Error listing S3 documents: {str(e)}")
            return self._list_local_documents(category)
    
    def _list_local_documents(self, category: Optional[DocumentCategory] = None) -> List[DocumentMetadata]:
        """List documents from local cache."""
        documents = []
        
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                stat = file_path.stat()
                
                # Determine category from path
                doc_category = DocumentCategory.EMPLOYEE
                rel_path = str(file_path.relative_to(self.cache_dir))
                for prefix, cat in self.prefix_categories.items():
                    if rel_path.startswith(prefix.rstrip("/")):
                        doc_category = cat
                        break
                
                if category and doc_category != category:
                    continue
                
                documents.append(DocumentMetadata(
                    key=rel_path,
                    name=file_path.name,
                    size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    etag=hashlib.md5(file_path.read_bytes()).hexdigest(),
                    extension=file_path.suffix.lower(),
                    category=doc_category,
                    local_path=str(file_path)
                ))
        
        logger.info(f"Found {len(documents)} documents in local cache")
        return documents
    
    def download_document(self, key: str, force: bool = False) -> Optional[Path]:
        """
        Download a document from S3 to local cache.
        
        Args:
            key: S3 object key
            force: Force re-download even if cached
            
        Returns:
            Path to local file or None on error
        """
        local_path = self.cache_dir / key
        
        # Check if already cached and up to date
        if not force and local_path.exists():
            cached_meta = self.file_metadata.get(key)
            if cached_meta:
                try:
                    if self.s3_client:
                        response = self.s3_client.head_object(
                            Bucket=self.bucket_name,
                            Key=key
                        )
                        current_etag = response["ETag"].strip('"')
                        if cached_meta.get("etag") == current_etag:
                            logger.debug(f"Using cached version of {key}")
                            return local_path
                except Exception:
                    pass
        
        # Download from S3
        if self.s3_client:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.s3_client.download_file(
                    self.bucket_name,
                    key,
                    str(local_path)
                )
                
                # Update metadata
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
                self.file_metadata[key] = {
                    "etag": response["ETag"].strip('"'),
                    "last_modified": response["LastModified"].isoformat() if response.get("LastModified") else None,
                    "downloaded_at": datetime.now().isoformat(),
                    "local_path": str(local_path)
                }
                self._save_metadata()
                
                logger.info(f"Downloaded {key} to {local_path}")
                return local_path
                
            except ClientError as e:
                logger.error(f"Error downloading {key}: {str(e)}")
                return local_path if local_path.exists() else None
        
        return local_path if local_path.exists() else None
    
    def load_document(self, key: str) -> Optional[str]:
        """
        Load document content as text.
        
        Args:
            key: S3 object key
            
        Returns:
            Document content as string or None
        """
        local_path = self.download_document(key)
        
        if not local_path or not local_path.exists():
            return None
        
        ext = local_path.suffix.lower()
        
        try:
            # Plain text formats
            if ext in {".txt", ".md", ".markdown", ".json"}:
                return local_path.read_text(encoding="utf-8")
            
            # PDF
            elif ext == ".pdf":
                return self._load_pdf(local_path)
            
            # Word documents
            elif ext in {".docx", ".doc"}:
                return self._load_docx(local_path)
            
            else:
                logger.warning(f"Unsupported file format: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading {key}: {str(e)}")
            return None
    
    def _load_pdf(self, path: Path) -> Optional[str]:
        """Load PDF content."""
        try:
            import pypdf
            
            reader = pypdf.PdfReader(path)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.warning("pypdf not installed. Run: pip install pypdf")
            return None
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {str(e)}")
            return None
    
    def _load_docx(self, path: Path) -> Optional[str]:
        """Load Word document content."""
        try:
            from docx import Document
            
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            return "\n\n".join(paragraphs)
            
        except ImportError:
            logger.warning("python-docx not installed. Run: pip install python-docx")
            return None
        except Exception as e:
            logger.error(f"Error reading DOCX {path}: {str(e)}")
            return None
    
    def sync_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Synchronize all documents from S3 to local cache.
        
        Args:
            force: Force re-download all files
            
        Returns:
            Sync statistics
        """
        # Check if cache is valid (skip download if not forcing)
        if not force:
            is_valid, reason = self._is_cache_valid()
            if is_valid:
                logger.info("Cache is valid, skipping sync")
                return {"skipped": True, "reason": "cache_valid"}
        
        documents = self.list_documents()
        version_hash, s3_metadata = self._get_s3_version_hash()
        
        stats = {
            "total": len(documents),
            "downloaded": 0,
            "cached": 0,
            "failed": 0,
            "files": []
        }
        
        manifest_files = []
        
        for doc in documents:
            key = doc.key
            result = self.download_document(key, force=force)
            
            if result:
                stats["downloaded"] += 1
                stats["files"].append(key)
                manifest_files.append({
                    "key": key,
                    "local_path": str(result),
                    "category": doc.category.value
                })
            else:
                stats["failed"] += 1
        
        # Save version hash
        if version_hash:
            with open(self.version_file, "w") as f:
                f.write(version_hash)
        
        # Save manifest
        with open(self.manifest_file, "w") as f:
            json.dump({
                "version": version_hash,
                "synced_at": datetime.now().isoformat(),
                "role": self.user_role.value,
                "files": manifest_files
            }, f, indent=2)
        
        logger.info(f"Sync complete: {stats['downloaded']} downloaded, {stats['failed']} failed")
        return stats
    
    def load_all_documents(
        self,
        category: Optional[DocumentCategory] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generator to load all documents with content.
        
        Args:
            category: Optional filter by category
            
        Yields:
            Dict with key, name, content, category, and metadata
        """
        documents = self.list_documents(category)
        
        for doc in documents:
            content = self.load_document(doc.key)
            
            if content:
                yield {
                    "key": doc.key,
                    "name": doc.name,
                    "content": content,
                    "extension": doc.extension,
                    "size": doc.size,
                    "last_modified": doc.last_modified,
                    "category": doc.category.value,
                    "is_executive": doc.category == DocumentCategory.EXECUTIVE
                }
    
    def can_access(self, document_key: str) -> bool:
        """
        Check if current user role can access a document.
        
        Args:
            document_key: S3 object key
            
        Returns:
            True if access is allowed
        """
        # Determine document category from key
        for prefix, category in self.prefix_categories.items():
            if document_key.startswith(prefix):
                if category == DocumentCategory.EXECUTIVE:
                    return self.user_role in [UserRole.EXECUTIVE, UserRole.ADMIN]
                return True
        
        # Default: allow access to unknown categories
        return True


# ============================================================================
# Factory Functions
# ============================================================================

_loader_instances: Dict[str, S3DocumentLoader] = {}


def get_s3_loader(
    user_role: str = "employee",
    domain: str = "hr"
) -> S3DocumentLoader:
    """
    Get or create S3 document loader instance for a user role.
    
    Args:
        user_role: "executive", "employee", or "admin"
        domain: Domain for multi-domain support
        
    Returns:
        S3DocumentLoader instance
    """
    cache_key = f"{domain}_{user_role}"
    
    if cache_key not in _loader_instances:
        _loader_instances[cache_key] = S3DocumentLoader(
            user_role=user_role,
            domain=domain
        )
    
    return _loader_instances[cache_key]


def load_hr_documents(user_role: str = "employee") -> S3DocumentLoader:
    """Get loader for HR documents with RBAC."""
    return get_s3_loader(user_role=user_role, domain="hr")


def load_executive_documents() -> S3DocumentLoader:
    """Get loader for executive-only access."""
    return get_s3_loader(user_role="executive", domain="hr")


def load_employee_documents() -> S3DocumentLoader:
    """Get loader for employee-level access."""
    return get_s3_loader(user_role="employee", domain="hr")


def load_master_documents() -> S3DocumentLoader:
    """Get loader for master documents (employee access)."""
    return get_s3_loader(user_role="employee", domain="hr")


__all__ = [
    "UserRole",
    "DocumentCategory",
    "DocumentMetadata",
    "S3DocumentLoader",
    "get_s3_loader",
    "load_hr_documents",
    "load_executive_documents",
    "load_employee_documents",
    "load_master_documents"
]
