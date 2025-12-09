"""
Utilities module for HR Bot V2.

Provides:
- Semantic response caching (SQLite/memory with similarity matching)
- S3 document loading with RBAC (Role-Based Access Control)
- Domain routing
- Response post-processing
- Content safety filtering
"""

from hr_bot.utils.cache import (
    ResponseCache,
    SQLiteCache,
    InMemoryCache,
    get_response_cache,
    cached_response,
    normalize_query,
    extract_keywords,
    calculate_similarity
)
from hr_bot.utils.s3_loader import (
    UserRole,
    DocumentCategory,
    DocumentMetadata,
    S3DocumentLoader,
    get_s3_loader,
    load_hr_documents,
    load_executive_documents,
    load_employee_documents,
    load_master_documents
)
from hr_bot.utils.domain_router import (
    Domain,
    DomainRouter,
    get_domain_router,
    route_query
)
from hr_bot.utils.response_formatter import (
    remove_document_evidence_section,
    clean_internal_references,
    format_professional_response,
    is_no_info_response,
    validate_response_quality
)
from hr_bot.utils.content_safety import (
    SafetyCategory,
    SafetyCheckResult,
    ContentSafetyFilter,
    get_safety_filter,
    is_query_safe
)

__all__ = [
    # Cache
    "ResponseCache",
    "SQLiteCache",
    "InMemoryCache",
    "get_response_cache",
    "cached_response",
    "normalize_query",
    "extract_keywords",
    "calculate_similarity",
    # S3 Loader with RBAC
    "UserRole",
    "DocumentCategory",
    "DocumentMetadata",
    "S3DocumentLoader",
    "get_s3_loader",
    "load_hr_documents",
    "load_executive_documents",
    "load_employee_documents",
    "load_master_documents",
    # Domain Router
    "Domain",
    "DomainRouter",
    "get_domain_router",
    "route_query",
    # Response Formatter
    "remove_document_evidence_section",
    "clean_internal_references",
    "format_professional_response",
    "is_no_info_response",
    "validate_response_quality",
    # Content Safety
    "SafetyCategory",
    "SafetyCheckResult",
    "ContentSafetyFilter",
    "get_safety_filter",
    "is_query_safe"
]
