"""
Production-ready configuration settings for HR Bot V2
Deep Agents Architecture with Manager + Subagents
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # ============================================================
    # AWS Bedrock Configuration
    # ============================================================
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # Specialist LLM (Amazon Nova Lite)
    specialist_llm_model: str = Field(
        default="bedrock/amazon.nova-lite-v1:0", alias="SPECIALIST_LLM_MODEL"
    )
    specialist_llm_temperature: float = Field(default=0.7, alias="SPECIALIST_LLM_TEMPERATURE")
    specialist_llm_max_tokens: int = Field(default=4000, alias="SPECIALIST_LLM_MAX_TOKENS")

    # Embedding model
    embedding_model: str = Field(default="amazon.titan-embed-text-v1", alias="EMBEDDING_MODEL")

    # ============================================================
    # Manager Agent (AWS Nova Lite)
    # ============================================================
    manager_llm_model: str = Field(default="amazon.nova-lite-v1:0", alias="MANAGER_LLM_MODEL")
    manager_llm_temperature: float = Field(default=0.2, alias="MANAGER_LLM_TEMPERATURE")
    manager_llm_max_tokens: int = Field(default=1000, alias="MANAGER_LLM_MAX_TOKENS")  # Reduced for speed

    # ============================================================
    # Deep Agents Configuration
    # ============================================================
    deep_agents_enabled: bool = Field(default=True, alias="DEEP_AGENTS_ENABLED")
    manager_timeout: int = Field(default=25, alias="MANAGER_TIMEOUT")  # Reduced
    subagent_timeout: int = Field(default=15, alias="SUBAGENT_TIMEOUT")  # Reduced
    rag_search_timeout: int = Field(default=10, alias="RAG_SEARCH_TIMEOUT")  # Reduced
    max_concurrent_subagents: int = Field(default=3, alias="MAX_CONCURRENT_SUBAGENTS")
    subagent_retry_attempts: int = Field(default=2, alias="SUBAGENT_RETRY_ATTEMPTS")
    
    # Web Search Toggle (user-controlled, default ON)
    web_search_enabled_default: bool = Field(default=True, alias="WEB_SEARCH_ENABLED_DEFAULT")

    # ============================================================
    # Cache Configuration (optimized for speed)
    # ============================================================
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")
    response_cache_enabled: bool = Field(default=True, alias="RESPONSE_CACHE_ENABLED")
    response_cache_ttl: int = Field(default=604800, alias="RESPONSE_CACHE_TTL")  # 7 days in seconds
    cache_ttl_hours: int = Field(default=168, alias="CACHE_TTL_HOURS")  # 7 days
    cache_similarity_threshold: float = Field(default=0.70, alias="CACHE_SIMILARITY_THRESHOLD")  # Lower = more hits
    cache_max_entries: int = Field(default=1000, alias="CACHE_MAX_ENTRIES")

    # Property aliases for backward compatibility
    @property
    def RESPONSE_CACHE_ENABLED(self) -> bool:
        return self.response_cache_enabled

    @property
    def RESPONSE_CACHE_TTL(self) -> int:
        return self.response_cache_ttl

    @property
    def CACHE_DIR(self) -> Path:
        return self.cache_dir

    # ============================================================
    # RAG Configuration
    # ============================================================
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=12, alias="TOP_K_RESULTS")
    bm25_weight: float = Field(default=0.5, alias="BM25_WEIGHT")
    vector_weight: float = Field(default=0.5, alias="VECTOR_WEIGHT")
    rrf_candidate_multiplier: int = Field(default=12, alias="RRF_CANDIDATE_MULTIPLIER")
    rrf_bm25_weight: float = Field(default=1.5, alias="RRF_BM25_WEIGHT")
    rrf_vector_weight: float = Field(default=1.0, alias="RRF_VECTOR_WEIGHT")

    # ============================================================
    # Finance Search Configuration
    # ============================================================
    serper_api_key: str = Field(default="", alias="SERPER_API_KEY")
    serper_endpoint: str = Field(
        default="https://google.serper.dev/search", alias="SERPER_ENDPOINT"
    )
    finance_serper_country: str = Field(default="in", alias="FINANCE_SERPER_COUNTRY")
    finance_serper_language: str = Field(default="en", alias="FINANCE_SERPER_LANGUAGE")

    # ============================================================
    # S3 Configuration
    # ============================================================
    s3_bucket_name: str = Field(default="", alias="S3_BUCKET_NAME")
    s3_hr_prefix: str = Field(default="hr-documents/", alias="S3_HR_PREFIX")
    s3_master_prefix: str = Field(default="master-documents/", alias="S3_MASTER_PREFIX")
    s3_executive_prefix: str = Field(default="executive-documents/", alias="S3_EXECUTIVE_PREFIX")
    
    # Property alias for backward compatibility
    @property
    def S3_BUCKET(self) -> str:
        return self.s3_bucket_name
    
    @property
    def AWS_ACCESS_KEY_ID(self) -> str:
        return self.aws_access_key_id
    
    @property
    def AWS_SECRET_ACCESS_KEY(self) -> str:
        return self.aws_secret_access_key
    
    @property
    def AWS_REGION(self) -> str:
        return self.aws_region
    
    @property
    def OPENAI_API_KEY(self) -> str:
        return self.openai_api_key

    # ============================================================
    # Authentication
    # ============================================================
    google_client_id: str = Field(default="", alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(default="", alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(default="http://localhost:8501/", alias="GOOGLE_REDIRECT_URI")
    executive_emails: str = Field(default="", alias="EXECUTIVE_EMAILS")
    employee_emails: str = Field(default="", alias="EMPLOYEE_EMAILS")
    allow_dev_login: bool = Field(default=False, alias="ALLOW_DEV_LOGIN")
    dev_test_email: str = Field(default="", alias="DEV_TEST_EMAIL")
    debug_auth: bool = Field(default=False, alias="DEBUG_AUTH")

    # ============================================================
    # Observability
    # ============================================================
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="hr-bot-v2", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")

    # ============================================================
    # Application & Branding
    # ============================================================
    # Bot/Product Configuration
    bot_name: str = Field(default="Inara", alias="BOT_NAME")
    bot_description: str = Field(
        default="Universal Enterprise Assistant powered by Deep Agents", alias="BOT_DESCRIPTION"
    )
    
    # Company Configuration
    company_name: str = Field(default="Your Company", alias="COMPANY_NAME")
    company_address: str = Field(default="", alias="COMPANY_ADDRESS")
    hr_email: str = Field(default="hr@company.com", alias="HR_EMAIL")
    support_contact_email: str = Field(default="support@company.com", alias="SUPPORT_CONTACT_EMAIL")
    
    # Legacy aliases for backward compatibility
    app_name: str = Field(default="Inara", alias="APP_NAME")
    app_description: str = Field(
        default="Your intelligent HR assistant powered by Deep Agents", alias="APP_DESCRIPTION"
    )

    # ============================================================
    # Paths
    # ============================================================
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def storage_dir(self) -> Path:
        return self.project_root / "storage"

    @property
    def cache_dir(self) -> Path:
        return self.storage_dir / "cache"

    @property
    def index_dir(self) -> Path:
        return self.storage_dir / "indexes"

    @property
    def memory_dir(self) -> Path:
        return self.storage_dir / "memory"

    def get_executive_emails(self) -> List[str]:
        """Parse executive emails from comma-separated string."""
        if not self.executive_emails:
            return []
        return [e.strip().lower() for e in self.executive_emails.split(",") if e.strip()]

    def get_employee_emails(self) -> List[str]:
        """Parse employee emails from comma-separated string."""
        if not self.employee_emails:
            return []
        return [e.strip().lower() for e in self.employee_emails.split(",") if e.strip()]

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.storage_dir, self.cache_dir, self.index_dir, self.memory_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
