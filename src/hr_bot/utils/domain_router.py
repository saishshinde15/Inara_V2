"""
Domain Router Utility

Routes queries to appropriate domains/agents based on intent classification.
Uses keyword matching and optional ML-based classification.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from hr_bot.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# Domain Definitions
# ============================================================================

class Domain(Enum):
    """Available domains/agents."""
    HR = "hr"
    FINANCE = "finance"
    GENERAL = "general"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class DomainConfig:
    """Configuration for a domain."""
    name: str
    keywords: List[str]
    patterns: List[str]
    description: str
    weight: float = 1.0


# Domain configurations
DOMAIN_CONFIGS: Dict[Domain, DomainConfig] = {
    Domain.HR: DomainConfig(
        name="HR",
        keywords=[
            # Leave
            "leave", "vacation", "pto", "time off", "sick", "holiday", "sabbatical",
            # Benefits
            "benefit", "insurance", "health", "dental", "vision", "401k", "retirement",
            "pension", "hsa", "fsa", "wellness", "gym", "coverage",
            # Compensation
            "salary", "pay", "compensation", "bonus", "raise", "promotion",
            "paycheck", "payroll", "overtime", "stipend",
            # Performance
            "performance", "review", "feedback", "goal", "objective", "appraisal",
            "evaluation", "kpi", "rating",
            # Employment
            "employee", "employment", "hire", "hiring", "onboarding", "offboarding",
            "termination", "resignation", "notice period", "probation",
            "contract", "full-time", "part-time", "intern", "contractor",
            # Policies
            "policy", "handbook", "guideline", "procedure", "code of conduct",
            "dress code", "remote work", "wfh", "hybrid", "attendance",
            # Workplace
            "workplace", "harassment", "discrimination", "grievance", "complaint",
            "disciplinary", "warning", "confidential", "hr department",
            # Company internal finance policies
            "expense", "reimbursement", "travel", "per diem", "allowance",
            "corporate card", "receipt", "claim"
        ],
        patterns=[
            r"how (do|can) i (apply|request|submit|get|claim)",
            r"what (is|are) (the|our|my) (policy|policies|benefit|benefits)",
            r"(leave|vacation|sick|holiday) (policy|balance|request)",
            r"(expense|reimbursement|travel) (report|claim|policy)",
            r"(performance|annual|quarterly) review",
            r"(health|dental|vision) (insurance|coverage|benefit)",
            r"(maternity|paternity|parental) leave",
            r"(employee|staff|team) (handbook|manual|guide)"
        ],
        description="HR policies, benefits, procedures, and company guidelines",
        weight=1.2  # Slight boost for HR domain (primary use case)
    ),
    
    Domain.FINANCE: DomainConfig(
        name="Finance",
        keywords=[
            # Markets
            "stock", "market", "invest", "trading", "portfolio", "dividend",
            "equity", "bond", "fund", "etf", "index", "nasdaq", "dow",
            # Banking
            "bank", "account", "interest", "loan", "mortgage", "credit",
            "debit", "savings", "checking", "apr", "apy",
            # Tax
            "tax", "irs", "deduction", "bracket", "filing", "return",
            "capital gains", "income tax", "tax credit",
            # Finance concepts
            "finance", "financial", "economy", "economic", "inflation",
            "recession", "gdp", "fed", "federal reserve",
            "compound interest", "roi", "return",
            # Personal finance
            "budget", "debt", "retirement planning", "wealth",
            "financial planning", "asset", "liability"
        ],
        patterns=[
            r"what is (a |an )?(stock|bond|etf|401k|ira|roth)",
            r"how (does|do) (the market|stocks|bonds|interest|taxes) work",
            r"(current|latest|today) (market|stock|economic)",
            r"(capital gains|income|property) tax",
            r"(investment|retirement|financial) (strategy|planning|advice)",
            r"(stock|bond|crypto) (price|value|performance)"
        ],
        description="General finance concepts, markets, investments, and taxes",
        weight=1.0
    ),
    
    Domain.OUT_OF_SCOPE: DomainConfig(
        name="Out of Scope",
        keywords=[
            # Entertainment
            "movie", "music", "game", "sport", "celebrity", "gossip",
            # Random
            "recipe", "cook", "weather", "joke", "funny",
            # Technical (not our domain)
            "code", "programming", "python", "javascript", "database",
            # Personal
            "date", "relationship", "love", "friend"
        ],
        patterns=[
            r"tell me (a joke|something funny)",
            r"what('s| is) the weather",
            r"(who won|score of|game between)",
            r"how (do|to) (cook|make|bake)"
        ],
        description="Questions outside the scope of HR and Finance",
        weight=0.5
    )
}


# ============================================================================
# Domain Router
# ============================================================================

class DomainRouter:
    """
    Route queries to appropriate domains based on content analysis.
    
    Uses keyword matching and regex patterns for classification.
    """
    
    def __init__(self, configs: Optional[Dict[Domain, DomainConfig]] = None):
        """
        Initialize domain router.
        
        Args:
            configs: Optional custom domain configurations
        """
        self.configs = configs or DOMAIN_CONFIGS
        
        # Compile patterns
        self._compiled_patterns: Dict[Domain, List[re.Pattern]] = {}
        for domain, config in self.configs.items():
            self._compiled_patterns[domain] = [
                re.compile(p, re.IGNORECASE) for p in config.patterns
            ]
        
        logger.info(f"Domain router initialized with {len(self.configs)} domains")
    
    def _calculate_keyword_score(
        self,
        query: str,
        keywords: List[str]
    ) -> float:
        """Calculate keyword match score."""
        query_lower = query.lower()
        matches = sum(1 for kw in keywords if kw in query_lower)
        return matches / len(keywords) if keywords else 0
    
    def _calculate_pattern_score(
        self,
        query: str,
        patterns: List[re.Pattern]
    ) -> float:
        """Calculate pattern match score."""
        matches = sum(1 for p in patterns if p.search(query))
        return matches / len(patterns) if patterns else 0
    
    def classify(
        self,
        query: str,
        threshold: float = 0.1
    ) -> Tuple[Domain, float, Dict[str, float]]:
        """
        Classify query into a domain.
        
        Args:
            query: User query
            threshold: Minimum score threshold
            
        Returns:
            Tuple of (domain, confidence, all_scores)
        """
        scores: Dict[str, float] = {}
        
        for domain, config in self.configs.items():
            # Calculate scores
            keyword_score = self._calculate_keyword_score(query, config.keywords)
            pattern_score = self._calculate_pattern_score(
                query, 
                self._compiled_patterns[domain]
            )
            
            # Combine scores (patterns weighted higher)
            combined_score = (keyword_score * 0.4 + pattern_score * 0.6) * config.weight
            scores[domain.value] = combined_score
        
        # Find best match
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]
        
        # Check threshold
        if best_score < threshold:
            return Domain.GENERAL, best_score, scores
        
        return Domain(best_domain), best_score, scores
    
    def route(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Route query and return routing decision.
        
        Args:
            query: User query
            
        Returns:
            Routing decision with domain, confidence, and metadata
        """
        domain, confidence, scores = self.classify(query)
        
        return {
            "domain": domain.value,
            "confidence": round(confidence, 3),
            "all_scores": {k: round(v, 3) for k, v in scores.items()},
            "description": self.configs.get(domain, DomainConfig("Unknown", [], [], "")).description,
            "is_in_scope": domain not in [Domain.OUT_OF_SCOPE, Domain.GENERAL]
        }
    
    def get_agent_for_domain(self, domain: Domain) -> str:
        """
        Get the agent name for a domain.
        
        Args:
            domain: Domain enum
            
        Returns:
            Agent identifier string
        """
        mapping = {
            Domain.HR: "hr_subagent",
            Domain.FINANCE: "finance_subagent",
            Domain.GENERAL: "manager",
            Domain.OUT_OF_SCOPE: "manager"
        }
        return mapping.get(domain, "manager")


# ============================================================================
# Singleton Instance
# ============================================================================

_router_instance: Optional[DomainRouter] = None


def get_domain_router() -> DomainRouter:
    """Get or create singleton domain router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = DomainRouter()
    return _router_instance


def route_query(query: str) -> Dict[str, Any]:
    """
    Convenience function to route a query.
    
    Args:
        query: User query
        
    Returns:
        Routing decision
    """
    router = get_domain_router()
    return router.route(query)


__all__ = [
    "Domain",
    "DomainConfig",
    "DomainRouter",
    "DOMAIN_CONFIGS",
    "get_domain_router",
    "route_query"
]
