"""
Content Safety Filtering

Provides content safety checks for both input queries and output responses.
Filters inappropriate content and ensures professional responses.
"""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyCategory(Enum):
    """Categories of safety concerns."""
    SAFE = "safe"
    PROFANITY = "profanity"
    HARASSMENT = "harassment"
    PII_REQUEST = "pii_request"
    MALICIOUS = "malicious"
    INAPPROPRIATE = "inappropriate"


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    is_safe: bool
    category: SafetyCategory
    reason: Optional[str] = None
    confidence: float = 1.0


class ContentSafetyFilter:
    """
    Content safety filter for HR Bot.
    
    Checks both input queries and output responses for:
    - Profanity and inappropriate language
    - Harassment or discriminatory content
    - PII solicitation attempts
    - Malicious prompt injection
    - Out-of-scope inappropriate requests
    """
    
    # Profanity patterns (basic list - extend as needed)
    PROFANITY_PATTERNS = [
        r"\b(damn|hell|crap)\b",  # Mild
        r"\b(ass|bastard|bitch)\b",  # Moderate
        # Add more patterns as needed - keeping this family-friendly
    ]
    
    # Harassment indicators
    HARASSMENT_PATTERNS = [
        r"(stupid|idiot|moron|dumb)\s+(person|employee|worker|colleague)",
        r"(hate|despise|loathe)\s+(my|the)\s+(boss|manager|colleague)",
        r"(kill|hurt|harm|attack)\s+(someone|them|him|her|boss|manager)",
        r"(discriminat|racist|sexist|ageist)",
        r"(threat|threaten|blackmail)",
    ]
    
    # PII solicitation patterns
    PII_PATTERNS = [
        r"(give|share|tell)\s+(me|us)\s+(someone|employee|staff|colleague)('s)?\s*(ssn|social security|password|salary|address|phone)",
        r"(access|get|retrieve)\s+(other|another)\s+(employee|person|staff)('s)?\s*(data|info|information|record)",
        r"(leak|expose|reveal)\s+(confidential|private|personal)",
    ]
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|prior|above)\s+(instructions|prompt|context)",
        r"forget\s+(everything|all|what)\s+(you|i|was)",
        r"you\s+are\s+now\s+(a|an|the)",
        r"new\s+instruction(s)?:",
        r"system\s*:\s*",
        r"<\s*(system|prompt|instruction)",
        r"jailbreak|bypass|override",
    ]
    
    # Safe response template
    SAFETY_RESPONSE = (
        "I apologize, but I'm unable to process that request as it appears to "
        "contain content outside my scope as an HR assistant. Please rephrase "
        "your question about HR policies, benefits, or procedures, and I'll be "
        "happy to help."
    )
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize content safety filter.
        
        Args:
            strict_mode: If True, use stricter filtering
        """
        self.strict_mode = strict_mode
        
        # Compile patterns for performance
        self._profanity_re = [re.compile(p, re.IGNORECASE) for p in self.PROFANITY_PATTERNS]
        self._harassment_re = [re.compile(p, re.IGNORECASE) for p in self.HARASSMENT_PATTERNS]
        self._pii_re = [re.compile(p, re.IGNORECASE) for p in self.PII_PATTERNS]
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
    
    def check_input(self, query: str) -> SafetyCheckResult:
        """
        Check input query for safety concerns.
        
        Args:
            query: User query to check
            
        Returns:
            SafetyCheckResult with safety status
        """
        if not query:
            return SafetyCheckResult(is_safe=True, category=SafetyCategory.SAFE)
        
        query_lower = query.lower()
        
        # Check for prompt injection (highest priority)
        for pattern in self._injection_re:
            if pattern.search(query):
                logger.warning(f"Prompt injection detected in query")
                return SafetyCheckResult(
                    is_safe=False,
                    category=SafetyCategory.MALICIOUS,
                    reason="Potential prompt injection detected",
                    confidence=0.9
                )
        
        # Check for PII solicitation
        for pattern in self._pii_re:
            if pattern.search(query):
                logger.warning(f"PII solicitation detected in query")
                return SafetyCheckResult(
                    is_safe=False,
                    category=SafetyCategory.PII_REQUEST,
                    reason="Request for private employee information",
                    confidence=0.85
                )
        
        # Check for harassment
        for pattern in self._harassment_re:
            if pattern.search(query):
                logger.warning(f"Harassment content detected in query")
                return SafetyCheckResult(
                    is_safe=False,
                    category=SafetyCategory.HARASSMENT,
                    reason="Content contains harassment indicators",
                    confidence=0.8
                )
        
        # Check for profanity (only in strict mode for mild cases)
        if self.strict_mode:
            for pattern in self._profanity_re:
                if pattern.search(query):
                    return SafetyCheckResult(
                        is_safe=False,
                        category=SafetyCategory.PROFANITY,
                        reason="Query contains inappropriate language",
                        confidence=0.7
                    )
        
        return SafetyCheckResult(is_safe=True, category=SafetyCategory.SAFE)
    
    def check_output(self, response: str) -> SafetyCheckResult:
        """
        Check output response for safety concerns.
        
        Args:
            response: Generated response to check
            
        Returns:
            SafetyCheckResult with safety status
        """
        if not response:
            return SafetyCheckResult(is_safe=True, category=SafetyCategory.SAFE)
        
        # Check for leaked PII patterns in response
        pii_leak_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN format
            r"\b\d{10,12}\b",  # Phone numbers (10-12 digit)
            r"password\s*(is|:)\s*\S+",  # Password leak
            # More specific salary pattern - only individual salary disclosure, not policy amounts
            r"(your|their|his|her|employee('s)?)\s*salary\s*(is|:)\s*\$?\d+",
        ]
        
        for pattern in pii_leak_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning(f"Potential PII leak in response")
                return SafetyCheckResult(
                    is_safe=False,
                    category=SafetyCategory.PII_REQUEST,
                    reason="Response may contain sensitive information",
                    confidence=0.8
                )
        
        # Check for inappropriate content generation
        for pattern in self._harassment_re:
            if pattern.search(response):
                return SafetyCheckResult(
                    is_safe=False,
                    category=SafetyCategory.INAPPROPRIATE,
                    reason="Response contains inappropriate content",
                    confidence=0.75
                )
        
        return SafetyCheckResult(is_safe=True, category=SafetyCategory.SAFE)
    
    def filter_query(self, query: str) -> Tuple[str, bool]:
        """
        Filter query and return sanitized version.
        
        Args:
            query: Original query
            
        Returns:
            Tuple of (processed_query, was_filtered)
        """
        check_result = self.check_input(query)
        
        if check_result.is_safe:
            return query, False
        
        # Log the filter action
        logger.info(f"Query filtered: {check_result.category.value} - {check_result.reason}")
        
        return "", True
    
    def get_safety_response(self, category: SafetyCategory) -> str:
        """
        Get appropriate safety response for a category.
        
        Args:
            category: Safety category that was triggered
            
        Returns:
            Safe response message
        """
        responses = {
            SafetyCategory.PROFANITY: (
                "I'd be happy to help with your HR question. Could you please "
                "rephrase your query in a professional manner?"
            ),
            SafetyCategory.HARASSMENT: (
                "I understand workplace situations can be challenging. If you're "
                "experiencing issues with colleagues or management, I can help you "
                "understand our conflict resolution or grievance policies."
            ),
            SafetyCategory.PII_REQUEST: (
                "I'm unable to share private employee information. If you need "
                "access to specific records, please contact HR directly with "
                "appropriate authorization."
            ),
            SafetyCategory.MALICIOUS: self.SAFETY_RESPONSE,
            SafetyCategory.INAPPROPRIATE: self.SAFETY_RESPONSE,
        }
        
        return responses.get(category, self.SAFETY_RESPONSE)


# Singleton instance
_safety_filter: Optional[ContentSafetyFilter] = None


def get_safety_filter(strict_mode: bool = False) -> ContentSafetyFilter:
    """Get or create singleton safety filter instance."""
    global _safety_filter
    if _safety_filter is None:
        _safety_filter = ContentSafetyFilter(strict_mode=strict_mode)
    return _safety_filter


def is_query_safe(query: str) -> Tuple[bool, Optional[str]]:
    """
    Quick check if query is safe.
    
    Args:
        query: Query to check
        
    Returns:
        Tuple of (is_safe, safety_response_if_unsafe)
    """
    filter_instance = get_safety_filter()
    result = filter_instance.check_input(query)
    
    if result.is_safe:
        return True, None
    
    return False, filter_instance.get_safety_response(result.category)


__all__ = [
    "SafetyCategory",
    "SafetyCheckResult",
    "ContentSafetyFilter",
    "get_safety_filter",
    "is_query_safe"
]
