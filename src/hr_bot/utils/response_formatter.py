"""
Response Post-Processing Utilities

Cleans and formats agent responses for production:
- Removes document evidence sections
- Removes stray source mentions
- Formats responses professionally
- Validates response quality
"""

import re
from typing import List, Optional, Dict, Any


def remove_document_evidence_section(response: str) -> str:
    """
    Remove 'Document Evidence:' sections from responses.
    Preserves the final 'Sources:' line which is part of the professional response format.
    
    Args:
        response: The raw response text from the agent
        
    Returns:
        Cleaned response without internal document evidence sections
    """
    if not response:
        return response
    
    lines = response.split("\n")
    filtered = []
    skip_document_evidence = False
    
    for line in lines:
        normalized = line.strip().lower()
        
        # Skip "Document Evidence:" sections (internal artifacts)
        if normalized.startswith("document evidence:"):
            skip_document_evidence = True
            continue
        
        # Stop skipping document evidence when we hit another section
        if skip_document_evidence:
            # End of document evidence section when we hit a new header or Sources
            if (normalized.startswith("**") or 
                normalized.startswith("sources:") or
                normalized.startswith("source:")):
                skip_document_evidence = False
                # Include this line if it's Sources (part of format)
                if normalized.startswith("sources:") or normalized.startswith("source:"):
                    filtered.append(line)
                    continue
        
        # Remove internal references like "I found this information in..."
        if ("i found this information in" in normalized or 
            "i found this in" in normalized or
            "found this information in" in normalized or
            "found this in" in normalized or
            normalized.startswith("i found this")):
            continue
        
        if not skip_document_evidence:
            filtered.append(line)
    
    return "\n".join(filtered).strip()


def clean_internal_references(response: str) -> str:
    """
    Remove internal tool references and agent artifacts from response.
    
    Args:
        response: Raw response text
        
    Returns:
        Cleaned response
    """
    if not response:
        return response
    
    # Patterns to remove
    patterns = [
        r"\[Tool:.*?\]",  # Tool references
        r"\[Action:.*?\]",  # Action markers
        r"Using tool:.*?\n",  # Tool usage logs
        r"Calling:.*?\n",  # Function call logs
        r"\*\*Result \d+\*\* \(Source:.*?\)\n",  # RAG result headers
        r"---\n",  # Section separators
        r"<thinking>.*?</thinking>",  # LLM thinking tokens
        r"<think>.*?</think>",  # Alternative thinking tags
        r"\[thinking\].*?\[/thinking\]",  # Bracket-style thinking
        r"^```\s*$",  # Code block start/end markers (standalone)
        r"^```\w*\s*$",  # Code block with language (e.g., ```markdown)
    ]
    
    cleaned = response
    for pattern in patterns:
        # Use DOTALL flag for multiline thinking blocks, MULTILINE for ^ patterns
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
    # Clean up excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    return cleaned.strip()


def format_professional_response(
    answer_text: str,
    sources: Optional[List[str]] = None,
    include_sources: bool = True,
    max_sources: int = 2
) -> str:
    """
    Format agent response for professional user-facing output.
    
    Args:
        answer_text: The main answer content
        sources: Optional list of source document names
        include_sources: Whether to append source attribution
        max_sources: Maximum number of sources to include
        
    Returns:
        Professionally formatted response
    """
    if not answer_text:
        return answer_text
    
    # Clean document evidence and internal references
    cleaned = remove_document_evidence_section(answer_text)
    cleaned = clean_internal_references(cleaned)
    
    # Trim whitespace
    cleaned = cleaned.strip()
    
    # Build final response
    parts = [cleaned]
    
    # Add source attribution if requested and available
    if include_sources and sources:
        top_sources = [s for s in sources if s][:max_sources]
        if top_sources:
            source_str = " • ".join(top_sources)
            parts.append(f"\n\n*Sources: {source_str}*")
    
    return "".join(parts)


def is_no_info_response(response: str) -> bool:
    """
    Check if response indicates no information was found.
    
    Args:
        response: Response text to check
        
    Returns:
        True if response indicates no relevant information
    """
    if not response:
        return True
    
    no_info_indicators = [
        "no_relevant_documents",
        "no relevant information",
        "could not find",
        "couldn't find",
        "unable to find",
        "don't have information",
        "no information available",
        "outside my knowledge",
        "beyond my scope",
    ]
    
    lower_response = response.lower()
    return any(indicator in lower_response for indicator in no_info_indicators)


def validate_response_quality(
    response: str,
    min_length: int = 50,
    max_length: int = 10000
) -> Dict[str, Any]:
    """
    Validate response quality metrics.
    
    Args:
        response: Response to validate
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        
    Returns:
        Dict with validation results
    """
    if not response:
        return {
            "valid": False,
            "reason": "Empty response",
            "length": 0
        }
    
    length = len(response)
    
    if length < min_length:
        return {
            "valid": False,
            "reason": f"Response too short ({length} < {min_length})",
            "length": length
        }
    
    if length > max_length:
        return {
            "valid": False,
            "reason": f"Response too long ({length} > {max_length})",
            "length": length
        }
    
    # Check for tool artifacts
    tool_artifacts = [
        "[Tool:", "[Action:", "Using tool:", "Calling:",
        "NO_RELEVANT_DOCUMENTS", "SEARCH_ERROR"
    ]
    
    for artifact in tool_artifacts:
        if artifact in response:
            return {
                "valid": False,
                "reason": f"Contains tool artifact: {artifact}",
                "length": length
            }
    
    return {
        "valid": True,
        "reason": "OK",
        "length": length
    }


__all__ = [
    "remove_document_evidence_section",
    "clean_internal_references", 
    "format_professional_response",
    "is_no_info_response",
    "validate_response_quality"
]
