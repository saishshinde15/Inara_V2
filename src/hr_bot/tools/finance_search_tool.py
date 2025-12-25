"""
Finance Web Search Tool - External finance/government regulations search
Uses Serper API for web search on GST, TDS, EPF, tax rates, RBI rules
"""

import json
from typing import Optional

import httpx
from langchain.tools import tool

from hr_bot.config.settings import get_settings


class FinanceSerperSearchTool:
    """
    Finance-focused web search using Serper API.
    Searches for external government regulations and finance information.
    """
    
    def __init__(
        self,
        country_code: str = "in",
        language: str = "en",
    ):
        """
        Initialize Finance Serper Search Tool.
        
        Args:
            country_code: Country for search results (default: India)
            language: Language for results
        """
        self.settings = get_settings()
        self.api_key = self.settings.serper_api_key
        self.endpoint = self.settings.serper_endpoint
        self.country_code = country_code
        self.language = language
        
        # Finance-specific search enhancements
        self.finance_keywords = [
            "India", "Indian", "government", "official",
            "2024", "2025", "latest", "current"
        ]
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with finance-specific context."""
        q_lower = query.lower()
        
        # Add context based on query type
        enhancements = []
        
        if "gst" in q_lower:
            enhancements.append("GST India government")
        elif "tds" in q_lower:
            enhancements.append("TDS India Income Tax")
        elif "epf" in q_lower or "pf" in q_lower or "provident fund" in q_lower:
            enhancements.append("EPF EPFO India")
        elif "tax" in q_lower:
            enhancements.append("India Income Tax 2024-25")
        elif "rbi" in q_lower:
            enhancements.append("RBI Reserve Bank India")
        
        if enhancements:
            return f"{query} {' '.join(enhancements)}"
        
        return f"{query} India government regulations"
    
    def search(self, query: str, num_results: int = 5) -> str:
        """
        Search for finance/government information.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Formatted search results or error message
        """
        if not self.api_key:
            return "SEARCH_ERROR: Serper API key not configured. Please set SERPER_API_KEY."
        
        enhanced_query = self._enhance_query(query)
        
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": enhanced_query,
                "gl": self.country_code,
                "hl": self.language,
                "num": num_results,
            }
            
            with httpx.Client(timeout=15.0) as client:
                response = client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
            
            # Format results
            results = []
            
            # Answer box (highest priority - direct answer)
            answer_box = data.get("answerBox", {})
            if answer_box:
                answer = answer_box.get("answer", "") or answer_box.get("snippet", "")
                title = answer_box.get("title", "")
                if answer:
                    results.append(f"**📊 Direct Answer{': ' + title if title else ''}**\n{answer}\n")
            
            # Knowledge graph (second priority)
            knowledge_graph = data.get("knowledgeGraph", {})
            if knowledge_graph:
                kg_title = knowledge_graph.get("title", "")
                kg_desc = knowledge_graph.get("description", "")
                kg_attributes = knowledge_graph.get("attributes", {})
                
                if kg_title:
                    kg_text = f"**{kg_title}**"
                    if kg_desc:
                        kg_text += f"\n{kg_desc}"
                    # Add any attributes (like current value, price, etc.)
                    for attr_name, attr_value in kg_attributes.items():
                        kg_text += f"\n• {attr_name}: {attr_value}"
                    results.append(kg_text + "\n")
            
            # Organic results
            organic = data.get("organic", [])
            for i, item in enumerate(organic[:num_results], 1):
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No description")
                link = item.get("link", "")
                source = item.get("source", "")
                
                # Format with source name and clickable markdown link
                source_text = f" ({source})" if source else ""
                link_text = f"[🔗 Read more]({link})" if link else ""
                results.append(f"**{i}. {title}{source_text}**\n{snippet}\n{link_text}\n")
            
            if not results:
                return "NO_RELEVANT_RESULTS: Could not find relevant finance information for this query."
            
            output = "\n".join(results)
            output += "\n\n⚠️ **Disclaimer:** This information is from external web sources. Please verify with official government websites or consult a tax professional before taking action."
            
            return output
            
        except httpx.TimeoutException:
            return "SEARCH_TIMEOUT: The search request timed out. Please try again."
        except httpx.HTTPStatusError as e:
            return f"SEARCH_ERROR: HTTP error {e.response.status_code}"
        except Exception as e:
            return f"SEARCH_ERROR: {str(e)}"


# Global search tool instance
_finance_search: Optional[FinanceSerperSearchTool] = None


def get_finance_search_tool() -> FinanceSerperSearchTool:
    """Get or create Finance Serper Search tool."""
    global _finance_search
    
    if _finance_search is None:
        settings = get_settings()
        _finance_search = FinanceSerperSearchTool(
            country_code=settings.finance_serper_country,
            language=settings.finance_serper_language,
        )
    
    return _finance_search


@tool
def finance_web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for ANY external information using Google search.
    
    Use this tool for ANY questions requiring current/external information:
    - Flight prices and travel deals
    - Hotel prices and booking information
    - Income tax slabs and tax calculations
    - GST rates on any product/service
    - Government regulations and schemes
    - Stock prices, currency rates, market data
    - Current events, news, and real-time information
    - Product prices, reviews, and comparisons
    - General knowledge that needs verification
    
    YOU MUST USE THIS TOOL to get accurate, up-to-date information.
    Your internal knowledge is outdated - ALWAYS search!
    
    Args:
        query: The search query - be specific about what you need
        num_results: Number of search results to fetch (default: 5)
    
    Returns:
        Search results with sources and links, or error message
    """
    search_tool = get_finance_search_tool()
    return search_tool.search(query, num_results=num_results)
