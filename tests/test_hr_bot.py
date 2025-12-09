"""
Unit tests for HR Bot V2.

Test coverage for:
- Tools (HR RAG, Master Actions, Finance Search)
- Subagents (HR, Finance)
- Manager Agent
- Utilities (Cache, Domain Router)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import modules under test
from hr_bot.config.settings import get_settings, Settings
from hr_bot.utils.domain_router import (
    Domain, DomainRouter, route_query, get_domain_router
)
from hr_bot.utils.cache import (
    InMemoryCache, ResponseCache, get_response_cache
)


# ============================================================================
# Settings Tests
# ============================================================================

class TestSettings:
    """Test settings configuration."""
    
    def test_settings_singleton(self):
        """Test that get_settings returns singleton."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
    
    def test_settings_has_required_fields(self):
        """Test that settings has all required fields."""
        settings = get_settings()
        
        # Check important fields exist
        assert hasattr(settings, "OPENAI_API_KEY")
        assert hasattr(settings, "DATA_DIR")
        assert hasattr(settings, "CACHE_DIR")
        assert hasattr(settings, "RAG_CHUNK_SIZE")
        assert hasattr(settings, "RAG_CHUNK_OVERLAP")
    
    def test_data_dir_is_path(self):
        """Test that DATA_DIR is a Path object."""
        settings = get_settings()
        from pathlib import Path
        assert isinstance(settings.DATA_DIR, Path)


# ============================================================================
# Domain Router Tests
# ============================================================================

class TestDomainRouter:
    """Test domain routing functionality."""
    
    @pytest.fixture
    def router(self):
        """Create router instance."""
        return DomainRouter()
    
    def test_hr_query_classification(self, router):
        """Test HR queries are correctly classified."""
        hr_queries = [
            "What is our leave policy?",
            "How do I submit an expense report?",
            "What are the health insurance benefits?",
            "Tell me about the performance review process",
            "How much PTO do I have?",
            "What is the dress code policy?"
        ]
        
        for query in hr_queries:
            result = router.route(query)
            assert result["domain"] == "hr", f"Query '{query}' should route to HR"
            assert result["is_in_scope"] is True
    
    def test_finance_query_classification(self, router):
        """Test finance queries are correctly classified."""
        finance_queries = [
            "What is a 401k?",
            "How does compound interest work?",
            "What are the current market trends?",
            "Explain capital gains tax",
            "How do stocks work?"
        ]
        
        for query in finance_queries:
            result = router.route(query)
            assert result["domain"] == "finance", f"Query '{query}' should route to Finance"
            assert result["is_in_scope"] is True
    
    def test_out_of_scope_classification(self, router):
        """Test out-of-scope queries are identified."""
        oos_queries = [
            "Tell me a joke",
            "What's the weather today?",
            "How do I cook pasta?",
            "Who won the game last night?"
        ]
        
        for query in oos_queries:
            result = router.route(query)
            # Should either be out_of_scope or general
            assert result["domain"] in ["out_of_scope", "general"]
    
    def test_classification_returns_scores(self, router):
        """Test that classification returns all domain scores."""
        result = router.route("What is our leave policy?")
        
        assert "all_scores" in result
        assert "hr" in result["all_scores"]
        assert "finance" in result["all_scores"]
    
    def test_get_agent_for_domain(self, router):
        """Test agent mapping for domains."""
        assert router.get_agent_for_domain(Domain.HR) == "hr_subagent"
        assert router.get_agent_for_domain(Domain.FINANCE) == "finance_subagent"
        assert router.get_agent_for_domain(Domain.GENERAL) == "manager"


# ============================================================================
# Cache Tests
# ============================================================================

class TestInMemoryCache:
    """Test in-memory cache functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        return InMemoryCache(default_ttl=3600)
    
    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        
        assert result is not None
        assert result["data"] == "value1"
    
    def test_get_nonexistent(self, cache):
        """Test getting non-existent key returns None."""
        result = cache.get("nonexistent")
        assert result is None
    
    def test_delete(self, cache):
        """Test delete operation."""
        cache.set("key1", {"data": "value1"})
        cache.delete("key1")
        result = cache.get("key1")
        
        assert result is None
    
    def test_clear(self, cache):
        """Test clear operation."""
        cache.set("key1", {"data": "value1"})
        cache.set("key2", {"data": "value2"})
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_stats(self, cache):
        """Test statistics tracking."""
        cache.set("key1", {"data": "value1"})
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["entries"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1


class TestResponseCache:
    """Test response cache manager."""
    
    @pytest.fixture
    def response_cache(self):
        """Create response cache instance."""
        return ResponseCache(cache_type="memory", ttl=3600)
    
    def test_cache_and_retrieve_response(self, response_cache):
        """Test caching and retrieving responses."""
        query = "What is our leave policy?"
        response = {"output": "Leave policy details...", "success": True}
        
        response_cache.cache_response(query, response)
        cached = response_cache.get_cached_response(query)
        
        assert cached is not None
        assert cached["output"] == response["output"]
    
    def test_key_generation_consistency(self, response_cache):
        """Test that key generation is consistent."""
        query1 = "What is our leave policy?"
        query2 = "what is our leave policy?"  # Different case
        
        response = {"output": "Leave policy details...", "success": True}
        response_cache.cache_response(query1, response)
        
        # Should retrieve with different casing
        cached = response_cache.get_cached_response(query2)
        assert cached is not None
    
    def test_invalidate(self, response_cache):
        """Test cache invalidation."""
        query = "What is our leave policy?"
        response = {"output": "Leave policy details...", "success": True}
        
        response_cache.cache_response(query, response)
        response_cache.invalidate(query)
        cached = response_cache.get_cached_response(query)
        
        assert cached is None


# ============================================================================
# Tool Tests (Mocked)
# ============================================================================

class TestHRRAGTool:
    """Test HR RAG tool functionality (mocked)."""
    
    @patch("hr_bot.tools.hr_rag_tool.HybridRAGRetriever")
    def test_hr_document_search_returns_string(self, mock_retriever):
        """Test that hr_document_search returns a string."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.search.return_value = [
            {"content": "Leave policy content", "metadata": {"source": "handbook.md"}}
        ]
        mock_retriever.return_value = mock_instance
        
        # Import after mocking
        from hr_bot.tools.hr_rag_tool import hr_document_search
        
        result = hr_document_search("leave policy")
        
        assert isinstance(result, str)


class TestFinanceSearchTool:
    """Test Finance search tool functionality (mocked)."""
    
    @patch("hr_bot.tools.finance_search_tool.FinanceWebSearcher")
    def test_finance_web_search_returns_string(self, mock_searcher):
        """Test that finance_web_search returns a string."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.search.return_value = [
            {"title": "What is 401k", "snippet": "401k explanation", "link": "http://example.com"}
        ]
        mock_searcher.return_value = mock_instance
        
        from hr_bot.tools.finance_search_tool import finance_web_search
        
        result = finance_web_search("what is 401k")
        
        assert isinstance(result, str)


# ============================================================================
# Integration Tests (Mocked)
# ============================================================================

class TestSubagentIntegration:
    """Test subagent integration (mocked)."""
    
    @patch("hr_bot.deep_agents.subagents.hr_subagent.ChatOpenAI")
    @patch("hr_bot.deep_agents.subagents.hr_subagent.create_openai_tools_agent")
    @patch("hr_bot.deep_agents.subagents.hr_subagent.AgentExecutor")
    def test_hr_subagent_invoke(
        self,
        mock_executor,
        mock_create_agent,
        mock_chat
    ):
        """Test HR subagent invoke method."""
        # Setup mocks
        mock_executor_instance = Mock()
        mock_executor_instance.invoke.return_value = {
            "output": "Here is the leave policy...",
            "intermediate_steps": []
        }
        mock_executor.return_value = mock_executor_instance
        
        from hr_bot.deep_agents.subagents.hr_subagent import HRSubagent
        
        agent = HRSubagent()
        result = agent.invoke("What is the leave policy?")
        
        assert result["success"] is True
        assert "output" in result
        assert result["agent"] == "hr_subagent"


class TestManagerAgentIntegration:
    """Test manager agent integration (mocked)."""
    
    @patch("hr_bot.deep_agents.manager.ChatOpenAI")
    @patch("hr_bot.deep_agents.manager.create_openai_tools_agent")
    @patch("hr_bot.deep_agents.manager.AgentExecutor")
    def test_manager_agent_invoke(
        self,
        mock_executor,
        mock_create_agent,
        mock_chat
    ):
        """Test manager agent invoke method."""
        # Setup mocks
        mock_executor_instance = Mock()
        mock_executor_instance.invoke.return_value = {
            "output": "Based on our HR policy...",
            "intermediate_steps": [
                (Mock(tool="delegate_to_hr_agent", tool_input="leave policy"), "HR response")
            ]
        }
        mock_executor.return_value = mock_executor_instance
        
        from hr_bot.deep_agents.manager import ManagerAgent
        
        manager = ManagerAgent()
        result = manager.invoke("What is our leave policy?")
        
        assert result["success"] is True
        assert "output" in result
        assert "delegations" in result


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
