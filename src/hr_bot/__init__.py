"""
HR Bot V2 - Deep Agents Architecture
=====================================

Production-ready HR assistant with Manager + Subagent orchestration.

Architecture:
- Manager Agent (Nova Lite): Routes queries, orchestrates subagents
- HR Subagent (Nova Lite): RAG search + Master Actions for HR policies
- General Subagent (Nova Lite): Web search for any external queries

Built on LangChain + LangGraph for durable execution and observability.
"""

__version__ = "2.1.0"
__author__ = "HR Bot Team"

from hr_bot.deep_agents.manager import ManagerAgent
from hr_bot.deep_agents.subagents.hr_subagent import HRSubagent
from hr_bot.deep_agents.subagents.general_subagent import GeneralSubagent

# Backward compatibility
FinanceSubagent = GeneralSubagent

__all__ = [
    "ManagerAgent",
    "HRSubagent", 
    "GeneralSubagent",
    "FinanceSubagent",  # Backward compatibility alias
    "__version__",
]
