"""
Deep Agents Module for HR Bot V2.

Enterprise Assistant architecture with:
- Manager Agent (GPT-5 Nano): Intelligent query routing and response synthesis
- HR Subagent: HR policies, procedures, and company handbook
- Finance Subagent: General finance questions via web search
"""

from hr_bot.deep_agents.manager import (
    ManagerAgent,
    ManagerAgentState,
    create_manager_graph,
    get_manager_agent,
    create_enterprise_assistant
)
from hr_bot.deep_agents.subagents import (
    HRSubagent,
    FinanceSubagent,
    get_hr_subagent,
    get_finance_subagent
)

__all__ = [
    # Manager
    "ManagerAgent",
    "ManagerAgentState",
    "create_manager_graph",
    "get_manager_agent",
    "create_enterprise_assistant",
    # Subagents
    "HRSubagent",
    "FinanceSubagent",
    "get_hr_subagent",
    "get_finance_subagent"
]
