"""
Deep Agents Subagents Module.

Contains specialized subagents:
- HR Subagent: Handles HR policy queries using RAG + MasterActions
- General Subagent: Handles any general knowledge queries using web search
"""

from hr_bot.deep_agents.subagents.hr_subagent import (
    HRSubagent,
    HRAgentState,
    create_hr_subagent_graph,
    get_hr_subagent
)
from hr_bot.deep_agents.subagents.general_subagent import (
    GeneralSubagent,
    GeneralAgentState,
    create_general_subagent_graph,
    get_general_subagent,
    # Backward compatibility aliases
    FinanceSubagent,
    FinanceAgentState,
    create_finance_subagent_graph,
    get_finance_subagent
)

__all__ = [
    # HR Subagent
    "HRSubagent",
    "HRAgentState",
    "create_hr_subagent_graph",
    "get_hr_subagent",
    # General Subagent (new)
    "GeneralSubagent",
    "GeneralAgentState",
    "create_general_subagent_graph",
    "get_general_subagent",
    # Backward compatibility (Finance aliases)
    "FinanceSubagent",
    "FinanceAgentState",
    "create_finance_subagent_graph",
    "get_finance_subagent"
]
