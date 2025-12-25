"""
General Subagent - Specialized agent for general knowledge queries using web search.

This subagent handles:
- Any questions NOT covered by company HR documents
- General knowledge and information queries
- Finance, market information and trends
- Current events and news
- Technical concepts and explanations
- Government regulations and guidelines

Uses:
- Serper API for web search
- Real-time information retrieval
- LangChain 1.1.2 create_agent API

Note: Company-specific HR POLICIES are handled by HR Subagent.
This subagent handles ALL external/generic questions.
"""

import logging
from typing import Optional, List, Dict, Any

import boto3
from langchain_core.messages import HumanMessage, AIMessage
from langchain_aws import ChatBedrock
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from hr_bot.config.settings import get_settings
from hr_bot.tools.finance_search_tool import finance_web_search

logger = logging.getLogger(__name__)
settings = get_settings()


def get_bedrock_client():
    """Create boto3 Bedrock client with explicit credentials."""
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )


class GeneralAgentState(TypedDict):
    """State for General subagent graph."""
    messages: List[Any]
    context: Optional[str]
    final_response: Optional[str]


class GeneralSubagent:
    """
    General Subagent using LangChain 1.1.2 create_agent pattern.
    
    Handles any general knowledge queries using web search.
    Uses Nova Lite model via Bedrock for fast, cost-effective inference.
    """
    
    # Primary model: Nova Lite
    NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.2,
        verbose: bool = False
    ):
        """
        Initialize General Subagent with Nova Lite model.
        
        Args:
            model_id: Bedrock model ID (defaults to Nova Lite)
            temperature: Model temperature for responses
            verbose: Enable verbose logging
        """
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize tools - web search for general queries
        self.tools = [finance_web_search]  # TODO: Rename tool to general_web_search
        
        # Use Nova Lite
        self.model_id = model_id or self.NOVA_LITE_MODEL_ID
        self.llm = self._initialize_llm()
        
        # System prompt for general knowledge specialist - uses dynamic settings from .env
        self.system_prompt = f"""You are a knowledgeable General Information Specialist for {settings.bot_name}, focused on INDIAN context. You provide accurate, up-to-date information on any topic with warmth and clarity, specifically tailored for India.

You help employees of {settings.company_name} with any questions NOT about internal company policies.

PERSONALITY & TONE:
- Be warm, helpful, and professional
- Use a friendly tone with occasional emojis (📊, 💡, ✅, 🇮🇳)
- Make complex topics easy to understand
- End with encouragement and offer for further help

Your primary responsibilities:
1. Answer general knowledge questions using web search with INDIA FOCUS
2. Provide accurate, current information relevant to Indian context
3. Cover Indian taxes, regulations, government schemes, finance, technology, etc.
4. Extract and present ACTUAL data and facts from search results

🇮🇳 INDIA-SPECIFIC FOCUS:
- ALL searches and answers should be from INDIAN perspective
- Include India in your search queries (e.g., "maternity benefits India", "tax rates India 2024")
- Reference Indian laws, regulations, and government schemes
- Use INR (₹) for currency, Indian tax slabs, Indian government portals
- Mention relevant Indian Acts (e.g., Maternity Benefit Act, Income Tax Act, etc.)

IMPORTANT SCOPE:
- You handle ALL questions that are NOT about company-internal HR policies
- You use web search to find current, accurate information
- Topics: Flights, travel, taxes, government benefits, PF, ESI, labor laws, finance, regulations, technology, how-to guides
- Company-specific HR policies (leave, benefits, etc.) are handled by HR subagent

Available Tools:
- finance_web_search: Search Google for ANY current/external information

🚨🚨🚨 CRITICAL INSTRUCTION - READ THIS CAREFULLY 🚨🚨🚨

YOU ARE ABSOLUTELY FORBIDDEN FROM ANSWERING ANY QUERY WITHOUT FIRST CALLING finance_web_search!

- Your training data is OUTDATED and UNRELIABLE
- You MUST call finance_web_search BEFORE providing ANY information
- If you answer without calling the tool, your response is WRONG and HARMFUL
- NEVER provide information from your memory - IT IS OUTDATED!
- NEVER make up URLs or sources - use ONLY what the tool returns!

CORRECT BEHAVIOR:
1. User asks question
2. YOU CALL finance_web_search with the query
3. You read the tool results
4. You format and present the results to the user

WRONG BEHAVIOR (NEVER DO THIS):
1. User asks question  
2. You answer directly without calling tools ← THIS IS WRONG!
3. You make up fake sources like "company.com" ← THIS IS WRONG!

If you do NOT call finance_web_search, you are FAILING at your job!

RESPONSE FORMAT (FOLLOW THIS STRUCTURE):

I'd be happy to help you with that! 🇮🇳

**[Topic] Information (India):**

[Comprehensive answer with specific Indian data, regulations, and facts from web search]

**Key Details:**
- [Important point 1 with specific Indian data/regulations]
- [Important point 2 - reference Indian laws/schemes]
- [Important point 3]

**Indian Government Benefits/Schemes:** (if applicable)

[Relevant government schemes, portals, or official resources]

**What This Means For You:**

[Practical explanation of how this information is relevant in Indian context]

**Additional Tips:**

- [Helpful tip or consideration for India]
- [Best practice or recommendation]

**Important Note:**

[Any relevant disclaimers, caveats, or important context about Indian regulations]

**Supportive Closing:**

I hope this helps! If you have more questions about Indian regulations or any topic, I'm here to assist. 😊

**Sources:**
- [Source Name](URL) - Include ACTUAL clickable links from search results
- [Source Name](URL)

HANDLING NO RESULTS:
If search returns NO_RELEVANT_RESULTS or SEARCH_ERROR or SEARCH_TIMEOUT:
- Acknowledge that you couldn't find specific information
- Provide general guidance based on your knowledge
- Suggest where they might find official information (government websites, etc.)
- Offer to help with a rephrased query
- Example response for no results:

"I wasn't able to find specific current information on this topic through web search. However, I can share some general guidance:

[General information based on common knowledge]

**Where to Find Official Information:**
- [Relevant government website](URL)
- [Official portal](URL)

I'd recommend checking these official sources for the most accurate and up-to-date information. Would you like me to try searching with different terms?"

MANDATORY REQUIREMENTS:
1. **ALWAYS add "India" to searches** - Ensure India-specific results
2. **Include Indian data** - Extract real values from Indian sources (₹, Indian laws, etc.)
3. **Warm tone** - Be friendly and supportive
4. **Indian context** - Explain what the information means in India
5. **CLICKABLE SOURCES** - Include actual URLs from search as markdown links: [Source Name](https://url.com)
6. **Be comprehensive** - Provide thorough, helpful answers for Indian audience
7. **PRESERVE URLs** - Copy the exact URLs from search results into your response as clickable markdown links"""

        # Create agent using LangChain 1.1.2 API
        self._create_agent()
        
        logger.info(f"General Subagent initialized with model: {self.model_id}")
    
    def _initialize_llm(self) -> ChatBedrock:
        """
        Initialize LLM.
        
        Returns:
            ChatBedrock instance
        """
        return ChatBedrock(
            model_id=self.model_id,
            client=get_bedrock_client(),
            model_kwargs={
                "temperature": self.temperature,
                "max_tokens": 4096
            }
        )
    
    def _create_agent(self) -> None:
        """Create the LangChain agent using create_agent API."""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            debug=self.verbose
        )
    
    def invoke(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Invoke General subagent with a query.
        
        Args:
            query: User's question
            chat_history: Optional conversation history
            
        Returns:
            Dict with output, intermediate_steps, and metadata
        """
        try:
            # Format messages
            messages = []
            if chat_history:
                for msg in chat_history:
                    if msg.get("role") == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    elif msg.get("role") == "assistant":
                        messages.append({"role": "assistant", "content": msg["content"]})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # Invoke agent
            result = self.agent.invoke({"messages": messages})
            
            # Extract response from messages
            output = ""
            if result.get("messages"):
                last_msg = result["messages"][-1]
                if hasattr(last_msg, "content"):
                    output = last_msg.content
                elif isinstance(last_msg, dict):
                    output = last_msg.get("content", "")
            
            return {
                "output": output,
                "tools_used": [],
                "success": True,
                "agent": "general_subagent"
            }
            
        except Exception as e:
            logger.error(f"General Subagent error: {str(e)}")
            return {
                "output": f"I encountered an error while processing your query: {str(e)}",
                "tools_used": [],
                "success": False,
                "error": str(e),
                "agent": "general_subagent"
            }
    
    async def ainvoke(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Async invoke General subagent with a query.
        
        Args:
            query: User's question
            chat_history: Optional conversation history
            
        Returns:
            Dict with output, intermediate_steps, and metadata
        """
        try:
            # Format messages
            messages = []
            if chat_history:
                for msg in chat_history:
                    if msg.get("role") == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    elif msg.get("role") == "assistant":
                        messages.append({"role": "assistant", "content": msg["content"]})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # Async invoke
            result = await self.agent.ainvoke({"messages": messages})
            
            # Extract response from messages
            output = ""
            if result.get("messages"):
                last_msg = result["messages"][-1]
                if hasattr(last_msg, "content"):
                    output = last_msg.content
                elif isinstance(last_msg, dict):
                    output = last_msg.get("content", "")
            
            return {
                "output": output,
                "tools_used": [],
                "success": True,
                "agent": "general_subagent"
            }
            
        except Exception as e:
            logger.error(f"General Subagent async error: {str(e)}")
            return {
                "output": f"I encountered an error while processing your query: {str(e)}",
                "tools_used": [],
                "success": False,
                "error": str(e),
                "agent": "general_subagent"
            }


def create_general_subagent_graph() -> StateGraph:
    """
    Create a LangGraph StateGraph for General subagent.
    
    Returns:
        Compiled StateGraph for General subagent
    """
    general_agent = GeneralSubagent()
    
    def process_query(state: GeneralAgentState) -> GeneralAgentState:
        """Process query using the agent."""
        messages = state.get("messages", [])
        
        # Extract latest user message
        query = ""
        chat_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                query = msg.content
            elif isinstance(msg, AIMessage):
                chat_history.append({"role": "assistant", "content": msg.content})
        
        # Invoke agent
        result = general_agent.invoke(query, chat_history if chat_history else None)
        
        # Update state
        state["final_response"] = result["output"]
        state["messages"] = messages + [AIMessage(content=result["output"])]
        
        return state
    
    def should_end(state: GeneralAgentState) -> str:
        """Determine if processing should end."""
        if state.get("final_response"):
            return END
        return "process"
    
    # Build graph
    workflow = StateGraph(GeneralAgentState)
    workflow.add_node("process", process_query)
    workflow.set_entry_point("process")
    workflow.add_conditional_edges("process", should_end)
    
    return workflow.compile()


# Singleton instance
_general_subagent_instance: Optional[GeneralSubagent] = None


def get_general_subagent() -> GeneralSubagent:
    """Get or create singleton General subagent instance."""
    global _general_subagent_instance
    if _general_subagent_instance is None:
        _general_subagent_instance = GeneralSubagent()
    return _general_subagent_instance


# Backward compatibility aliases
FinanceSubagent = GeneralSubagent
FinanceAgentState = GeneralAgentState
get_finance_subagent = get_general_subagent
create_finance_subagent_graph = create_general_subagent_graph


# Export for manager integration
__all__ = [
    "GeneralSubagent",
    "GeneralAgentState", 
    "create_general_subagent_graph",
    "get_general_subagent",
    # Backward compatibility
    "FinanceSubagent",
    "FinanceAgentState",
    "get_finance_subagent",
    "create_finance_subagent_graph",
]
