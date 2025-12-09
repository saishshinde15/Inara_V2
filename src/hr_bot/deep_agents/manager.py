"""
Manager Agent - Deep Agent orchestrating HR and Finance subagents.

Uses AWS Nova Lite model to intelligently route queries to appropriate
subagents and synthesize responses.

Architecture:
- Manager Agent (AWS Nova Lite) with SubAgentMiddleware
- HR Subagent (RAG + MasterActions)
- Finance Subagent (Web Search)

Features:
- Content safety filtering (query + response)
- Semantic response caching
- Response post-processing for professional output
- LangChain 1.1.2 create_agent API
- Persistent conversation memory with SQLite checkpointer
- Semantic memory for cross-conversation context
"""

import asyncio
import logging
import os
import uuid
from typing import Optional, List, Dict, Any, Callable
from functools import lru_cache

import boto3
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from hr_bot.config.settings import get_settings
from hr_bot.deep_agents.subagents import (
    get_hr_subagent,
    get_general_subagent,
    HRSubagent,
    GeneralSubagent
)
from hr_bot.utils.content_safety import (
    ContentSafetyFilter,
    get_safety_filter,
    is_query_safe
)
from hr_bot.utils.response_formatter import (
    remove_document_evidence_section,
    clean_internal_references,
    format_professional_response,
    validate_response_quality
)
from hr_bot.utils.cache import get_response_cache
from hr_bot.utils.memory import get_memory_manager, get_checkpointer

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


# ============================================================================
# Subagent Tools for Manager
# ============================================================================

@tool
def delegate_to_hr_agent(query: str) -> str:
    """
    Delegate HR-related queries to the HR specialist subagent.
    
    Use this tool for questions about:
    - Company HR policies (leave, benefits, compensation)
    - Employee handbook and guidelines
    - Performance reviews and feedback processes
    - Onboarding and offboarding procedures
    - Workplace conduct and compliance
    - Any company-internal HR matters
    
    Args:
        query: The HR-related question to delegate
        
    Returns:
        HR subagent's response with policy information
    """
    try:
        hr_agent = get_hr_subagent()
        result = hr_agent.invoke(query)
        
        if result.get("success"):
            response = result["output"]
            tools_used = result.get("tools_used", [])
            
            # Add metadata about tools used
            if tools_used:
                tool_names = [t["tool"] for t in tools_used]
                response += f"\n\n[Sources consulted: {', '.join(tool_names)}]"
            
            return response
        else:
            return f"HR Agent encountered an issue: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error delegating to HR agent: {str(e)}")
        return f"Unable to process HR query: {str(e)}"


@tool
def delegate_to_general_agent(query: str) -> str:
    """
    Delegate general knowledge queries to the General subagent.
    
    Use this tool for ANY question that is NOT about internal company HR policies:
    - General knowledge questions
    - Current events and news
    - Finance, tax, and economic information
    - Technology, science, health topics
    - External organization policies (NGOs, government, etc.)
    - Any topic requiring web search
    
    This agent uses web search to find up-to-date information.
    
    Args:
        query: The question to search for
        
    Returns:
        General subagent's response with web search results
    """
    try:
        general_agent = get_general_subagent()
        result = general_agent.invoke(query)
        
        if result.get("success"):
            response = result["output"]
            sources = result.get("sources", [])
            
            # Add source attribution
            if sources:
                response += "\n\n[Information sourced from web search]"
            
            return response
        else:
            return f"General Agent encountered an issue: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error delegating to General agent: {str(e)}")
        return f"Unable to process general query: {str(e)}"


@tool
def delegate_to_both_agents(hr_query: str, general_query: str) -> str:
    """
    Delegate to BOTH HR and General agents IN PARALLEL for multi-domain queries.
    
    Use this tool when a question requires information from BOTH:
    - Internal company HR policies AND
    - External/general knowledge (web search)
    
    This runs both agents concurrently for faster response times.
    
    Examples of when to use this:
    - "What is our leave policy and what are general tax rates in India?" 
    - "Tell me about company onboarding and what tax I need to pay on my salary"
    - Questions combining internal HR policies with external information
    
    Args:
        hr_query: The internal HR policy portion of the question
        general_query: The general/external knowledge portion of the question
        
    Returns:
        Combined response from both agents
    """
    import concurrent.futures
    
    def call_hr():
        try:
            hr_agent = get_hr_subagent()
            result = hr_agent.invoke(hr_query)
            if result.get("success"):
                return ("hr", result["output"])
            return ("hr", f"HR Agent issue: {result.get('error', 'Unknown')}")
        except Exception as e:
            return ("hr", f"HR error: {str(e)}")
    
    def call_general():
        try:
            general_agent = get_general_subagent()
            result = general_agent.invoke(general_query)
            if result.get("success"):
                return ("general", result["output"])
            return ("general", f"General Agent issue: {result.get('error', 'Unknown')}")
        except Exception as e:
            return ("general", f"General error: {str(e)}")
    
    # Run both agents in parallel
    hr_response = ""
    general_response = ""
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_hr), executor.submit(call_general)]
        
        for future in concurrent.futures.as_completed(futures, timeout=20):
            try:
                agent_type, response = future.result()
                if agent_type == "hr":
                    hr_response = response
                else:
                    general_response = response
            except Exception as e:
                logger.error(f"Parallel agent error: {e}")
    
    # Combine responses
    combined = ""
    if hr_response:
        combined += f"**HR Policy Information:**\n{hr_response}\n\n"
    if general_response:
        combined += f"**General Information:**\n{general_response}"
    
    return combined.strip() if combined else "Could not retrieve information from either agent."


# ============================================================================
# Manager Agent State
# ============================================================================

class ManagerAgentState(TypedDict):
    """State for Manager agent graph."""
    messages: List[Any]
    chat_history: List[Dict[str, str]]
    current_query: str
    delegated_to: Optional[str]
    final_response: Optional[str]
    metadata: Dict[str, Any]


# ============================================================================
# Manager Agent Class
# ============================================================================

class ManagerAgent:
    """
    Manager Deep Agent orchestrating specialized subagents.
    
    Uses AWS Nova lite model to:
    1. Understand user intent
    2. Route to appropriate subagent (HR or Finance)
    3. Handle multi-domain queries
    4. Synthesize responses
    
    Uses LangChain 1.1.2 create_agent API.
    """
    
    # AWS Nova Lite model identifier for Bedrock
    MANAGER_MODEL = "amazon.nova-lite-v1:0"  # Using AWS Nova Lite for consistency with subagents
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize Manager Agent.
        
        Args:
            model_name: Override model (defaults to AWS Nova Lite)
            temperature: Model temperature
            verbose: Enable verbose logging
        """
        # Use AWS Nova Lite as specified, with fallback
        self.model_name = model_name or self.MANAGER_MODEL
        self.temperature = temperature
        self.verbose = verbose
        
        # Subagent delegation tools (including parallel execution tool)
        self.tools = [
            delegate_to_hr_agent,
            delegate_to_general_agent,
            delegate_to_both_agents  # Parallel execution for multi-domain queries
        ]
        
        # Initialize manager LLM (AWS Nova Lite via Bedrock)
        self.llm = ChatBedrock(
            model_id=self.model_name,
            client=get_bedrock_client(),
            model_kwargs={
                "temperature": self.temperature,
                "max_tokens": 2000  # Reduced for speed
            }
        )
        
        # Manager system prompt - uses dynamic settings from .env
        self.system_prompt = f"""You are the Manager Agent for {settings.bot_name}, a Universal Enterprise Assistant that helps users with a wide range of questions.

IMPORTANT CONTEXT:
- {settings.bot_name} is the NAME OF THE PRODUCT (Universal Enterprise Assistant), not a company name
- You help employees of {settings.company_name}
- You coordinate specialized subagents to answer user questions
- {settings.bot_name} can help with BOTH internal company matters AND general knowledge questions
- For HR queries, employees can contact: {settings.hr_email}

Your Role:
You are the intelligent router that:
1. Understands the user's question
2. Routes it to the appropriate specialist subagent
3. Presents comprehensive responses

CONVERSATION MEMORY:
- You have access to the conversation history
- When users provide additional context, USE that information
- Do NOT repeatedly ask for the same information
- Build on previous exchanges to provide complete answers

Available Specialist Subagents:

1. **HR Subagent** (delegate_to_hr_agent):
   - Internal company HR policies and procedures
   - Leave policies, benefits, compensation
   - Employee handbook queries
   - Performance management
   - Onboarding/offboarding
   - Company-internal finance policies (expense reports, reimbursements)
   - Workplace conduct
   - Any question about YOUR COMPANY'S internal policies

2. **General Subagent** (delegate_to_general_agent):
   - ANY question NOT about internal company HR policies
   - General knowledge questions
   - External organization policies (NGOs, government, other companies)
   - Finance, tax, investment information
   - Current events and news
   - Technology, science, health topics
   - Market trends and economic data
   - Uses WEB SEARCH to find up-to-date information

3. **PARALLEL Mode** (delegate_to_both_agents) - USE FOR MULTI-DOMAIN QUERIES:
   - When query needs BOTH internal HR policies AND external/general information
   - Runs both subagents IN PARALLEL for faster response
   - Example: "What is our leave policy and what are general tax rates in India?"
   - Example: "Tell me about company onboarding and what tax I pay on 20 LPA"

Routing Guidelines:
- Internal company HR policies → delegate_to_hr_agent
- EVERYTHING ELSE (general knowledge, external info, web search) → delegate_to_general_agent
- **MULTI-DOMAIN queries (internal + external)** → delegate_to_both_agents (FASTER!)
- If unsure whether it's internal HR → use delegate_to_general_agent (it can search the web)

CRITICAL: ALWAYS USE A SUBAGENT
- Do NOT answer questions from your own knowledge
- ALWAYS delegate to either HR or General subagent
- The General subagent can handle ANY non-HR question via web search

RESPONSE REQUIREMENTS:
1. **ALWAYS delegate** - never answer directly without using a subagent
2. **DETAILED responses** - provide comprehensive information
3. **CITE SOURCES** - include source documents or web sources
4. **STRUCTURED format** - use bullet points, numbered lists, clear sections

Remember: You coordinate and route - the subagents have the expertise and tools. ALWAYS delegate!"""

        # Create agent using LangChain 1.1.2 API with checkpointer for memory
        self._create_agent()
        
        # Conversation memory - now using proper checkpointer
        self.conversation_history: List[Dict[str, str]] = []
        self.memory_manager = get_memory_manager()
        
        # Default thread ID (can be overridden per conversation)
        self.current_thread_id: Optional[str] = None
        
        logger.info(f"Manager Agent initialized with model: {self.model_name}")
    
    def _create_agent(self) -> None:
        """Create the LangChain agent using create_agent API with checkpointer."""
        # Get SQLite checkpointer for persistent memory
        try:
            checkpointer = get_checkpointer()
        except Exception as e:
            logger.warning(f"Could not initialize SQLite checkpointer: {e}, using in-memory")
            checkpointer = MemorySaver()
        
        # Create agent using new LangChain 1.1.2 API with checkpointer
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=checkpointer,
            debug=self.verbose
        )
        
        self.checkpointer = checkpointer
    
    def set_thread(self, thread_id: str) -> None:
        """Set the current conversation thread."""
        self.current_thread_id = thread_id
        logger.info(f"Set conversation thread: {thread_id}")
    
    def new_thread(self) -> str:
        """Create a new conversation thread."""
        self.current_thread_id = str(uuid.uuid4())
        self.conversation_history.clear()
        logger.info(f"Created new conversation thread: {self.current_thread_id}")
        return self.current_thread_id
    
    def invoke(
        self,
        query: str,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Invoke Manager agent with a query.
        
        Args:
            query: User's question
            thread_id: Conversation thread ID for memory persistence
            session_id: Optional session ID for tracking (legacy)
            include_history: Whether to include conversation history
            
        Returns:
            Dict with output, metadata, and conversation info
        """
        # Use provided thread_id or current thread or create new one
        effective_thread_id = thread_id or self.current_thread_id or str(uuid.uuid4())
        if not self.current_thread_id:
            self.current_thread_id = effective_thread_id
        
        try:
            # =============================================
            # STEP 1: Content Safety Check on Input
            # =============================================
            safety_filter = get_safety_filter()
            safety_check = safety_filter.check_input(query)
            
            if not safety_check.is_safe:
                logger.warning(f"Blocked unsafe query: {safety_check.reason}")
                return {
                    "output": f"I'm sorry, but I can't help with that request. {safety_check.reason}",
                    "delegations": [],
                    "thread_id": effective_thread_id,
                    "session_id": session_id,
                    "success": False,
                    "blocked": True,
                    "safety_category": safety_check.category.value if safety_check.category else None
                }
            
            # =============================================
            # STEP 2: Get Memory Context
            # =============================================
            memory_context = self.memory_manager.get_context_for_query(
                query=query,
                thread_id=effective_thread_id
            )
            
            # Build context string from memory
            context_parts = []
            if memory_context.get("summary"):
                context_parts.append(f"Previous conversation summary: {memory_context['summary']}")
            
            if memory_context.get("relevant_memories"):
                facts = [m["content"] for m in memory_context["relevant_memories"][:3]]
                if facts:
                    context_parts.append(f"Relevant context from earlier: {'; '.join(facts)}")
            
            memory_context_str = "\n".join(context_parts) if context_parts else ""
            
            # =============================================
            # STEP 3: Check Response Cache (disabled for conversations with memory)
            # =============================================
            # Skip cache if we have conversation context to ensure continuity
            cache = get_response_cache()
            if not memory_context_str:
                cached = cache.get_cached_response(query)
                
                if cached:
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    cached_response = cached.get("response", "")
                    return {
                        "output": cached_response,
                        "delegations": [],
                        "thread_id": effective_thread_id,
                        "session_id": session_id,
                        "success": True,
                        "cached": True,
                        "metadata": {
                            "model": self.model_name,
                            "from_cache": True
                        }
                    }
            
            # =============================================
            # STEP 4: Format messages with memory context
            # =============================================
            messages = []
            
            # Add memory context as system message if available
            if memory_context_str:
                messages.append({
                    "role": "system",
                    "content": f"CONVERSATION CONTEXT (use this to maintain continuity):\n{memory_context_str}"
                })
            
            # Add conversation history
            if include_history and self.conversation_history:
                for msg in self.conversation_history[-10:]:  # Last 10 messages
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # =============================================
            # STEP 5: Invoke agent with thread config for checkpointer
            # =============================================
            config = {"configurable": {"thread_id": effective_thread_id}}
            result = self.agent.invoke({"messages": messages}, config=config)
            
            # Extract response from messages
            output = ""
            delegations = []
            if result.get("messages"):
                last_msg = result["messages"][-1]
                if hasattr(last_msg, "content"):
                    output = last_msg.content
                elif isinstance(last_msg, dict):
                    output = last_msg.get("content", "")
            
            # =============================================
            # STEP 6: Post-process Response
            # =============================================
            # Remove document evidence sections (internal references)
            output = remove_document_evidence_section(output)
            
            # Clean internal references and format professionally
            output = clean_internal_references(output)
            output = format_professional_response(output)
            
            # =============================================
            # STEP 6: Content Safety Check on Response
            # =============================================
            response_safety = safety_filter.check_output(output)
            if not response_safety.is_safe:
                logger.warning(f"Blocked unsafe response: {response_safety.reason}")
                output = "I apologize, but I encountered an issue generating a safe response. Please rephrase your question."
            
            # =============================================
            # STEP 7: Validate Response Quality
            # =============================================
            validation_result = validate_response_quality(output)
            is_valid = validation_result.get("valid", True)
            if not is_valid:
                logger.warning(f"Response quality issues: {validation_result.get('reason', 'Unknown')}")
                # Still return the response but log issues
            
            # =============================================
            # STEP 8: Cache the Response
            # =============================================
            if output and response_safety.is_safe:
                cache.cache_response(query, {"response": output})
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": output})
            
            # =============================================
            # STEP 9: Store in Persistent Memory
            # =============================================
            # Store conversation turn in semantic memory for cross-conversation retrieval
            if output and response_safety.is_safe:
                try:
                    self.memory_manager.add_memory(
                        thread_id=effective_thread_id,
                        content=f"User asked: {query}\nAssistant answered: {output[:500]}...",
                        memory_type="conversation",
                        metadata={
                            "query": query,
                            "response_preview": output[:200] if len(output) > 200 else output,
                            "delegations": [d["agent"] for d in delegations] if delegations else []
                        }
                    )
                    logger.debug(f"Stored conversation turn in memory for thread {effective_thread_id}")
                except Exception as mem_error:
                    logger.warning(f"Could not store to memory: {mem_error}")
            
            return {
                "output": output,
                "delegations": delegations,
                "thread_id": effective_thread_id,
                "session_id": session_id,
                "success": True,
                "metadata": {
                    "model": self.model_name,
                    "agents_consulted": [d["agent"] for d in delegations] if delegations else [],
                    "response_quality_valid": is_valid,
                    "memory_enabled": True
                }
            }
            
        except Exception as e:
            logger.error(f"Manager Agent error: {str(e)}")
            return {
                "output": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "delegations": [],
                "thread_id": effective_thread_id if 'effective_thread_id' in dir() else None,
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }
    
    async def ainvoke(
        self,
        query: str,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Async invoke Manager agent with a query.
        
        Args:
            query: User's question
            thread_id: Conversation thread ID for memory persistence
            session_id: Optional session ID for tracking (legacy)
            include_history: Whether to include conversation history
            
        Returns:
            Dict with output, metadata, and conversation info
        """
        # Use provided thread_id or current thread or create new one
        effective_thread_id = thread_id or self.current_thread_id or str(uuid.uuid4())
        if not self.current_thread_id:
            self.current_thread_id = effective_thread_id
            
        try:
            # =============================================
            # STEP 1: Content Safety Check on Input
            # =============================================
            safety_filter = get_safety_filter()
            safety_check = safety_filter.check_input(query)
            
            if not safety_check.is_safe:
                logger.warning(f"Blocked unsafe query: {safety_check.reason}")
                return {
                    "output": f"I'm sorry, but I can't help with that request. {safety_check.reason}",
                    "delegations": [],
                    "thread_id": effective_thread_id,
                    "session_id": session_id,
                    "success": False,
                    "blocked": True,
                    "safety_category": safety_check.category.value if safety_check.category else None
                }
            
            # =============================================
            # STEP 2: Get Memory Context
            # =============================================
            memory_context = self.memory_manager.get_context_for_query(
                query=query,
                thread_id=effective_thread_id
            )
            
            # Build context string from memory
            context_parts = []
            if memory_context.get("summary"):
                context_parts.append(f"Previous conversation summary: {memory_context['summary']}")
            
            if memory_context.get("relevant_memories"):
                facts = [m["content"] for m in memory_context["relevant_memories"][:3]]
                if facts:
                    context_parts.append(f"Relevant context from earlier: {'; '.join(facts)}")
            
            memory_context_str = "\n".join(context_parts) if context_parts else ""
            
            # =============================================
            # STEP 3: Check Response Cache (disabled for conversations with memory)
            # =============================================
            cache = get_response_cache()
            if not memory_context_str:
                cached = cache.get_cached_response(query)
                
                if cached:
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    cached_response = cached.get("response", "")
                    return {
                        "output": cached_response,
                        "delegations": [],
                        "thread_id": effective_thread_id,
                        "session_id": session_id,
                        "success": True,
                        "cached": True,
                        "metadata": {
                            "model": self.model_name,
                            "from_cache": True
                        }
                    }
            
            # =============================================
            # STEP 4: Format messages with memory context
            # =============================================
            messages = []
            
            # Add memory context as system message if available
            if memory_context_str:
                messages.append({
                    "role": "system",
                    "content": f"CONVERSATION CONTEXT (use this to maintain continuity):\n{memory_context_str}"
                })
            
            if include_history and self.conversation_history:
                for msg in self.conversation_history[-10:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # =============================================
            # STEP 5: Async invoke agent with thread config for checkpointer
            # =============================================
            config = {"configurable": {"thread_id": effective_thread_id}}
            result = await self.agent.ainvoke({"messages": messages}, config=config)
            
            # Extract response from messages
            output = ""
            delegations = []
            if result.get("messages"):
                last_msg = result["messages"][-1]
                if hasattr(last_msg, "content"):
                    output = last_msg.content
                elif isinstance(last_msg, dict):
                    output = last_msg.get("content", "")
            
            # =============================================
            # STEP 5: Post-process Response
            # =============================================
            output = remove_document_evidence_section(output)
            output = clean_internal_references(output)
            output = format_professional_response(output)
            
            # =============================================
            # STEP 6: Content Safety Check on Response
            # =============================================
            response_safety = safety_filter.check_output(output)
            if not response_safety.is_safe:
                logger.warning(f"Blocked unsafe response: {response_safety.reason}")
                output = "I apologize, but I encountered an issue generating a safe response. Please rephrase your question."
            
            # =============================================
            # STEP 7: Validate Response Quality
            # =============================================
            validation_result = validate_response_quality(output)
            is_valid = validation_result.get("valid", True)
            if not is_valid:
                logger.warning(f"Response quality issues: {validation_result.get('reason', 'Unknown')}")
            
            # =============================================
            # STEP 8: Cache the Response
            # =============================================
            if output and response_safety.is_safe:
                cache.cache_response(query, {"response": output})
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": output})
            
            # =============================================
            # STEP 9: Store in Persistent Memory
            # =============================================
            if output and response_safety.is_safe:
                try:
                    self.memory_manager.add_memory(
                        thread_id=effective_thread_id,
                        content=f"User asked: {query}\nAssistant answered: {output[:500]}...",
                        memory_type="conversation",
                        metadata={
                            "query": query,
                            "response_preview": output[:200] if len(output) > 200 else output,
                            "delegations": [d["agent"] for d in delegations] if delegations else []
                        }
                    )
                    logger.debug(f"Stored conversation turn in memory for thread {effective_thread_id}")
                except Exception as mem_error:
                    logger.warning(f"Could not store to memory: {mem_error}")
            
            return {
                "output": output,
                "delegations": delegations,
                "thread_id": effective_thread_id,
                "session_id": session_id,
                "success": True,
                "metadata": {
                    "model": self.model_name,
                    "agents_consulted": [d["agent"] for d in delegations] if delegations else [],
                    "response_quality_valid": is_valid,
                    "memory_enabled": True
                }
            }
            
        except Exception as e:
            logger.error(f"Manager Agent async error: {str(e)}")
            return {
                "output": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "delegations": [],
                "thread_id": effective_thread_id if 'effective_thread_id' in dir() else None,
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
        self.current_thread_id = None
        logger.info("Conversation history and thread reset")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.conversation_history.copy()
    
    def get_thread_id(self) -> Optional[str]:
        """Get current thread ID."""
        return self.current_thread_id


# ============================================================================
# LangGraph Manager Implementation
# ============================================================================

def create_manager_graph() -> StateGraph:
    """
    Create a LangGraph StateGraph for the Manager agent.
    
    This provides a more sophisticated control flow with:
    - Intent classification
    - Multi-agent routing
    - Response synthesis
    
    Returns:
        Compiled StateGraph for Manager agent
    """
    manager = ManagerAgent()
    
    def classify_intent(state: ManagerAgentState) -> ManagerAgentState:
        """Classify the user's intent and determine routing."""
        query = state.get("current_query", "")
        
        # Simple keyword-based classification (manager LLM does sophisticated routing)
        hr_keywords = ["policy", "leave", "benefit", "hr", "employee", "handbook", 
                       "onboarding", "performance", "expense", "reimbursement"]
        finance_keywords = ["invest", "market", "stock", "tax", "401k", "finance",
                           "economy", "interest rate", "inflation"]
        
        query_lower = query.lower()
        
        hr_score = sum(1 for kw in hr_keywords if kw in query_lower)
        finance_score = sum(1 for kw in finance_keywords if kw in query_lower)
        
        if hr_score > finance_score:
            state["delegated_to"] = "hr"
        elif finance_score > hr_score:
            state["delegated_to"] = "finance"
        else:
            state["delegated_to"] = "manager"  # Let manager decide
        
        return state
    
    def process_with_manager(state: ManagerAgentState) -> ManagerAgentState:
        """Process query using the full manager agent."""
        query = state.get("current_query", "")
        
        result = manager.invoke(query)
        state["final_response"] = result["output"]
        state["metadata"] = result.get("metadata", {})
        
        return state
    
    def should_end(state: ManagerAgentState) -> str:
        """Determine if processing should end."""
        if state.get("final_response"):
            return END
        return "process"
    
    # Build graph
    workflow = StateGraph(ManagerAgentState)
    
    # Add nodes
    workflow.add_node("classify", classify_intent)
    workflow.add_node("process", process_with_manager)
    
    # Add edges
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "process")
    workflow.add_conditional_edges("process", should_end)
    
    return workflow.compile()


# ============================================================================
# Singleton and Factory
# ============================================================================

_manager_instance: Optional[ManagerAgent] = None


def get_manager_agent(reset: bool = False) -> ManagerAgent:
    """
    Get or create singleton Manager agent instance.
    
    Args:
        reset: Force create new instance
        
    Returns:
        ManagerAgent instance
    """
    global _manager_instance
    if _manager_instance is None or reset:
        _manager_instance = ManagerAgent()
    return _manager_instance


def create_enterprise_assistant(
    model_name: Optional[str] = None,
    verbose: bool = False
) -> ManagerAgent:
    """
    Factory function to create a configured enterprise assistant.
    
    Args:
        model_name: Optional model override
        verbose: Enable verbose logging
        
    Returns:
        Configured ManagerAgent
    """
    return ManagerAgent(
        model_name=model_name,
        verbose=verbose
    )


# Export
__all__ = [
    "ManagerAgent",
    "ManagerAgentState",
    "create_manager_graph",
    "get_manager_agent",
    "create_enterprise_assistant",
    "delegate_to_hr_agent",
    "delegate_to_finance_agent",
    "delegate_to_both_agents"
]
