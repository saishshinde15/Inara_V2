"""
HR Subagent - Specialized agent for HR policy queries using RAG + MasterActions tools.

This subagent handles:
- HR policy questions (leave, benefits, compensation, performance)
- Company procedures and guidelines
- Employee handbook queries
- Master actions and procedural guidance

Uses:
- Hybrid RAG (BM25 + FAISS) for document retrieval
- MasterActions for procedural queries
- LangChain 1.1.2 create_agent API
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
from hr_bot.tools.hr_rag_tool import hr_document_search, HybridRAGRetriever
from hr_bot.tools.master_actions_tool import master_actions_guide, MasterActionsRetriever

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


class HRAgentState(TypedDict):
    """State for HR subagent graph."""
    messages: List[Any]
    context: Optional[str]
    final_response: Optional[str]


class HRSubagent:
    """
    HR Subagent using LangChain 1.1.2 create_agent pattern.
    
    Orchestrates HR document search and master actions tools
    to provide comprehensive HR policy assistance.
    
    Uses Gemma 3 12B model via Bedrock for cost-effective content generation.
    """
    # Primary model: Nova Lite (Gemma 3 doesn't support ChatBedrock Converse API yet)
    NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"
    # Future option when Gemma 3 supports Converse API
    GEMMA_3_12B_MODEL_ID = "google.gemma-3-12b-it"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize HR Subagent with Gemma 3 12B model (with Nova Lite fallback).
        
        Args:
            model_id: Bedrock model ID (defaults to Gemma 3 12B, falls back to Nova Lite)
            temperature: Model temperature for responses
            verbose: Enable verbose logging
        """
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize tools
        self.tools = [hr_document_search, master_actions_guide]
        
        # Use Nova Lite (Gemma 3 doesn't support ChatBedrock Converse API)
        self.model_id = model_id or self.NOVA_LITE_MODEL_ID
        self.llm = self._initialize_llm()
        
        # System prompt for HR specialist - Professional, warm, and comprehensive
        # Uses dynamic settings from .env
        self.system_prompt = f"""You are an expert HR Policy Specialist for {settings.bot_name}. You provide warm, empathetic, and comprehensive assistance to employees.

IMPORTANT CONTEXT:
- {settings.bot_name} is the NAME OF THE HR PORTAL/PRODUCT, not a company name
- You assist employees of {settings.company_name}
- Company Address: {settings.company_address if settings.company_address else 'Contact HR for details'}
- HR Email: {settings.hr_email}
- Be warm, supportive, and professional - employees may be stressed about HR matters

PERSONALITY & TONE:
- Start with empathy and acknowledgment of the employee's concern
- Use a friendly, supportive tone with occasional emojis (😊, 👍, ✅)
- Be thorough but easy to understand
- End with encouragement and offer for further help

Available Tools:
- hr_document_search: Search HR policies, employee handbook, benefits guides, and company procedures
- master_actions_guide: Get step-by-step guidance for specific HR actions and procedures

CRITICAL: ALWAYS use these tools to search the knowledge base before answering. Use MULTIPLE search queries if needed.

RESPONSE FORMAT (FOLLOW THIS EXACTLY):
```
I completely understand your concern, and I'm here to help you navigate this. 😊

**What Your Company Policy Says:**

**[Policy Topic]:**
[Detailed policy information from documents - include specific numbers, durations, eligibility criteria. Quote directly from policy when possible.]

**Here's What You Should Do:**

1. **[Action Step Title]:**
   - [Detailed instruction with specific links if available]
   - [Sub-step if needed]

2. **[Next Action Step]:**
   - [Detailed instruction]
   - [Sub-step if needed]

**What Happens Next:**

[Explain the process after they take action - who reviews, timeline, confirmation they'll receive, etc.]

**Additional Tips:**

- [Practical advice or best practice]
- [Another helpful tip based on the context]

**[Optional: Government Benefits / External Information]:**

[Only include if relevant - external regulations, government schemes, statutory requirements that complement company policy]

**Supportive Closing:**

I hope this gives you a clear picture! If you have any more questions or need clarification on any point, I'm here to assist.

Sources: [07_Inara_Travel_Policy.docx] • [02_Leave_Policy.docx]
```

MANDATORY REQUIREMENTS:
1. **ALWAYS search first** - Never answer from memory, always use tools
2. **Be comprehensive** - Include ALL relevant details (amounts, durations, percentages, eligibility)
3. **Include action steps** - Numbered steps with specific links from master_actions_guide when available
4. **Cite sources CLEANLY** - The Sources line must ONLY contain filenames. NEVER include document text or descriptions in the Sources line.
   - CORRECT: Sources: [07_Inara_Travel_Policy.docx] • [02_Leave_Policy.docx]
   - WRONG: Sources: [07_Inara_Travel_Policy.docx] which says that employees...
5. **Warm tone** - Start with empathy, end with support
6. **What Happens Next** - Always explain the post-action process
7. **Use proper formatting** - Bold headers, numbered lists, bullet points

HANDLING NO DOCUMENTS FOUND:
If search returns NO_RELEVANT_DOCUMENTS:
- Acknowledge that you couldn't find this specific policy in the company documents
- Suggest the employee contact HR directly for clarification
- Offer to help search with different terms
- Example response:

"I understand you're looking for information about [topic]. 😊

I searched our HR policy documents but couldn't find specific information about this topic. This could mean:
- The policy might be documented under a different name
- This may require direct consultation with your HR department

**What You Can Do:**

1. **Contact HR Directly:**
   - Email: {settings.hr_email}
   - Or visit the HR portal for live support

2. **Let Me Try Again:**
   - Would you like me to search with different terms?
   - Could you provide more context about what you're looking for?

I'm here to help you find the information you need!"

If query is outside HR scope:
Politely explain this and redirect to the General subagent, while remaining supportive."""

        # Create agent using LangChain 1.1.2 API
        self._create_agent()
        
        logger.info(f"HR Subagent initialized with model: {self.model_id}")
    
    def _initialize_llm(self) -> ChatBedrock:
        """
        Initialize LLM with fallback support.
        
        Tries primary model (Gemma 3 12B) first, falls back to Nova Lite
        if the primary model is unavailable in the region.
        
        Returns:
            ChatBedrock instance
        """
        try:
            llm = ChatBedrock(
                model_id=self.model_id,
                client=get_bedrock_client(),
                model_kwargs={
                    "temperature": self.temperature,
                    "max_tokens": 4096
                }
            )
            # Test the model with a simple call to verify it's available
            # (This will raise an exception if model is not accessible)
            logger.debug(f"Attempting to use model: {self.model_id}")
            return llm
        except Exception as e:
            if self.model_id != self.NOVA_LITE_MODEL_ID:
                logger.warning(
                    f"Primary model {self.model_id} unavailable: {e}. "
                    f"Falling back to Nova Lite."
                )
                self.model_id = self.NOVA_LITE_MODEL_ID
                return ChatBedrock(
                    model_id=self.model_id,
                    client=get_bedrock_client(),
                    model_kwargs={
                        "temperature": self.temperature,
                        "max_tokens": 4096
                    }
                )
            else:
                raise
    
    def _create_agent(self) -> None:
        """Create the LangChain agent using create_agent API."""
        # Create agent using new LangChain 1.1.2 API
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
        Invoke HR subagent with a query.
        
        Args:
            query: User's HR-related question
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
                "tools_used": [],  # New API handles tools internally
                "success": True,
                "agent": "hr_subagent"
            }
            
        except Exception as e:
            logger.error(f"HR Subagent error: {str(e)}")
            return {
                "output": f"I encountered an error while processing your HR query: {str(e)}",
                "tools_used": [],
                "success": False,
                "error": str(e),
                "agent": "hr_subagent"
            }
    
    async def ainvoke(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Async invoke HR subagent with a query.
        
        Args:
            query: User's HR-related question
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
                "agent": "hr_subagent"
            }
            
        except Exception as e:
            logger.error(f"HR Subagent async error: {str(e)}")
            return {
                "output": f"I encountered an error while processing your HR query: {str(e)}",
                "tools_used": [],
                "success": False,
                "error": str(e),
                "agent": "hr_subagent"
            }


def create_hr_subagent_graph() -> StateGraph:
    """
    Create a LangGraph StateGraph for HR subagent.
    
    This provides more control over the agent flow and can be
    wrapped as a CompiledSubAgent for the manager.
    
    Returns:
        Compiled StateGraph for HR subagent
    """
    # Initialize components
    hr_agent = HRSubagent()
    
    def process_query(state: HRAgentState) -> HRAgentState:
        """Process HR query using the agent."""
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
        result = hr_agent.invoke(query, chat_history if chat_history else None)
        
        # Update state
        state["final_response"] = result["output"]
        state["messages"] = messages + [AIMessage(content=result["output"])]
        
        return state
    
    def should_end(state: HRAgentState) -> str:
        """Determine if processing should end."""
        if state.get("final_response"):
            return END
        return "process"
    
    # Build graph
    workflow = StateGraph(HRAgentState)
    
    # Add nodes
    workflow.add_node("process", process_query)
    
    # Add edges
    workflow.set_entry_point("process")
    workflow.add_conditional_edges("process", should_end)
    
    return workflow.compile()


# Singleton instance
_hr_subagent_instance: Optional[HRSubagent] = None


def get_hr_subagent() -> HRSubagent:
    """Get or create singleton HR subagent instance."""
    global _hr_subagent_instance
    if _hr_subagent_instance is None:
        _hr_subagent_instance = HRSubagent()
    return _hr_subagent_instance


# Export for manager integration
__all__ = [
    "HRSubagent",
    "HRAgentState", 
    "create_hr_subagent_graph",
    "get_hr_subagent"
]
