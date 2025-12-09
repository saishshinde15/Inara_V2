"""
HR Bot V2 - Main Entry Point

Enterprise Assistant using Deep Agents + LangChain architecture.
Manager agent (GPT-5 Nano) orchestrates HR and Finance subagents.

Usage:
    # As module
    from hr_bot.main import run, chat
    
    # Interactive mode
    python -m hr_bot.main --interactive
    
    # Single query
    python -m hr_bot.main --query "What is our leave policy?"
    
    # Start API server
    python -m hr_bot.main --serve
"""

import argparse
import asyncio
import logging
import sys
import uuid
from typing import Optional, Dict, Any

from hr_bot.config.settings import get_settings
from hr_bot.deep_agents import (
    get_manager_agent,
    create_enterprise_assistant,
    ManagerAgent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# Core Functions
# ============================================================================

def chat(
    query: str,
    session_id: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process a single chat query.
    
    Args:
        query: User's question
        session_id: Optional session ID for tracking
        verbose: Enable verbose output
        
    Returns:
        Dict with response, metadata, and session info
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Sanitize smart/curly quotes to standard ASCII quotes
    # This prevents encoding issues with the Bedrock Converse API
    query = query.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("—", "-")
    
    try:
        manager = get_manager_agent()
        result = manager.invoke(query, session_id=session_id)
        
        return {
            "response": result["output"],
            "success": result.get("success", True),
            "session_id": session_id or str(uuid.uuid4()),
            "agents_consulted": result.get("metadata", {}).get("agents_consulted", []),
            "delegations": result.get("delegations", [])
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {
            "response": f"Error processing query: {str(e)}",
            "success": False,
            "session_id": session_id,
            "error": str(e)
        }


async def achat(
    query: str,
    session_id: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process a single chat query asynchronously.
    
    Args:
        query: User's question
        session_id: Optional session ID for tracking
        verbose: Enable verbose output
        
    Returns:
        Dict with response, metadata, and session info
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        manager = get_manager_agent()
        result = await manager.ainvoke(query, session_id=session_id)
        
        return {
            "response": result["output"],
            "success": result.get("success", True),
            "session_id": session_id or str(uuid.uuid4()),
            "agents_consulted": result.get("metadata", {}).get("agents_consulted", []),
            "delegations": result.get("delegations", [])
        }
        
    except Exception as e:
        logger.error(f"Async chat error: {str(e)}")
        return {
            "response": f"Error processing query: {str(e)}",
            "success": False,
            "session_id": session_id,
            "error": str(e)
        }


def run(query: str, verbose: bool = False) -> str:
    """
    Simple run function - returns just the response string.
    
    Args:
        query: User's question
        verbose: Enable verbose output
        
    Returns:
        Response string
    """
    result = chat(query, verbose=verbose)
    return result["response"]


# ============================================================================
# Interactive Mode
# ============================================================================

def interactive_mode(verbose: bool = False) -> None:
    """
    Run interactive chat session in terminal.
    
    Args:
        verbose: Enable verbose output
    """
    print("\n" + "="*60)
    print("🤖 HR Bot V2 - Enterprise Assistant")
    print("="*60)
    print("Powered by GPT-5 Nano with HR and Finance specialists")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'reset' to clear conversation history")
    print("Type 'help' for available commands")
    print("="*60 + "\n")
    
    manager = get_manager_agent()
    session_id = str(uuid.uuid4())
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye! Have a great day.")
                break
            
            if user_input.lower() == "reset":
                manager.reset_conversation()
                session_id = str(uuid.uuid4())
                print("🔄 Conversation reset. Starting fresh!\n")
                continue
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            if user_input.lower() == "history":
                history = manager.get_conversation_history()
                if history:
                    print("\n📜 Conversation History:")
                    for i, msg in enumerate(history, 1):
                        role = "You" if msg["role"] == "user" else "Bot"
                        print(f"  {i}. [{role}]: {msg['content'][:100]}...")
                    print()
                else:
                    print("No conversation history yet.\n")
                continue
            
            # Process query
            print("\n🤔 Processing...\n")
            
            result = manager.invoke(user_input, session_id=session_id)
            
            # Display response
            print(f"🤖 Assistant: {result['output']}")
            
            # Show delegations if any
            if result.get("delegations"):
                agents = [d["agent"].upper() for d in result["delegations"]]
                print(f"\n   [Consulted: {', '.join(agents)} specialists]")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Interactive mode error: {str(e)}")
            print(f"\n❌ Error: {str(e)}\n")


def print_help() -> None:
    """Print help information."""
    print("""
📚 HR Bot V2 - Help
==================

Commands:
  quit, exit, q  - End the session
  reset          - Clear conversation history
  history        - Show conversation history
  help           - Show this help message

Example Questions:
  HR Policy:
    - "What is our annual leave policy?"
    - "How do I submit an expense report?"
    - "What are the health insurance benefits?"
    - "Tell me about the performance review process"
  
  Finance:
    - "What is a 401k?"
    - "How does compound interest work?"
    - "What are the current market trends?"
    - "Explain capital gains tax"

Tips:
  - Be specific in your questions for better answers
  - The bot will route your question to the right specialist
  - HR policies come from company documents
  - Finance info comes from web search
""")


# ============================================================================
# API Server (Optional)
# ============================================================================

def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start FastAPI server for API access.
    
    Args:
        host: Server host
        port: Server port
    """
    try:
        import uvicorn
        from hr_bot.ui.app import create_app
        
        app = create_app()
        
        logger.info(f"Starting HR Bot V2 API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        logger.error("FastAPI/Uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HR Bot V2 - Enterprise Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python -m hr_bot.main --interactive
    
  Single query:
    python -m hr_bot.main --query "What is our leave policy?"
    
  Start API server:
    python -m hr_bot.main --serve --port 8000
    
  Verbose mode:
    python -m hr_bot.main --interactive --verbose
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive chat mode"
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Process a single query"
    )
    
    parser.add_argument(
        "-s", "--serve",
        action="store_true",
        help="Start the API server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    # Handle modes
    if args.serve:
        start_server(host=args.host, port=args.port)
    elif args.query:
        result = chat(args.query, verbose=args.verbose)
        print(f"\n{result['response']}\n")
        if result.get("agents_consulted"):
            print(f"[Consulted: {', '.join(result['agents_consulted'])}]")
    elif args.interactive:
        interactive_mode(verbose=args.verbose)
    else:
        # Default to interactive mode
        interactive_mode(verbose=args.verbose)


if __name__ == "__main__":
    main()
