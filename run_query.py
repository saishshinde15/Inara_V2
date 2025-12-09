#!/usr/bin/env python3
"""
HR Bot V2 - Test Runner

Run queries through the Deep Agents + LangChain architecture.
Similar to `crewai run` but for LangChain/Deep Agents.

Usage:
    python run_query.py "Your question here"
    python run_query.py  # Uses default test query
"""

import os
import sys
from pathlib import Path

# Load .env file BEFORE any LangChain imports (required for LangSmith tracing)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables fallback
os.environ.setdefault("AWS_REGION", "us-east-1")

from datetime import datetime
from typing import Dict, Any
import json

# =========================================
# PRE-WARMING - Load all expensive resources at startup
# =========================================
print("🔄 Pre-warming resources (one-time startup cost)...")

# 1. Pre-load embeddings (saves ~4s)
from hr_bot.utils.embeddings import preload_embeddings
preload_embeddings()

# 2. Pre-load subagent singletons (saves ~3s each on first query)
from hr_bot.deep_agents.subagents import get_hr_subagent, get_general_subagent
_ = get_hr_subagent()
_ = get_general_subagent()

# 3. Pre-load RAG indexes (saves ~2s per index)
from hr_bot.tools.hr_rag_tool import create_hr_rag_retriever
from hr_bot.tools.master_actions_tool import create_master_actions_retriever
_ = create_hr_rag_retriever()
_ = create_master_actions_retriever()

print("✅ Pre-warming complete!")


def print_banner():
    """Print startup banner."""
    print("\n" + "="*70)
    print("🤖 HR Bot V2 - Deep Agents + LangChain Architecture")
    print("="*70)
    print(f"   Manager: Amazon Nova Lite (Orchestration)")
    print(f"   HR Subagent: Amazon Nova Lite + Hybrid RAG")
    print(f"   General Subagent: Amazon Nova Lite + Web Search")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def format_result(result: Dict[str, Any]) -> str:
    """Format the result for display."""
    output = []
    
    # Response
    output.append("📝 RESPONSE:")
    output.append("-" * 60)
    output.append(result.get("response", "No response"))
    output.append("-" * 60)
    
    # Metadata
    if result.get("agents_consulted"):
        output.append(f"\n🔗 Agents Consulted: {', '.join(result['agents_consulted'])}")
    
    if result.get("delegations"):
        output.append("\n📋 Delegations:")
        for d in result["delegations"]:
            output.append(f"   → {d.get('agent', 'unknown')}: {d.get('query', '')[:80]}...")
    
    output.append(f"\n✅ Success: {result.get('success', False)}")
    output.append(f"🔑 Session: {result.get('session_id', 'N/A')}")
    
    if result.get("cached"):
        output.append("💾 Response served from cache")
    
    if result.get("blocked"):
        output.append(f"🚫 Query blocked: {result.get('safety_category', 'unknown')}")
    
    return "\n".join(output)


def run_query(query: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run a query through HR Bot V2.
    
    Args:
        query: The user's question
        verbose: Enable verbose logging
        
    Returns:
        Result dictionary with response and metadata
    """
    # Sanitize smart/curly quotes to standard ASCII quotes
    # This prevents encoding issues with the Bedrock Converse API
    query = query.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("—", "-")
    
    print(f"📨 Query: {query}\n")
    print("⏳ Processing...\n")
    
    try:
        # Import here to allow environment setup first
        from hr_bot.main import chat
        
        result = chat(query, verbose=verbose)
        return result
        
    except ImportError as e:
        return {
            "response": f"Import error: {e}\n\nMake sure all dependencies are installed:\npip install -e .",
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "success": False,
            "error": str(e)
        }


def main():
    """Main entry point."""
    # Default test query - use a valid HR-related question
    default_query = "What is the leave policy for employees?"
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = default_query
    
    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if verbose:
        query = query.replace("--verbose", "").replace("-v", "").strip()
    
    # Print banner
    print_banner()
    
    # Run the query
    result = run_query(query, verbose=verbose)
    
    # Display formatted result
    print(format_result(result))
    print("\n" + "="*70 + "\n")
    
    # Return exit code based on success
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
