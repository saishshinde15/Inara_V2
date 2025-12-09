#!/usr/bin/env python3
"""
HR Bot V2 - Interactive Conversation Demo

Demonstrates multi-turn conversation with persistent memory.
Memory persists across sessions using SQLite + Titan embeddings.

Usage:
    python run_conversation.py                    # New conversation
    python run_conversation.py --thread abc123    # Resume existing thread
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ.setdefault("AWS_REGION", "us-east-1")

from datetime import datetime


def print_banner(thread_id: str):
    """Print startup banner."""
    print("\n" + "="*70)
    print("🤖 HR Bot V2 - Interactive Conversation Mode")
    print("="*70)
    print(f"   Manager: GPT-5 Nano (with persistent memory)")
    print(f"   HR Subagent: AWS Nova Lite + Hybrid RAG")
    print(f"   Finance Subagent: AWS Nova Lite + Web Search")
    print(f"   Thread ID: {thread_id[:20]}...")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\n💡 Commands:")
    print("   'quit' or 'exit' - End conversation")
    print("   'clear' - Reset conversation memory (start new thread)")
    print("   'history' - Show conversation history")
    print("   'thread' - Show current thread ID")
    print("="*70 + "\n")


def main():
    """Interactive conversation loop with persistent memory."""
    parser = argparse.ArgumentParser(description="HR Bot Interactive Conversation")
    parser.add_argument("--thread", type=str, help="Resume existing conversation thread")
    args = parser.parse_args()
    
    from hr_bot.deep_agents import get_manager_agent
    
    # Get singleton manager
    manager = get_manager_agent()
    
    # Set or create thread ID
    if args.thread:
        manager.set_thread(args.thread)
        thread_id = args.thread
    else:
        thread_id = manager.new_thread()
    
    print_banner(thread_id)
    turn_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for using Inara HR Portal.")
                print(f"📝 Your thread ID: {thread_id}")
                print("   Use --thread to resume this conversation later.")
                break
            
            if user_input.lower() == 'clear':
                thread_id = manager.new_thread()
                print(f"🗑️ Started new conversation thread: {thread_id[:20]}...")
                turn_count = 0
                continue
            
            if user_input.lower() == 'thread':
                print(f"🧵 Current Thread ID: {manager.get_thread_id()}")
                continue
            
            if user_input.lower() == 'history':
                if manager.conversation_history:
                    print("\n📜 Conversation History:")
                    print("-" * 40)
                    for i, msg in enumerate(manager.conversation_history):
                        role = "👤 You" if msg["role"] == "user" else "🤖 Bot"
                        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        print(f"{i+1}. {role}: {content}")
                    print("-" * 40)
                else:
                    print("📜 No conversation history yet.")
                continue
            
            # Process the query with thread_id for persistent memory
            turn_count += 1
            print(f"\n⏳ Processing (turn {turn_count})...\n")
            
            result = manager.invoke(user_input, thread_id=thread_id)
            
            # Display response
            print("🤖 Inara HR Assistant:")
            print("-" * 60)
            print(result.get("output", "No response"))
            print("-" * 60)
            
            # Show metadata
            if result.get("metadata", {}).get("agents_consulted"):
                agents = result["metadata"]["agents_consulted"]
                print(f"\n📋 Agents consulted: {', '.join(agents)}")
            
            if result.get("cached"):
                print("💾 (Response from cache)")
            
            if result.get("metadata", {}).get("memory_enabled"):
                print("🧠 (Memory enabled - context preserved)")
                
        except KeyboardInterrupt:
            print("\n\n👋 Conversation ended.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n📊 Session Summary:")
    print(f"   - Total turns: {turn_count}")
    print(f"   - Messages in history: {len(manager.conversation_history)}")
    print(f"   - Thread ID: {thread_id}")


if __name__ == "__main__":
    main()
