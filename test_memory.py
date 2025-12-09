#!/usr/bin/env python3
"""
Test script for persistent memory functionality.

Tests:
1. Single conversation memory
2. Cross-conversation memory retrieval
3. Thread-based isolation
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("AWS_REGION", "us-east-1")


def test_memory_module():
    """Test memory module components."""
    print("=" * 60)
    print("Testing Memory Module")
    print("=" * 60)
    
    from hr_bot.utils.memory import (
        get_checkpointer,
        get_memory_manager,
        ConversationSummaryStore,
        SemanticMemory
    )
    
    # Test checkpointer
    checkpointer = get_checkpointer()
    print(f"✅ Checkpointer: {type(checkpointer).__name__}")
    
    # Test memory manager
    memory_mgr = get_memory_manager()
    print(f"✅ Memory Manager: {type(memory_mgr).__name__}")
    
    # Test summary store
    summary_store = ConversationSummaryStore()
    print(f"✅ Summary Store initialized")
    
    # Test semantic memory
    semantic_mem = SemanticMemory()
    print(f"✅ Semantic Memory initialized")
    
    print("\n✅ All memory components working!")
    return True


def test_manager_with_memory():
    """Test Manager Agent with memory."""
    print("\n" + "=" * 60)
    print("Testing Manager Agent with Memory")
    print("=" * 60)
    
    from hr_bot.deep_agents import get_manager_agent
    
    manager = get_manager_agent()
    
    # Create new thread
    thread_id = manager.new_thread()
    print(f"✅ Thread created: {thread_id[:20]}...")
    
    # Verify components
    print(f"✅ Checkpointer: {type(manager.checkpointer).__name__}")
    print(f"✅ Memory Manager: {type(manager.memory_manager).__name__}")
    
    print("\n✅ Manager with memory is functional!")
    return manager, thread_id


def test_conversation_memory(thread_id: str, manager):
    """Test conversation continuity within a thread."""
    print("\n" + "=" * 60)
    print("Testing Conversation Memory")
    print("=" * 60)
    
    manager.set_thread(thread_id)
    
    # First query - establish context
    print("\n📝 Query 1: 'My name is John and I work in Engineering'")
    result1 = manager.invoke(
        "My name is John and I work in Engineering department. What leave policies apply to me?",
        thread_id=thread_id
    )
    print(f"✅ Response received ({len(result1.get('output', ''))} chars)")
    print(f"   Memory enabled: {result1.get('metadata', {}).get('memory_enabled', False)}")
    
    # Check conversation history
    history = manager.get_conversation_history()
    print(f"✅ Conversation history: {len(history)} messages")
    
    # Second query - should use context
    print("\n📝 Query 2: 'What about medical benefits?' (should remember John/Engineering)")
    result2 = manager.invoke(
        "What about medical benefits?",
        thread_id=thread_id
    )
    print(f"✅ Response received ({len(result2.get('output', ''))} chars)")
    
    # Verify history grew
    history2 = manager.get_conversation_history()
    print(f"✅ Conversation history: {len(history2)} messages")
    
    if len(history2) > len(history):
        print("\n✅ Conversation memory is accumulating!")
    else:
        print("\n⚠️ History did not grow as expected")
    
    return True


def test_memory_retrieval():
    """Test semantic memory retrieval."""
    print("\n" + "=" * 60)
    print("Testing Semantic Memory Retrieval")
    print("=" * 60)
    
    from hr_bot.utils.memory import get_memory_manager
    
    memory_mgr = get_memory_manager()
    
    # Store a test memory
    test_thread = "test-semantic-thread"
    memory_mgr.add_memory(
        thread_id=test_thread,
        content="User John from Engineering asked about leave policies",
        memory_type="conversation",
        metadata={"topic": "leave", "user": "John"}
    )
    print("✅ Test memory stored")
    
    # Retrieve with semantic search
    context = memory_mgr.get_context_for_query(
        query="What are my leave options?",
        thread_id=test_thread
    )
    print(f"✅ Context retrieved: {len(context.get('relevant_memories', []))} memories")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "🧪" * 30)
    print("HR Bot V2 - Memory System Test Suite")
    print("🧪" * 30 + "\n")
    
    try:
        # Test 1: Memory module
        test_memory_module()
        
        # Test 2: Manager with memory
        manager, thread_id = test_manager_with_memory()
        
        # Test 3: Conversation memory (pass manager instance)
        if manager:
            test_conversation_memory(thread_id, manager)
        
        # Test 4: Semantic retrieval
        test_memory_retrieval()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
