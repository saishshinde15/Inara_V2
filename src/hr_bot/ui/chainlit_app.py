"""
HR Bot V2 - Chainlit Application

Interactive chat interface for Inara HR Assistant powered by the Deep Agent architecture.

Run with:
    chainlit run src/hr_bot/ui/chainlit_app.py -w

Features:
- Role-based access control (executive/employee)
- Persistent conversation memory with thread support
- Real-time streaming responses
- Admin actions (refresh docs, clear cache, reset conversation)
- Source citations from policy documents
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Any

import chainlit as cl
from starlette.datastructures import Headers

from hr_bot.config.settings import get_settings
from hr_bot.deep_agents import get_manager_agent, ManagerAgent
from hr_bot.utils.s3_loader import S3DocumentLoader
from hr_bot.utils.cache import get_response_cache

# Pre-warm expensive resources at import time
# This runs when Chainlit starts, making first queries faster
from hr_bot.utils.embeddings import preload_embeddings
preload_embeddings()

# Pre-load subagent singletons
from hr_bot.deep_agents.subagents import get_hr_subagent, get_general_subagent
_hr = get_hr_subagent()
_general = get_general_subagent()

# Pre-load RAG indexes
from hr_bot.tools.hr_rag_tool import create_hr_rag_retriever
from hr_bot.tools.master_actions_tool import create_master_actions_retriever
_rag = create_hr_rag_retriever()
_master = create_master_actions_retriever()

logger = logging.getLogger("hr_bot.chainlit")
logger.info("✅ Pre-warming complete - embeddings, subagents, and indexes loaded")
settings = get_settings()

# ---------------------------------------------------------------------------
# Centralized branding configuration
# ---------------------------------------------------------------------------

APP_NAME = os.getenv("APP_NAME", "Inara")
APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "your intelligent HR assistant")
APP_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Environment wiring for OAuth
# ---------------------------------------------------------------------------

_ENV_ALIAS_MAP = {
    "OAUTH_GOOGLE_CLIENT_ID": "GOOGLE_CLIENT_ID",
    "OAUTH_GOOGLE_CLIENT_SECRET": "GOOGLE_CLIENT_SECRET",
}
for target, source in _ENV_ALIAS_MAP.items():
    if not os.getenv(target) and os.getenv(source):
        os.environ[target] = os.environ[source]

if not os.getenv("CHAINLIT_AUTH_SECRET"):
    os.environ["CHAINLIT_AUTH_SECRET"] = os.getenv("CHAINLIT_JWT_SECRET", "inara-demo-secret")
    logger.warning(
        "CHAINLIT_AUTH_SECRET missing. A temporary value was injected for development. "
        "Run `chainlit create-secret` and set CHAINLIT_AUTH_SECRET before production deploys."
    )

SUPPORT_CONTACT_EMAIL = os.getenv("SUPPORT_CONTACT_EMAIL", "support@company.com")
DEFAULT_PLACEHOLDER = "Ask me anything about HR policies, benefits, or general finance questions."

# Thread pool for blocking operations
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("CHAINLIT_MAX_WORKERS", "4")))

# Manager cache per role
MANAGER_CACHE: Dict[str, ManagerAgent] = {}
MANAGER_LOCKS: Dict[str, asyncio.Lock] = {}
WARMED_ROLES: set[str] = set()

# Warmup queries for RAG index
WARMUP_QUERIES: List[str] = [
    "What is the sick leave policy?",
    "How do I request paternity leave?",
    "What are my vacation entitlements?",
    "How do I report an absence?",
    "What health benefits are available?",
    "How do I submit an expense claim?",
    "What is the remote work policy?",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _env_flag(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).strip().lower() in {"1", "true", "yes"}


def _env_list(key: str) -> List[str]:
    values = os.getenv(key, "")
    return [item.strip().lower() for item in values.split(",") if item.strip()]


def _normalize_email(email: Optional[str]) -> str:
    return (email or "").strip().lower()


def _derive_role(email: Optional[str]) -> str:
    """Derive user role from email."""
    normalized = _normalize_email(email)
    if not normalized:
        return "unauthorized"
    if normalized in _env_list("EXECUTIVE_EMAILS"):
        return "executive"
    if normalized in _env_list("EMPLOYEE_EMAILS"):
        return "employee"
    # Default to employee for demo purposes if no lists configured
    if not _env_list("EXECUTIVE_EMAILS") and not _env_list("EMPLOYEE_EMAILS"):
        return "employee"
    return "unauthorized"


def _display_name(email: str, fallback: Optional[str]) -> str:
    if fallback:
        return fallback
    if "@" in email:
        return email.split("@", 1)[0].title()
    return email or "Colleague"


def _author_as_string(author: Optional[object]) -> str:
    """Return a JSON-serializable string representation for the author."""
    if author is None:
        return APP_NAME
    try:
        if isinstance(author, cl.User):
            return getattr(author, "display_name", None) or getattr(author, "identifier", str(author))
    except Exception:
        pass
    return str(author)


def _history() -> List[Dict[str, str]]:
    return cl.user_session.get("history", [])


def _set_history(history: List[Dict[str, str]]) -> None:
    cl.user_session.set("history", history)


def _get_thread_id() -> str:
    """Get or create thread ID for this session."""
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", thread_id)
    return thread_id


def _get_lock(role: str) -> asyncio.Lock:
    lock = MANAGER_LOCKS.get(role)
    if lock is None:
        lock = asyncio.Lock()
        MANAGER_LOCKS[role] = lock
    return lock


async def _run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, partial(func, *args, **kwargs))


# ---------------------------------------------------------------------------
# Manager Agent Management
# ---------------------------------------------------------------------------

async def _get_manager(role: str) -> ManagerAgent:
    """Get or create manager agent for a role."""
    role_key = (role or "employee").lower()
    if cached := MANAGER_CACHE.get(role_key):
        return cached
    lock = _get_lock(role_key)
    async with lock:
        if cached := MANAGER_CACHE.get(role_key):
            return cached
        logger.info("Initializing ManagerAgent for role=%s", role_key)
        manager = await _run_blocking(ManagerAgent, verbose=False)
        MANAGER_CACHE[role_key] = manager
        return manager


def _warm_manager(manager: ManagerAgent) -> None:
    """Warm up the manager's RAG index."""
    try:
        # The manager uses subagents that have RAG tools
        # Invoke a simple query to warm up
        for query in WARMUP_QUERIES[:3]:
            try:
                manager.invoke(query)
                break  # One successful warm-up is enough
            except Exception:
                continue
    except Exception as exc:
        logger.warning("Warm-up skipped: %s", exc)


async def _ensure_warm(role: str, manager: ManagerAgent) -> None:
    """Ensure manager is warmed up."""
    role_key = (role or "employee").lower()
    if role_key in WARMED_ROLES:
        return
    await _run_blocking(_warm_manager, manager)
    WARMED_ROLES.add(role_key)


# ---------------------------------------------------------------------------
# Response Formatting
# ---------------------------------------------------------------------------

def format_sources(answer: str) -> str:
    """Format source citations nicely and move them to the end."""
    import re
    
    if not answer:
        return answer
    
    lines = answer.splitlines()
    
    # Collect all sources from the response
    all_sources = set()
    source_patterns = [
        r"Source:\s*([^\n]+)",  # "Source: filename.docx"
        r"\(Source:\s*([^)]+)\)",  # "(Source: filename.docx)"
    ]
    
    for pattern in source_patterns:
        for match in re.finditer(pattern, answer, re.IGNORECASE):
            source = match.group(1).strip()
            # Clean up the source name
            source = re.sub(r"^\d+\.\s*", "", source)
            source = re.sub(r"^\[\d+\]\s*", "", source)
            if source and len(source) > 3:
                all_sources.add(source)
    
    # Find and remove existing Sources sections (we'll add a clean one at the end)
    cleaned_lines = []
    skip_source_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect start of Sources section
        if (line_lower.startswith("**sources:**") or 
            line_lower.startswith("sources:") or
            line_lower == "**source documents:**" or
            line_lower.startswith("source documents:")):
            skip_source_section = True
            continue
        
        # Detect end of Sources section (next major section or empty line after items)
        if skip_source_section:
            # Continue skipping source items (lines starting with - or bullet)
            if line.strip().startswith("-") or line.strip().startswith("•"):
                # Extract source from bullet point
                source_text = line.strip().lstrip("-•").strip()
                if source_text:
                    all_sources.add(source_text)
                continue
            # If it's an empty line, continue skipping
            if not line.strip():
                continue
            # If it's a new section (starts with ** or ##), stop skipping
            if line.strip().startswith("**") or line.strip().startswith("##"):
                skip_source_section = False
        
        if not skip_source_section:
            cleaned_lines.append(line)
    
    # Build clean output
    output = "\n".join(cleaned_lines).strip()
    
    # Remove inline "(Source: ...)" references for cleaner text
    output = re.sub(r"\s*\(Source:\s*[^)]+\)", "", output)
    
    # Add consolidated Sources section at the end
    if all_sources:
        formatted_sources = []
        for source in sorted(all_sources):
            # Clean and format source name
            display_name = source.replace("_", " ").strip()
            # Remove .docx extension for cleaner display
            display_name = re.sub(r"\.docx$", "", display_name, flags=re.IGNORECASE)
            if display_name:
                formatted_sources.append(f"`{display_name}`")
        
        if formatted_sources:
            output += f"\n\n---\n\n**Sources:** {' · '.join(formatted_sources)}"
    
    return output


def clean_markdown_artifacts(text: str) -> str:
    """Clean up markdown code block artifacts and formatting issues."""
    import re
    text = text.replace("```markdown", "").replace("```", "")
    
    # Remove common internal artifacts
    artifacts_to_remove = [
        r"^\[Direct answer to the question\]\s*\n?",
        r"^\[Direct answer\]\s*\n?",
        r"^\[Answer\]\s*\n?",
        r"\[Sources consulted:.*?\]\s*\n?",
        r"\[Information sourced from.*?\]\s*\n?",
    ]
    for pattern in artifacts_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up broken source formatting artifacts
    broken_source_patterns = [
        r"\*\*Sources:\*\*\s*`\*\*`",  # **Sources:** `**`
        r"\*\*Sources:\*\*\s*\*\*\s*$",  # **Sources:** ** at end
        r"\*\*Sources:\*\*\s*\*\*\n",  # **Sources:** ** with newline
        r"`\*\*`",  # standalone `**`
        r"---\s*\n\s*\*\*Sources:\*\*\s*\n\s*---",  # Empty sources section
        r"---\s*\n\s*---",  # Double horizontal rules
    ]
    for pattern in broken_source_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    
    # Clean lines that are just "**" or "---"
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "**" or stripped == "`**`":
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    
    # Clean up excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def remove_document_evidence_section(text: str) -> str:
    """Remove internal document evidence sections."""
    lines = text.splitlines()
    output: List[str] = []
    in_block = False
    
    for line in lines:
        if "document evidence" in line.lower() and ("##" in line or "**" in line):
            in_block = True
            continue
        if in_block:
            if (
                line.lower().startswith("sources:")
                or "**sources:**" in line.lower()
                or (line.startswith("##") and "document evidence" not in line.lower())
            ):
                in_block = False
                output.append(line)
            continue
        output.append(line)
    
    return "\n".join(output)


def enhance_professional_formatting(text: str) -> str:
    """Enhance response formatting for professional appearance."""
    import re
    
    # Add visual emphasis to key section headers
    section_headers = [
        "Policy Details:",
        "Eligibility Requirements:",
        "Procedure/Process:",
        "Coverage Details:",
        "Important Notes:",
        "Next Steps:",
        "Key Points:",
        "Summary:",
    ]
    
    for header in section_headers:
        # If header exists but isn't bold, make it bold
        text = re.sub(
            rf"(?<!\*\*){re.escape(header)}",
            f"**{header}**",
            text
        )
    
    # Add horizontal rule before "Next Steps" section for visual separation
    text = re.sub(
        r"(\n\*\*Next Steps:\*\*)",
        r"\n\n---\1",
        text
    )
    
    # Clean up excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    
    # Ensure proper spacing after bullet points
    text = re.sub(r"^(\s*[-•])\s+", r"\1 ", text, flags=re.MULTILINE)
    
    return text.strip()


def format_answer(answer: str) -> str:
    """Format the answer for professional display."""
    if not answer:
        return "I'm sorry, I couldn't generate a response."
    
    # Clean markdown artifacts and internal references
    answer = clean_markdown_artifacts(answer)
    answer = remove_document_evidence_section(answer)
    
    # Enhance professional formatting
    answer = enhance_professional_formatting(answer)
    
    # Format source citations nicely
    answer = format_sources(answer)
    
    return answer


# ---------------------------------------------------------------------------
# Dashboard and Actions
# ---------------------------------------------------------------------------

def _compose_dashboard(display_name: str, role: str) -> str:
    """Compose the welcome dashboard message."""
    return (
        f"### Welcome back, {display_name}\n"
        f"Access level: **{role.title()}**\n\n"
        f"{APP_DESCRIPTION} — available 24/7.\n\n"
        f"Need help? Contact [{SUPPORT_CONTACT_EMAIL}](mailto:{SUPPORT_CONTACT_EMAIL})."
    )


async def _send_dashboard(display_name: str, role: str) -> None:
    """Send the dashboard with action buttons."""
    actions = [
        cl.Action(
            name="refresh_s3_docs",
            payload={"role": role},
            label="🔄 Refresh S3 Docs",
            tooltip="Download the latest HR policies and rebuild indexes",
        ),
        cl.Action(
            name="clear_response_cache",
            payload={},
            label="🧹 Clear Cache",
            tooltip="Purge semantic cache for troubleshooting",
        ),
        cl.Action(
            name="reset_conversation",
            payload={},
            label="🔁 New Conversation",
            tooltip="Start a fresh conversation thread",
        ),
    ]
    message = cl.Message(
        author=_author_as_string(f"{APP_NAME} Control Center"),
        content=_compose_dashboard(display_name, role),
        actions=actions,
    )
    await message.send()


async def _refresh_s3_documents(role: str) -> str:
    """Refresh documents from S3."""
    def _refresh() -> str:
        loader = S3DocumentLoader(user_role=role)
        
        # Load documents from S3 (force_refresh not needed - loader handles caching)
        doc_count = 0
        for doc_data in loader.load_all_documents():
            doc_count += 1
        
        # Clear RAG index to force rebuild
        from hr_bot.config.settings import get_settings
        settings = get_settings()
        rag_index_dir = settings.index_dir / "hr_rag"
        if rag_index_dir.exists():
            shutil.rmtree(rag_index_dir)
        
        # Clear manager cache to force rebuild
        MANAGER_CACHE.pop(role, None)
        WARMED_ROLES.discard(role)
        
        return f"✅ Refreshed {doc_count} HR documents from S3 and cleared RAG index cache."

    return await _run_blocking(_refresh)


async def _clear_response_cache() -> str:
    """Clear the response cache."""
    cache = get_response_cache()
    await _run_blocking(cache.clear_all)
    return "🧹 Response cache cleared successfully."


async def _query_manager(prompt: str, thread_id: str) -> Dict[str, Any]:
    """Query the manager agent."""
    manager: ManagerAgent = cl.user_session.get("manager")
    if manager is None:
        role = cl.user_session.get("role", "employee")
        manager = await _get_manager(role)
        cl.user_session.set("manager", manager)
    
    # Set thread for memory continuity
    manager.set_thread(thread_id)
    
    def _call_manager() -> Dict[str, Any]:
        return manager.invoke(prompt, thread_id=thread_id)

    return await _run_blocking(_call_manager)


# ---------------------------------------------------------------------------
# Authentication callbacks
# ---------------------------------------------------------------------------

@cl.oauth_callback
async def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
    id_token: Optional[str] = None,
) -> Optional[cl.User]:
    """Handle OAuth authentication."""
    email = _normalize_email(raw_user_data.get("email") or default_user.identifier)
    role = _derive_role(email)
    
    if role == "unauthorized":
        logger.warning("OAuth login rejected for %s", email)
        return None
    
    display_name = raw_user_data.get("name") or raw_user_data.get("given_name")
    default_user.identifier = email
    default_user.display_name = _display_name(email, display_name)
    metadata = default_user.metadata or {}
    metadata.update({"role": role, "email": email, "provider": provider_id})
    default_user.metadata = metadata
    return default_user


@cl.header_auth_callback
async def header_auth(headers: Headers) -> Optional[cl.User]:
    """Handle header-based authentication for development."""
    if not _env_flag("ALLOW_DEV_LOGIN"):
        return None
    
    email = _normalize_email(
        headers.get("x-forwarded-email")
        or headers.get("x-dev-email")
        or headers.get("x-user-email")
    )
    if not email:
        return None
    
    role = _derive_role(email)
    if role == "unauthorized":
        return None
    
    return cl.User(identifier=email, metadata={"role": role, "provider": "header"})


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------

@cl.on_app_shutdown
async def shutdown() -> None:
    """Clean up on app shutdown."""
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize chat session."""
    session_user = cl.user_session.get("user")
    email = _normalize_email(getattr(session_user, "identifier", None) if session_user else None)
    
    # Determine role
    role = "employee"  # Default for demo
    if session_user:
        role = session_user.metadata.get("role") or _derive_role(email)
    
    # For demo mode without auth, allow access
    if role == "unauthorized" and _env_flag("DEMO_MODE", "true"):
        role = "employee"
        email = "demo@company.com"
    
    if role == "unauthorized":
        await cl.ErrorMessage(
            content=(
                "Access denied. This workspace is limited to approved corporate emails. "
                f"Need assistance? Reach out to {SUPPORT_CONTACT_EMAIL}."
            )
        ).send()
        return
    
    display_name = _display_name(email, getattr(session_user, "display_name", None) if session_user else None)
    
    # Set session variables
    cl.user_session.set("email", email)
    cl.user_session.set("role", role)
    cl.user_session.set("display_name", display_name)
    cl.user_session.set("history", [])
    
    # Create new thread for this conversation
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    
    # Get manager for this session
    manager = await _get_manager(role)
    manager.set_thread(thread_id)
    cl.user_session.set("manager", manager)
    
    # Warm up in background (don't wait)
    asyncio.create_task(_ensure_warm(role, manager))
    
    # Send dashboard
    await _send_dashboard(display_name, role)
    
    # Welcome message
    await cl.Message(
        author=_author_as_string(APP_NAME),
        content=(
            f"🚀 **I'm ready to help!**\n\n"
            f"I can assist with:\n"
            f"- 📋 **HR policies** – leave, benefits, onboarding, expenses\n"
            f"- 🌏 **General knowledge** – government schemes, regulations, any topic\n\n"
            f"_Try: \"What is the maternity leave policy and how do I apply?\"_"
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming messages with streaming-like updates."""
    prompt = (message.content or "").strip()
    if not prompt:
        return
    
    # Sanitize smart/curly quotes to standard ASCII quotes
    # This prevents encoding issues with the Bedrock Converse API
    prompt = prompt.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("—", "-")
    
    # Add to history
    history = _history()
    history.append({"role": "user", "content": prompt})
    _set_history(history)
    
    # Get thread ID
    thread_id = _get_thread_id()
    
    # Show progress with stages
    progress = cl.Message(
        author=_author_as_string(APP_NAME),
        content="🔍 Checking cache...",
    )
    await progress.send()
    
    import time
    start_time = time.time()
    
    try:
        # Update progress as we go
        async def update_progress(stage: str):
            progress.content = stage
            await progress.update()
        
        # Check cache first (fast path)
        from hr_bot.utils.cache import get_response_cache
        cache = get_response_cache()
        cached = cache.get_cached_response(prompt)
        
        if cached:
            elapsed = time.time() - start_time
            logger.info(f"Cache hit! Response in {elapsed:.2f}s")
            answer = cached.get("response", "")
            is_cached = True
        else:
            await update_progress("🧠 **Understanding your question...**\n\n_Routing to the best specialist agent_")
            await asyncio.sleep(0.3)  # Brief pause for UX
            await update_progress("🔍 **Searching knowledge base...**\n\n_Checking HR policies and external sources_")
            result = await _query_manager(prompt, thread_id)
            answer = result.get("output", "")
            is_cached = result.get("cached", False)
        
        elapsed = time.time() - start_time
        
    except Exception as exc:
        logger.exception("Failed to answer question", exc_info=exc)
        progress.content = (
            "⚠️ Technical issue while generating a response. Please try again or use "
            "'Clear Cache' if the issue persists."
        )
        await progress.update()
        return
    
    # Format and display
    formatted = format_answer(answer)
    
    # Add timing info in debug mode
    if _env_flag("DEBUG_TIMING"):
        cache_status = "✅ cached" if is_cached else "🔄 generated"
        formatted += f"\n\n---\n*⏱️ {elapsed:.1f}s ({cache_status})*"
    
    # Update history
    history.append({"role": "assistant", "content": formatted})
    _set_history(history)
    
    # Update message
    progress.content = formatted
    await progress.update()


# ---------------------------------------------------------------------------
# Action callbacks
# ---------------------------------------------------------------------------

@cl.action_callback("refresh_s3_docs")
async def refresh_docs(action: cl.Action) -> None:
    """Refresh documents from S3."""
    role = cl.user_session.get("role", "employee")
    await cl.Message(content="🔄 Refreshing S3 documents. This may take a minute...").send()
    try:
        status = await _refresh_s3_documents(role)
        await cl.Message(content=status).send()
    except Exception as exc:
        logger.exception("S3 refresh failed", exc_info=exc)
        await cl.Message(content=f"⚠️ Error refreshing documents: {exc}").send()


@cl.action_callback("clear_response_cache")
async def clear_cache(action: cl.Action) -> None:
    """Clear the response cache."""
    try:
        status = await _clear_response_cache()
        await cl.Message(content=status).send()
    except Exception as exc:
        logger.exception("Cache clear failed", exc_info=exc)
        await cl.Message(content=f"⚠️ Error clearing cache: {exc}").send()


@cl.action_callback("reset_conversation")
async def reset_conversation(action: cl.Action) -> None:
    """Reset conversation and start a new thread."""
    # Create new thread
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("history", [])
    
    # Reset manager's conversation
    manager = cl.user_session.get("manager")
    if manager:
        manager.new_thread()
        manager.set_thread(thread_id)
    
    await cl.Message(
        content="🔁 **New Conversation Started**\n\nYour conversation history has been cleared. Feel free to start a new topic!"
    ).send()


@cl.on_stop
async def on_stop() -> None:
    """Handle stop button click."""
    cl.user_session.set("history", [])
    await cl.Message(content="⏹️ Response generation stopped.").send()
