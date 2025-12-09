# HR Bot V2

**Enterprise HR Assistant** powered by Deep Agents + LangChain architecture.

A production-ready HR assistant using a manager-subagent architecture with **Amazon Nova Lite** for intelligent routing and content generation.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Manager Agent (Amazon Nova Lite)                    │
│         Intelligent Query Routing & Orchestration                │
└───────────────────────┬───────────────────┬─────────────────────┘
                        │                   │
           ┌────────────▼────────┐ ┌────────▼────────────┐
           │    HR Subagent      │ │  General Subagent   │
           │  (Nova Lite + RAG)  │ │ (Nova Lite + Search)│
           └────────────┬────────┘ └────────┬────────────┘
                        │                   │
           ┌────────────▼────────┐ ┌────────▼────────────┐
           │ • HR Document RAG   │ │ • Serper Web Search │
           │ • Master Actions    │ │ • Any External Info │
           │ • Policy Retrieval  │ │ • General Knowledge │
           └─────────────────────┘ └─────────────────────┘
```

### Components

| Component | Description | Model |
|-----------|-------------|-------|
| **Manager Agent** | Routes queries, synthesizes responses | Amazon Nova Lite |
| **HR Subagent** | Company HR policies, benefits, procedures | Amazon Nova Lite |
| **General Subagent** | Any general knowledge (finance, tech, etc.) | Amazon Nova Lite |

### Why Two Subagents?
- **HR Agent**: Uses RAG to search company documents (internal knowledge)
- **General Agent**: Uses web search for any external question (finance, tech, news, etc.)

### Tools

- **HR Document Search**: Hybrid RAG (BM25 + FAISS) for company documents
- **Master Actions Guide**: Procedural guidance for HR actions
- **General Web Search**: Serper API for any external information

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- AWS credentials (for Bedrock - Nova Lite access)
- Serper API Key (for web search)
- S3 bucket (for HR documents)

### Installation

```bash
# Clone and navigate
cd HR_BOT_V2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key  # For finance search
```

### Running

#### Interactive Mode
```bash
python -m hr_bot.main --interactive
```

#### Single Query
```bash
python -m hr_bot.main --query "What is our leave policy?"
```

#### API Server
```bash
python -m hr_bot.main --serve --port 8000
```

#### Chainlit UI
```bash
chainlit run src/hr_bot/ui/chainlit_app.py -w
```

## 📁 Project Structure

```
HR_BOT_V2/
├── pyproject.toml              # Project configuration
├── .env.example                # Environment template
├── README.md                   # This file
│
├── src/hr_bot/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   │
│   ├── config/
│   │   ├── settings.py         # Pydantic settings
│   │   └── agents.yaml         # Agent configurations
│   │
│   ├── deep_agents/
│   │   ├── __init__.py
│   │   ├── manager.py          # Manager Agent (GPT-5 Nano)
│   │   └── subagents/
│   │       ├── hr_subagent.py      # HR specialist
│   │       └── finance_subagent.py # Finance specialist
│   │
│   ├── tools/
│   │   ├── hr_rag_tool.py      # Hybrid RAG search
│   │   ├── master_actions_tool.py  # Procedural guide
│   │   └── finance_search_tool.py  # Serper web search
│   │
│   ├── ui/
│   │   ├── app.py              # FastAPI application
│   │   └── chainlit_app.py     # Chainlit UI
│   │
│   └── utils/
│       ├── cache.py            # Response caching
│       ├── s3_loader.py        # S3 document loader
│       └── domain_router.py    # Query routing
│
├── data/                       # Document storage
│   ├── Regular-Employee-Documents/
│   ├── Executive-Only-Documents/
│   └── Master-Document/
│
└── tests/
    └── test_hr_bot.py          # Unit tests
```

## 🔧 API Reference

### REST API

#### Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
  "message": "What is our leave policy?",
  "session_id": "optional-session-id",
  "include_history": true
}
```

Response:
```json
{
  "response": "Our leave policy provides...",
  "session_id": "abc123",
  "success": true,
  "agents_consulted": ["hr"],
  "metadata": {}
}
```

#### Health Check
```http
GET /health
```

### Python API

```python
from hr_bot.main import chat, run

# Full response with metadata
result = chat("What is our leave policy?")
print(result["response"])
print(result["agents_consulted"])

# Simple string response
response = run("How do I submit expenses?")
print(response)
```

### Async Usage

```python
import asyncio
from hr_bot.main import achat

async def main():
    result = await achat("What are the health benefits?")
    print(result["response"])

asyncio.run(main())
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hr_bot --cov-report=html

# Run specific test file
pytest tests/test_hr_bot.py -v
```

## 📊 Query Routing

The system intelligently routes queries:

| Query Type | Routed To | Examples |
|------------|-----------|----------|
| HR Policies | HR Subagent | Leave policy, benefits, handbook |
| Company Procedures | HR Subagent | Expense reports, onboarding |
| General Finance | Finance Subagent | 401k, taxes, investments |
| Market Information | Finance Subagent | Stock prices, market trends |

### Routing Examples

```
"What is our PTO policy?" → HR Subagent
"How do I submit an expense report?" → HR Subagent
"What is a 401k?" → Finance Subagent
"How does compound interest work?" → Finance Subagent
```

## 🔐 Security

- API keys stored in environment variables
- Session isolation for multi-user support
- Optional authentication via Chainlit
- Configurable CORS for API access

## 📈 Performance

- **Response Caching**: SQLite-backed cache reduces redundant API calls
- **Hybrid RAG**: BM25 + FAISS ensemble for optimal retrieval
- **Async Support**: Non-blocking I/O for concurrent requests

## 🛠️ Development

### Adding New Tools

1. Create tool in `src/hr_bot/tools/`
2. Define with `@tool` decorator
3. Add to appropriate subagent

```python
from langchain_core.tools import tool

@tool("my_new_tool")
def my_new_tool(query: str) -> str:
    """Tool description for LLM."""
    return "Tool response"
```

### Adding New Subagents

1. Create agent in `src/hr_bot/deep_agents/subagents/`
2. Define tools and system prompt
3. Register with manager's delegation tools

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for agent framework
- OpenAI for language models
- Serper for web search API
- FAISS for vector search
