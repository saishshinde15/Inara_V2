<p align="center">
  <img src="https://img.shields.io/badge/Version-2.0.0-blue?style=for-the-badge" alt="Version 2.0.0"/>
  <img src="https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/AWS-Bedrock-orange?style=for-the-badge&logo=amazon-aws" alt="AWS Bedrock"/>
  <img src="https://img.shields.io/badge/LangChain-1.1.2-purple?style=for-the-badge" alt="LangChain"/>
</p>

# 🌟 Inara – Universal Enterprise Assistant

**Inara** is a production-ready AI assistant that combines internal company knowledge with real-time web search capabilities. Powered by a sophisticated **Manager-Subagent architecture** and **Amazon Nova Lite**, it delivers accurate, well-synthesized answers for any question.

> **Version 2.0.0** – Now with Web Search Toggle, Multi-Domain Response Synthesis, and Semantic Caching.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎛️ **Web Search Toggle** | Switch between HR-only (fast, private) and full web search modes |
| 🧠 **Intelligent Routing** | Manager Agent automatically routes queries to the right specialist |
| ⚡ **Parallel Execution** | Multi-domain queries run HR + General agents simultaneously |
| 🔄 **Semantic Caching** | Repeat queries served in <1s via intelligent response cache |
| 📚 **Clean Source Citations** | Professional, filename-only citations at the end of every response |
| 💬 **Conversation Memory** | Persistent thread-based memory with SQLite checkpointing |
| 🛡️ **Content Safety** | Built-in input/output filtering for safe responses |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Manager Agent (Amazon Nova Lite)                   │
│            Intelligent Query Routing & Response Synthesis            │
└───────────────────────┬───────────────────┬─────────────────────────┘
                        │                   │
           ┌────────────▼────────┐ ┌────────▼────────────┐
           │    HR Subagent      │ │  General Subagent   │
           │  (Nova Lite + RAG)  │ │ (Nova Lite + Search)│
           └────────────┬────────┘ └────────┬────────────┘
                        │                   │
           ┌────────────▼────────┐ ┌────────▼────────────┐
           │ • HR Document RAG   │ │ • Serper Web Search │
           │ • Master Actions    │ │ • Finance/Tax Info  │
           │ • Policy Retrieval  │ │ • General Knowledge │
           └─────────────────────┘ └─────────────────────┘
```

### Routing Logic

| Query Type | Web Search ON | Web Search OFF |
|------------|---------------|----------------|
| **HR Policy** | → HR Subagent | → HR Subagent |
| **External Info** | → General Subagent (Web) | ❌ Not Available (suggests enabling) |
| **Mixed Query** | → **Parallel** (Both agents) | → HR Subagent only |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **AWS Credentials** – For Bedrock (Amazon Nova Lite)
- **Serper API Key** – For web search
- **S3 Bucket** – For HR document storage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Inara_V2.git
cd Inara_V2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required Environment Variables:**

```env
# AWS Bedrock
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1

# Web Search
SERPER_API_KEY=your_serper_key

# S3 Document Storage
S3_BUCKET_NAME=your-hr-docs-bucket

# Application
BOT_NAME=Inara
COMPANY_NAME=YourCompany
HR_EMAIL=hr@yourcompany.com
```

### Running

#### 🖥️ Chainlit UI (Recommended)

```bash
chainlit run src/hr_bot/ui/chainlit_app.py -w --port 8501
```

Then open [http://localhost:8501](http://localhost:8501)

#### 💻 CLI Interactive Mode

```bash
python -m hr_bot.main --interactive
```

#### 🔌 Single Query

```bash
python -m hr_bot.main --query "What is our maternity leave policy?"
```

---

## 🎛️ Web Search Toggle

The UI includes a toggle button to control web search behavior:

| Mode | Behavior | Use Case |
|------|----------|----------|
| 🌐 **ON** | Full architecture with web search | Complex queries, external info |
| 🔒 **OFF** | HR documents only (faster) | Confidential policy lookups |

When OFF, if the HR Subagent can't find an answer, it will suggest enabling web search.

---

## 📁 Project Structure

```
Inara_V2/
├── src/hr_bot/
│   ├── main.py                    # CLI entry point
│   ├── config/
│   │   └── settings.py            # Pydantic settings
│   ├── deep_agents/
│   │   ├── manager.py             # Manager Agent (routing + synthesis)
│   │   └── subagents/
│   │       ├── hr_subagent.py     # HR specialist (RAG)
│   │       └── general_subagent.py # General specialist (Web Search)
│   ├── tools/
│   │   ├── hr_rag_tool.py         # Hybrid RAG search
│   │   ├── master_actions_tool.py # Procedural guidance
│   │   └── finance_search_tool.py # Serper web search
│   ├── ui/
│   │   └── chainlit_app.py        # Chainlit UI
│   └── utils/
│       ├── cache.py               # Semantic response caching
│       ├── memory.py              # Conversation memory
│       ├── content_safety.py      # Input/output filtering
│       └── s3_loader.py           # S3 document loader
├── data/                          # HR document storage
├── tests/                         # Unit tests
└── pyproject.toml                 # Project configuration
```

---

## 🔧 API Reference

### Python API

```python
from hr_bot.main import chat, run

# Full response with metadata
result = chat(
    query="What is our expense policy?",
    web_search_enabled=True
)
print(result["output"])
print(result["agents_consulted"])

# Simple string response
response = run("How do I submit a leave request?")
print(response)
```

### Async Usage

```python
import asyncio
from hr_bot.main import achat

async def main():
    result = await achat("What are the health benefits?")
    print(result["output"])

asyncio.run(main())
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hr_bot --cov-report=html
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **HR-Only Query** | ~6-8 seconds |
| **Web Search Query** | ~10-15 seconds |
| **Parallel (Both Agents)** | ~15-20 seconds |
| **Cached Response** | <1 second |

---

## 🔐 Security

- ✅ API keys stored in environment variables
- ✅ Session isolation for multi-user support
- ✅ Content safety filtering (input/output)
- ✅ OAuth authentication via Chainlit
- ✅ Role-based access control (Executive/Employee)

---

## 🛠️ Development

### Adding New Tools

```python
from langchain_core.tools import tool

@tool
def my_new_tool(query: str) -> str:
    """Tool description for LLM."""
    return "Tool response"
```

### Adding New Subagents

1. Create agent in `src/hr_bot/deep_agents/subagents/`
2. Define tools and system prompt
3. Register with Manager's delegation tools

---

## 📝 Changelog

### v2.0.0 (2025-12-25)

- ✨ **Web Search Toggle** – Switch between HR-only and full web search modes
- ✨ **Multi-Domain Response Synthesis** – Cohesive answers for mixed queries
- ✨ **Semantic Caching** – Intelligent response caching for repeat queries
- ✨ **Clean Source Citations** – Professional filename-only citations
- 🐛 **Fixed Memory Pollution** – Warmup queries no longer affect user sessions
- 🐛 **Fixed Source Visibility** – Sources now properly appear in UI
- ⚡ **Improved Parallel Execution** – Faster multi-domain query handling

### v1.0.0

- Initial release with Manager-Subagent architecture
- HR RAG and General Web Search capabilities

---

## 📄 License

MIT License – See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) – Agent framework
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) – Nova Lite model
- [Serper](https://serper.dev) – Web search API
- [FAISS](https://faiss.ai) – Vector search
- [Chainlit](https://chainlit.io) – Chat UI framework

---

<p align="center">
  <b>Built with ❤️ for enterprise AI</b>
</p>
