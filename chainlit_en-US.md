# Inara HR Assistant V2

Welcome to the Chainlit edition of the **Inara HR Assistant V2**. This UI wraps the new Deep Agent architecture with:

## Architecture
- **Manager Agent** (GPT-5 Nano): Intelligent query routing and response synthesis
- **HR Subagent** (AWS Nova Lite): HR policies, procedures, and employee handbook queries
- **Finance Subagent** (AWS Nova Lite): General finance questions via web search

## Features
- 🔐 **Role-based access control** driven by `EXECUTIVE_EMAILS` and `EMPLOYEE_EMAILS` environment variables
- 📚 **Secure document retrieval** from S3 with Hybrid RAG (semantic + keyword search)
- 🧠 **Persistent conversation memory** using SQLite checkpointer with Titan embeddings
- 🔄 **Admin actions** to refresh the HR knowledge base and clear semantic cache
- 📝 **Inline source citations** pulled directly from policy documents

## Getting Started
Sign in with your corporate Google account to get started.

## What I Can Help With

### 📋 HR Policies & Procedures
- Leave policies (PTO, sick leave, vacation, maternity/paternity)
- Employee benefits (health, dental, vision, 401k)
- Company handbook questions
- Performance reviews and management
- Expense reports and reimbursements
- Onboarding/offboarding processes

### 💰 General Finance Information
- Investment concepts and terminology
- Tax regulations and guidelines
- Market trends and economic news
- Financial concepts explained

---

**Just type your question below to get started!**

_Tip: Be specific for better answers. Example: "What is the process to request maternity leave?"_

Need help? Contact the HR Tech team.
