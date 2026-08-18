# LangSmith Masterclass

A practical, hands-on repository demonstrating LangChain and LangSmith capabilities, ranging from fundamental LLM invocations to sequential LCEL chains and RAG (Retrieval-Augmented Generation) workflows.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| [`1_simple_llm_call.py`](file:///c:/D_Drive/Python%20Projects/langsmith/1_simple_llm_call.py) | Basic LLM call using LCEL (`PromptTemplate | ChatOpenAI | StrOutputParser`). |
| [`2_sequential_chain.py`](file:///c:/D_Drive/Python%20Projects/langsmith/2_sequential_chain.py) | Two-step sequential chain: generates a detailed report on a topic and then produces a 5-point summary. |
| [`3_rag_v1_notracing.py`](file:///c:/D_Drive/Python%20Projects/langsmith/3_rag_v1_notracing.py) | RAG pipeline over a PDF document using FAISS vector store, OpenAI embeddings, and LCEL parallel runnables. |

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone or open the project and create a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

Install the required packages:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith Tracing Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT="langsmith-masterclass"
```

---

## 🏃 Usage & Examples

### 1. Simple LLM Call
Demonstrates a minimal prompt-to-response chain using LangChain Expression Language (LCEL).
```bash
python 1_simple_llm_call.py
```

### 2. Sequential Chain
Demonstrates chaining multiple LLM tasks sequentially (Report Generation ➔ Summarization).
```bash
python 2_sequential_chain.py
```

### 3. RAG Pipeline (PDF Question Answering)
Loads a PDF document (e.g., `introtoml.pdf`), chunks it, embeds with `text-embedding-3-small`, indexes into FAISS, and answers questions strictly based on the retrieved context.
```bash
python 3_rag_v1_notracing.py
```
> **Note**: Make sure to place your target PDF (e.g. `introtoml.pdf`) in the project directory before running the RAG script.

---

## 🛠️ LangSmith Features Explored
- **LLM Run Tracing & Observability**: Inspect prompts, token usage, latency, and intermediate outputs.
- **LCEL Chain Debugging**: Detailed visibility into sequential steps and parallel runnables (`RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`).
- **RAG Debugging**: Trace document retrieval, context formatting, and grounded LLM answers.