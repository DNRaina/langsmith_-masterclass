# LangSmith Masterclass

A practical, hands-on repository demonstrating LangChain and LangSmith capabilities, ranging from fundamental LLM invocations to sequential LCEL chains and RAG (Retrieval-Augmented Generation) workflows.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| [`1_simple_llm_call.py`](file:///c:/D_Drive/Python%20Projects/langsmith/1_simple_llm_call.py) | Basic LLM call using LCEL (`PromptTemplate | ChatOpenAI | StrOutputParser`). |
| [`2_sequential_chain.py`](file:///c:/D_Drive/Python%20Projects/langsmith/2_sequential_chain.py) | Two-step sequential chain: generates a detailed report on a topic and then produces a 5-point summary. |
| [`3_rag_v1_notracing.py`](file:///c:/D_Drive/Python%20Projects/langsmith/3_rag_v1_notracing.py) | RAG pipeline over a PDF document using FAISS vector store, OpenAI embeddings, and LCEL parallel runnables. |
| [`3_rag_v2.py`](file:///c:/D_Drive/Python%20Projects/langsmith/3_rag_v2.py) | RAG pipeline with custom LangSmith `@traceable` annotations for tracking document loading, splitting, vector store building, and custom run naming. |
| [`3_rag_v3.py`](file:///c:/D_Drive/Python%20Projects/langsmith/3_rag_v3.py) | RAG pipeline demonstrating run nesting, custom tagging (`tags=["setup"]`), and wrapping the entire setup and query flow inside a traceable run. |
| [`3_rag_v4.py`](file:///c:/D_Drive/Python%20Projects/langsmith/3_rag_v4.py) | Advanced RAG pipeline with local index persistence and caching. Uses file fingerprinting and config hashing to skip rebuilds, demonstrating conditional trace paths (`load_index` vs `build_index`) in LangSmith. |

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

### 4. RAG Pipeline with Custom Tracing
Enhances the RAG pipeline using LangSmith's `@traceable` decorator to log non-LangChain steps (PDF loading, document splitting, and vector database building) and sets custom run names.
```bash
python 3_rag_v2.py
```

### 5. RAG Pipeline with Nesting and Tagging
Demonstrates run nesting (wrapping setup and query flows inside a parent traceable run `pdf_rag_full_run`) and applying custom tags (such as `tags=["setup"]`) for run categorisation and filtering in LangSmith.
```bash
python 3_rag_v3.py
```

### 6. RAG Pipeline with Index Caching & Local Persistence
Demonstrates locally caching and loading the FAISS vector database based on PDF file fingerprints (SHA-256, size, mtime) and indexing configurations (chunk size, overlap, embedding model). Conditionally routes traces to either `load_index_run` or `build_index_run` in LangSmith depending on cache status.
```bash
python 3_rag_v4.py
```

> **Note**: Make sure to place your target PDF (e.g. `introtoml.pdf`) in the project directory before running any of the RAG scripts.

---

## 🛠️ LangSmith Features Explored
- **LLM Run Tracing & Observability**: Inspect prompts, token usage, latency, and intermediate outputs.
- **LCEL Chain Debugging**: Detailed visibility into sequential steps and parallel runnables (`RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`).
- **RAG Debugging**: Trace document retrieval, context formatting, and grounded LLM answers.
- **Custom Python Tracing (`@traceable`)**: Instrument custom/standalone Python functions (e.g., document loading, splitting, and vector database building) to see them in the trace tree.
- **Run Custom Naming & Metadata**: Set custom names for LCEL runs using the configuration object (e.g., `run_name: pdf_rag_query`).
- **Run Nesting & Tagging**: Nest multiple custom traced runs inside a parent run (`pdf_rag_full_run`) and add custom tags (`tags=["setup"]`) to filter runs easily.
- **Conditional Branch Tracing**: Observe separate conditional paths (e.g., loading from vs. building an index) traced in real time, keeping your trace histories accurate to the code paths actually executed.