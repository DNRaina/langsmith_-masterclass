import os
from dotenv import load_dotenv
from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith env (make sure these are set) ---
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=pdf_rag_demo

load_dotenv()

PDF_PATH = "introtoml.pdf"

@traceable(name = 'load_pdf')
def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)
    return loader.load() #list [Document]

@traceable(name = 'split_documents')
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap = 100
    )
    return splitter.split_documents(docs)

@traceable(name = "build_vs")
def build_vs(splits):
    emb = OpenAIEmbeddings(model = "text-embedding-3-small")
    vs = FAISS.from_documents(splits, emb)
    return vs

@traceable(name = 'setup_pipeline')
def setup_pipeline(pdf_path):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vs(splits)
    return vs

llm = ChatOpenAI(model = 'gpt-4o-mini', temperature= 0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "answer only from the provided context. If you don't know, say I don't know."), # <-- Added comma here
    ("human", "Question: {question} \n\n Context  \n {context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type = 'similarity', search_kwargs = {"k":4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question" : RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name": "pdf_rag_query"
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)