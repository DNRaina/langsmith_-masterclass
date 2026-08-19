import os
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

pdf_path = "introtoml.pdf"


@traceable(name = 'load_pdf')
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

@traceable(name = 'split_documents')
def split_documents(docs, chunk_size = 1000, chunk_overlap = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap = 100
    )
    return splitter.split_documents(docs)

@traceable(name = 'build_vectorstore')
def build_vectorstore(splits):
    emb = OpenAIEmbeddings(model = 'text-embedding-3-small')
    return FAISS.from_documents(emb, splits)

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150):
    
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str):

    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | StrOutputParser()

    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)

if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(pdf_path, q)
    print("\nA:", ans)