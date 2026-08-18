import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_path = "introtoml.pdf"

loader = PyPDFLoader(PDF_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
splits = splitter.split_documents(docs)

emb = OpenAIEmbeddings(model = "text-embedding-3-small")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type = 'similarity', search_kwargs = {"k" : 4})

prompt = ChatPromptTemplate.from_messages([
    ('system', "answer only from the provided context. if not found, say you don't know"),
    ("human", "Question : {question} \n \n Context : \n {context}")
])

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context" : retriever | RunnableLambda(format_docs),
    "question" : RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

print("pdf rag only")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)