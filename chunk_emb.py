# pdf_to_pinecone.py

from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import re
import os
from dotenv import load_dotenv
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
index_name = "bajaj-chat"

loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path="D:\ASSIGNMENT\Aj_assigment\data",
        glob="*.pdf",
    ),
    blob_parser=PyMuPDFParser(),
)
pdf_docs = loader.load()
print(f"Loaded {len(pdf_docs)} PDFs")

def fully_clean(text):
    # remove extra blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # strip spaces
    return text.strip()
def recursive_chunk(text, chunk_size=1500, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?"] )
    return splitter.split_text(text)


pc = Pinecone(api_key=pinecone_api_key)

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,  # matches text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)
# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

for pdf in pdf_docs:
    raw_text = pdf.page_content
    metadata = pdf.metadata or {}
    cleaned_text = fully_clean(raw_text)
    chunks = recursive_chunk(cleaned_text)
    doc_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk.split()) < 10:  # skip tiny chunks
            continue
        doc_chunks.append(
            Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_id": f"{metadata.get('source', 'pdf')}_chunk_{i+1}"
                }
            
            )
        )
    
    if doc_chunks:
        vector_store.add_documents(doc_chunks)
        print(f"Added {len(doc_chunks)} chunks from PDF: {metadata.get('source', 'unknown')}")

