
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

system_prompt = """
You are an assistant. 
Return the financial report in JSON format ONLY with the following structure:
[
  {"Company": "Bajaj Finance", "Profit": 1200, "Loss": 0},
  {"Company": "Bajaj Housing Finance", "Profit": 800, "Loss": 0}
]
Do not include extra text.
"""



pinecone_api_key = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
index_name = "bajaj-chat"

query = "summary of thebajaj"

def retrieve_similar_documents(query: str, top_k=10):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    dense_encoder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    vector_store = PineconeVectorStore(embedding=dense_encoder, index=index)
    results = vector_store.similarity_search(query, k=top_k)
    return results
results = retrieve_similar_documents(query)
retrieved_texts = ""
for doc in results:
    metadata = doc.metadata
    retrieved_texts += f"\n[{metadata}] {doc.page_content}\n\n"

system_prompt = "You are a financial analyst. Answer precisely based on the provided context."

def query_llm_with_context(user_query: str, results, system_prompt: str = None):
    context = "\n\n".join([f"[{doc.metadata}] {doc.page_content}" for doc in results])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Question: {user_query}"}
    ]
    
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.5)
    response = llm.invoke(messages)
    return response.content

answer = query_llm_with_context(query, results, system_prompt)

print("-------------------Bajaj report --------------------")
print(answer)