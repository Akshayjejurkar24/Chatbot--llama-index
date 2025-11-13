from fastapi import FastAPI
from pydantic import BaseModel
from rag import *

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_bajaj(request: QueryRequest):
    results = retrieve_similar_documents(request.query)
    answer = query_llm_with_context(request.query, results, system_prompt)
    return {"answer": answer}
