# 🧠 Bajaj Financial Report Chatbot (RAG Pipeline)

This project is an **end-to-end Retrieval-Augmented Generation (RAG)** pipeline that loads PDF reports, extracts and embeds their content into **Pinecone**, and allows users to query them using **OpenAI GPT-4** via both **FastAPI** and **Streamlit UI**.

---

## 📁 Project Structure

├── .env # API keys and environment variables
├── chunk_emb.py # Loads, cleans, chunks, and embeds PDFs into Pinecone
├── rag.py # Handles document retrieval and LLM querying
├── fast.py # FastAPI backend endpoint (/ask)
├── ui_app.py # Streamlit-based user interface
├── requirements.txt # Project dependencies



## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bajaj-rag-chatbot.git
cd bajaj-rag-chatbot
2. Create a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
3. Install Dependencies
pip install -r requirements.txt
🔑 Environment Variables (.env)
Your .env file should contain:


export PINECONE_API_KEY="your-pinecone-key"
export OPENAI_API_KEY="your-openai-key"
Load it using:

from dotenv import load_dotenv
load_dotenv()
🧩 Pipeline Overview
1. Data Ingestion and Embedding (chunk_emb.py)
Loads all PDF files from a directory.

Cleans text and splits it into overlapping chunks.

Embeds each chunk using text-embedding-3-large from OpenAI.

Stores the embeddings in Pinecone with metadata.

Run:
python chunk_emb.py
2. Retrieval and Generation (rag.py)
Retrieves top-k similar chunks from Pinecone.

Combines them into a context.

Passes the query and context to gpt-4o for generating a precise financial report summary.
Run:
python rag.py
3. FastAPI Backend (fast.py)
A simple REST API that handles user queries:

Endpoint:
POST /ask
Request Body:
json
Copy code
{
  "query": "summary of Bajaj Finance profit"
}
Response:

json
Copy code
{
  "answer": "Bajaj Finance reported a profit of 1200 and no losses."
}
Run:
uvicorn fast:app --reload
4. Streamlit User Interface (ui_app.py)
A simple chatbot-style UI for querying the embedded PDFs interactively.

Run:
streamlit run ui_app.py
🧠 How It Works
Data Loading: PDFs are read and parsed using PyMuPDFParser.

Chunking: Text is split using RecursiveCharacterTextSplitter for optimal embedding quality.

Embedding: Each chunk is embedded using OpenAIEmbeddings.

Storage: Chunks are saved to Pinecone as vectors.

Querying: User inputs are embedded and matched against stored vectors.

Response Generation: Retrieved context is fed into GPT-4o for accurate, context-aware answers.

🧾 Example Query Flow
User:

“What is the financial summary of Bajaj Finance?”

System:

Retrieves relevant chunks from Pinecone.

Generates a concise JSON-style report using the LLM.

🚀 Future Improvements
Add authentication to FastAPI endpoints.

Enable support for multiple companies and document types.

Integrate caching for frequent queries.

Add visual analytics in Streamlit dashboard.

🧑‍💻 Author
Akshay Jejurkar
Data Engineer / ML Developer
📧 [akshayjejurkar24@gmail.com]
