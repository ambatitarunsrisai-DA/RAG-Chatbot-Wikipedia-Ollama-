# RAG Chatbot using Wikipedia & Ollama

## 🚀 Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot that retrieves relevant information from a knowledge base (Wikipedia data) and generates responses using an LLM powered by Ollama.

## 🛠 Tech Stack
- Python
- LangChain
- Ollama (Local LLM)
- ChromaDB (Vector Database)

## ⚙️ Features
- Semantic search using embeddings
- Context-aware response generation
- Local LLM inference (no API cost)

## 📂 Project Structure
- app.py → Main application
- ingest.py → Data ingestion & embedding
- query.py → Query handling
- chroma_db/ → Vector database storage

## ▶️ How to Run
```bash
pip install -r requirements.txt
python ingest.py
python query.py
