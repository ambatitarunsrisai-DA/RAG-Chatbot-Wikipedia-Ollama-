# -------------------------------
# CLEAN LOGS (MUST BE FIRST)
# -------------------------------
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# -------------------------------
# Imports
# -------------------------------
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🧠 RAG Chatbot (Wikipedia + Ollama)")
st.write("Ask questions from your local knowledge base")

# -------------------------------
# Load system (cached)
# -------------------------------
@st.cache_resource
def load_system():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    llm = OllamaLLM(
        model="llama3.2:3b",
        temperature=0.2
    )

    return retriever, llm


retriever, llm = load_system()

# -------------------------------
# Chat history
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------------
# User input
# -------------------------------
query = st.chat_input("Ask something...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                # Retrieve docs
                docs = retriever.invoke(query)

                st.caption(f"Retrieved docs: {len(docs)}")

                # Filter docs
                filtered_docs = []
                for d in docs:
                    if d.page_content:
                        text = d.page_content.strip()
                        if len(text) > 50:
                            filtered_docs.append(text)

                context = "\n\n".join(filtered_docs[:5])

                # No context case
                if not context:
                    answer = "I don't know (no relevant context found)."
                else:
                    prompt = f"""
You are a precise AI assistant.

Rules:
- Use ONLY the provided context
- Do NOT use outside knowledge
- Answer in 3-5 clear sentences
- If answer not present, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""
                    answer = llm.invoke(prompt)

            except Exception as e:
                answer = f"Error: {str(e)}"

            st.write(answer)

            # Show sources
            if docs:
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(docs):
                        if doc.page_content:
                            st.write(f"**Source {i+1}:**")
                            st.write(doc.page_content[:300] + "...")
                            st.divider()

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )