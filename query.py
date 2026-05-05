from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

# ✅ Load embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ✅ Load vector DB
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ✅ Better retriever (MMR = smarter search)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 12}
)

# ✅ Load LLM
llm = OllamaLLM(model="llama3.2:3b")


def ask_question(query):
    # 🔍 Retrieve docs
    docs = retriever.invoke(query)

    print(f"\nRetrieved docs: {len(docs)}")

    # ✅ Remove weak chunks
    docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]

    # 🔍 Debug (see what model reads)
    for i, doc in enumerate(docs[:2]):
        print(f"\n--- DOC {i} ---\n{doc.page_content[:300]}")

    # ✅ Build context
    context = "\n\n".join([doc.page_content for doc in docs])

    if not context.strip():
        return "No relevant context found. Try increasing dataset size."

    # ✅ Strong prompt
    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question clearly.

Rules:
- Answer in 3-5 sentences
- Be simple and informative
- Use only the context
- If partial info exists, still try to answer
- Only say "I don't know" if nothing is relevant

Context:
{context}

Question:
{query}

Answer:
"""

    # 🤖 Generate answer
    response = llm.invoke(prompt)

    return response


# ✅ Run test
if __name__ == "__main__":
    print(ask_question("What is artificial intelligence?"))