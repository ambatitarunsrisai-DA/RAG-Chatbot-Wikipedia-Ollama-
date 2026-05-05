from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ✅ Load dataset (increase size for better answers)
ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.simple",
    split="train[:5%]"   # increase if your system can handle it
)

# ✅ Extract clean text
texts = [item["text"] for item in ds if item["text"]]

# ✅ Limit for performance (adjust based on your PC)
texts = texts[:1000]

print(f"Using {len(texts)} documents")

# ✅ Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.create_documents(texts)

print(f"Created {len(docs)} chunks")

# ✅ Embeddings (Ollama must be running)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ✅ Store in Chroma
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ Ingestion complete")