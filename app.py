import os
import shutil
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
)

print("API Key loaded:", api_key[:10], "...")

DATA_PATH = "data"
DB_PATH = "chroma_db"

# Cleanup old database
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("Old database removed.")

# Load documents
loaders = [
    DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader),
    DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
]

documents = []
for loader in loaders:
    documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Clean metadata
for chunk in chunks:
    chunk.metadata = {}

# Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=DB_PATH
)

# db.persist()
print("Vector database created successfully.")
