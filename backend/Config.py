
import os
from dotenv import load_dotenv

# Load environment variables from the project's .env file.
load_dotenv()

# OpenAI configuration.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Neo4j graph database configuration.
NEO4J_URI = os.getenv("NEO4J_URI", "NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# FAISS vector database configuration.
FAISS_PATH = os.getenv("FAISS_PATH", "./faiss_store")

# OpenAI embedding model for document and query vectors.
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Document chunking configuration.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Number of vector search results per question.
TOP_K = int(os.getenv("TOP_K", "4"))
