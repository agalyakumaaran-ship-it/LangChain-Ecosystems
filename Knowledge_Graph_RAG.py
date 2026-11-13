import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# 1. Load the document
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

# 2. Initialize the text splitter & split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(documents)

# 3. Initialize the LLM with your OpenAI API key
llm = ChatOpenAI(
    api_key="Open_API_Key",
    temperature=0,
    model="gpt-4o-mini"
)

# 4. Extract nodes and relationships using the graph transformer
print("Extracting Graph data from text chunks...")
graph_transformer = LLMGraphTransformer(llm=llm)
graph_documents = graph_transformer.convert_to_graph_documents(chunks)

print(f"\nExtracted {len(graph_documents)} graph document structures.\n")
for i, doc in enumerate(graph_documents):
    print(f"--- Chunk {i+1} Graph ---")
    print("Nodes:", [f"{node.id} ({node.type})" for node in doc.nodes])
    print("Edges (Relationships):")
    for rel in doc.relationships:
        print(f"  {rel.source.id} --[{rel.type}]--> {rel.target.id}")

# 5. Connect to Neo4j AuraDB with self-signed certificate acceptance & store graph
NEO4J_URI = "neo4j+ssc://956d9e58.databases.neo4j.io"
NEO4J_USERNAME = "956d9e58"
NEO4J_PASSWORD = "1KrIY6hv6ow3vIwnwnpDAUG_-R_gjN0sgirOM2XFV_k"

graph = Neo4jGraph(
    url="neo4j+ssc://956d9e58.databases.neo4j.io",
    username="956d9e58",
    password="1KrIY6hv6ow3vIwnwnpDAUG_-R_gjN0sgirOM2XFV_k",
    database="956d9e58"
)

graph.add_graph_documents(graph_documents)
print("Successfully connected to Neo4j AuraDB and stored graph!")


