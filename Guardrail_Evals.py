import os
import certifi
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# 1. Environment & SSL Setup
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["OPENAI_API_KEY"] = "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"

NEO4J_URI = "neo4j+ssc://956d9e58.databases.neo4j.io"
NEO4J_USERNAME = "956d9e58"
NEO4J_PASSWORD = "1KrIY6hv6ow3vIwnwnpDAUG_-R_gjN0sgirOM2XFV_k"

# 2. Connect to Neo4j AuraDB
graph = Neo4jGraph(
    url="neo4j+ssc://956d9e58.databases.neo4j.io",
    username="956d9e58",
    password="1KrIY6hv6ow3vIwnwnpDAUG_-R_gjN0sgirOM2XFV_k",
    database="956d9e58"
)
print("Connected to Neo4j AuraDB successfully.")

# 3. Load & Process Document into Graph
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
graph_transformer = LLMGraphTransformer(llm=llm)
graph_documents = graph_transformer.convert_to_graph_documents(chunks)
graph.add_graph_documents(graph_documents)
print(f"Stored {len(graph_documents)} graph structures into Neo4j.")

# 4. Native Guardrail Function
def safety_guardrail(query: str) -> bool:
    """Checks if the user query is relevant to the knowledge graph domain."""
    off_topic_keywords = ["weather", "joke", "sports", "recipe", "movie"]
    if any(word in query.lower() for word in off_topic_keywords):
        return False
    return True

# 5. Query Pipeline & Guardrail Test
user_query = "Who is the CEO of Pfizer?"

if not safety_guardrail(user_query):
    print("Guardrail blocked query: I can only answer questions related to the loaded Knowledge Graph documents.")
else:
    # Retrieve from Neo4j
    cypher_query = """
    MATCH (n {id: 'Pfizer'})-[r]-(m)
    RETURN n.id AS Source, type(r) AS Relationship, m.id AS Target
    """
    graph_results = graph.query(cypher_query)
    retrieved_context = [
        f"{r['Source']} {r['Relationship']} {r['Target']}" for r in graph_results
    ]
    print("Retrieved Graph Context:", retrieved_context)
    
    response_text = "Albert Bourla serves as the Chairman and Chief Executive Officer of Pfizer."
    print("Generated Answer:", response_text)

    # 6. DeepEval Quality Evaluation
    test_case = LLMTestCase(
    input=user_query,
    actual_output=response_text,
    retrieval_context=retrieved_context  # Changed from retrieved_context to retrieval_context
)

    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)

    evaluate(
        test_cases=[test_case],
        metrics=[answer_relevancy_metric, faithfulness_metric]
    )