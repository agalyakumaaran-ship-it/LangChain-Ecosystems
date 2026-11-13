
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

PDF_PATH = BASE_DIR / (
    "Duster_Hybrid_Features.pdf"
)

FAISS_PATH = BASE_DIR / "faiss_index"

load_dotenv(ENV_PATH, override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Check your .env file."
    )


# --------------------------------------------------
# 2. INITIALIZE OPENAI MODELS
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)


# --------------------------------------------------
# 3. LOAD OR CREATE THE FAISS VECTOR DATABASE
# --------------------------------------------------

def create_or_load_vector_db():

    index_file = FAISS_PATH / "index.faiss"
    metadata_file = FAISS_PATH / "index.pkl"

    # Reuse the existing local vector database
    if index_file.exists() and metadata_file.exists():

        print("Loading existing FAISS vector database...")

        vector_db = FAISS.load_local(
            str(FAISS_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("FAISS database loaded successfully.")
        return vector_db

    # Build the database from the PDF on the first run
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place the PDF in the same folder as rag_app.py."
        )

    print("Loading PDF document...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    if not documents:
        raise ValueError(
            "No text was extracted from the PDF."
        )

    print(f"Loaded {len(documents)} PDF pages.")

    # Split the document into searchable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks.")
    print("Generating embeddings and building FAISS database...")

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save the vector database locally
    FAISS_PATH.mkdir(parents=True, exist_ok=True)

    vector_db.save_local(str(FAISS_PATH))

    print("FAISS database created and saved successfully.")

    return vector_db


# --------------------------------------------------
# 4. RETRIEVE RELEVANT INFORMATION
# --------------------------------------------------

def retrieve_context(vector_db, question):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    relevant_documents = retriever.invoke(question)

    context_parts = []

    for doc in relevant_documents:
        page_number = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"Source: PDF page {page_number}\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    return context, relevant_documents


# --------------------------------------------------
# 5. GENERATE AN ANSWER USING THE OPENAI LLM
# --------------------------------------------------

def ask_question(vector_db, question):

    context, source_documents = retrieve_context(
        vector_db,
        question
    )

    prompt = f"""
You are a helpful assistant answering questions
about the supplied hybrid SUV reference document.

Instructions:
1. Answer using only the retrieved document context.
2. Do not invent facts or specifications.
3. If the answer is not supported by the context,
   clearly say that the document does not provide
   enough information to answer.
4. Give a clear, concise answer.
5. Refer to the relevant PDF page numbers when possible.

Retrieved document context:
{context}

User question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n" + "=" * 55)
    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(response.content)

    print("\nSOURCE PAGES:")

    pages = sorted({
        doc.metadata.get("page", 0) + 1
        for doc in source_documents
    })

    print(", ".join(str(page) for page in pages))

    print("=" * 55)


# --------------------------------------------------
# 6. MAIN APPLICATION
# --------------------------------------------------

def main():

    print("\nHYBRID SUV DOCUMENT RAG APPLICATION")
    print("-" * 40)

    vector_db = create_or_load_vector_db()

    print("\nYou can now ask questions about the PDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        question = input("Ask a question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("RAG application closed.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            ask_question(vector_db, question)

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
