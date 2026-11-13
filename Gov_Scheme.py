import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set page config
st.set_page_config(page_title="TN Government Schemes Chatbot", page_icon="🏛️")
st.title("🏛️ Tamil Nadu Government Schemes Assistant")

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load and cache the vector store so it scrapes only once
@st.cache_resource
def initialize_rag():
    url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = initialize_rag()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """You are a helpful assistant answering questions about Tamil Nadu government welfare schemes. 
Answer strictly based on the provided context. If you don't know, say you don't know.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt_text := st.chat_input("Ask about any welfare scheme (e.g., UYEGP)..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching schemes..."):
            response = rag_chain.invoke(prompt_text)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})