import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from a .env file if it exists
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Chef Assistant", page_icon="👩‍🍳")
st.title("👩‍🍳 AI Chef Assistant")

# Check for API key in environment variables first
api_key = os.getenv("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")

# If not found in environment, provide a secure sidebar input as a fallback
if not api_key:
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar or set it in your .env file to get started.")
        st.stop()

st.write("Give me your ingredients or ask follow-up questions!")

# Initialize the Chat model using the detected key
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

# Define the prompt template with chat history placeholder
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert chef. Create quick recipes using the ingredients provided and remember all previous context and follow-ups."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{ingredients}")
])

# Create the base chain
base_chain = prompt_template | llm | StrOutputParser()

# Use Streamlit session state to persist the chat store across reruns
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

# Wrap the chain with message history capability
conversational_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="ingredients",
    history_messages_key="chat_history"
)

config = {"configurable": {"session_id": "chef_streamlit_session"}}

# Initialize message history display in Streamlit UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input from the chat box
if user_input := st.chat_input("Type your ingredients or ask a follow-up question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        for chunk in conversational_chain.stream(
            {"ingredients": user_input},
            config=config
        ):
            full_response += chunk
            response_container.markdown(full_response + "▌")
            
        response_container.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})