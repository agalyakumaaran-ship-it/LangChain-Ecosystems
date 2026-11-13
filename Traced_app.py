
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load API key from .env beside this script
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

# Define three prompts before using them
prompts = [
    "What are tokens in large language models?",
    "What is the difference between tokens and words?",
    "Why does token count matter when using an LLM?"
]

# Send each prompt and print its answer
for i, prompt in enumerate(prompts, start=1):
    print(f"\n--- Run {i} ---", flush=True)
    print(f"Prompt: {prompt}", flush=True)

    response = model.invoke(prompt)

    print(f"Answer: {response.content}", flush=True)
