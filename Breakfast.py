import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Set your API key directly
os.environ["OPENAI_API_KEY"] = ""
# Initialize the Chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define the prompt template for breakfast ideas
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert breakfast chef. Create quick, delicious, and"
            " nutritious breakfast ideas using the ingredients provided."
        ),
    ),
    ("user", "I want to make a breakfast using: {ingredients}"),
])

# Format the prompt with real breakfast ingredients to see how it looks
formatted_prompt = prompt_template.format_messages(
    ingredients="Eggs, bread, avocado, spinach, cheese"
)
print("Formatted Prompt Look:")
for message in formatted_prompt:
    print(f"{message.type.upper()}: {message.content}")

# Create the chain using LCEL
chain = prompt_template | llm | StrOutputParser()

# Run (invoke) the chain with breakfast ingredients
print("\n--- Generating Breakfast Recipe ---")
response = chain.invoke({"ingredients": "Eggs, bread, avocado, spinach, cheese"})
print(response)