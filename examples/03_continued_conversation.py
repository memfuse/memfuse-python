from memfuse.llm import OpenAI
from memfuse import MemFuse
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Initialize MemFuse client
memfuse = MemFuse()
 
# Create a memory instance with a specific user identifier
memory = memfuse.init(user="bob", agent="astro_bot", session="astro_bot_session")
 
# Initialize OpenAI client with memory
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    memory=memory
)

print(f"Using memory for conversation: {memory}")

# First message - the initial query
response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are three interesting facts about the Moon?"}],
)
 
print("\n--- First Response ---")
print(response1.choices[0].message.content)

# Second message - follow-up question (without needing to repeat context)
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "How does that compare to Mars?"}],
)

print("\n--- Second Response ---")
print(response2.choices[0].message.content)

# Third message - another follow-up
response3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Which would be easier to establish a human colony on?"}],
)

print("\n--- Third Response ---")
print(response3.choices[0].message.content)

# Fourth message - follow-up about challenges
response4 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are the biggest challenges humans would face on each?"}],
)

print("\n--- Fourth Response ---")
print(response4.choices[0].message.content)

# Fifth message - follow-up about resources
response5 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What resources could be harvested from either location?"}],
)

print("\n--- Fifth Response ---")
print(response5.choices[0].message.content)

# Sixth message - testing persistent memory with a related follow-up about Europa
response6 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What about Europa, Jupiter's moon? How would it compare with the two we discussed?"}],
)

print("\n--- Sixth Response (Testing Persistent Memory) ---")
print(response6.choices[0].message.content)

# The LLM remembers the entire conversation context thanks to MemFuse
print("\nMemFuse automatically maintained context across all interactions!") 