from langchain_ollama import ChatOllama

# initialize local llama3 model
llm = ChatOllama(model="llama3", temperature=0.7)

# test prompt
response = llm.invoke("Explain AI in one sentence.")

print("\nResponse from Llama3:\n")
print(response.content)
