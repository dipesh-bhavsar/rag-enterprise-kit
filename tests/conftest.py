import os

# Ensure mock keys are always set during tests
os.environ["OPENAI_API_KEY"] = "mock-openai-key"
os.environ["COHERE_API_KEY"] = "mock-cohere-key"
