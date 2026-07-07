import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv(".env")  # Load environment variables from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Groq API key

EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"  # Model for generating embeddings, Sentence Transformers (dimension 512)


SG_LLM = "openai/gpt-oss-safeguard-20b"  # Model for safety moderation, Groq Provider
PROMPT_MODERATOR_SYSTEM_FILE = "prompts\\prompt_moderator_system.txt"  # Path to the system prompt file for the moderator


RAG_LLM = "llama-3.3-70b-versatile"
PROMPT_RAG_SYSTEM_FILE = "prompts\\prompt_rag_system.txt"  # Path to the system prompt file for RAG

