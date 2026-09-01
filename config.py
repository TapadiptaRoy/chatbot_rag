from dotenv import load_dotenv
import os

load_dotenv()  # reads .env and loads it into os.environ

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
CHROMA_TENANT = os.environ.get("CHROMA_TENANT")
CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")