import os
from dotenv import load_dotenv

load_dotenv()

def _setting(name: str, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return value
    except Exception:
        pass
    return default

GROQ_API_KEY = _setting("GROQ_API_KEY")
MODEL = _setting("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = _setting("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(_setting("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(_setting("SIMILARITY_THRESHOLD", "0.34"))

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to .env for local use "
        "or Streamlit Cloud Secrets for deployment."
    )
