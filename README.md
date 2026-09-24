# AI Enterprise Knowledge Assistant

An AI-powered enterprise knowledge assistant built with RAG, FAISS semantic search, Sentence Transformers, Groq LLMs, LangGraph and Streamlit.

## Features

- HR, Technical and Project knowledge bases
- PDF ingestion and extraction
- Chunking with overlap
- Sentence Transformer embeddings
- FAISS semantic retrieval
- Manager Agent classification
- HR / Technical / Project / General specialist routing
- LangGraph orchestration
- Grounded Groq responses
- Source filename, page, department and similarity
- Conversation context
- Additional PDF upload during a session
- Unsupported-question handling

## Architecture

```text
User
  |
  v
Streamlit
  |
  v
Manager Agent
  |
  +---- HR Specialist
  +---- Technical Specialist
  +---- Project Specialist
  +---- General Specialist
  |
  v
FAISS + Sentence Transformers
  |
  v
Relevant PDF Chunks
  |
  v
Groq LLM
  |
  v
Grounded Answer + Sources
```

## Local setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Create `.env` using `.env.example` and add your Groq API key.

## Example questions

- What is the leave policy?
- How many casual leaves do employees receive?
- How do I deploy the application?
- What Python version should be used?
- What technology does Project Alpha use?
- Who owns the Project Alpha API?

For an unsupported question, the assistant should respond:

`I could not find this information in the available company documents.`

## Security

Never commit `.env` or API keys. For Streamlit Community Cloud, use the app Secrets configuration.
