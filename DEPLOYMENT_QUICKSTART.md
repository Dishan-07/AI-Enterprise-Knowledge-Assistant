# Deployment Quickstart

## Local

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## GitHub

Repository name:

`AI-Enterprise-Knowledge-Assistant`

Upload the project except `.env` and `venv`.

## Streamlit Community Cloud

Deploy `app.py` from the `main` branch.

Add these secrets:

```toml
GROQ_API_KEY = "your_key"
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = "4"
SIMILARITY_THRESHOLD = "0.34"
```

## Google Drive

Upload the complete working project folder and set:

`General access → Anyone with the link → Viewer`

## Final checklist

- Working local project
- Google Drive folder accessible
- GitHub repository accessible
- Live Streamlit URL works
- HR test works
- Technical test works
- Project test works
- Unsupported-question test works
- No API key is public
