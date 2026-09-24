# AI Enterprise Knowledge Assistant — Project Report

## 1. Problem Statement

Organizations store policies, technical documentation and project information across multiple documents. Manual searching is slow, while a generic chatbot can answer without reliable company evidence.

This project combines RAG, semantic search, LLMs and agent routing to provide grounded enterprise question answering.

## 2. Objectives

1. Process enterprise PDF documents.
2. Build semantic retrieval using Sentence Transformers and FAISS.
3. Route questions to HR, Technical, Project or General specialist agents.
4. Orchestrate the workflow with LangGraph.
5. Generate grounded responses using Groq.
6. Display source information.
7. Reduce hallucinations for unsupported questions.
8. Support session-based PDF uploads.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | UI |
| Groq | LLM inference |
| Sentence Transformers | Embeddings |
| FAISS | Vector search |
| PyPDF | PDF extraction |
| LangGraph | Agent workflow |
| NumPy | Numerical operations |
| python-dotenv | Configuration |

## 4. Architecture

```text
User
 ↓
Streamlit Chat UI
 ↓
Manager Agent
 ↓
HR / Technical / Project / General
 ↓
Document Retriever
 ↓
FAISS Vector Search
 ↓
Relevant Chunks
 ↓
Groq Answer Agent
 ↓
Grounded Answer
 ↓
Source Citation
```

## 5. RAG Pipeline

PDF pages are extracted, cleaned and divided into overlapping chunks. Each chunk is embedded with Sentence Transformers and indexed in FAISS. The user's question is embedded in the same vector space. High-similarity chunks are retrieved and passed as context to the answer agent.

## 6. Hallucination Control

The answer agent is explicitly instructed to use only retrieved company-document context. If sufficient evidence is unavailable, the application returns:

`I could not find this information in the available company documents.`

## 7. Agent Workflow

The Manager Agent classifies each question as HR, TECHNICAL, PROJECT or GENERAL. The graph uses that classification to select the appropriate document department before retrieval. If the specialist route has no evidence, a global retrieval retry prevents an incorrect route from hiding a relevant document.

## 8. Source Transparency

Each source contains filename, page, department and similarity score. This makes the answer traceable to the indexed documents.

## 9. Testing

HR test: `How many casual leaves do employees receive?`

Technical test: `How is the application deployed?`

Project test: `What technology does Project Alpha use?`

Unsupported test: `What is the company's international travel allowance?`

## 10. Future Enhancements

- Authentication and authorization
- Persistent chat history
- Admin document management
- OCR for scanned PDFs
- Evaluation dashboard
- Feedback collection
- Persistent vector database
- Enterprise database integration

## 11. Conclusion

The project demonstrates a practical enterprise GenAI architecture where routing, retrieval and response generation are separated. The resulting assistant provides document-grounded answers with transparent source information instead of relying on an unrestricted chatbot.
