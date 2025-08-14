# 📄 DocBot – Your Personal Document Q&A Assistant

Ever wished you could **ask questions directly to your documents** and get instant answers? Meet **DocBot**, a lightweight, **Windows-friendly Retrieval-Augmented Generation (RAG) chatbot** that lets you interact with PDFs, DOCX, and TXT files in a fun, intelligent way – all on your laptop, no cloud required.  

---

## 🚀 Features

- **Upload multiple documents** (PDF, DOCX, TXT) and build your own knowledge base instantly.  
- **Ask questions naturally** and get either:
  - **Extractive answers:** Exact text snippets from your documents.  
  - **Generative answers:** Summarized, context-aware responses.  
- **Citations included:** Know where your answers are coming from.  
- **OCR support:** Read scanned PDFs with Tesseract (optional).  
- **Local vector search:** Uses FAISS for fast, offline embeddings.  
- **Multiple LLM models supported:** FLAN-T5, BART, DistilBERT QA, RoBERTa QA.  

---

## 🛠 Stack

| Component | Technology |
|-----------|------------|
| **UI** | Streamlit |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Vector DB** | FAISS (CPU-based, local) |
| **LLM / Generator** | Google FLAN-T5 (small/base), BART Large CNN, DistilBERT / RoBERTa QA |
| **Document Parsers** | PyMuPDF (PDF), python-docx (DOCX), plain TXT |
| **OCR (optional)** | Tesseract |

---

## ⚡ Quickstart (Windows 10)

1. Install Python 3.10+  
2. Open **Command Prompt**:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_streamlit.py
