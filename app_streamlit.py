import os
import io
import json
import time
import pickle
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd

# Parsing
import fitz  # PyMuPDF
import docx
# Optional OCR (disabled by default)
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Embeddings & Vector store
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Multiple model options - all free
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5Tokenizer, 
    T5ForConditionalGeneration,
    BartTokenizer, 
    BartForConditionalGeneration,
    pipeline
)

###############################################################################
# Storage Layout (per-workspace)
# workspaces/
#   <workspace_name>/
#     index.faiss
#     chunks.pkl
#     meta.json
#     chats/
#       <chat_id>.json
###############################################################################

BASE_DIR = "workspaces"  # root folder for everything (indexes + chats)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_workspaces() -> List[str]:
    ensure_dir(BASE_DIR)
    return sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

def workspace_path(ws: str) -> str:
    return os.path.join(BASE_DIR, ws)

def chats_dir(ws: str) -> str:
    return os.path.join(workspace_path(ws), "chats")

def list_chats(ws: str) -> List[Dict]:
    """Return list of chats as dicts: {id, title, updated_at} (sorted by updated desc)."""
    ensure_dir(chats_dir(ws))
    chats = []
    for fn in os.listdir(chats_dir(ws)):
        if fn.endswith(".json"):
            cid = fn[:-5]
            try:
                with open(os.path.join(chats_dir(ws), fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
                chats.append({
                    "id": cid,
                    "title": data.get("title", f"Chat {cid}"),
                    "updated_at": data.get("updated_at", 0)
                })
            except Exception:
                pass
    chats.sort(key=lambda x: x["updated_at"], reverse=True)
    return chats

def load_chat(ws: str, chat_id: str) -> Dict:
    path = os.path.join(chats_dir(ws), f"{chat_id}.json")
    if not os.path.exists(path):
        return {"id": chat_id, "title": f"Chat {chat_id}", "messages": [], "updated_at": time.time()}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chat(ws: str, chat: Dict):
    ensure_dir(chats_dir(ws))
    chat["updated_at"] = time.time()
    path = os.path.join(chats_dir(ws), f"{chat['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=False, indent=2)

def delete_chat(ws: str, chat_id: str):
    path = os.path.join(chats_dir(ws), f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)

def new_chat(ws: str, title: str = "New Chat") -> Dict:
    ensure_dir(chats_dir(ws))
    cid = str(int(time.time()))  # simple unique id
    chat = {"id": cid, "title": title, "messages": [], "updated_at": time.time()}
    save_chat(ws, chat)
    return chat

###############################################################################
# Index I/O
###############################################################################

def index_files(ws: str):
    wpath = workspace_path(ws)
    return (
        os.path.join(wpath, "index.faiss"),
        os.path.join(wpath, "chunks.pkl"),
        os.path.join(wpath, "meta.json"),
    )

def persist_index(ws: str, index: faiss.IndexFlatIP, chunks: List[str], meta: List[Dict]):
    wpath = workspace_path(ws)
    ensure_dir(wpath)
    faiss.write_index(index, os.path.join(wpath, "index.faiss"))
    with open(os.path.join(wpath, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    with open(os.path.join(wpath, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index(ws: str):
    idx_path, ch_path, meta_path = index_files(ws)
    if not (os.path.exists(idx_path) and os.path.exists(ch_path) and os.path.exists(meta_path)):
        return None, None, None
    index = faiss.read_index(idx_path)
    with open(ch_path, "rb") as f:
        chunks = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, chunks, meta

###############################################################################
# File Readers + Improved Chunking
###############################################################################

def read_txt_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    return [(1, text)]

def read_docx_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    f = io.BytesIO(file_bytes)
    d = docx.Document(f)
    text = "\n".join([p.text for p in d.paragraphs])
    return [(1, text)]

def read_pdf_pages(file_bytes: bytes, use_ocr: bool = False) -> List[Tuple[int, str]]:
    pages = []
    doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    for pno, page in enumerate(doc, start=1):
        if use_ocr and OCR_AVAILABLE:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                pages.append((pno, ocr_text))
                continue
        pages.append((pno, page.get_text("text")))
    return pages

def normalize_spaces(s: str) -> str:
    return " ".join(s.split())

def smart_chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Improved chunking that respects sentence boundaries"""
    import re
    
    text = normalize_spaces(text)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

###############################################################################
# Improved Model Loading with Multiple Options
###############################################################################

def load_embedder(name: str):
    return SentenceTransformer(name)

def load_model_and_tokenizer(model_type: str, model_name: str):
    """Load different types of models based on selection"""
    try:
        if model_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif model_type == "bart":
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
        elif model_type == "qa_pipeline":
            # For extractive QA
            qa_pipeline = pipeline("question-answering", model=model_name)
            return qa_pipeline, None
        else:
            # Default to auto tokenizer/model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        # Fallback to smallest working model
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model

def score_context_relevance(question: str, context: str) -> float:
    """Score how relevant a context is to the question"""
    question_words = set(word.lower() for word in question.split() if len(word) > 2)
    context_words = set(word.lower() for word in context.split() if len(word) > 2)
    
    if not question_words:
        return 0.5
    
    intersection = question_words.intersection(context_words)
    return len(intersection) / len(question_words)

def filter_relevant_contexts(question: str, contexts: List[str], min_score: float = 0.1) -> List[str]:
    """Filter contexts based on relevance to question"""
    scored_contexts = []
    for context in contexts:
        score = score_context_relevance(question, context)
        if score >= min_score:
            scored_contexts.append((score, context))
    
    # Sort by relevance score (descending)
    scored_contexts.sort(reverse=True, key=lambda x: x[0])
    
    # Return top contexts (max 3 for manageable size)
    return [context for score, context in scored_contexts[:3]]

def extractive_qa_answer(qa_pipeline, question: str, contexts: List[str]) -> str:
    """Use extractive QA model to find direct answers"""
    best_answer = None
    best_score = 0
    
    for context in contexts:
        # Truncate context for BERT-based models (512 token limit)
        if len(context) > 2000:  # rough character limit
            context = context[:2000] + "..."
        
        try:
            result = qa_pipeline(question=question, context=context)
            if result['score'] > best_score and result['score'] > 0.1:
                best_score = result['score']
                best_answer = result['answer']
        except Exception:
            continue
    
    return best_answer if best_answer else "No relevant answer found in the provided context."

def generative_answer(tokenizer, model, question: str, contexts: List[str], max_new_tokens: int = 200) -> str:
    """Improved generative answer with better prompts and filtering"""
    
    # Filter contexts for relevance
    relevant_contexts = filter_relevant_contexts(question, contexts)
    
    if not relevant_contexts:
        return "I could not find relevant information to answer your question in the provided documents."
    
    # Join contexts with clear separation
    joined_context = "\n---\n".join(relevant_contexts)
    
    # Truncate if too long
    if len(joined_context) > 1200:
        joined_context = joined_context[:1200] + "..."
    
    # Create focused prompt
    prompt = f"""Context Information:
{joined_context}

Question: {question}

Instructions: Answer the question using ONLY the information from the context above. Be concise and specific. If the answer is not clearly stated in the context, say "The provided context does not contain enough information to answer this question."

Answer:"""
    
    try:
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True
        )
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean response - remove the prompt part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        # Remove common unwanted prefixes
        unwanted_prefixes = [
            "Based on the context,", "According to the context,", 
            "The context states that", "From the information provided,",
            "Context Information:", "Question:"
        ]
        
        for prefix in unwanted_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response if response else "Could not generate a proper answer."
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

###############################################################################
# Streamlit UI Setup
###############################################################################

st.set_page_config(page_title="Enhanced Document Q&A Chatbot", page_icon="📄", layout="wide")

# Session state
if "workspace" not in st.session_state:
    st.session_state.workspace = "default"
    ensure_dir(workspace_path(st.session_state.workspace))
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "meta" not in st.session_state:
    st.session_state.meta = []
if "model_handler" not in st.session_state:
    st.session_state.model_handler = None
if "model_type" not in st.session_state:
    st.session_state.model_type = None
# Active chat
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

###############################################################################
# Sidebar: Workspaces & Chats
###############################################################################

st.sidebar.title("📁 Workspaces & Chats")

# Workspaces
all_ws = list_workspaces()
if st.session_state.workspace not in all_ws:
    ensure_dir(workspace_path(st.session_state.workspace))
    all_ws = list_workspaces()

ws_idx = 0 if st.session_state.workspace not in all_ws else all_ws.index(st.session_state.workspace)
selected_ws = st.sidebar.selectbox("Workspace", all_ws + ["➕ Create new..."], index=ws_idx)

if selected_ws == "➕ Create new...":
    new_ws_name = st.sidebar.text_input("New workspace name", value="project-1")
    if st.sidebar.button("Create workspace"):
        ensure_dir(workspace_path(new_ws_name))
        st.session_state.workspace = new_ws_name
        st.session_state.index = None
        st.session_state.active_chat_id = None
        st.session_state.active_chat = None
        st.rerun()
else:
    if selected_ws != st.session_state.workspace:
        st.session_state.workspace = selected_ws
        st.session_state.index = None
        st.session_state.active_chat_id = None
        st.session_state.active_chat = None
        st.rerun()

# Chats management
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Chats")

ws_chats = list_chats(st.session_state.workspace)
chat_titles = [f"{c['title']}" for c in ws_chats]
chat_ids = [c["id"] for c in ws_chats]

if ws_chats:
    default_idx = 0
    if st.session_state.active_chat_id in chat_ids:
        default_idx = chat_ids.index(st.session_state.active_chat_id)
    selected_chat_title = st.sidebar.selectbox("Select chat", chat_titles, index=default_idx)
    selected_chat_id = ws_chats[chat_titles.index(selected_chat_title)]["id"]
else:
    selected_chat_id = None

colA, colB, colC = st.sidebar.columns([1, 1, 1])
with colA:
    if st.button("➕ New"):
        chat = new_chat(st.session_state.workspace, title="New Chat")
        st.session_state.active_chat_id = chat["id"]
        st.session_state.active_chat = chat
        st.rerun()
with colB:
    if st.button("🧹 Clear"):
        if st.session_state.active_chat_id:
            chat = load_chat(st.session_state.workspace, st.session_state.active_chat_id)
            chat["messages"] = []
            save_chat(st.session_state.workspace, chat)
            st.session_state.active_chat = chat
            st.rerun()
with colC:
    if st.button("🗑 Delete"):
        if selected_chat_id:
            delete_chat(st.session_state.workspace, selected_chat_id)
            st.session_state.active_chat_id = None
            st.session_state.active_chat = None
            st.rerun()

# Load selected chat
if selected_chat_id and selected_chat_id != st.session_state.active_chat_id:
    st.session_state.active_chat_id = selected_chat_id
    st.session_state.active_chat = load_chat(st.session_state.workspace, selected_chat_id)

# Ensure at least one chat exists
if st.session_state.active_chat is None:
    if ws_chats:
        st.session_state.active_chat_id = ws_chats[0]["id"]
        st.session_state.active_chat = load_chat(st.session_state.workspace, st.session_state.active_chat_id)
    else:
        chat = new_chat(st.session_state.workspace, title="New Chat")
        st.session_state.active_chat_id = chat["id"]
        st.session_state.active_chat = chat

# Chat title management
st.sidebar.text_input("Chat title", key="chat_title_input", value=st.session_state.active_chat.get("title", "New Chat"))
if st.sidebar.button("💾 Save Title"):
    st.session_state.active_chat["title"] = st.session_state.chat_title_input or "Untitled"
    save_chat(st.session_state.workspace, st.session_state.active_chat)
    st.rerun()

###############################################################################
# Sidebar: Build Knowledge Base
###############################################################################

st.sidebar.markdown("---")
st.sidebar.header("📄 Build Knowledge Base")

use_ocr = st.sidebar.checkbox("Use OCR for scanned PDFs", value=False, disabled=(not OCR_AVAILABLE))

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT (multiple files supported)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

chunk_size = st.sidebar.slider("Chunk size (chars)", 400, 1500, 800, step=50)
chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", 0, 400, 100, step=10)

build_btn = st.sidebar.button("🔨 Build / Rebuild Index")

###############################################################################
# Sidebar: Enhanced Model Configuration
###############################################################################

st.sidebar.markdown("---")
st.sidebar.header("🤖 Model Configuration")

embed_model_name = st.sidebar.selectbox(
    "Embedding model", 
    [
        "sentence-transformers/all-MiniLM-L6-v2",  # Fast and good
        "sentence-transformers/all-mpnet-base-v2",  # Better quality
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
    ]
)

# Enhanced model selection with free options
model_options = {
    "FLAN-T5 Base (Recommended)": ("t5", "google/flan-t5-base"),
    "FLAN-T5 Small (Fast)": ("t5", "google/flan-t5-small"),
    "BART Large CNN (Good for summarization)": ("bart", "facebook/bart-large-cnn"),
    "DistilBERT QA (Extractive QA)": ("qa_pipeline", "distilbert-base-cased-distilled-squad"),
    "RoBERTa QA (Better Extractive)": ("qa_pipeline", "deepset/roberta-base-squad2")
}

selected_model = st.sidebar.selectbox("Generator model", list(model_options.keys()))
model_type, model_name = model_options[selected_model]

answer_mode = st.sidebar.radio(
    "Answer mode",
    ["Generative (Creative)", "Extractive (Precise)"],
    help="Generative creates answers, Extractive finds exact text from documents"
)

top_k = st.sidebar.slider("Top-K context chunks", 1, 10, 5)

###############################################################################
# Main Area
###############################################################################

st.title("🚀 Enhanced Document Q&A Chatbot")

# Two columns: chat + status
col1, col2 = st.columns([3, 2])

with col2:
    st.subheader("📊 System Status")
    
    # Index status
    idx, ch, mt = load_index(st.session_state.workspace)
    if idx is not None:
        st.success(f"✅ Index loaded: {idx.ntotal} chunks")
        if len(mt) > 0:
            sources = set(m.get('source', '') for m in mt)
            st.info(f"📚 Sources: {len(sources)} files")
        if st.session_state.index is None:
            st.session_state.index = idx
            st.session_state.chunks = ch
            st.session_state.meta = mt
    else:
        st.warning("⚠️ No index found. Upload files and build index.")
    
    # Model status
    if st.session_state.model_handler is not None:
        st.success(f"✅ Model loaded: {selected_model}")
    else:
        st.info("🤖 Model will load on first query")
    
    # Workspace info
    st.subheader("🗂️ Workspace Info")
    st.code(f"Path: {workspace_path(st.session_state.workspace)}")

# Build index
if build_btn:
    if not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        with st.spinner("Processing documents..."):
            all_chunks = []
            meta = []

            progress_bar = st.progress(0)
            for idx, f in enumerate(uploaded_files):
                fname = f.name
                ext = os.path.splitext(fname)[1].lower()

                if ext == ".pdf":
                    pages = read_pdf_pages(f.read(), use_ocr=use_ocr)
                elif ext == ".docx":
                    pages = read_docx_pages(f.read())
                else:
                    pages = read_txt_pages(f.read())

                for (pno, page_text) in pages:
                    pieces = smart_chunk_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap)
                    for i, chnk in enumerate(pieces):
                        if len(chnk.strip()) > 50:  # Only add substantial chunks
                            all_chunks.append(chnk)
                            meta.append({"source": fname, "page": int(pno), "chunk_id": i})
                
                progress_bar.progress((idx + 1) / len(uploaded_files))

        if all_chunks:
            with st.spinner("Building vector index..."):
                st.session_state.embedder = load_embedder(embed_model_name)
                vectors = st.session_state.embedder.encode(
                    all_chunks,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    batch_size=32
                )
                faiss.normalize_L2(vectors)
                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)

            st.session_state.index = index
            st.session_state.chunks = all_chunks
            st.session_state.meta = meta
            persist_index(st.session_state.workspace, index, all_chunks, meta)
            st.success(f"🎉 Index built with {len(all_chunks)} chunks from {len(uploaded_files)} files!")
            st.rerun()

###############################################################################
# Chat Interface
###############################################################################

with col1:
    # Display chat messages
    for msg in st.session_state.active_chat.get("messages", []):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        citations = msg.get("citations", [])
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and citations:
                st.caption("📚 Sources: " + " | ".join(citations))

    # Chat input
    user_question = st.chat_input("Ask anything about your documents...")
    
    if user_question:
        # Add user message
        st.session_state.active_chat["messages"].append({"role": "user", "content": user_question})
        save_chat(st.session_state.workspace, st.session_state.active_chat)
        
        with st.chat_message("user"):
            st.markdown(user_question)

        # Check if index exists
        if st.session_state.index is None or not st.session_state.chunks:
            with st.chat_message("assistant"):
                st.error("Please build an index first by uploading documents in the sidebar.")
        else:
            # Load model if needed
            if st.session_state.model_handler is None or st.session_state.model_type != model_type:
                with st.spinner(f"Loading {selected_model}..."):
                    tokenizer_or_pipeline, model = load_model_and_tokenizer(model_type, model_name)
                    st.session_state.model_handler = tokenizer_or_pipeline
                    st.session_state.model_companion = model  # Store model separately for generative models
                    st.session_state.model_type = model_type

            # Retrieve relevant contexts
            with st.spinner("Searching documents..."):
                if st.session_state.embedder is None:
                    st.session_state.embedder = load_embedder(embed_model_name)
                
                q_vec = st.session_state.embedder.encode([user_question], convert_to_numpy=True)
                faiss.normalize_L2(q_vec)
                D, I = st.session_state.index.search(q_vec, min(top_k, len(st.session_state.chunks)))

                # Get contexts and citations
                contexts = []
                citations = []
                for idx_pos in I[0]:
                    if idx_pos < len(st.session_state.chunks):  # Safety check
                        contexts.append(st.session_state.chunks[idx_pos])
                        meta = st.session_state.meta[idx_pos]
                        citation = f"{meta.get('source', 'unknown')} (p.{meta.get('page', '?')})"
                        citations.append(citation)

            # Generate answer
            with st.spinner("Generating answer..."):
                if answer_mode == "Extractive (Precise)" and model_type == "qa_pipeline":
                    answer = extractive_qa_answer(st.session_state.model_handler, user_question, contexts)
                else:
                    # Use generative approach
                    if model_type == "qa_pipeline":
                        # Fallback to simple context matching for QA models in generative mode
                        best_contexts = filter_relevant_contexts(user_question, contexts)
                        if best_contexts:
                            answer = f"Based on the documents: {' '.join(best_contexts[:2])}"[:500] + "..."
                        else:
                            answer = "No relevant information found."
                    else:
                        answer = generative_answer(
                            st.session_state.model_handler, 
                            st.session_state.model_companion, 
                            user_question, 
                            contexts
                        )

            # Display answer
            with st.chat_message("assistant"):
                st.markdown(answer)
                if citations:
                    unique_citations = list(dict.fromkeys(citations))  # Remove duplicates
                    st.caption("📚 Sources: " + " | ".join(unique_citations[:5]))

            # Save assistant message
            assistant_msg = {
                "role": "assistant", 
                "content": answer, 
                "citations": list(dict.fromkeys(citations))
            }
            st.session_state.active_chat["messages"].append(assistant_msg)
            save_chat(st.session_state.workspace, st.session_state.active_chat)

            # Show source preview in expander
            if contexts:
                with st.expander("🔍 View source contexts"):
                    for i, (ctx, cit) in enumerate(zip(contexts[:3], citations[:3])):
                        st.markdown(f"**{cit}**")
                        preview = ctx[:300] + "..." if len(ctx) > 300 else ctx
                        st.text_area(f"Context {i+1}", value=preview, height=100, disabled=True)
                        st.markdown("---")

###############################################################################
# Additional Features
###############################################################################

# Footer with usage tips
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Usage Tips")
st.sidebar.markdown("""
- **Extractive mode**: Finds exact answers from text
- **Generative mode**: Creates comprehensive answers
- **FLAN-T5**: Best for general questions
- **BART**: Great for summarization
- **QA models**: Best for factual questions
- Use specific keywords for better results
- Shorter chunks = more precise answers
""")

# Model comparison info
# if st.sidebar.expander("📋 Model Comparison"):
#     st.markdown("""
#     **Free Models Available:**
    
#     🏆 **Recommended for most users:**
#     - FLAN-T5 Base: Best balance of quality and speed
    
#     ⚡ **For fast responses:**
#     - FLAN-T5 Small: Fastest, good for simple questions
    
#     📝 **For summarization:**
#     - BART Large CNN: Specialized for creating summaries
    
#     🎯 **For precise answers:**
#     - DistilBERT QA: Finds exact text from documents
#     - RoBERTa QA: More accurate extractive QA
    
#     **Note:** All models run locally and are completely free!
#     """)

# Performance monitoring
if st.sidebar.button("🔧 Clear Model Cache"):
    st.session_state.model_handler = None
    st.session_state.model_companion = None
    st.session_state.embedder = None
    st.success("Model cache cleared. Models will reload on next query.")

# Export functionality
st.sidebar.markdown("---")
if st.sidebar.button("📤 Export Chat"):
    if st.session_state.active_chat:
        chat_data = json.dumps(st.session_state.active_chat, indent=2, ensure_ascii=False)
        st.sidebar.download_button(
            label="💾 Download Chat JSON",
            data=chat_data,
            file_name=f"chat_{st.session_state.active_chat['id']}.json",
            mime="application/json"
        )

###############################################################################
# Error Handling and Optimization
###############################################################################

# Add error handling wrapper for the main chat processing
def safe_process_query(question: str) -> tuple:
    """Safely process a query with error handling"""
    try:
        # Your existing query processing logic here
        return True, "Success"
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        st.error(error_msg)
        return False, error_msg

# Memory optimization - clear old embeddings if too many files
def optimize_memory():
    """Clear memory if too many chunks loaded"""
    if len(st.session_state.chunks) > 10000:  # Arbitrary threshold
        st.warning("Large number of chunks detected. Consider splitting into multiple workspaces for better performance.")

# Auto-save functionality
def auto_save_chat():
    """Auto-save chat periodically"""
    if st.session_state.active_chat:
        save_chat(st.session_state.workspace, st.session_state.active_chat)

# Performance metrics (optional display)
if st.sidebar.checkbox("Show Performance Metrics", value=False):
    if st.session_state.index:
        st.sidebar.metric("Index Size", f"{st.session_state.index.ntotal:,} vectors")
    if st.session_state.chunks:
        avg_chunk_size = sum(len(c) for c in st.session_state.chunks) / len(st.session_state.chunks)
        st.sidebar.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} chars")
    
    # Memory usage (rough estimate)
    import sys
    total_size = sys.getsizeof(st.session_state.chunks) if st.session_state.chunks else 0
    st.sidebar.metric("Memory Usage", f"{total_size / 1024 / 1024:.1f} MB")

###############################################################################
# Advanced Features
###############################################################################

# Conversation context awareness
def get_conversation_context(messages: List[Dict], max_context: int = 3) -> str:
    """Get recent conversation context for better answers"""
    if len(messages) <= 1:
        return ""
    
    recent_messages = messages[-max_context:]
    context_parts = []
    
    for msg in recent_messages:
        if msg.get("role") == "user":
            context_parts.append(f"Previous question: {msg.get('content', '')}")
        elif msg.get("role") == "assistant":
            # Only include short responses to avoid overwhelming context
            content = msg.get('content', '')
            if len(content) < 200:
                context_parts.append(f"Previous answer: {content}")
    
    return "\n".join(context_parts)

# Smart query expansion
def expand_query(question: str) -> List[str]:
    """Expand query with related terms for better retrieval"""
    expansions = [question]
    
    # Simple synonym/related term expansion
    synonyms = {
        "what": ["describe", "explain", "define"],
        "how": ["process", "method", "way", "procedure"],
        "when": ["time", "date", "period"],
        "where": ["location", "place", "position"],
        "why": ["reason", "cause", "purpose"]
    }
    
    words = question.lower().split()
    for word in words:
        if word in synonyms:
            for syn in synonyms[word]:
                expanded = question.replace(word, syn, 1)
                if expanded != question:
                    expansions.append(expanded)
    
    return expansions[:3]  # Limit to avoid too many queries

# Document structure analysis
def analyze_document_structure(chunks: List[str], meta: List[Dict]) -> Dict:
    """Analyze document structure for better chunking insights"""
    structure = {
        "total_chunks": len(chunks),
        "sources": {},
        "avg_chunk_length": 0,
        "page_distribution": {}
    }
    
    if not chunks:
        return structure
    
    # Calculate average chunk length
    structure["avg_chunk_length"] = sum(len(c) for c in chunks) / len(chunks)
    
    # Analyze sources and pages
    for m in meta:
        source = m.get("source", "unknown")
        page = m.get("page", 1)
        
        if source not in structure["sources"]:
            structure["sources"][source] = {"chunks": 0, "pages": set()}
        
        structure["sources"][source]["chunks"] += 1
        structure["sources"][source]["pages"].add(page)
        
        if page not in structure["page_distribution"]:
            structure["page_distribution"][page] = 0
        structure["page_distribution"][page] += 1
    
    return structure

# Display document analysis in sidebar
if st.session_state.chunks and st.sidebar.checkbox("📊 Document Analysis", value=False):
    analysis = analyze_document_structure(st.session_state.chunks, st.session_state.meta)
    
    st.sidebar.markdown("**Document Structure:**")
    st.sidebar.json(analysis)

###############################################################################
# Keyboard Shortcuts and UX Improvements
###############################################################################

# Add some CSS for better styling
st.markdown("""
<style>
.stChat > div {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
}

.success-metric {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 10px;
    margin: 5px 0;
}

.warning-metric {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)



###############################################################################
# Final optimizations and cleanup
###############################################################################

# Cleanup function for session management
def cleanup_session():
    """Clean up session state to prevent memory leaks"""
    keys_to_preserve = [
        "workspace", "active_chat_id", "active_chat", 
        "index", "chunks", "meta", "embedder"
    ]
    
    all_keys = list(st.session_state.keys())
    for key in all_keys:
        if key not in keys_to_preserve and key.startswith("temp_"):
            del st.session_state[key]

# Run cleanup periodically
if len(st.session_state.get("messages", [])) % 10 == 0:
    cleanup_session()

# Add version info
st.sidebar.markdown("---")
st.sidebar.caption("Enhanced Document Q&A v2.0 | Free & Open Source")