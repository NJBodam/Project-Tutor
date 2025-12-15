

import json
import os
from typing import List, Dict

import numpy as np
import faiss
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ==============================================================================
# Configuration and Initialization
# ==============================================================================

INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

# ------------------------------------
# Load Index and Metadata
# ------------------------------------
try:
    if not os.path.exists(FAISS_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Index files not found in '{INDEX_DIR}'. Run the ingestion script first.")

    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    print(f"Loaded FAISS index with {index.ntotal} vectors and {len(meta)} metadata entries.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    # In a production setup, you might want to raise the exception,
    # but for a simple service, we initialize variables to None
    index = None
    meta = []

# ------------------------------------
# Embedding Model (must be the same as ingestion)
# ------------------------------------
EMBEDDER_MODEL_NAME = "BAAI/bge-m3"
try:
    print(f"Loading embedding model: {EMBEDDER_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME)
except Exception as e:
    print(f"FATAL ERROR: Could not load SentenceTransformer: {e}")
    embedder = None

# ------------------------------------
# LLM Configuration
# ------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"

# ------------------------------------
# FastAPI Setup
# ------------------------------------
app = FastAPI()

class AskReq(BaseModel):
    """Request model for the /ask endpoint."""
    question: str
    top_k: int = 6
    temperature: float = 0.2
    language: str = "en"  # "en" or "de"

# ==============================================================================
# Retrieval and Answering Logic
# ==============================================================================

def retrieve(question: str, top_k: int = 6) -> List[Dict]:
    """
    Performs vector search on the FAISS index.
    """
    if index is None or embedder is None:
        return []

    # Encode the question and normalize the embedding
    q_emb = embedder.encode([question], normalize_embeddings=True)

    # Perform the FAISS search: D=Distances/Scores, I=Indices
    D, I = index.search(q_emb.astype(np.float32), top_k)

    results = []
    for rank, idx in enumerate(I[0]):
        # idx is the index into the 'meta' list
        item = meta[idx]

        results.append({
            "rank": rank,
            "score": float(D[0][rank]),
            "doc_id": item["doc_id"],
            "title": item["title"],
            "page": item["page"],
            "text": item["text"]
        })
    return results

def build_prompt(question: str, passages: List[Dict], language: str = "en") -> str:
    """
    Creates a strict RAG prompt with instructions and context passages.
    """
    # Multilingual instruction set
    instr_en = (
        "You are an academic tutor. Answer only using the provided passages. "
        "If the passages are insufficient, say you don't know. "
        "Cite each claim with (title, page). Be concise; offer more detail if asked."
    )
    instr_de = (
        "Du bist ein akademischer Tutor. Beantworte nur anhand der bereitgestellten Textausschnitte. "
        "Wenn die Ausschnitte nicht ausreichen, sage, dass du es nicht weißt. "
        "Zitiere jede Aussage mit (Titel, Seite). Antworte zunächst prägnant; biete Details auf Nachfrage an."
    )

    instr = instr_en if language.lower() == "en" else instr_de

    # Format the passages for context, including citation info
    context = "\n\n".join([
        f"[{p['title']} | p.{p['page']}]\n{p['text']}" for p in passages
    ])

    return f"{instr}\n\nQuestion:\n{question}\n\nPassages:\n{context}\n\nAnswer:"

def call_llm(prompt: str, temperature: float = 0.2) -> str:
    """
    Sends the prompt to the Ollama API and returns the generated response.
    """
    if not OLLAMA_URL:
        return "LLM service URL is not configured."

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False # We want the final answer, not a stream
            },
            timeout=120 # Set a generous timeout
        )
        resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return resp.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama ({LLM_MODEL}): {e}"

# ==============================================================================
# API Endpoint
# ==============================================================================

@app.post("/ask")
def ask(req: AskReq):
    """
    API endpoint to retrieve relevant documents and get an LLM-generated answer.
    """
    if index is None or embedder is None:
        return {"answer": "Service is not initialized. Check server logs.", "citations": []}

    # 1. Retrieve relevant passages
    passages = retrieve(req.question, req.top_k)

    # 2. Build the context-aware prompt
    prompt = build_prompt(req.question, passages, req.language)

    # 3. Call the LLM with the RAG prompt
    answer = call_llm(prompt, req.temperature)

    # 4. Return the answer and the metadata of the retrieved passages
    return {
        "answer": answer,
        "citations": [{
            "title": p["title"],
            "page": p["page"],
            "score": p["score"]
        } for p in passages]
    }

# ==============================================================================
# Run Command
# ==============================================================================

# To run this service, save it as 'server.py' and execute the following command
# in your terminal (make sure your Python environment is active):
"""
uvicorn server:app --host 127.0.0.1 --port 8000
"""