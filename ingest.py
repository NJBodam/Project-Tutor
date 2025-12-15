import os
import json
import re
from typing import List, Dict

import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss

# ==============================================================================
# Step 1: Document Loading and Chunking
# ==============================================================================

def extract_text_with_pages(pdf_path: str) -> List[Dict]:
    """
    Extracts text from a PDF, performing basic cleanup, and returns a list
    of page-based text chunks.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        # Extract text
        text = doc[page_num].get_text("text")

        # Basic text cleaning
        text = re.sub(r'-\n', '', text)       # Dehyphenate (e.g., "word-\ncontinuation" -> "wordcontinuation")
        text = re.sub(r'\s+\n', '\n', text)   # Remove leading spaces before newlines
        text = re.sub(r'\n{2,}', '\n\n', text) # Consolidate multiple newlines

        chunks.append({
            "page": page_num + 1,
            "text": text
        })
    doc.close()
    return chunks

def chunk_page_text(text: str, target_chars: int = 4000, overlap: int = 800) -> List[str]:
    """
    Splits text from a single page into overlapping chunks based on character count.

    A more robust approach would use sentence-based or token-based chunking.
    """
    spans = []
    i = 0
    while i < len(text):
        chunk = text[i:i + target_chars]
        spans.append(chunk)
        # Move the index forward by the chunk size minus the overlap
        i += target_chars - overlap

        # Stop if the next step would go beyond the text length without overlap
        if i < len(text) and len(text) - i <= overlap:
            break

    return spans

def load_documents(folder: str) -> List[Dict]:
    """
    Loads all PDF documents from a folder, extracts text by page, and then
    further chunks each page, returning a list of all final document chunks.
    """
    items = []

    if not os.path.isdir(folder):
        print(f"Error: Directory '{folder}' not found.")
        return items

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder, fname)
        print(f"Processing: {fname}")

        pages = extract_text_with_pages(path)

        for p in pages:
            # Chunk the text from the single page
            for chunk in chunk_page_text(p["text"]):
                items.append({
                    "doc_id": os.path.splitext(fname)[0], # Filename without extension
                    "title": fname,
                    "page": p["page"],
                    "text": chunk
                })
    return items

# ==============================================================================
# Step 2: Indexing (Embedding and FAISS)
# ==============================================================================

def build_faiss_index(texts: List[str], model_name: str = "BAAI/bge-m3"):
    """
    Generates sentence embeddings for a list of texts and creates a FAISS index
    for efficient retrieval. Cosine similarity is achieved using normalized
    embeddings and the Inner Product (IP) index.
    """
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True # Normalize for Cosine Similarity
    )

    dim = embeddings.shape[1]

    # Initialize FAISS index: IndexFlatIP is Inner Product
    print(f"Building FAISS IndexFlatIP (Dimension: {dim})...")
    index = faiss.IndexFlatIP(dim)

    # FAISS requires float32
    index.add(embeddings.astype(np.float32))

    print("FAISS index built successfully.")
    return index, embeddings

# ==============================================================================
# Main Execution
# ==============================================================================

def main(data_dir: str = "data", out_dir: str = "index"):
    """
    Main function to load documents, build the index, and save the index files.
    """
    print(f"Starting ingestion process. Data folder: {data_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load, extract, and chunk documents
    docs = load_documents(data_dir)
    if not docs:
        print("No documents were loaded. Exiting.")
        return

    # Separate texts for embedding
    texts = [d["text"] for d in docs]

    # 2. Build the FAISS index
    index, _ = build_faiss_index(texts)

    # 3. Save the index and metadata
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(docs, f, indent=4) # Use indent for readability

    print("-" * 50)
    print(f"✅ Ingestion complete.")
    print(f"Ingested {len(docs)} chunks.")
    print(f"FAISS index saved to: {os.path.join(out_dir, 'faiss.index')}")
    print(f"Metadata saved to: {os.path.join(out_dir, 'meta.json')}")

if __name__ == "__main__":
    import sys
    # Fix the missing colon and variable name: name == "__main__"
    # Execute main, taking the first command-line argument as data_dir, or 'data' by default
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
