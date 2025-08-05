# scripts/query/query_with_lm_studio.py
#
# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant
# Description: Part of UC Berkeley MICS Machine Learning Course (2025)
# GitHub: https://github.com/isnakie
# Description: This script allows a user to query a combined cybersecurity knowledge base (STIGs + MITRE CWEs)
# using a FAISS index with Sentence-Transformer embeddings and local LLM (via LM Studio).
# The script retrieves the most relevant knowledge snippets and crafts a prompt to ask a local LLM for a focused response.
# License: MIT

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import textwrap
import re

# --- File paths for the FAISS index and corresponding metadata
INDEX_PATH = "data/embeddings/combined_faiss.index"
METADATA_PATH = "data/embeddings/combined_metadata.pkl"

# --- Load the vector index and metadata file
print(":: Loading FAISS index and metadata ...")
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# --- Load the SentenceTransformer model used to embed queries
print(":: Loading sentence transformer model ...")
embedder = SentenceTransformer("all-mpnet-base-v2")

# --- LLM Studio API configuration
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "mistral"

# --- Helper function to strip extra whitespace and truncate long text blocks
def clean_text(text, max_chars=1800):
    return " ".join(text.split())[:max_chars]

# --- Main RAG query logic: retrieve from FAISS, build context, and send to LLM
def query_lm(user_question, k=5, max_context_chars=3500):
    print(":: Searching FAISS index ...")

    # If the user mentions CWE IDs, include them again in the question to increase relevance in FAISS
    cwe_ids = re.findall(r'\bCWE-(\d+)\b', user_question.upper())
    if cwe_ids:
        user_question += " Related CWE IDs: " + " ".join([f"CWE-{cwe_id}" for cwe_id in cwe_ids])

    # Embed the query and search the index
    query_vec = embedder.encode([user_question])
    D, I = index.search(np.array(query_vec), k=k)

    # Build up the context prompt using the top matching entries
    context_blocks = []
    total_chars = 0
    for idx in I[0]:
        entry = metadata[idx]
        title = entry.get("title", "Untitled")
        source = entry.get("source", "Unknown")
        doc_id = entry.get("id", "N/A")
        text = clean_text(entry.get("text", ""))

        # Build a readable block with header
        header = f"[{source}] {doc_id} - {title}"
        block = f"{header}\n{textwrap.indent(text, '  ')}"
        block_chars = len(block)

        # Enforce total character limit to stay within LLM context window
        if total_chars + block_chars > max_context_chars:
            break

        context_blocks.append(block)
        total_chars += block_chars

    if not context_blocks:
        return ":: No usable entries found. Try a simpler query."

    # Join context blocks with separators
    context = "\n\n====\n\n".join(context_blocks)

    # The full prompt the LLM will see, including the user question and retrieved context
    full_prompt = f"""You are a focused cybersecurity assistant. Use only the provided context from STIGs or MITRE CWEs to answer the user's question.
- Do not speculate or hallucinate.
- If the answer is not in the context, say so.

User Question: {user_question}

Context:
{context}

Answer:""".strip()

    # Send the request to the LLM hosted in LM Studio
    response = requests.post(LLM_API_URL, json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.5
    })

    try:
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"!! Error contacting LLM :: {e}\n{response.text}"

# --- CLI loop: allows the user to type questions interactively
if __name__ == "__main__":
    print("\n=== Cybersecurity RAG Query ===")
    while True:
        q = input("\n>> Ask your question (or type 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("\n:: Generating response ...\n")
        answer = query_lm(q)
        print("\n" + answer)
        print("\n" + "-"*80)
