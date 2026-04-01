"""
query_system.py

Interactive CLI for querying the prescription RAG system.

Per query:
1. Embed the question
2. Retrieve top 3 matching prescriptions from FAISS
3. Send context + question to Gemini
4. Print answer and which prescriptions were used

Usage:
    python scripts\query_system.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.embeddings.embedder import embed_text
from src.vectordb.faiss_store import load_index, search
from src.ocr.gemini_vision import query_gemini

TOP_K = 3


def build_context(results: list[dict]) -> str:
    """Combine retrieved prescription texts into a single context block."""
    parts = []
    for i, result in enumerate(results, 1):
        parts.append(f"--- Prescription {i} ({result.get('file', 'unknown')}) ---\n{result.get('text', '').strip()}")
    return "\n\n".join(parts)


def run_query(question: str, index, metadata: list[dict]) -> tuple[str, list[str]]:
    """
    Route the message — conversational goes directly to chat_gemini,
    prescription queries go through the full RAG pipeline.
    Returns (answer, list of source filenames used).
    """
    from src.ocr.gemini_vision import is_conversational, chat_gemini
    if is_conversational(question):
        return chat_gemini(question), []

    query_embedding = embed_text(question)
    results = search(index, metadata, query_embedding, top_k=TOP_K)

    if not results:
        return "No relevant prescriptions found in the index.", []

    sources = [r["file"] for r in results]
    context = build_context(results)
    answer = query_gemini(question, context)
    return answer, sources


def main():
    print("=" * 55)
    print("  Medical Prescription RAG — Query System")
    print("=" * 55)

    try:
        index, metadata = load_index()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print(f"Index loaded — {index.ntotal} prescription(s) available.")
    print("Type your question. Enter 'exit' to quit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Query] Exiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("[Query] Goodbye.")
            break

        answer, sources = run_query(question, index, metadata)

        print("\n" + "-" * 55)
        print("Answer:")
        print(answer)
        if sources:
            print(f"\nSources used: {', '.join(sources)}")
        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()