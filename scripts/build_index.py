"""
build_index.py

Reads all JSON files from data/structured_json/,
generates embeddings, and stores them in a FAISS index.

Run this every time after process_images.py adds new files.

Usage:
    python scripts\build_index.py
"""

import sys
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.embeddings.embedder import embed_texts
from src.vectordb.faiss_store import save_index

JSON_DIR = _PROJECT_ROOT / "data" / "structured_json"


def load_all_json() -> list[dict]:
    """Load all prescription JSON records."""
    records = []
    for filepath in sorted(JSON_DIR.glob("*.json")):
        with open(filepath, "r") as f:
            record = json.load(f)
        records.append(record)
        chars = len(record.get("text", ""))
        date = record.get("processed_at", "unknown date")
        print(f"[Index] Loaded: {filepath.name}  ({chars} chars, processed {date})")
    return records


def main():
    if not JSON_DIR.exists():
        print(f"[Index] ERROR: JSON directory not found: {JSON_DIR}")
        print("[Index] Run process_images.py first.")
        sys.exit(1)

    records = load_all_json()

    if not records:
        print("[Index] No JSON files found. Run process_images.py first.")
        sys.exit(0)

    print(f"\n[Index] Building embeddings for {len(records)} prescription(s)...")
    texts = [r["text"] if r["text"] else "(empty)" for r in records]
    embeddings = embed_texts(texts)

    metadata = [{"file": r["file"], "text": r["text"]} for r in records]
    save_index(embeddings, metadata)

    print("\n[Index] Done. Ready to query: python scripts\\query_system.py")


if __name__ == "__main__":
    main()