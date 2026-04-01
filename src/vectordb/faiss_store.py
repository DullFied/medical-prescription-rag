import os
import json
import faiss
import numpy as np
from pathlib import Path

# Anchor paths to project root so scripts work from any working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VECTOR_STORE_DIR = _PROJECT_ROOT / "vector_store"
_METADATA_FILE = _VECTOR_STORE_DIR / "metadata.json"

# FAISS on Windows cannot handle non-ASCII characters in file paths
# (e.g. Japanese folder names like ドキュメント). We use the Windows
# short path (8.3 format) as a workaround when running on Windows.
def _safe_path(p: Path) -> str:
    """Return a FAISS-safe file path string. On Windows, converts to short 8.3 path."""
    path_str = str(p)
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(32767)
            ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 32767)
            short = buf.value
            if short:
                return short
        except Exception:
            pass
    return path_str


def save_index(embeddings: np.ndarray, metadata: list[dict]):
    """Build a FAISS flat L2 index from embeddings and save to disk."""
    _VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    dimension = embeddings.shape[1]
    print(f"[FAISS] Building index: {len(embeddings)} vectors, dim={dimension}")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    index_path = _safe_path(_VECTOR_STORE_DIR / "faiss.index")
    print(f"[FAISS] Writing index to: {index_path}")
    faiss.write_index(index, index_path)
    print(f"[FAISS] Index saved.")

    with open(_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[FAISS] Metadata saved.")


def load_index():
    """Load FAISS index and metadata from disk. Returns (index, metadata)."""
    index_path = _safe_path(_VECTOR_STORE_DIR / "faiss.index")

    if not Path(index_path).exists() and not (_VECTOR_STORE_DIR / "faiss.index").exists():
        raise FileNotFoundError("FAISS index not found. Run build_index.py first.")

    print(f"[FAISS] Loading index...")
    index = faiss.read_index(index_path)

    with open(_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[FAISS] Loaded {index.ntotal} vectors, {len(metadata)} records.")
    return index, metadata


def search(index, metadata: list[dict], query_embedding: np.ndarray, top_k: int = 3):
    """Search FAISS index. Returns top_k metadata dicts with distance scores."""
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        result = metadata[idx].copy()
        result["distance"] = float(distances[0][rank])
        results.append(result)
        print(f"[FAISS] Match #{rank+1}: {result['file']} (distance={result['distance']:.4f})")

    return results