from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Lazy-load the model — only instantiated on first actual use,
# not at import time. Avoids unnecessary load when the module
# is imported by scripts that don't need embeddings.
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Generate a vector embedding for a single text string."""
    embedding = _get_model().encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts. Returns shape (N, dim)."""
    print(f"[Embedder] Embedding {len(texts)} texts...")
    embeddings = _get_model().encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype(np.float32)