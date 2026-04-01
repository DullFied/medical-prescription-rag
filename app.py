import sys
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="EY Medical Assistant",
    layout="centered",
)

# ── EY LOGO (FIXED) ─────────────────────────────────────────
EY_LOGO_DARK_BG = '<img src="https://upload.wikimedia.org/wikipedia/commons/3/34/EY_logo_2019.svg" style="height:32px; width:auto; filter: brightness(0) invert(1);" />'
EY_LOGO_LIGHT_BG = '<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/EY_logo_2019.svg/120px-EY_logo_2019.svg.png" width="88" style="display:block;">'

# ── STYLING ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, .stApp { font-family: 'IBM Plex Sans', sans-serif !important; background-color: #FFFFFF !important; color: #2E2E38 !important; }
div[role="tooltip"] { display: none !important; }
.block-container { max-width: 780px; padding-top: 0 !important; padding-bottom: 5rem; }

section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 3px solid #FFE600 !important; }
section[data-testid="stSidebar"] * { color: #2E2E38 !important; }
section[data-testid="stSidebar"] .stButton > button { width: 100%; background: #FFE600 !important; color: #2E2E38 !important; border: none !important; border-radius: 2px !important; font-weight: 500 !important; letter-spacing: 0.04em; transition: all 0.2s ease !important; }
section[data-testid="stSidebar"] .stButton > button:hover { background: #2E2E38 !important; color: #FFE600 !important; }

.stButton > button { background: #FFFFFF !important; color: #2E2E38 !important; border: 1.5px solid #DCDCE0 !important; border-radius: 2px !important; padding: 0.65rem 1.1rem !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #FFE600 !important; border-color: #FFE600 !important; box-shadow: 0 0 0 2px rgba(255,230,0,0.2); }

[data-testid="stChatMessage"] { background: #FFFFFF !important; border: 1px solid #E6E6EA !important; border-left: 3px solid #FFE600 !important; border-radius: 6px !important; margin-bottom: 0.65rem !important; }
[data-testid="stChatInput"] { border: 1.5px solid #DCDCE0 !important; border-radius: 6px !important; }
[data-testid="stChatInput"]:focus-within { border-color: #2E2E38 !important; }

.header-container {
    background: linear-gradient(135deg, #1f1f27, #2E2E38); padding: 3rem 2.5rem 2.2rem 2.5rem; margin-left: -4rem; margin-right: -4rem; margin-bottom: 2.5rem; border-bottom: 2px solid rgba(255,230,0,0.8); box-shadow: 0 8px 30px rgba(0,0,0,0.15); overflow: visible;}
.stCaption { color: #9090A0 !important; }
.stSpinner > div { border-top-color: #FFE600 !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ── HELPERS ─────────────────────────────────────────────────
@st.cache_resource
def load_index():
    from src.vectordb.faiss_store import load_index
    return load_index()

def get_records():
    folder = _PROJECT_ROOT / "data" / "structured_json"
    if not folder.exists():
        return []
    out = []
    for f in folder.glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except:
            pass
    return out

def run_query(q, top_k):
    from src.embeddings.embedder import embed_text
    from src.vectordb.faiss_store import search
    from src.ocr.gemini_vision import query_gemini
    index, metadata = load_index()
    results = search(index, metadata, embed_text(q), top_k)
    context = "\n\n".join([r.get("text", "") for r in results])
    answer = query_gemini(q, context)
    sources = [r.get("file", "?") for r in results]
    return answer, sources

# ── LOAD STATE ──────────────────────────────────────────────
index_ok = False
n_docs = 0
try:
    idx, _ = load_index()
    n_docs = idx.ntotal
    index_ok = True
except:
    pass

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="padding:1.8rem 0 0.4rem 0;">{EY_LOGO_LIGHT_BG}</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.15em; color:#747480;">Medical Assistant</p>', unsafe_allow_html=True)
    st.metric("Indexed Documents", n_docs)
    st.divider()
    top_k = st.slider("Sources to retrieve", 1, max(1, min(5, n_docs)), min(3, n_docs) if n_docs else 1)
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('<div style="margin-top:2.5rem; padding-top:1rem; border-top:1px solid #EBEBED;"><p style="font-size:0.62rem; color:#AAAAAA;">© Ernst &amp; Young LLP.<br>For internal use only.<br>Not a substitute for professional medical advice.</p></div>', unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────
# Single unbroken string — no newlines inside HTML attributes
st.markdown(f'<div class="header-container"><div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.4rem;">{EY_LOGO_DARK_BG}<div style="width:1px; height:28px; background:rgba(255,255,255,0.2);"></div><span style="font-size:0.67rem; text-transform:uppercase; letter-spacing:0.2em; color:#9090A0;">Medical Intelligence</span></div><h1 style="font-size:1.9rem; font-weight:600; color:#FFFFFF; margin:0 0 0.6rem 0; line-height:1.35; padding-top:2px;">EY Medical Assistant</h1><p style="font-size:0.88rem; color:#B0B0BC; max-width:480px; line-height:1.6; margin:0;">Analyze prescriptions, verify medications, and surface patient insights — powered by EY\'s secure, local document intelligence platform.</p></div>', unsafe_allow_html=True)

# ── SUGGESTIONS ─────────────────────────────────────────────
SUGGESTIONS = ["What medicines are prescribed?", "Dosage instructions", "Patient details"]

if not st.session_state.messages and index_ok:
    st.markdown('<p style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.15em; color:#9090A0;">Suggested queries</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, s in zip(cols, SUGGESTIONS):
        with col:
            if st.button(s, use_container_width=True):
                st.session_state.pending = s
                st.rerun()

# ── CHAT HISTORY ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("Sources  ·  " + "  ·  ".join(msg["sources"]))

# ── QUERY HANDLER ───────────────────────────────────────────
def handle(q):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            answer, sources = run_query(q, top_k)
        st.markdown(answer)
        if sources:
            st.caption("Sources  ·  " + "  ·  ".join(sources))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# ── INPUT ───────────────────────────────────────────────────
if not index_ok:
    st.info("⚠️ No indexed documents found. Please run the data pipeline first.")
else:
    q = st.chat_input("Ask about prescriptions, medications, or patient records...")
    if q:
        handle(q)
    elif st.session_state.pending:
        handle(st.session_state.pending)
        st.session_state.pending = None