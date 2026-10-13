"""
ui/app.py
---------
TELUSEARCH: Streamlit Web Interface (Phase 5)

An intuitive, responsive semantic search interface for Telugu news articles.
Features:
- Streamlit application loading precomputed FAISS vector index & metadata.
- Sub-100ms query retrieval using multilingual-e5-base.
- Telugu query search bar, preset query buttons, and configurable top-k slider.
- Rich result cards displaying headlines, relevance scores, snippets, and full article expanders.
"""

import os
import sys
import time
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from search.semantic import SemanticSearcher
from search.bm25 import BM25Searcher
from search.hybrid import HybridSearcher

# Page configuration
st.set_page_config(
    page_title="TELUSEARCH | తెలుగు న్యూస్ సెమాంటిక్ సెర్చ్",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Ramabhadra&display=swap');
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.4rem;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-color: #bfdbfe;
    }
    .headline-text {
        font-family: 'Ramabhadra', sans-serif, system-ui;
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    .score-badge {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        border: 1px solid #a7f3d0;
        margin-right: 0.5rem;
    }
    .id-badge {
        display: inline-block;
        background-color: #f3f4f6;
        color: #374151;
        font-size: 0.8rem;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
    }
    .snippet-text {
        font-size: 0.95rem;
        color: #374151;
        line-height: 1.6;
        margin-top: 0.6rem;
    }
    .stat-pill {
        display: inline-block;
        background: #eff6ff;
        color: #1e40af;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models & index")
def load_all_searchers():
    """Cached loader for Semantic, BM25, and Hybrid searchers."""
    index_path = os.path.join(PROJECT_ROOT, "artifacts", "faiss_index.bin")
    metadata_path = os.path.join(PROJECT_ROOT, "artifacts", "metadata.parquet")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None, None

    # Semantic searcher
    semantic_searcher = SemanticSearcher(
        index_path=index_path,
        metadata_path=metadata_path
    )
    semantic_searcher.load()

    # BM25 searcher
    bm25_searcher = BM25Searcher(metadata_path=metadata_path)
    bm25_searcher.load()

    # Hybrid searcher
    hybrid_searcher = HybridSearcher(
        semantic_searcher=semantic_searcher,
        bm25_searcher=bm25_searcher,
        alpha=0.65
    )

    return semantic_searcher, bm25_searcher, hybrid_searcher


def main():
    # Sidebar
    st.sidebar.title("⚙️ Search Settings")
    st.sidebar.markdown("---")

    # Search Mode Selection
    search_mode = st.sidebar.radio(
        "🔍 Search Mode:",
        [
            "🧠 Semantic - FAISS + E5",
            "⚡ Hybrid - Dense + BM25",
            "🔤 Keyword - BM25"
        ],
        index=0
    )

    top_k = st.sidebar.slider(
        "Number of Results:",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    st.sidebar.markdown("### 💡 నమూనా ప్రశ్నలు (Sample Queries)")
    sample_queries = [
        "ఘోర రోడ్డు ప్రమాదం మృతులు",
        "కరోనా వైరస్ వ్యాక్సినేషన్ డోస్",
        "టాలీవుడ్ హీరో నాగశౌర్య కొత్త సినిమా",
        "సానియా మీర్జా టెన్నిస్ యుఎస్ ఓపెన్",
        "వ్యవసాయ విద్యార్థులకు ప్రభుత్వ ఉద్యోగాలు జీఓ 16",
        "ముఖ్యమంత్రి వైఎస్ జగన్ విశాఖపట్నం పర్యటన"
    ]

    selected_sample = None
    for sq in sample_queries:
        if st.sidebar.button(sq, use_container_width=True):
            selected_sample = sq

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Specs")
    st.sidebar.caption("• Model: `intfloat/multilingual-e5-base`")
    st.sidebar.caption("• Vector Dimensions: `768 Dimensions`")
    st.sidebar.caption("• Index Metric: `FAISS IndexFlatIP (Cosine)`")
    st.sidebar.caption("• Lexical Engine: `BM25 Okapi`")
    st.sidebar.caption("• Articles in Index: `1,000 Articles`")

    # Main UI Header
    st.markdown('<div class="main-title">TELUSEARCH 🔍</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Semantic Search for Telugu News Articles — Artificial Intelligence Based Meaningful Search</div>',
        unsafe_allow_html=True
    )

    # Load Searchers
    semantic_searcher, bm25_searcher, hybrid_searcher = load_all_searchers()
    if semantic_searcher is None:
        st.error(
            "FAISS Index or Metadata Files Not Found! "
            "Please run `python indexing/build_index.py` first to build the index."
        )
        st.info("Artifacts path: `artifacts/faiss_index.bin`, `artifacts/metadata.parquet`")
        return

    # Select active searcher
    if "Semantic" in search_mode:
        active_searcher = semantic_searcher
        mode_label = "Semantic (FAISS + E5)"
    elif "Hybrid" in search_mode:
        active_searcher = hybrid_searcher
        mode_label = "Hybrid (Dense + BM25, Alpha=0.65)"
    else:
        active_searcher = bm25_searcher
        mode_label = "Keyword (BM25 Okapi)"

    # Search Bar
    query_input = st.text_input(
        "Enter query in Telugu:",
        value=selected_sample if selected_sample else "",
        placeholder="ఉదా: రోడ్డు ప్రమాదం, రైతు ఆందోళన, సినిమా విశేషాలు...",
        key="query_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Execute Search
    active_query = query_input.strip()
    if search_clicked or selected_sample or (active_query and len(active_query) > 2):
        if not active_query:
            st.warning("Please enter a query.")
            return

        with st.spinner(f"[{mode_label}] Searching for articles..."):
            results = active_searcher.search(active_query, top_k=top_k)

        if not results:
            st.info("No matching articles found.")
            return

        retrieval_ms = results[0]["retrieval_time_ms"]
        st.markdown(
            f'<div class="stat-pill"> <b>{len(results)}</b> Results Found | Engine: <b>{mode_label}</b> | Time: <b>{retrieval_ms:.1f} ms</b></div>',
            unsafe_allow_html=True
        )

        # Render Results
        for r in results:
            if r["score"] > 1.0:
                score_label = f"BM25 Score: {r['score']:.2f}"
            else:
                score_label = f"Similarity Score: {r['score'] * 100:.1f}% ({r['score']:.4f})"

            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div>
                        <span class="score-badge">{score_label}</span>
                        <span class="id-badge">ID: {r['doc_id']}</span>
                    </div>
                    <span style="color: #6b7280; font-size: 0.85rem;">Rank: #{r['rank']}</span>
                </div>
                <div class="headline-text">{r['headline']}</div>
                <div class="snippet-text">{r['snippet']}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"📄 Read Full Story: {r['headline'][:40]}..."):
                st.write(r["full_article"])
                st.caption(f"Original Story ID: `{r['orig_story_id']}`")


if __name__ == "__main__":
    main()
