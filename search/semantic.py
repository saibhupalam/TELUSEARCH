"""
search/semantic.py
------------------
TELUSEARCH: Semantic Search Engine (Phase 4)

Key objectives:
1. Load persisted FAISS index and article metadata from artifacts/.
2. Accept a user's Telugu search query.
3. Encode the query using multilingual-e5-base with official 'query: ' prefix.
4. Execute vector similarity search using FAISS IndexFlatIP.
5. Return top-k relevant articles with headlines, snippets, and similarity scores.
6. Provide robust error handling (empty query, out-of-bounds k).
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss

# Windows UTF-8 console output support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class SemanticSearcher:
    """
    Manages FAISS index loading, embedding model caching, and semantic retrieval.
    """

    def __init__(
        self,
        index_path: str = "artifacts/faiss_index.bin",
        metadata_path: str = "artifacts/metadata.parquet",
        model_name: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.device = device
        self.index = None
        self.metadata_df = None
        self.model = None

    def load(self):
        """Loads FAISS index, metadata, and model into memory."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at '{self.index_path}'. Run indexing/build_index.py first.")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at '{self.metadata_path}'. Run indexing/build_index.py first.")

        # Load FAISS index
        print(f"Loading FAISS index from '{self.index_path}'...")
        self.index = faiss.read_index(self.index_path)

        # Load metadata
        print(f"Loading metadata from '{self.metadata_path}'...")
        self.metadata_df = pd.read_parquet(self.metadata_path)

        if self.index.ntotal != len(self.metadata_df):
            raise ValueError(
                f"Index size ({self.index.ntotal}) does not match metadata size ({len(self.metadata_df)})!"
            )

        # Load SentenceTransformer model
        import torch
        from sentence_transformers import SentenceTransformer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model '{self.model_name}' on [{self.device.upper()}]...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print("SemanticSearcher ready.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top-k semantically relevant Telugu articles for a query.

        Args:
            query: User's Telugu search query string.
            top_k: Number of results to return (default: 5).

        Returns:
            List of result dictionaries containing doc_id, headline, snippet, article, and score.
        """
        if self.index is None or self.metadata_df is None or self.model is None:
            self.load()

        # Input validation
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        if top_k <= 0:
            top_k = 5
        top_k = min(top_k, self.index.ntotal)

        # E5 model requirement: format query with 'query: ' prefix
        formatted_query = f"query: {clean_query}"

        start_time = time.time()
        # Encode and normalize query vector
        query_vector = self.model.encode(
            [formatted_query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        # Query FAISS index
        scores, indices = self.index.search(query_vector, top_k)
        retrieval_time = time.time() - start_time

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0 or idx >= len(self.metadata_df):
                continue

            row = self.metadata_df.iloc[idx]
            article_text = str(row.get("article", ""))
            snippet = article_text[:220].replace("\n", " ").strip()
            if len(article_text) > 220:
                snippet += "..."

            results.append({
                "rank": rank,
                "doc_id": row.get("doc_id"),
                "orig_story_id": row.get("orig_story_id"),
                "headline": row.get("headline"),
                "snippet": snippet,
                "full_article": article_text,
                "score": float(score),
                "retrieval_time_ms": round(retrieval_time * 1000, 2)
            })

        return results


def print_search_results(query: str, results: List[Dict[str, Any]]):
    """Prints formatted search results."""
    print("\n" + "=" * 75)
    print(f" TELUSEARCH RESULTS FOR QUERY: '{query}'")
    print("=" * 75)

    if not results:
        print("No matching articles found.")
        return

    retrieval_ms = results[0].get("retrieval_time_ms", 0)
    print(f"Found {len(results)} results in {retrieval_ms:.1f} ms:\n")

    for r in results:
        print(f"[{r['rank']}] Score: {r['score']:.4f} | ID: {r['doc_id']} (Story: {r['orig_story_id']})")
        print(f"    Headline: {r['headline']}")
        print(f"    Snippet : {r['snippet']}\n")


def main():
    parser = argparse.ArgumentParser(description="TELUSEARCH Semantic Retrieval")
    parser.add_argument("--query", type=str, default=None, help="Telugu search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--index", type=str, default="artifacts/faiss_index.bin", help="Path to faiss index")
    parser.add_argument("--metadata", type=str, default="artifacts/metadata.parquet", help="Path to metadata")
    args = parser.parse_args()

    searcher = SemanticSearcher(index_path=args.index, metadata_path=args.metadata)
    searcher.load()

    if args.query:
        results = searcher.search(args.query, top_k=args.top_k)
        print_search_results(args.query, results)
    else:
        # Default test queries
        test_queries = [
            "రోడ్డు ప్రమాదం మృతులు",
            "రైతు సమస్యలు పంట నష్టం",
            "సినీ నటులు మరియు దర్శకులు"
        ]
        print("\nRunning test queries across indexed Telugu news articles...")
        for q in test_queries:
            res = searcher.search(q, top_k=args.top_k)
            print_search_results(q, res)


if __name__ == "__main__":
    main()
