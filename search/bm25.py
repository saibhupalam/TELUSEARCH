"""
search/bm25.py
--------------
TELUSEARCH: BM25 Keyword Retrieval Engine (Phase 6 Optional Extension)

Key objectives:
1. Implement BM25 lexical ranking using rank_bm25.BM25Okapi.
2. Perform Telugu tokenization without corrupting Unicode characters.
3. Retrieve top-k keyword-matched articles with BM25 scores.
4. Support standalone keyword search and hybrid integration.
"""

import os
import sys
import re
import time
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
from rank_bm25 import BM25Okapi

# Windows UTF-8 console output support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def tokenize_telugu(text: str) -> List[str]:
    """
    Tokenizes Telugu text into whitespace-delimited words,
    stripping punctuation while preserving Telugu Unicode letters.
    """
    if not text or not isinstance(text, str):
        return []
    cleaned = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
    return tokens


class BM25Searcher:
    """
    In-memory BM25 index over the prepared corpus.
    """

    def __init__(self, metadata_path: str = "artifacts/metadata.parquet"):
        self.metadata_path = metadata_path
        self.metadata_df = None
        self.bm25 = None
        self.corpus_tokens = []

    def load(self):
        """Loads metadata and builds the BM25 index."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at '{self.metadata_path}'.")

        print(f"Loading metadata from '{self.metadata_path}' for BM25...")
        self.metadata_df = pd.read_parquet(self.metadata_path)

        print(f"Tokenizing {len(self.metadata_df):,} Telugu documents for BM25...")
        start_time = time.time()
        self.corpus_tokens = [
            tokenize_telugu(str(row.get("search_text", "")))
            for _, row in self.metadata_df.iterrows()
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        build_time = time.time() - start_time
        print(f"BM25 index built in {build_time:.2f} seconds.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes BM25 keyword search for a Telugu query.
        """
        if self.bm25 is None or self.metadata_df is None:
            self.load()

        if not query or not query.strip():
            return []

        clean_query = query.strip()
        query_tokens = tokenize_telugu(clean_query)
        if not query_tokens:
            return []

        top_k = min(max(1, top_k), len(self.metadata_df))

        start_time = time.time()
        doc_scores = self.bm25.get_scores(query_tokens)
        retrieval_time = time.time() - start_time

        top_indices = doc_scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(doc_scores[idx])
            if score <= 0:
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
                "score": round(score, 4),
                "retrieval_time_ms": round(retrieval_time * 1000, 2)
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="TELUSEARCH BM25 Keyword Search")
    parser.add_argument("--query", type=str, default="రోడ్డు ప్రమాదం", help="Telugu search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    searcher = BM25Searcher()
    searcher.load()
    results = searcher.search(args.query, top_k=args.top_k)

    print("\n" + "=" * 75)
    print(f" BM25 SEARCH RESULTS FOR QUERY: '{args.query}'")
    print("=" * 75)
    if not results:
        print("No keyword matches found.")
        return

    for r in results:
        print(f"[{r['rank']}] BM25 Score: {r['score']} | ID: {r['doc_id']}")
        print(f"    Headline: {r['headline']}")
        print(f"    Snippet : {r['snippet']}\n")


if __name__ == "__main__":
    main()
