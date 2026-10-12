"""
search/hybrid.py
----------------
TELUSEARCH: Hybrid Retrieval Engine (Phase 6 Optional Extension)

Key objectives:
1. Combine dense semantic retrieval (FAISS + E5) with sparse lexical retrieval (BM25).
2. Apply min-max score normalization to bridge score scales.
3. Compute weighted hybrid score: alpha * dense_score + (1 - alpha) * bm25_score.
4. Support Reciprocal Rank Fusion (RRF) as an alternative rank fusion method.
5. Return top-k hybrid-ranked articles.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from search.semantic import SemanticSearcher
from search.bm25 import BM25Searcher

# Windows UTF-8 console output support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class HybridSearcher:
    """
    Hybrid retriever fusing FAISS semantic retrieval with BM25 keyword matching.
    """

    def __init__(
        self,
        semantic_searcher: Optional[SemanticSearcher] = None,
        bm25_searcher: Optional[BM25Searcher] = None,
        alpha: float = 0.7
    ):
        """
        Args:
            semantic_searcher: Instance of SemanticSearcher.
            bm25_searcher: Instance of BM25Searcher.
            alpha: Weight for dense semantic score (1 - alpha for BM25). Default: 0.7.
        """
        self.semantic_searcher = semantic_searcher or SemanticSearcher()
        self.bm25_searcher = bm25_searcher or BM25Searcher()
        self.alpha = alpha

    def load(self):
        """Loads both semantic and BM25 indexers."""
        print("Loading semantic searcher...")
        self.semantic_searcher.load()
        print("Loading BM25 searcher...")
        self.bm25_searcher.load()
        print("HybridSearcher ready.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
        method: str = "weighted"
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search combining semantic and BM25 scores.

        Args:
            query: Telugu search query.
            top_k: Number of final results to return.
            alpha: Custom alpha weight (overrides default).
            method: 'weighted' (score interpolation) or 'rrf' (reciprocal rank fusion).
        """
        if alpha is None:
            alpha = self.alpha

        start_time = time.time()
        # Retrieve candidate pool from both methods (fetch top_k * 3 candidates)
        pool_size = max(top_k * 3, 20)
        dense_results = self.semantic_searcher.search(query, top_k=pool_size)
        bm25_results = self.bm25_searcher.search(query, top_k=pool_size)

        if not dense_results and not bm25_results:
            return []

        if method == "rrf":
            return self._fuse_rrf(dense_results, bm25_results, top_k, start_time)
        else:
            return self._fuse_weighted(dense_results, bm25_results, alpha, top_k, start_time)

    def _fuse_weighted(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        alpha: float,
        top_k: int,
        start_time: float
    ) -> List[Dict[str, Any]]:
        """Min-max normalized linear score interpolation."""
        # Map doc_id -> item data
        doc_store = {}
        dense_scores = {}
        bm25_scores = {}

        for r in dense_results:
            did = r["doc_id"]
            doc_store[did] = r
            dense_scores[did] = r["score"]

        for r in bm25_results:
            did = r["doc_id"]
            if did not in doc_store:
                doc_store[did] = r
            bm25_scores[did] = r["score"]

        # Min-max normalization helper
        def min_max_norm(score_dict):
            if not score_dict:
                return {}
            vals = list(score_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return {k: 1.0 for k in score_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

        dense_norm = min_max_norm(dense_scores)
        bm25_norm = min_max_norm(bm25_scores)

        all_doc_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        fused = []

        for did in all_doc_ids:
            d_score = dense_norm.get(did, 0.0)
            b_score = bm25_norm.get(did, 0.0)
            hybrid_score = (alpha * d_score) + ((1.0 - alpha) * b_score)

            base = doc_store[did]
            fused.append({
                "doc_id": did,
                "orig_story_id": base["orig_story_id"],
                "headline": base["headline"],
                "snippet": base["snippet"],
                "full_article": base["full_article"],
                "score": round(hybrid_score, 4),
                "dense_score": round(dense_scores.get(did, 0.0), 4),
                "bm25_score": round(bm25_scores.get(did, 0.0), 4),
            })

        # Sort descending by hybrid score
        fused.sort(key=lambda x: x["score"], reverse=True)
        retrieval_time = time.time() - start_time

        results = []
        for rank, item in enumerate(fused[:top_k], 1):
            item["rank"] = rank
            item["retrieval_time_ms"] = round(retrieval_time * 1000, 2)
            results.append(item)

        return results

    def _fuse_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
        start_time: float,
        k_constant: int = 60
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        doc_store = {}

        for rank, r in enumerate(dense_results, 1):
            did = r["doc_id"]
            doc_store[did] = r
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (k_constant + rank))

        for rank, r in enumerate(bm25_results, 1):
            did = r["doc_id"]
            if did not in doc_store:
                doc_store[did] = r
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (k_constant + rank))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        retrieval_time = time.time() - start_time

        results = []
        for rank, (did, score) in enumerate(sorted_docs[:top_k], 1):
            base = doc_store[did]
            results.append({
                "rank": rank,
                "doc_id": did,
                "orig_story_id": base["orig_story_id"],
                "headline": base["headline"],
                "snippet": base["snippet"],
                "full_article": base["full_article"],
                "score": round(score, 5),
                "retrieval_time_ms": round(retrieval_time * 1000, 2)
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="TELUSEARCH Hybrid Search")
    parser.add_argument("--query", type=str, default="రైతు సమస్యలు పంట నష్టం", help="Telugu search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--alpha", type=float, default=0.7, help="Dense weight (default 0.7)")
    args = parser.parse_args()

    searcher = HybridSearcher(alpha=args.alpha)
    searcher.load()
    results = searcher.search(args.query, top_k=args.top_k)

    print("\n" + "=" * 75)
    print(f" HYBRID SEARCH RESULTS FOR QUERY: '{args.query}' (Alpha={args.alpha})")
    print("=" * 75)
    for r in results:
        print(f"[{r['rank']}] Hybrid Score: {r['score']} (Dense: {r.get('dense_score')}, BM25: {r.get('bm25_score')})")
        print(f"    Headline: {r['headline']}")
        print(f"    Snippet : {r['snippet']}\n")


if __name__ == "__main__":
    main()
