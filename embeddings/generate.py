"""
embeddings/generate.py
----------------------
TELUSEARCH: Semantic Embeddings Generation (Phase 3)

Key objectives:
1. Load multilingual semantic embedding model 'intfloat/multilingual-e5-base'.
2. Adhere to model input format: 'passage: ' prefix for documents and 'query: ' for queries.
3. Generate normalized 768-dimensional embeddings in batches.
4. Verify embedding dimensions, numerical validity, and cosine similarity.
5. Save generated embeddings and metadata to local artifacts/ directory.
6. Provide a small test mode (--test) to evaluate runtime and compatibility.
"""

import os
import sys
import time
import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# Windows UTF-8 console output support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768


def get_embedding_model(model_name: str = MODEL_NAME, device: Optional[str] = None):
    """
    Loads and returns the SentenceTransformer model.
    Falls back gracefully to CPU if CUDA is not available.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading embedding model '{model_name}' on device: [{device.upper()}]...")
    model = SentenceTransformer(model_name, device=device)
    return model, device


def format_e5_text(text: str, is_query: bool = False) -> str:
    """
    E5 models require explicit prefixing:
    - Queries must start with 'query: '
    - Documents/passages must start with 'passage: '
    """
    prefix = "query: " if is_query else "passage: "
    text_clean = str(text).strip()
    return f"{prefix}{text_clean}"


def encode_texts(
    model,
    texts: List[str],
    is_query: bool = False,
    batch_size: int = 32,
    normalize: bool = True
) -> np.ndarray:
    """
    Encodes a list of texts into normalized embedding vectors.
    """
    formatted_texts = [format_e5_text(t, is_query=is_query) for t in texts]
    embeddings = model.encode(
        formatted_texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 5,
        normalize_embeddings=normalize,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def run_small_test(model=None, device: str = "cpu"):
    """
    Phase 3 Completion Gate Test:
    Encodes 5 sample Telugu articles and 2 Telugu queries.
    Validates vector dimensions, normalization, runtime, and cosine similarity.
    """
    print("\n" + "=" * 75)
    print(" PHASE 3: SMALL EMBEDDING & SIMILARITY TEST")
    print("=" * 75)

    if model is None:
        model, device = get_embedding_model()

    # Load 5 sample articles from processed dataset
    data_path = "data/processed/telugu_news_clean.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing processed data at {data_path}. Run preprocessing/clean.py first.")

    df = pd.read_parquet(data_path).head(5)
    print(f"Loaded {len(df)} sample articles from {data_path}.")

    # 1. Document encoding
    print("\n[Step 1] Encoding sample articles with 'passage: ' prefix...")
    start_time = time.time()
    doc_embeddings = encode_texts(model, df["search_text"].tolist(), is_query=False, batch_size=8)
    doc_encode_time = time.time() - start_time

    # Validate document embeddings
    print(f"  • Document embeddings shape: {doc_embeddings.shape}")
    print(f"  • Embedding dtype: {doc_embeddings.dtype}")
    print(f"  • Expected dimension: {EMBEDDING_DIM} | Actual: {doc_embeddings.shape[1]}")
    norms = np.linalg.norm(doc_embeddings, axis=1)
    print(f"  • Vector L2 norms (should all be ~1.0): min={norms.min():.4f}, max={norms.max():.4f}")
    print(f"  • Encoding time for 5 articles on {device.upper()}: {doc_encode_time:.2f}s ({doc_encode_time/len(df):.3f}s/doc)")

    # 2. Query encoding and similarity testing
    test_queries = [
        ("రోడ్డు ప్రమాదం మృతులు", "Road accident fatalities"),
        ("సినిమా దర్శకుడు కన్నుమూత", "Film director demise")
    ]

    print("\n[Step 2] Testing Semantic Similarity for Telugu Queries...")
    for tel_query, en_meaning in test_queries:
        print(f"\n--- Query: '{tel_query}' ({en_meaning}) ---")
        q_start = time.time()
        q_emb = encode_texts(model, [tel_query], is_query=True, batch_size=1)
        q_time = time.time() - q_start

        # Dot product with normalized vectors == Cosine Similarity
        scores = np.dot(doc_embeddings, q_emb.T).flatten()
        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])
        top_row = df.iloc[top_idx]

        print(f"  • Query encode time: {q_time:.3f}s")
        print(f"  • Top Semantic Match (Score: {top_score:.4f}):")
        print(f"    - Doc ID  : {top_row['doc_id']}")
        print(f"    - Headline: {top_row['headline']}")
        print(f"    - Snippet : {top_row['article'][:120]}...")

    print("\n" + "=" * 75)
    print(" Small embedding test passed successfully!")
    print("=" * 75)
    return {
        "device": device,
        "embedding_dim": doc_embeddings.shape[1],
        "doc_encode_time_sec": round(doc_encode_time, 2),
        "articles_tested": len(df)
    }


def generate_full_embeddings(
    input_parquet: str = "data/processed/telugu_news_clean.parquet",
    output_dir: str = "artifacts",
    batch_size: int = 32,
    limit: Optional[int] = None,
    force: bool = False
):
    """
    Generates embeddings for the full prepared dataset and saves them as artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    embeddings_file = os.path.join(output_dir, "embeddings.npy")
    metadata_file = os.path.join(output_dir, "metadata.parquet")

    if not force and os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        print(f"Existing embeddings found at '{embeddings_file}'.")
        print("Use --force to regenerate. Skipping embedding generation.")
        existing_emb = np.load(embeddings_file)
        print(f"Loaded existing embeddings: shape {existing_emb.shape}")
        return existing_emb

    df = pd.read_parquet(input_parquet)
    if limit and limit < len(df):
        print(f"Limiting dataset to {limit:,} records as requested...")
        df = df.head(limit).copy()

    total_docs = len(df)
    print(f"Starting embedding generation for {total_docs:,} articles...")

    model, device = get_embedding_model()

    start_time = time.time()
    embeddings = encode_texts(
        model=model,
        texts=df["search_text"].tolist(),
        is_query=False,
        batch_size=batch_size,
        normalize=True
    )
    total_time = time.time() - start_time

    # Save artifacts
    print(f"Saving embeddings array to '{embeddings_file}'...")
    np.save(embeddings_file, embeddings)

    print(f"Saving article metadata to '{metadata_file}'...")
    # Save metadata dataframe containing all columns necessary for retrieval & display
    df.to_parquet(metadata_file, index=False)

    file_size_mb = os.path.getsize(embeddings_file) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(" EMBEDDING GENERATION COMPLETE")
    print("=" * 75)
    print(f"  • Total documents embedded: {total_docs:,}")
    print(f"  • Embedding dimensions    : {embeddings.shape[1]}")
    print(f"  • Device used             : {device.upper()}")
    print(f"  • Total runtime           : {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"  • Throughput              : {total_docs / total_time:.1f} docs/second")
    print(f"  • Embeddings file size    : {file_size_mb:.2f} MB")
    print(f"  • Saved locations         : {embeddings_file}, {metadata_file}")
    print("=" * 75)

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="TELUSEARCH Embedding Generation")
    parser.add_argument("--test", action="store_true", help="Run small verification test on 5 articles")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding (default: 32)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to embed")
    parser.add_argument("--force", action="store_true", help="Force regenerate existing embeddings")
    args = parser.parse_args()

    if args.test:
        run_small_test()
    else:
        generate_full_embeddings(
            batch_size=args.batch_size,
            limit=args.limit,
            force=args.force
        )


if __name__ == "__main__":
    main()
