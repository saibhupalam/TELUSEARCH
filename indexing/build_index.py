"""
indexing/build_index.py
-----------------------
TELUSEARCH: FAISS Index Builder (Phase 4)

Key objectives:
1. Load normalized embeddings and document metadata from artifacts/.
2. Build an exact Inner Product FAISS index (faiss.IndexFlatIP) matching cosine similarity.
3. Verify alignment: ensure index.ntotal exactly equals metadata record count.
4. Persist the FAISS index to disk (artifacts/faiss_index.bin).
"""

import os
import sys
import time
import argparse
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


def build_faiss_index(
    embeddings_path: str = "artifacts/embeddings.npy",
    metadata_path: str = "artifacts/metadata.parquet",
    output_index_path: str = "artifacts/faiss_index.bin"
) -> faiss.IndexFlatIP:
    """
    Builds and persists a FAISS IndexFlatIP vector index from saved embeddings.
    """
    print("=" * 75)
    print(" TELUSEARCH: Building FAISS Index (Phase 4)")
    print("=" * 75)

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at '{embeddings_path}'. Run embeddings/generate.py first.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'. Run embeddings/generate.py first.")

    # 1. Load embeddings & metadata
    print(f"Loading embeddings from '{embeddings_path}'...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    num_vectors, dim = embeddings.shape
    print(f"-> Loaded {num_vectors:,} vectors of dimension {dim}.")

    print(f"Loading metadata from '{metadata_path}'...")
    df_meta = pd.read_parquet(metadata_path)
    print(f"-> Loaded {len(df_meta):,} metadata records.")

    if len(df_meta) != num_vectors:
        raise ValueError(
            f"Mismatch between metadata rows ({len(df_meta):,}) and vectors ({num_vectors:,})!"
        )

    # 2. Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("Vectors were not strictly normalized. Normalizing vectors with faiss.normalize_L2...")
        faiss.normalize_L2(embeddings)
    else:
        print("-> Confirmed: All vectors are unit normalized (L2 norm = 1.0).")

    # 3. Create FAISS IndexFlatIP
    # For normalized vectors, Inner Product (dot product) == Cosine Similarity
    print(f"Creating FAISS IndexFlatIP (metric=Inner Product, dimension={dim})...")
    start_time = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    build_time = time.time() - start_time

    print(f"-> Added {index.ntotal:,} vectors to FAISS index in {build_time:.4f} seconds.")

    # 4. Save index to disk
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    print(f"Saving FAISS index to '{output_index_path}'...")
    faiss.write_index(index, output_index_path)

    index_size_mb = os.path.getsize(output_index_path) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(" FAISS INDEX SUMMARY")
    print("=" * 75)
    print(f"  • Total indexed vectors : {index.ntotal:,}")
    print(f"  • Vector dimension       : {dim}")
    print(f"  • Metric                 : Inner Product (Cosine Similarity)")
    print(f"  • Index file size        : {index_size_mb:.2f} MB")
    print(f"  • Index path             : {output_index_path}")
    print("=" * 75)

    return index


def main():
    parser = argparse.ArgumentParser(description="TELUSEARCH Build FAISS Index")
    parser.add_argument("--embeddings", type=str, default="artifacts/embeddings.npy", help="Path to embeddings.npy")
    parser.add_argument("--metadata", type=str, default="artifacts/metadata.parquet", help="Path to metadata.parquet")
    parser.add_argument("--output", type=str, default="artifacts/faiss_index.bin", help="Path to output index.bin")
    args = parser.parse_args()

    build_faiss_index(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        output_index_path=args.output
    )


if __name__ == "__main__":
    main()
