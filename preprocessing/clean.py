"""
preprocessing/clean.py
----------------------
TELUSEARCH: Data Preparation & Preprocessing Pipeline (Phase 2)

Key objectives:
1. Retain stable article IDs, headlines, and articles.
2. Handle missing and empty text safely.
3. Remove exact duplicate articles while preserving traceability to original records.
4. Apply conservative text cleaning without damaging Telugu Unicode characters.
5. Preserve original text for user display alongside search-ready text.
6. Save processed dataset reproducibly in Parquet and CSV formats.
"""

import os
import sys
import re
import argparse
from typing import Optional, Tuple, Dict, Any
import pandas as pd

# Windows UTF-8 console output support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def conservative_clean_text(text: Optional[str]) -> str:
    """
    Conservatively cleans Telugu text without damaging Unicode characters:
    - Normalizes Unicode spaces (e.g., non-breaking space, zero-width space).
    - Replaces consecutive whitespace characters with a single standard space.
    - Strips leading and trailing whitespace.
    - Preserves all Telugu characters (U+0C00 - U+0C7F), punctuation, and numbers.
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace zero-width spaces, non-breaking spaces, and BOM
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")
    
    # Normalize tabs, newlines, and multiple spaces into a single space
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def prepare_dataset(
    split_name: str = "validation",
    sample_size: Optional[int] = None,
    output_dir: str = "data/processed",
    save_outputs: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw Telugu news articles, validates quality, applies conservative cleaning,
    removes duplicates, and assigns unique document IDs.

    Args:
        split_name: Hugging Face dataset split (default: 'validation' which has 5,962 unique articles)
        sample_size: Optional number of records to sample for quick testing
        output_dir: Directory where processed files will be saved
        save_outputs: Whether to write Parquet/CSV files to disk

    Returns:
        Tuple of (processed_dataframe, summary_statistics_dict)
    """
    from datasets import load_dataset

    print(f"Loading '{split_name}' split from saidines12/telugu_news_dataset...")
    raw_dataset = load_dataset("saidines12/telugu_news_dataset", split=split_name)
    total_raw = len(raw_dataset)
    print(f"Loaded {total_raw:,} records from '{split_name}' split.")

    if sample_size and sample_size < total_raw:
        print(f"Sampling first {sample_size:,} records for testing...")
        records = [raw_dataset[i] for i in range(sample_size)]
    else:
        records = list(raw_dataset)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    initial_count = len(df)

    # 1. Handle missing/null values
    df["headline_raw"] = df["headline"].fillna("").astype(str)
    df["article_raw"] = df["article"].fillna("").astype(str)
    df["orig_story_id"] = df["story_id"]

    # Filter out empty or whitespace-only records
    df["cleaned_headline"] = df["headline_raw"].apply(conservative_clean_text)
    df["cleaned_article"] = df["article_raw"].apply(conservative_clean_text)

    non_empty_mask = (df["cleaned_headline"].str.len() > 0) & (df["cleaned_article"].str.len() > 0)
    empty_dropped = int((~non_empty_mask).sum())
    df = df[non_empty_mask].copy()

    # 2. Deduplication based on cleaned article content
    # In Phase 1 inspection, we observed identical articles scraped under different runs.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["cleaned_article"], keep="first").copy()
    duplicates_dropped = before_dedup - len(df)

    # Reset index and generate guaranteed unique, sequential doc_ids
    df.reset_index(drop=True, inplace=True)
    df["doc_id"] = [f"telugu_news_{i:05d}" for i in range(len(df))]

    # 3. Create composite search text for embeddings & hybrid search
    # E5 models embed headline + article as a single contextual passage
    df["search_text"] = df["cleaned_headline"] + " -- " + df["cleaned_article"]

    # 4. Organize final schema
    # Preserve original raw text for faithful display in UI, plus clean/search fields
    final_cols = [
        "doc_id",
        "orig_story_id",
        "headline_raw",
        "article_raw",
        "cleaned_headline",
        "cleaned_article",
        "search_text"
    ]
    final_df = df[final_cols].copy()
    # Rename display fields for user convenience
    final_df.rename(columns={"headline_raw": "headline", "article_raw": "article"}, inplace=True)

    stats = {
        "split": split_name,
        "input_records": initial_count,
        "empty_dropped": empty_dropped,
        "duplicates_dropped": duplicates_dropped,
        "final_records": len(final_df),
        "headline_avg_len": round(final_df["cleaned_headline"].str.len().mean(), 1),
        "article_avg_len": round(final_df["cleaned_article"].str.len().mean(), 1),
    }

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
        parquet_path = os.path.join(output_dir, "telugu_news_clean.parquet")
        csv_path = os.path.join(output_dir, "telugu_news_clean.csv")

        # Save to Parquet (best for UTF-8 preservation & fast loading)
        final_df.to_parquet(parquet_path, index=False)
        # Also save CSV with utf-8-sig for Excel/universal viewing
        final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        stats["parquet_path"] = parquet_path
        stats["csv_path"] = csv_path
        stats["parquet_size_mb"] = round(os.path.getsize(parquet_path) / (1024 * 1024), 2)
        stats["csv_size_mb"] = round(os.path.getsize(csv_path) / (1024 * 1024), 2)

    return final_df, stats


def print_before_after_samples(df_sample: pd.DataFrame, num_samples: int = 2):
    """Prints clear before-and-after comparison of cleaned records."""
    print("\n" + "=" * 75)
    print(" BEFORE-AND-AFTER CLEANING VERIFICATION (Preserving Telugu Unicode)")
    print("=" * 75)
    for i in range(min(num_samples, len(df_sample))):
        row = df_sample.iloc[i]
        print(f"\n[Sample #{i + 1}] Doc ID: {row['doc_id']} | Story ID: {row['orig_story_id']}")
        print("  RAW HEADLINE      :", repr(row['headline'][:70]))
        print("  CLEANED HEADLINE  :", repr(row['cleaned_headline'][:70]))
        print("  RAW ARTICLE (snippet)    :", repr(row['article'][:100]))
        print("  CLEANED ARTICLE (snippet):", repr(row['cleaned_article'][:100]))
        print("  SEARCH TEXT (passage)    :", repr(row['search_text'][:120]))


def main():
    parser = argparse.ArgumentParser(description="TELUSEARCH Preprocessing Pipeline")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to process")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for testing (e.g., 100)")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    print("=" * 75)
    print(" TELUSEARCH: Data Preparation Pipeline (Phase 2)")
    print("=" * 75)

    df_clean, stats = prepare_dataset(
        split_name=args.split,
        sample_size=args.sample,
        output_dir=args.output_dir,
        save_outputs=True
    )

    print("\n" + "=" * 75)
    print(" PREPROCESSING SUMMARY & DATA QUALITY METRICS")
    print("=" * 75)
    for k, v in stats.items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")

    print_before_after_samples(df_clean, num_samples=2)

    print("\n" + "=" * 75)
    print(" Phase 2 Preprocessing execution completed.")
    print("=" * 75)


if __name__ == "__main__":
    main()
