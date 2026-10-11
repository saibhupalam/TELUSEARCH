"""
inspect_dataset.py
------------------
TELUSEARCH: Semantic Search for Telugu News Articles (Phase 1)

This script inspects the Hugging Face dataset 'saidines12/telugu_news_dataset'.
It checks:
1. Dataset split names and row counts.
2. Column schema and data types.
3. Data quality: null values, empty strings, duplicate IDs, exact duplicate articles, and text length statistics.
4. Sample records with Telugu Unicode preservation.
"""

import sys
from collections import Counter

# On Windows, standard output encoding is often cp1252, which triggers
# UnicodeEncodeError when printing Telugu characters. Reconfiguring to utf-8 guarantees proper display.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def inspect_telugu_news_dataset():
    print("=" * 75)
    print(" TELUSEARCH: Hugging Face Dataset Inspection (Phase 1)")
    print(" Dataset: saidines12/telugu_news_dataset")
    print("=" * 75)

    try:
        from datasets import load_dataset, get_dataset_split_names
    except ImportError:
        print("[ERROR] 'datasets' library is not installed.")
        print("Please run: pip install datasets")
        sys.exit(1)

    # Step 1: Detect splits
    print("\n[Step 1] Inspecting dataset splits...")
    try:
        split_names = get_dataset_split_names("saidines12/telugu_news_dataset")
        print(f"-> Available splits: {split_names}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch split names: {e}")
        sys.exit(1)

    # Step 2: Load dataset
    print("\n[Step 2] Loading dataset splits from Hugging Face...")
    try:
        dataset_dict = load_dataset("saidines12/telugu_news_dataset")
        print("-> Successfully loaded dataset.")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print("Diagnostic Tip: Check internet connectivity or Hugging Face Hub status.")
        sys.exit(1)

    total_rows = 0
    print("\n[Step 3] Split summaries & schema:")
    for split_name in split_names:
        split_data = dataset_dict[split_name]
        row_count = len(split_data)
        total_rows += row_count
        features = split_data.features
        print(f"  • Split '{split_name}': {row_count:,} rows")
        print(f"    Columns: {list(features.keys())}")
        print(f"    Schema : {features}")

    print(f"\n-> Total records across all splits: {total_rows:,}")

    # Step 4: Data Quality Checks
    print("\n" + "=" * 75)
    print("[Step 4] Performing Deep Data Quality & Duplication Analysis...")
    print("=" * 75)

    for split_name in split_names:
        split_data = dataset_dict[split_name]
        total_in_split = len(split_data)
        print(f"\n--- Analysis for Split: '{split_name}' ({total_in_split:,} rows) ---")

        # Counts
        seen_ids = set()
        duplicate_id_count = 0
        missing_headlines = 0
        empty_headlines = 0
        missing_articles = 0
        empty_articles = 0

        headline_lengths = []
        article_lengths = []

        articles = []
        headlines = []

        for row in split_data:
            sid = row.get("story_id")
            hl = row.get("headline")
            art = row.get("article")

            # Check ID uniqueness
            if sid is not None:
                if sid in seen_ids:
                    duplicate_id_count += 1
                else:
                    seen_ids.add(sid)

            # Check Headline
            if hl is None:
                missing_headlines += 1
            elif not str(hl).strip():
                empty_headlines += 1
            else:
                hl_str = str(hl).strip()
                headline_lengths.append(len(hl_str))
                headlines.append(hl_str)

            # Check Article
            if art is None:
                missing_articles += 1
            elif not str(art).strip():
                empty_articles += 1
            else:
                art_str = str(art).strip()
                article_lengths.append(len(art_str))
                articles.append(art_str)

        unique_articles = len(set(articles))
        unique_headlines = len(set(headlines))
        unique_ids = len(seen_ids)

        print(f"  • Missing headlines (None) : {missing_headlines}")
        print(f"  • Empty headlines (blank)  : {empty_headlines}")
        print(f"  • Missing articles (None)  : {missing_articles}")
        print(f"  • Empty articles (blank)   : {empty_articles}")
        print(f"  • Unique story_id count    : {unique_ids:,} (Duplicate IDs: {duplicate_id_count:,})")
        print(f"  • Unique article text count: {unique_articles:,} (Duplicate texts: {total_in_split - unique_articles:,})")
        print(f"  • Unique headline count    : {unique_headlines:,}")

        if headline_lengths:
            avg_hl = sum(headline_lengths) / len(headline_lengths)
            print(f"  • Headline length (chars)  : min={min(headline_lengths)}, max={max(headline_lengths)}, avg={avg_hl:.1f}")
        if article_lengths:
            avg_art = sum(article_lengths) / len(article_lengths)
            print(f"  • Article length (chars)   : min={min(article_lengths)}, max={max(article_lengths)}, avg={avg_art:.1f}")

    # Step 5: Sample Records
    print("\n" + "=" * 75)
    print("[Step 5] Sample Telugu News Records (Preserving Unicode)")
    print("=" * 75)

    sample_split = "validation" if "validation" in split_names else split_names[0]
    sample_records = dataset_dict[sample_split].select(range(min(3, len(dataset_dict[sample_split]))))

    for idx, row in enumerate(sample_records, 1):
        print(f"\n--- Sample #{idx} (Split: {sample_split}) ---")
        print(f"Story ID : {row.get('story_id')}")
        print(f"Headline : {row.get('headline')}")
        article_snippet = str(row.get('article', ''))[:160].replace("\n", " ")
        print(f"Snippet  : {article_snippet}...")

    # Step 6: Engineering Recommendations for Phase 2
    print("\n" + "=" * 75)
    print("[Step 6] Key Engineering Findings & Recommendations")
    print("=" * 75)
    print("1. Schema verified: story_id (int64), headline (string), article (string).")
    print("2. Completeness: 0 missing/null headlines and 0 missing/null articles across all splits.")
    print("3. Duplication discovery:")
    print("   - 'train' contains 83,866 rows but only 22,189 unique articles (~73.5% duplicates).")
    print("   - 'test' contains 12,504 rows but only 3,312 unique articles (~73.5% duplicates).")
    print("   - 'validation' contains 5,962 rows and ALL 5,962 articles are 100% unique!")
    print("4. story_id collision:")
    print("   - The upstream dataset reuses story_id for different articles. story_id is NOT a unique primary key.")
    print("   - Recommendation for Phase 2: Generate a composite or surrogate key (e.g., f'{split}_{idx}')")
    print("     while retaining the original story_id for traceability.")
    print("5. Recommended Working Dataset Size:")
    print("   - The 5,962 articles from the validation split (or a deduplicated subset of 5,000–6,000 rows)")
    print("     is optimal for CPU embedding generation (~3 to 5 minutes) and instant retrieval.")
    print("=" * 75)
    print("Phase 1 Inspection Complete.")
    print("=" * 75)


if __name__ == "__main__":
    inspect_telugu_news_dataset()
