import json
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

def load_raw_data(filename="movies_raw.json"):
    input_path = DATA_RAW_DIR / filename
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_genres(genre_list):
    """
    Transforms [{'id': 28, 'name': 'Action'}, ...] -> ['Action', ...]
    """
    if isinstance(genre_list, list):
        return [g["name"] for g in genre_list if isinstance(g, dict) and "name" in g]
    return []

def extract_keywords(keywords_dict):
    """
    Transforms {'keywords': [{'id': 1, 'name': 'hero'}, ...]} -> ['hero', ...]
    """
    if isinstance(keywords_dict, dict) and "keywords" in keywords_dict:
        kw_list = keywords_dict["keywords"]
        if isinstance(kw_list, list):
            return [k["name"] for k in kw_list if isinstance(k, dict) and "name" in k]
    return []

def clean_movie_data(raw_data):
    """
    Main cleaning pipeline:
    - Flattens nested JSON
    - Deduplicates
    - Fixes data types & missing values
    """
    cleaned_rows = []
    
    for item in raw_data:
        cleaned_rows.append({
            "movie_id": item.get("id"),
            "title": item.get("title", ""),
            "overview": item.get("overview") or "",  # Prevent None for NLP
            "release_date": item.get("release_date"),
            "runtime": item.get("runtime"),
            "original_language": item.get("original_language", "en"),
            "genres": extract_genres(item.get("genres")),
            "keywords": extract_keywords(item.get("keywords")),
            "budget": item.get("budget", 0),
            "revenue": item.get("revenue", 0),
            "popularity": item.get("popularity", 0.0),
            "vote_average": item.get("vote_average", 0.0),
            "vote_count": item.get("vote_count", 0),
        })
        
    df = pd.DataFrame(cleaned_rows)
    
    # 1. Deduplication
    initial_count = len(df)
    df.drop_duplicates(subset=["movie_id"], inplace=True)
    print(f"[*] Removed {initial_count - len(df)} duplicate records.")
    
    # 2. Date conversion
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    # Drop records that have no valid release date
    df.dropna(subset=["release_date"], inplace=True)
    
    # 3. Numeric coercions
    numeric_cols = ["budget", "revenue", "popularity", "vote_average", "vote_count", "runtime"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
    # 4. Impute runtime 0 with median of valid runtimes (> 0)
    valid_runtimes = df[df["runtime"] > 0]["runtime"]
    median_runtime = valid_runtimes.median() if not valid_runtimes.empty else 100
    df.loc[df["runtime"] == 0, "runtime"] = median_runtime
    print(f"[*] Imputed missing runtimes with median: {median_runtime:.1f} minutes.")
    
    # 5. Reset index
    df.reset_index(drop=True, inplace=True)
    
    print(f"[✓] Final clean dataset shape: {df.shape}")
    return df

def save_clean_data(df, output_csv="movies_cleaned.csv", output_json="movies_cleaned.json"):
    """
    Saves clean data in CSV (for Pandas & ML) and JSON (for MongoDB).
    """
    csv_path = DATA_PROCESSED_DIR / output_csv
    json_path = DATA_PROCESSED_DIR / output_json
    
    # Save CSV (converting lists to strings for tabular format)
    df_csv = df.copy()
    df_csv["genres"] = df_csv["genres"].apply(lambda g: ", ".join(g) if isinstance(g, list) else "")
    df_csv["keywords"] = df_csv["keywords"].apply(lambda k: ", ".join(k) if isinstance(k, list) else "")
    df_csv.to_csv(csv_path, index=False)
    
    # Save JSON (preserves native arrays for MongoDB)
    df.to_json(json_path, orient="records", date_format="iso", indent=2)
    
    print(f"[✓] Saved cleaned data to:")
    print(f"    - {csv_path}")
    print(f"    - {json_path}")

def run_cleaning_pipeline():
    raw_data = load_raw_data()
    df_clean = clean_movie_data(raw_data)
    save_clean_data(df_clean)
    return df_clean

if __name__ == "__main__":
    run_cleaning_pipeline()