import pandas as pd
import numpy as np
from src.config import DATA_PROCESSED_DIR


def create_features(df):
    df = df.copy()

    df["release_date"] = pd.to_datetime(df["release_date"])

    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_decade"] = (df["release_year"] // 10) * 10


    def count_items(text):
        if str(text).strip():
            return len(str(text).split(", "))
        return 0
    
    df["num_genres"] = df["genres"].apply(count_items)
    df["num_keywords"] = df["keywords"].apply(count_items)


    # labeling each movie depends on its duration
    df["runtime_category"] = pd.cut(df["runtime"], bins=[0, 89, 120, np.inf],
                                                   labels=["short", "standard", "long"])

    # astype(int) turns true 1 , false 0
    df["has_budget"] = (df["budget"] > 0).astype(int)



    median_votes = df["vote_count"].median()

    df["high_engagement"] = (df["vote_count"] >= median_votes).astype(int)



    return df


def run_feature_pipeline():
    df_clean = pd.read_csv("data/processed/movies_cleaned.csv")

    df_features = create_features(df_clean)

    df_features.to_csv("data/processed/movies_features.csv", index=False)



run_feature_pipeline()