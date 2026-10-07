import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity



def load_recommender_artifacts():
    df = pd.read_csv("data/processed/movies_cleaned.csv")
    tfidf_matrix = joblib.load("models/tfidf_matrix.joblib")

    return df, tfidf_matrix



def recommend_movies(title, df, tfidf_matrix, top_n=5):
    matches = df[df["title"].str.lower() == title.lower()]

    if matches.empty:
        print(f"Movie '{title}' not found!")
        return None

    idx = matches.index[0]

    similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    sorted_indices = similarity.argsort()[::-1]

    recommend_indices = [i for i in sorted_indices if i != idx][:top_n]

    results = df.iloc[recommend_indices][["title", "genres", "release_date"]].copy()

    results["similarity_score"] = similarity[recommend_indices]

    return results




if __name__ == "__main__":
    df, tfidf_matrix = load_recommender_artifacts()

    recs = recommend_movies("Captain America: The First Avenger", df, tfidf_matrix)
    print(recs)