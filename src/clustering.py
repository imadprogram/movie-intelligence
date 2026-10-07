import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PROCESSED_DIR





def prepare_clustering_data():
    df = pd.read_csv("data/processed/movies_features.csv")

    numeric_cols = ["budget", "revenue", "runtime", "popularity", "release_year", "num_genres", "num_keywords"]
    numerics = df[numeric_cols]

    x_scaled = StandardScaler().fit_transform(numerics)


    return x_scaled, df





def find_optimal_k(X_scaled):
    best_k = 2
    best_score = -1
    for k in range(2,8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        print(f"K = {k} | Silhouette Score: {score:.4f}")

        if(score > best_score):
            best_score = score
            best_k = k
        

    print(f"best K = {best_k} with silhouette score: {best_score: .4f}")

    return best_k





def fit_and_profile_clusters(X_scaled, df, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    df["cluster"] = kmeans.labels_
    numeric_cols = ["budget", "revenue", "runtime", "popularity", "release_year", "num_genres", "num_keywords"]
    profile = df.groupby("cluster")[numeric_cols].mean()
    print("\n--- Cluster Profiles (Averages) ---")
    print(profile)
    joblib.dump(kmeans, "models/kmeans.joblib")

    return kmeans, df




def reduce_dimensions_and_save(X_scaled, df):
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X_scaled)

    df["svd_x"] = coords[:, 0]
    df["svd_y"] = coords[:, 1]

    df.to_csv("data/processed/movies_clustered.csv", index=False)





if __name__ == "__main__":
    X_scaled, df = prepare_clustering_data()
    best_k = find_optimal_k(X_scaled)
    kmeans, df = fit_and_profile_clusters(X_scaled, df, best_k)
    reduce_dimensions_and_save(X_scaled, df)