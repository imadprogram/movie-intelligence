import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import DATA_PROCESSED_DIR
import joblib





def clean_text(text):
    if not text or pd.isna(text) :
        return ""

    text = str(text).lower()

    # Replace all non-letters with a space
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = ' '.join(text.split())

    return text




def build_tfidf(df, max_features=5000, ngram_range=(1, 2)):
    cleaned_text = df["overview"].apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=ngram_range)

    tfidf_matrix = vectorizer.fit_transform(cleaned_text)
    

    return vectorizer, tfidf_matrix



def run_nlp_pipeline():
    df = pd.read_csv('data/processed/movies_features.csv')
    vectorizer, tfidf_matrix = build_tfidf(df, max_features=5000, ngram_range=(1, 2))
    words = vectorizer.get_feature_names_out()

    bigrams = [w for w in words if " " in w][:10]
    print("sample 2-word phrases:", bigrams)
    # this will print (1312, 5000): 1312 movies x 5000 features
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    joblib.dump(tfidf_matrix, "models/tfidf_matrix.joblib")
    print("[✓] Saved TF-IDF artifacts to models/")


run_nlp_pipeline()