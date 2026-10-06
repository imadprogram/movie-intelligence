import pandas as pd
from src.config import DATA_PROCESSED_DIR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import GridSearchCV
import joblib





def prepare_data():

    df = pd.read_csv('data/processed/movies_features.csv')

    y = df["high_engagement"]

    numeric_cols = ["budget", "revenue", "runtime", "popularity", "release_year", "release_month", "num_genres", "num_keywords", "has_budget"]
    categorical_cols = ["runtime_category"]

    feature_cols = numeric_cols + categorical_cols + ["overview"]
    X = df[feature_cols]

    return X , y, numeric_cols, categorical_cols



def build_preprocessor(numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("text", TfidfVectorizer(max_features=1000, stop_words="english"), "overview")
        ]
    )
    return preprocessor



def build_pipelines(preprocessor):
    models = {
        "Logistic Regression": Pipeline(steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(random_state=42, max_iter=1000))
        ]),

        "Random Forest": Pipeline(steps=[
            ("prep", preprocessor),
            ("model", RandomForestClassifier(random_state=42, n_estimators=100))
        ]),

        "Linear SVM": Pipeline(steps=[
            ("prep", preprocessor),
            ("model", LinearSVC(random_state=42, max_iter=2000, dual=False))
        ])
    }

    return models





def evaluate_models(models, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = []
    for name , pipeline in models.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

        results.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1-Score": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean()
        })

    results_df = pd.DataFrame(results)
    print(results_df)

    return results_df






def run_classification():
    X, y, numeric_cols, categorical_cols = prepare_data()
    preprocessor = build_preprocessor(numeric_cols , categorical_cols)
    pipeline = build_pipelines(preprocessor)
    evaluate = evaluate_models(pipeline, X, y)
    tune_random_forest(preprocessor, X, y)






def tune_random_forest(preprocessor, X, y):
    pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])


    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X, y)


    print("\n--- GridSearchCV Tuning Results ---")
    print("Best Parameters:", grid.best_params_)
    print(f"Best Cross-Validation F1-Score: {grid.best_score_:.4f}")

    joblib.dump(grid.best_estimator_, "models/best_classifier.joblib")
    print("[✓] Saved best tuned classifier to models/best_classifier.joblib")

    return grid.best_estimator_


    
run_classification()