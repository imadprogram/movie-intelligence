from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


# Note: We import src modules inside functions so Airflow's scheduler doesn't load heavy ML libraries on startup
def task_extract():
    from src.extraction import extract_and_save
    extract_and_save(num_pages=5)

def task_clean():
    from src.cleaning import run_cleaning_pipeline
    run_cleaning_pipeline()

def task_features():
    from src.features import run_feature_pipeline
    from src.nlp import run_nlp_pipeline
    run_feature_pipeline()
    run_nlp_pipeline()

def task_mongodb():
    from src.database import insert_movies
    insert_movies()

def task_ml():
    from src.classification import run_classification
    from src.clustering import (
        prepare_clustering_data,
        find_optimal_k,
        fit_and_profile_clusters,
        reduce_dimensions_and_save,
    )
    run_classification()
    X_scaled, df = prepare_clustering_data()
    best_k = find_optimal_k(X_scaled)
    kmeans, df = fit_and_profile_clusters(X_scaled, df, best_k)
    reduce_dimensions_and_save(X_scaled, df)


default_args = {
    "owner": "movie_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="movie_intelligence_pipeline",
    default_args=default_args,
    description="Automated TMDB ETL, MongoDB storage, and ML retraining pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    t1_extract = PythonOperator(
        task_id="extract_movies_tmdb",
        python_callable=task_extract,
    )

    t2_clean = PythonOperator(
        task_id="clean_and_deduplicate",
        python_callable=task_clean,
    )

    t3_features = PythonOperator(
        task_id="engineer_features_and_nlp",
        python_callable=task_features,
    )

    t4_mongodb = PythonOperator(
        task_id="load_catalog_mongodb",
        python_callable=task_mongodb,
    )

    t5_ml = PythonOperator(
        task_id="train_and_save_ml_models",
        python_callable=task_ml,
    )

    # '>>' specifies execution order: Extraction -> Nettoyage -> Features -> MongoDB -> ML
    t1_extract >> t2_clean >> t3_features >> t4_mongodb >> t5_ml