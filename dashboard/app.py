import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Movie Intelligence Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #E50914;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0A0A0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222A;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2E3440;
    }
    .recommend-card {
        background: linear-gradient(135deg, #1f2430 0%, #2b3345 100%);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        border-left: 5px solid #E50914;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED DATA & ARTIFACT LOADERS ---
@st.cache_data
def load_data():
    df_clean = pd.read_csv("data/processed/movies_cleaned.csv")
    df_clustered = pd.read_csv("data/processed/movies_clustered.csv")
    return df_clean, df_clustered

@st.cache_resource
def load_models():
    classifier = joblib.load("models/best_classifier.joblib")
    kmeans = joblib.load("models/kmeans.joblib")
    tfidf_matrix = joblib.load("models/tfidf_matrix.joblib")
    return classifier, kmeans, tfidf_matrix

df_clean, df_clustered = load_data()
classifier, kmeans, tfidf_matrix = load_models()

# --- SIDEBAR CATALOG SUMMARY ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.title("🎬 Movie Intelligence")
    st.caption("End-to-End Film Intelligence Platform")
    st.markdown("---")
    st.markdown(f"**Catalog Size:** `{len(df_clean):,}` films")
    st.markdown(f"**Date Range:** `{int(df_clean['release_date'].dropna().str[:4].astype(int).min())}` — `{int(df_clean['release_date'].dropna().str[:4].astype(int).max())}`")
    st.markdown(f"**Model In Use:** `Tuned Random Forest`")
    st.markdown(f"**Cluster Mode:** `K-Means (K=2 Archetypes)`")
    st.markdown("---")
    st.info("💡 Switch tabs at the top to explore analytics, make predictions, explore clusters, or get recommendations.")

# --- TITLE BANNER ---
st.markdown('<div class="main-header">🎬 Movie Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data Science, Predictive Machine Learning, and Semantic Recommender System</div>', unsafe_allow_html=True)

# --- 4 CORE TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Catalog Analytics & EDA",
    "🎯 Engagement Predictor",
    "🌌 Movie Archetypes (Clusters)",
    "🎬 Smart Recommender"
])

# ==============================================================================
# TAB 1: CATALOG ANALYTICS & EDA
# ==============================================================================
with tab1:
    st.subheader("High-Level Catalog Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Catalog Titles", f"{len(df_clean):,}")
    with col2:
        median_budget = df_clean[df_clean["budget"] > 0]["budget"].median()
        st.metric("Median Budget", f"${median_budget / 1e6:.1f}M")
    with col3:
        median_rev = df_clean[df_clean["revenue"] > 0]["revenue"].median()
        st.metric("Median Box Office", f"${median_rev / 1e6:.1f}M")
    with col4:
        avg_rating = df_clean["vote_average"].mean()
        st.metric("Average Rating", f"{avg_rating:.2f} / 10 ⭐")

    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("##### 💰 Budget vs. Box Office Revenue")
        # Filter non-zero for meaningful visual
        commercial_df = df_clean[(df_clean["budget"] > 0) & (df_clean["revenue"] > 0)].copy()
        commercial_df["budget_millions"] = commercial_df["budget"] / 1e6
        commercial_df["revenue_millions"] = commercial_df["revenue"] / 1e6
        
        fig_scatter = px.scatter(
            commercial_df,
            x="budget_millions",
            y="revenue_millions",
            hover_name="title",
            color="vote_average",
            color_continuous_scale="Viridis",
            labels={"budget_millions": "Budget ($M)", "revenue_millions": "Revenue ($M)", "vote_average": "Rating"},
            title="Budget vs. Revenue (Hover for Movie Name)"
        )
        # Add break-even line y = x
        max_val = max(commercial_df["budget_millions"].max(), commercial_df["revenue_millions"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Break-Even (1x ROI)"
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.markdown("##### 🎭 Top 10 Film Genres in Catalog")
        # Split and count genres
        genre_series = df_clean["genres"].dropna().apply(lambda x: [g.strip() for g in str(x).split(",") if g.strip()])
        all_genres = [g for sublist in genre_series for g in sublist]
        top_genres = pd.Series(all_genres).value_counts().head(10).reset_index()
        top_genres.columns = ["Genre", "Film Count"]
        
        fig_bar = px.bar(
            top_genres,
            x="Film Count",
            y="Genre",
            orientation="h",
            color="Film Count",
            color_continuous_scale="Reds",
            title="Most Represented Genres"
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        st.markdown("##### ⏱️ Movie Runtime Distribution")
        fig_runtime = px.histogram(
            df_clean[df_clean["runtime"] > 0],
            x="runtime",
            nbins=40,
            title="Distribution of Film Runtimes (Minutes)",
            color_discrete_sequence=["#4C72B0"]
        )
        st.plotly_chart(fig_runtime, use_container_width=True)

    with col_hist2:
        st.markdown("##### 📈 Release Year Timeline")
        df_clean["release_year"] = pd.to_datetime(df_clean["release_date"], errors="coerce").dt.year
        yearly_counts = df_clean["release_year"].dropna().value_counts().sort_index().reset_index()
        yearly_counts.columns = ["Year", "Count"]
        fig_timeline = px.line(
            yearly_counts,
            x="Year",
            y="Count",
            title="Number of Catalog Releases Over Time",
            markers=True
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


# ==============================================================================
# TAB 2: ENGAGEMENT PREDICTOR (SUPERVISED CLASSIFICATION)
# ==============================================================================
with tab2:
    st.subheader("Predict Audience Engagement for a New Film")
    st.markdown(
        "Enter metadata for an upcoming or theoretical film. Our **Tuned Random Forest Classifier** "
        "(trained with 5-fold Stratified Cross-Validation) will predict whether this film will achieve **High Audience Engagement**."
    )
    
    with st.form("prediction_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            input_title = st.text_input("Movie Title", value="Project Horizon")
            input_budget = st.number_input("Budget (USD)", min_value=0, max_value=500000000, value=75000000, step=5000000)
            input_revenue = st.number_input("Expected Box Office Revenue (USD)", min_value=0, max_value=3000000000, value=220000000, step=10000000)
            input_runtime = st.slider("Runtime (minutes)", min_value=45, max_value=240, value=118)
            input_popularity = st.slider("TMDB Popularity Score", min_value=0.0, max_value=500.0, value=45.0, step=1.0)
            
        with col_f2:
            input_year = st.number_input("Release Year", min_value=1950, max_value=2030, value=2026)
            input_month = st.selectbox("Release Month", list(range(1, 13)), index=6)
            input_num_genres = st.slider("Number of Genres", min_value=1, max_value=7, value=3)
            input_num_keywords = st.slider("Number of Keywords / Tags", min_value=0, max_value=40, value=15)
            input_overview = st.text_area(
                "Plot Overview / Synopsis",
                value="An elite group of scientists and astronauts embark on a perilous voyage beyond the edge of the known universe to stop an interstellar anomaly from collapsing Earth's gravitational field."
            )
        
        submit_btn = st.form_submit_button("🚀 Run Engagement Prediction", use_container_width=True)

    if submit_btn:
        # Compute runtime category
        if input_runtime < 90:
            runtime_cat = "short"
        elif input_runtime <= 120:
            runtime_cat = "standard"
        else:
            runtime_cat = "long"
            
        has_budget = 1 if input_budget > 0 else 0

        # Construct single-row DataFrame matching the ColumnTransformer schema
        input_data = pd.DataFrame([{
            "budget": float(input_budget),
            "revenue": float(input_revenue),
            "runtime": float(input_runtime),
            "popularity": float(input_popularity),
            "release_year": int(input_year),
            "release_month": int(input_month),
            "num_genres": int(input_num_genres),
            "num_keywords": int(input_num_keywords),
            "has_budget": int(has_budget),
            "runtime_category": runtime_cat,
            "overview": input_overview
        }])

        prediction = classifier.predict(input_data)[0]
        probabilities = classifier.predict_proba(input_data)[0]
        high_prob = probabilities[1] * 100

        st.markdown("### Prediction Result")
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            if prediction == 1:
                st.success(f"### 🎉 HIGH ENGAGEMENT PREDICTED!\n\n**'{input_title}'** is projected to surpass the catalog median engagement threshold.")
            else:
                st.warning(f"### ⚠️ MODERATE / LOW ENGAGEMENT\n\n**'{input_title}'** is projected to achieve average or niche catalog engagement.")
            
            # Additional contextual breakdown
            st.markdown(f"""
            - **Predicted Label:** `{'High (1)' if prediction == 1 else 'Low/Average (0)'}`
            - **Engagement Probability:** `{high_prob:.1f}%`
            - **Decision Threshold:** `50.0%`
            """)

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=high_prob,
                number={'suffix': "%", 'font': {'size': 36}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>High Engagement Probability</b>", 'font': {'size': 17, 'color': '#E0E0E0'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#888"},
                    'bar': {'color': "#E50914" if prediction == 1 else "#F59E0B", 'thickness': 0.75},
                    'bgcolor': "#1F232B",
                    'steps': [
                        {'range': [0, 50], 'color': "#2A2E39"},
                        {'range': [50, 100], 'color': "#343A46"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.8,
                        'value': 50.0
                    }
                }
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=30, r=30, t=65, b=25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# ==============================================================================
# TAB 3: MOVIE ARCHETYPES (UNSUPERVISED CLUSTERING)
# ==============================================================================
with tab3:
    st.subheader("Unsupervised Discovery of Film Archetypes")
    st.markdown(
        "Using **K-Means Clustering** evaluated via Silhouette Score, our catalog has been segmented into distinct industrial archetypes. "
        "The 7-dimensional feature space was compressed into 2D using **TruncatedSVD** for interactive exploration."
    )
    
    col_map, col_profile = st.columns([2, 1])
    
    with col_map:
        cluster_names = {
            0: "Cluster 0: Mega-Budget Commercial Blockbusters",
            1: "Cluster 1: Mainstream, Mid-Budget & Indie Productions"
        }
        df_clustered_vis = df_clustered.copy()
        df_clustered_vis["Archetype"] = df_clustered_vis["cluster"].map(cluster_names)
        
        fig_clusters = px.scatter(
            df_clustered_vis,
            x="svd_x",
            y="svd_y",
            color="Archetype",
            hover_name="title",
            hover_data={
                "svd_x": False,
                "svd_y": False,
                "budget": ":$,.0f",
                "revenue": ":$,.0f",
                "popularity": ":.1f"
            },
            color_discrete_map={
                "Cluster 0: Mega-Budget Commercial Blockbusters": "#E50914",
                "Cluster 1: Mainstream, Mid-Budget & Indie Productions": "#00D2D3"
            },
            title="2D SVD Semantic Projection of Movie Clusters"
        )
        fig_clusters.update_traces(marker=dict(size=8, opacity=0.8))
        fig_clusters.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_clusters, use_container_width=True)

    with col_profile:
        st.markdown("##### 📋 Archetype Comparison")
        numeric_cols = ["budget", "revenue", "runtime", "popularity", "release_year", "num_genres", "num_keywords"]
        profile_df = df_clustered.groupby("cluster")[numeric_cols].mean().round(1)
        profile_df["budget"] = profile_df["budget"].apply(lambda x: f"${x/1e6:.1f}M")
        profile_df["revenue"] = profile_df["revenue"].apply(lambda x: f"${x/1e6:.1f}M")
        profile_df.index = ["Cluster 0 (Blockbusters)", "Cluster 1 (Mid/Indie)"]
        st.dataframe(profile_df.T, use_container_width=True)
        
        st.caption("Notice how Cluster 0 represents movies with over 4x larger budgets and nearly 5x greater box office return.")


# ==============================================================================
# TAB 4: SMART RECOMMENDER SYSTEM
# ==============================================================================
with tab4:
    st.subheader("Semantic Content-Based Movie Recommender")
    st.markdown(
        "Choose any movie from the TMDB catalog. The engine computes **Cosine Similarity** "
        "over our pre-trained **TF-IDF synopsis vectors** to discover films with similar thematic DNA."
    )
    
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        movie_titles = sorted(df_clean["title"].dropna().unique().tolist())
        selected_title = st.selectbox("Select a film from catalog:", movie_titles, index=movie_titles.index("Inception") if "Inception" in movie_titles else 0)
    with col_sel2:
        top_k = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5)

    if selected_title:
        # Pull original movie synopsis
        selected_row = df_clean[df_clean["title"] == selected_title].iloc[0]
        
        with st.expander(f"📖 Selected Film Overview: **{selected_title}**", expanded=True):
            st.write(f"**Genres:** {selected_row.get('genres', 'N/A')} | **Release Date:** {selected_row.get('release_date', 'N/A')}")
            st.info(selected_row.get("overview", "No synopsis available."))

        # Import recommend_movies directly from src.recommender
        from src.recommender import recommend_movies
        recs = recommend_movies(selected_title, df_clean, tfidf_matrix, top_n=top_k)

        st.markdown(f"### 🎯 Top {top_k} Recommended Titles")
        
        if recs is not None and not recs.empty:
            for idx, r in recs.iterrows():
                similarity_pct = float(r["similarity_score"]) * 100
                st.markdown(f"""
                <div class="recommend-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #FFFFFF;">🎬 {r['title']}</h4>
                        <span style="background-color: #E50914; color: white; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 0.9rem;">
                            Match: {similarity_pct:.1f}%
                        </span>
                    </div>
                    <p style="margin: 6px 0 0 0; color: #A0A0A0; font-size: 0.9rem;">
                        <strong>Genres:</strong> {r.get('genres', 'N/A')} &nbsp;|&nbsp; <strong>Release:</strong> {r.get('release_date', 'N/A')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found.")
