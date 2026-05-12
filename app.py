"""
Streamlit dashboard for football match predictions.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
from datetime import datetime, timedelta
import asyncio
import os

from src.config import setup_logging, DB_PATH
from src.database import db
from src.data_pipeline import run_pipeline
from src.feature_engineering import prepare_training_data, FeatureEngineer
from src.models import ModelTrainer, time_series_cross_validation
from src.predictor import MatchPredictor
from src.utils import setup_logging

# Setup
os.makedirs('logs', exist_ok=True)
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Football Prediction System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .main { background-color: #0d1117; }
    h1, h2, h3 { color: #58a6ff; }
    .metric-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #58a6ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("⚽ Football Match Prediction System")
st.markdown("**Advanced ML-based prediction engine with ensemble modeling**")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Predictions", "📈 Analysis", "🔧 Model Training", "📉 Performance"]
)

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to Football Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        matches_df = db.get_all_completed_matches()
        st.metric(
            "Total Matches",
            len(matches_df),
            f"{len(matches_df) // 365} years of data" if len(matches_df) > 0 else "No data"
        )
    
    with col2:
        upcoming = db.get_upcoming_matches()
        st.metric("Upcoming Matches", len(upcoming))
    
    with col3:
        leagues = matches_df["league"].unique() if len(matches_df) > 0 else []
        st.metric("Leagues Tracked", len(leagues))
    
    st.divider()
    
    st.subheader("System Overview")
    
    st.markdown("""
    ### Features
    - **Data Pipeline**: Real-time ingestion from legal APIs (Football-Data.org)
    - **Feature Engineering**: 100+ advanced features per match
    - **Ensemble Models**: XGBoost + LightGBM + Logistic Regression
    - **Predictions**: Home/Draw/Away outcomes with confidence scores
    - **Explainability**: Top feature drivers for each prediction
    
    ### Target Accuracy Range
    - **Premier League & Top Leagues**: 58-63%
    - **Overall**: 58-62%
    - **Baseline (Random)**: 33.3%
    
    ### Current Data Status
    """)
    
    # Show data summary
    if len(matches_df) > 0:
        summary_data = {
            "Metric": [
                "Total Matches",
                "Date Range",
                "Completed Matches",
                "Upcoming Matches",
            ],
            "Value": [
                len(matches_df),
                f"{matches_df['date'].min().date()} to {matches_df['date'].max().date()}",
                len(matches_df[matches_df['result'].notna()]),
                len(upcoming),
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    else:
        st.warning("❌ No data loaded. Use 'Model Training' tab to fetch data.")

# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================
elif page == "📊 Predictions":
    st.header("Match Predictions")
    
    # Check if models exist
    try:
        matches_df = db.get_all_completed_matches()
        
        if len(matches_df) < 100:
            st.warning("⚠️  Need more historical data. Train models first.")
        else:
            st.subheader("Upcoming Matches")
            
            upcoming = db.get_upcoming_matches()
            
            if len(upcoming) == 0:
                st.info("No upcoming matches in database")
            else:
                # Display predictions
                for idx, match in upcoming.head(10).iterrows():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    
                    with col1:
                        st.metric(
                            "Home Team",
                            f"Team {match['home_team_id']}",
                            f"{match['date'].strftime('%Y-%m-%d')}"
                        )
                    
                    with col2:
                        st.metric("Away Team", f"Team {match['away_team_id']}")
                    
                    with col3:
                        st.metric("Prediction", "H/D/A", "Requires Model")
                    
                    with col4:
                        st.metric("Confidence", "N/A", "Requires Model")
    
    except Exception as e:
        st.error(f"Error loading predictions: {e}")

# ============================================================================
# PAGE: ANALYSIS
# ============================================================================
elif page == "📈 Analysis":
    st.header("Statistical Analysis")
    
    matches_df = db.get_all_completed_matches()
    
    if len(matches_df) == 0:
        st.warning("No data available for analysis")
    else:
        # Filter by league
        leagues = sorted(matches_df["league"].unique())
        selected_league = st.selectbox("Select League", leagues)
        
        league_data = matches_df[matches_df["league"] == selected_league]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            home_wins = (league_data["result"] == "H").sum()
            st.metric("Home Wins", home_wins, f"{home_wins/len(league_data)*100:.1f}%" if len(league_data) > 0 else "0%")
        
        with col2:
            draws = (league_data["result"] == "D").sum()
            st.metric("Draws", draws, f"{draws/len(league_data)*100:.1f}%" if len(league_data) > 0 else "0%")
        
        with col3:
            away_wins = (league_data["result"] == "A").sum()
            st.metric("Away Wins", away_wins, f"{away_wins/len(league_data)*100:.1f}%" if len(league_data) > 0 else "0%")
        
        st.divider()
        
        # Result distribution
        results_count = league_data["result"].value_counts()
        if len(results_count) > 0:
            fig = px.pie(
                values=results_count.values,
                names=["Home Win" if n == "H" else "Draw" if n == "D" else "Away Win" for n in results_count.index],
                title=f"Match Outcomes - {selected_league}",
                color_discrete_sequence=["#58a6ff", "#79c0ff", "#a371f7"]
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: MODEL TRAINING
# ============================================================================
elif page == "🔧 Model Training":
    st.header("Model Training & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Data Pipeline")
        
        if st.button("Fetch Data from APIs"):
            with st.spinner("Fetching data... (this may take a few minutes)"):
                try:
                    asyncio.run(run_pipeline())
                    st.success("✅ Data fetched successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        matches_df = db.get_all_completed_matches()
        st.info(f"📊 Current data: {len(matches_df)} matches")
    
    with col2:
        st.subheader("2. Feature Engineering")
        
        if st.button("Prepare Training Data"):
            if len(matches_df) < 100:
                st.error("Need at least 100 historical matches")
            else:
                with st.spinner("Computing features..."):
                    try:
                        features_df = prepare_training_data(matches_df)
                        st.success(f"✅ Created {len(features_df)} feature vectors")
                        st.dataframe(features_df.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("3. Train Models")
        
        if st.button("Train Ensemble Models"):
            with st.spinner("Training models..."):
                try:
                    matches_df = db.get_all_completed_matches()
                    features_df = prepare_training_data(matches_df)
                    
                    # Train/test split
                    split_idx = int(len(features_df) * 0.8)
                    train_df = features_df.iloc[:split_idx]
                    test_df = features_df.iloc[split_idx:]
                    
                    trainer = ModelTrainer()
                    X_train, y_train = trainer._get_xy(train_df)
                    X_test, y_test = trainer._get_xy(test_df)
                    
                    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
                    
                    trainer.save_models("data/models.pkl")
                    
                    st.success("✅ Models trained and saved!")
                    
                    # Show results
                    for model_name, metrics in results.items():
                        st.metric(
                            f"{model_name.replace('_', ' ').title()} Accuracy",
                            f"{metrics['accuracy']:.4f}"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col4:
        st.subheader("4. Cross-Validation")
        
        if st.button("Run Time-Series CV"):
            with st.spinner("Running cross-validation..."):
                try:
                    matches_df = db.get_all_completed_matches()
                    features_df = prepare_training_data(matches_df)
                    mean_acc, std_acc = time_series_cross_validation(features_df, n_splits=3)
                    
                    st.success("✅ Cross-validation complete!")
                    st.metric("Mean Accuracy", f"{mean_acc:.4f}")
                    st.metric("Std Dev", f"{std_acc:.4f}")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================================================
# PAGE: PERFORMANCE
# ============================================================================
elif page == "📉 Performance":
    st.header("Model Performance & Metrics")
    
    st.subheader("Expected Accuracy Range")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Baseline (Random)",
            "33.3%",
            "Always predicting one outcome"
        )
    
    with col2:
        st.metric(
            "Target Range",
            "58-63%",
            "With feature engineering"
        )
    
    with col3:
        st.metric(
            "Advanced Ensemble",
            "60-65%",
            "With optimal hyperparameters"
        )
    
    st.divider()
    
    st.subheader("Model Components")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Logistic Regression**
        - Baseline model
        - Fast inference
        - ~56-58% accuracy
        """)
    
    with col2:
        st.markdown("""
        **XGBoost**
        - Gradient boosting
        - Feature importance
        - ~59-62% accuracy
        """)
    
    with col3:
        st.markdown("""
        **LightGBM**
        - Fast training
        - Memory efficient
        - ~59-62% accuracy
        """)
    
    st.divider()
    
    st.subheader("Ensemble Strategy")
    
    st.markdown("""
    **Soft Voting Ensemble:**
    - Average probabilities from all models
    - Confidence = max probability across classes
    - Result = argmax of averaged probabilities
    
    **Why Ensemble?**
    - Reduces overfitting
    - Captures different patterns
    - More robust predictions
    - Better calibration
    """)

if __name__ == "__main__":
    st.markdown("---")
    st.caption("Football Prediction System v1.0 | Built with Streamlit & ML")