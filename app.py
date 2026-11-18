import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import list_trips, load_trip
from src.candidate_generator import CandidateGenerator
from eda.visualize_map import create_folium_map
from train_and_evaluate import calculate_route_continuity

st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching",
    page_icon="🗺️",
    layout="wide"
)

# Header
st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.caption("PES University | Department of CSE | Course UE24CS352A Machine Learning")

# Sidebar
st.sidebar.header("⚙️ Evaluation Controls")
split_option = st.sidebar.selectbox("Dataset Partition", ["student_test", "train"], index=0)
available_trips = list_trips(split=split_option)
selected_trip_id = st.sidebar.selectbox("Select Trip ID", available_trips, index=0)

model_options = {
    "Random Forest (Ensemble Bagging)": "model_random_forest.pkl",
    "LightGBM (Gradient Boosting)": "model_lightgbm.pkl",
    "Multi-Layer Perceptron (Neural Net)": "model_multi_layer_perceptron.pkl",
    "Decision Tree (Tree-Based)": "model_decision_tree.pkl",
    "Logistic Regression (Linear)": "model_logistic_regression.pkl"
}
selected_model_name = st.sidebar.selectbox("Prediction Model", list(model_options.keys()), index=0)

@st.cache_resource
def load_model_and_scaler(model_file):
    scaler_path = "artifacts/scaler.pkl"
    model_path = os.path.join("artifacts", model_file)
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        return None, None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_model_and_scaler(model_options[selected_model_name])

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Live Route Reconstruction", "📊 5-Model Benchmark Table", "🔍 Dataset EDA & Quality"])

with tab1:
    st.subheader(f"Trip `{selected_trip_id}` Evaluation ({split_option})")
    
    # Load Trip
    trip = load_trip(selected_trip_id, split=split_option)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GPS Observations", len(trip["track"]))
    col2.metric("Road Nodes", len(trip["nodes"]))
    col3.metric("Road Arcs", len(trip["arcs"]))
    col4.metric("Reference Route Arcs", len(trip["route"]) if trip["route"] is not None else "N/A")

    if st.button("▶️ Run Map-Matching Prediction", type="primary"):
        if model is None or scaler is None:
            st.error("Model artifacts not found! Please run `python train_and_evaluate.py` first.")
        else:
            with st.spinner("Processing GPS points, querying candidates, and running inference..."):
                gen = CandidateGenerator(top_k=5)
                X_trip, y_trip, meta, miss_count, n_obs = gen.process_trip(trip, include_gt=(trip["route"] is not None))
                X_scaled = scaler.transform(X_trip)
                probs = model.predict_proba(X_scaled)[:, 1]

                # Observation Level Predictions
                pred_arcs = []
                correct_obs = 0
                valid_obs = 0
                for obs_idx in range(n_obs):
                    start = obs_idx * 5
                    end = start + 5
                    obs_probs = probs[start:end]
                    best_cand_idx = int(np.argmax(obs_probs))
                    pred_arc = meta[start + best_cand_idx]["candidate_arc_id"]
                    pred_arcs.append(pred_arc)

                    if trip["route"] is not None:
                        obs_true = y_trip[start:end]
                        if np.sum(obs_true) > 0:
                            valid_obs += 1
                            if obs_true[best_cand_idx] == 1:
                                correct_obs += 1

                # Metrics
                top1_acc = (correct_obs / valid_obs * 100.0) if valid_obs > 0 else 0.0
                miss_rate = (miss_count / n_obs * 100.0)
                continuity = calculate_route_continuity(pred_arcs, trip["arcs"]) * 100.0

                m1, m2, m3 = st.columns(3)
                m1.metric("Top-1 Observation Accuracy", f"{top1_acc:.2f}%")
                m2.metric("Candidate Miss Rate", f"{miss_rate:.2f}%")
                m3.metric("Route Continuity", f"{continuity:.1f}%")

                # Generate Map
                map_path = f"maps/live_trip_{selected_trip_id}.html"
                create_folium_map(trip, predicted_route=pred_arcs, output_html=map_path)

                with open(map_path, "r", encoding="utf-8") as f:
                    map_html = f.read()
                st.components.v1.html(map_html, height=550)
                st.success("✅ Map matching completed successfully!")
    else:
        # Show default baseline map
        base_map_path = f"maps/trip_{selected_trip_id}_preview.html"
        create_folium_map(trip, output_html=base_map_path)
        with open(base_map_path, "r", encoding="utf-8") as f:
            base_html = f.read()
        st.components.v1.html(base_html, height=550)
        st.info("Click **'Run Map-Matching Prediction'** above to compute predictions and view the predicted route on map.")

with tab2:
    st.subheader("Official 5-Model Benchmark Leaderboard")
    if os.path.exists("outputs/model_comparison.csv"):
        df_comp = pd.read_csv("outputs/model_comparison.csv")
        st.dataframe(df_comp, use_container_width=True)

        st.subheader("Benchmark Comparison Visualizations")
        if os.path.exists("outputs/model_comparison_plots.png"):
            st.image("outputs/model_comparison_plots.png", use_column_width=True)
    else:
        st.warning("Benchmark results not yet generated.")

with tab3:
    st.subheader("Dataset Exploratory Data Analysis & Quality Assessment")
    if os.path.exists("outputs/dataset_inventory.csv"):
        df_inv = pd.read_csv("outputs/dataset_inventory.csv")
        st.write(f"Total Trips Indexed: {len(df_inv)} (80 Train, 10 Student-Test)")
        st.dataframe(df_inv.describe(), use_container_width=True)

    if os.path.exists("outputs/eda_analysis_plots.png"):
        st.image("outputs/eda_analysis_plots.png", use_column_width=True)
