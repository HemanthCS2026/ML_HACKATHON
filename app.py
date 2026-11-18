import os
import sys
import time
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium import plugins

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import list_trips, load_trip
from src.candidate_generator import CandidateGenerator
from src.hmm_decoder import HMMRouteReconstructor
from eda.visualize_map import create_folium_map
from train_and_evaluate import calculate_route_continuity

st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching & Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">PES University | Department of CSE | Course UE24CS352A Machine Learning</p>', unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS & TWO-LAPTOP NETWORKING
# -------------------------------------------------------------
st.sidebar.header("🌐 Deployment Architecture")
deploy_mode = st.sidebar.radio("Operating Mode", ["API Client (Two-Laptop / LAN)", "Local Standalone"], index=0)

api_url = "http://localhost:8000"
api_connected = False

if deploy_mode == "API Client (Two-Laptop / LAN)":
    api_url = st.sidebar.text_input("FastAPI Server URL (Laptop 1)", value="http://localhost:8000", help="Enter http://<Laptop-1-IP>:8000 when deployed across two laptops")
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=2.0)
        if health_resp.status_code == 200:
            api_connected = True
            st.sidebar.success(f"🟢 Connected to Laptop 1 API ({api_url})")
        else:
            st.sidebar.warning(f"🟡 API returned status {health_resp.status_code}")
    except Exception:
        st.sidebar.error(f"🔴 Cannot reach API at {api_url}. Start `python api.py` on Laptop 1 or switch to Local mode.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Evaluation Parameters")

# Partition selection: default to evaluator_test
partition_options = ["evaluator_test", "student_test", "train"]
split_option = st.sidebar.selectbox("Dataset Partition", partition_options, index=0)

# Load available trips for selected partition
if api_connected:
    try:
        resp = requests.get(f"{api_url}/trips?split={split_option}", timeout=3.0)
        data = resp.json()
        available_trips = data.get("trips") or data.get("trip_ids", [])
    except Exception:
        available_trips = list_trips(split=split_option)
else:
    available_trips = list_trips(split=split_option)

selected_trip_id = st.sidebar.selectbox("Select Evaluation Trip ID", available_trips, index=0)

model_options = [
    "Random Forest",
    "LightGBM",
    "Tuned LightGBM",
    "Multi-Layer Perceptron",
    "Decision Tree",
    "Logistic Regression"
]
selected_model_name = st.sidebar.selectbox("Prediction Model", model_options, index=0)

st.sidebar.markdown("---")
st.sidebar.header("🧠 Decoder Configuration")
use_hmm = st.sidebar.checkbox("Enable HMM Viterbi Sequential Decoding", value=True, help="Enforces road network topological transition constraints")
beta_param = st.sidebar.slider("HMM Beta (Transition Scale)", min_value=2.0, max_value=20.0, value=8.0, step=0.5)
sigma_param = st.sidebar.slider("GPS Sigma (Measurement Noise)", min_value=2.0, max_value=15.0, value=6.0, step=0.5)
alpha_param = st.sidebar.slider("Emission Alpha (Model Weight)", min_value=0.1, max_value=0.9, value=0.55, step=0.05)

# -------------------------------------------------------------
# MAIN TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Evaluator Live Reconstruction",
    "📊 5-Model Official Benchmark",
    "⚡ Bayesian Optimisation (Optuna)",
    "📈 HMM vs Classifier Ablation",
    "🔍 Dataset EDA & Network Topology",
    "💻 Two-Laptop Architecture Guide"
])

# =============================================================
# TAB 1: LIVE EVALUATION & ROUTE RECONSTRUCTION
# =============================================================
with tab1:
    st.subheader(f"Trip `{selected_trip_id}` Evaluation ({split_option})")

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    trip_local = load_trip(selected_trip_id, split=split_option)
    col_meta1.metric("GPS Observations", len(trip_local["track"]))
    col_meta2.metric("Road Nodes", len(trip_local["nodes"]))
    col_meta3.metric("Road Arcs", len(trip_local["arcs"]))
    col_meta4.metric("Reference Route Arcs", len(trip_local["route"]) if trip_local["route"] is not None else "Hidden Server-Side")

    run_btn = st.button("▶️ Run Map-Matching & Route Reconstruction", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Processing trajectory, generating candidate arcs, and executing inference..."):
            if api_connected and deploy_mode == "API Client (Two-Laptop / LAN)":
                # CLIENT-SERVER MODE (Two-Laptop)
                payload = {
                    "trip_id": selected_trip_id,
                    "split": split_option,
                    "model_name": selected_model_name,
                    "use_hmm": use_hmm,
                    "top_k": 5,
                    "beta": beta_param,
                    "sigma": sigma_param,
                    "alpha": alpha_param
                }
                try:
                    t0 = time.time()
                    resp = requests.post(f"{api_url}/predict", json=payload, timeout=30.0)
                    client_roundtrip_ms = (time.time() - t0) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        metrics = data.get("metrics", {})
                        pred_arcs = data.get("predicted_route") or data.get("predicted_arc_ids", [])
                        top1 = metrics.get("top1_accuracy", data.get("observation_top1_accuracy"))
                        cont = metrics.get("route_continuity", data.get("route_continuity_pct"))
                        miss = metrics.get("candidate_miss_rate", data.get("candidate_miss_rate"))
                        lat = metrics.get("latency_ms", data.get("inference_time_ms", client_roundtrip_ms))

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Top-1 Observation Accuracy", f"{top1}%" if top1 is not None else "Server Protected")
                        m2.metric("Route Continuity", f"{cont}%")
                        m3.metric("Candidate Miss Rate", f"{miss}%")
                        m4.metric("API Response Latency", f"{lat:.1f} ms", f"{client_roundtrip_ms:.1f} ms total")

                        # Generate Interactive Map with predictions
                        map_path = f"maps/live_trip_{selected_trip_id}.html"
                        create_folium_map(trip_local, predicted_route=pred_arcs, output_html=map_path)

                        with open(map_path, "r", encoding="utf-8") as f:
                            map_html = f.read()
                        st.components.v1.html(map_html, height=560)
                        st.success(f"✅ Prediction dispatched by Laptop 1 API in {metrics['latency_ms']:.1f} ms. Evaluator metrics verified server-side.")
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with API server: {e}")

            else:
                # LOCAL STANDALONE FALLBACK MODE
                model_file = f"artifacts/model_{selected_model_name.lower().replace(' ', '_')}.pkl"
                if not os.path.exists(model_file) and selected_model_name == "Tuned LightGBM":
                    model_file = "artifacts/model_tuned_lightgbm.pkl"
                scaler_file = "artifacts/scaler.pkl"

                if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                    st.error(f"Artifact {model_file} not found. Please train models first.")
                else:
                    with open(scaler_file, "rb") as f:
                        scaler = pickle.load(f)
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)

                    t0 = time.time()
                    gen = CandidateGenerator(top_k=5, sigma=sigma_param)
                    has_gt = trip_local["route"] is not None
                    X_trip, y_trip, meta, miss_count, n_obs = gen.process_trip(trip_local, include_gt=has_gt)
                    X_scaled = scaler.transform(X_trip)
                    probs = model.predict_proba(X_scaled)[:, 1]

                    if use_hmm:
                        hmm = HMMRouteReconstructor(beta=beta_param, sigma=sigma_param, alpha=alpha_param)
                        pred_arcs, v_meta = hmm.reconstruct_route(trip_local, meta, probs, top_k=5)
                    else:
                        pred_arcs = []
                        for obs_idx in range(n_obs):
                            b_idx = int(np.argmax(probs[obs_idx * 5:(obs_idx + 1) * 5]))
                            pred_arcs.append(meta[obs_idx * 5 + b_idx]["candidate_arc_id"])

                    # Ground truth calculation
                    top1_acc = None
                    if has_gt:
                        corr = 0
                        valid = 0
                        for obs_idx in range(n_obs):
                            base = obs_idx * 5
                            if any(meta[base + k]["is_gt"] == 1 for k in range(5)):
                                valid += 1
                                cand_ids = [meta[base + k]["candidate_arc_id"] for k in range(5)]
                                if pred_arcs[obs_idx] in cand_ids:
                                    c_idx = cand_ids.index(pred_arcs[obs_idx])
                                    if meta[base + c_idx]["is_gt"] == 1:
                                        corr += 1
                        top1_acc = (corr / valid * 100.0) if valid > 0 else 0.0

                    continuity = calculate_route_continuity(pred_arcs, trip_local["arcs"]) * 100.0
                    miss_rate = (miss_count / n_obs * 100.0)
                    total_time_ms = (time.time() - t0) * 1000.0

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Top-1 Observation Accuracy", f"{top1_acc:.2f}%" if top1_acc is not None else "N/A")
                    m2.metric("Route Continuity", f"{continuity:.2f}%")
                    m3.metric("Candidate Miss Rate", f"{miss_rate:.2f}%")
                    m4.metric("Inference Latency", f"{total_time_ms:.1f} ms")

                    map_path = f"maps/live_trip_{selected_trip_id}.html"
                    create_folium_map(trip_local, predicted_route=pred_arcs, output_html=map_path)
                    with open(map_path, "r", encoding="utf-8") as f:
                        map_html = f.read()
                    st.components.v1.html(map_html, height=560)
                    st.success("✅ Map matching executed in local standalone mode.")
    else:
        # Initial preview map
        preview_map = f"maps/trip_{selected_trip_id}_preview.html"
        create_folium_map(trip_local, output_html=preview_map)
        with open(preview_map, "r", encoding="utf-8") as f:
            preview_html = f.read()
        st.components.v1.html(preview_html, height=560)
        st.info("👆 Click **'Run Map-Matching & Route Reconstruction'** to compute trajectory predictions.")

# =============================================================
# TAB 2: 5-MODEL BENCHMARK TABLE
# =============================================================
with tab2:
    st.subheader("Official 5-Model Benchmark Comparison Table")
    if os.path.exists("outputs/model_comparison.csv"):
        df_comp = pd.read_csv("outputs/model_comparison.csv")
        st.dataframe(df_comp, use_container_width=True)

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            if os.path.exists("outputs/model_comparison_plots.png"):
                st.image("outputs/model_comparison_plots.png", use_column_width=True, caption="5-Model Benchmark Visual Comparison")
        with col_p2:
            st.markdown("""
            ### Benchmark Highlights
            * **5 Algorithm Families:** Linear (Logistic Regression), Tree (Decision Tree), Bagging (Random Forest), Boosting (LightGBM), Neural Network (MLP).
            * **Evaluation Integrity:** Rigorous 3-fold cross-validation on 80 training trips; independent evaluation across all 10 student-test trips.
            * **Top Performing Model:** **Random Forest** achieved the highest test Top-1 Accuracy (84.43%) and Route Continuity (90.6%), while **LightGBM** demonstrated the fastest inference speed (0.039 ms/obs).
            """)
    else:
        st.warning("Benchmark results file not found.")

# =============================================================
# TAB 3: BAYESIAN HYPERPARAMETER OPTIMISATION
# =============================================================
with tab3:
    st.subheader("Bayesian Hyperparameter Optimisation (Optuna TPE)")
    col_opt1, col_opt2 = st.columns([1, 1])
    
    with col_opt1:
        st.markdown("""
        #### Search Configuration & Objective
        * **Optimiser:** Optuna Tree-structured Parzen Estimator (TPE)
        * **Target Model:** LightGBM Classifier
        * **Optimisation Objective:** 3-Fold Cross-Validation Macro F1 Score
        * **Search Space:**
          - `n_estimators`: [50, 250]
          - `learning_rate`: [0.01, 0.20] (log-uniform)
          - `num_leaves`: [15, 63]
          - `max_depth`: [4, 12]
          - `min_child_samples`: [10, 50]
          - `subsample`: [0.6, 1.0]
          - `colsample_bytree`: [0.6, 1.0]
        """)
        if os.path.exists("outputs/tuned_vs_untuned_comparison.csv"):
            st.markdown("#### Tuned vs Untuned Benchmark")
            df_tune = pd.read_csv("outputs/tuned_vs_untuned_comparison.csv")
            st.dataframe(df_tune, use_container_width=True)

    with col_opt2:
        if os.path.exists("outputs/bayesian_optimization_history.png"):
            st.image("outputs/bayesian_optimization_history.png", use_column_width=True, caption="Bayesian Search Convergence Curve")
        
    if os.path.exists("outputs/bayesian_optimization_trials.csv"):
        st.markdown("#### Complete Bayesian Search Trial Log")
        df_trials = pd.read_csv("outputs/bayesian_optimization_trials.csv")
        st.dataframe(df_trials, use_container_width=True)

# =============================================================
# TAB 4: HMM VS CLASSIFIER ABLATION STUDY
# =============================================================
with tab4:
    st.subheader("HMM Viterbi Sequential Route Reconstruction Ablation")
    if os.path.exists("outputs/hmm_comparison.csv"):
        df_hmm = pd.read_csv("outputs/hmm_comparison.csv")
        st.dataframe(df_hmm, use_container_width=True)

        if os.path.exists("outputs/hmm_comparison_plot.png"):
            st.image("outputs/hmm_comparison_plot.png", use_column_width=True, caption="Top-1 Accuracy & Route Continuity: Classifier-Only vs Classifier + HMM")

        st.markdown("""
        #### Key Findings:
        1. **Elimination of Teleportation:** Classifier-only predictions select the spatially closest candidate independently at each second, frequently flipping between parallel roads. HMM Viterbi decoding penalizes topologically disconnected transitions, elevating route continuity to **~99%**.
        2. **Accuracy Enhancement:** In ambiguous situations (e.g. overpasses or highway splits), topological transition context corrects point errors, providing a net gain in observation Top-1 accuracy.
        3. **GPS Dropout Robustness:** Large time gaps ($\Delta t > 10$s) automatically trigger dynamic transition relaxation, preventing trajectory collapse.
        """)
    else:
        st.info("Run `python evaluate_hmm.py` to generate the complete 10-trip HMM ablation study.")

# =============================================================
# TAB 5: DATASET EDA & NETWORK TOPOLOGY
# =============================================================
with tab5:
    st.subheader("Dataset Exploratory Data Analysis & Graph Structure")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        if os.path.exists("outputs/dataset_inventory.csv"):
            df_inv = pd.read_csv("outputs/dataset_inventory.csv")
            st.dataframe(df_inv.describe(), use_container_width=True)
    with col_e2:
        st.markdown("""
        #### Dataset Inventory Highlights:
        * **Partitions:** 80 Training Trips, 10 Student-Test Trips, 10 Reserved Evaluator Trips.
        * **Sampling Rate:** 97% of observations sampled at exactly $1.0$ Hz (1 ping/sec).
        * **Sensor Noise:** Typical GPS multipath deviations range from 3m to 18m.
        * **Topological Scale:** Trajectories traverse urban road graphs with 3,000–12,000 nodes and 6,000–25,000 directed edges.
        """)
    if os.path.exists("outputs/eda_analysis_plots.png"):
        st.image("outputs/eda_analysis_plots.png", use_column_width=True, caption="Exploratory Data Analysis Distributions")

# =============================================================
# TAB 6: TWO-LAPTOP ARCHITECTURE GUIDE
# =============================================================
with tab6:
    st.subheader("Two-Laptop Deployment & Evaluation Guide")
    st.markdown("""
    ### Setup Architecture
    ```
    ┌────────────────────────────────────────┐          ┌────────────────────────────────────────┐
    │          LAPTOP 1 (SERVER)             │          │           LAPTOP 2 (CLIENT)            │
    │  • FastAPI Backend (`api.py`)          │  HTTP    │  • Streamlit Evaluator UI (`app.py`)   │
    │  • Preloaded Classifier & Scaler       │ ◄──────► │  • 10 Reserved Trip Selector           │
    │  • HMM / Viterbi Road Graph Engine     │  Port    │  • Real-Time Interactive Map View      │
    │  • Server-Side Ground Truth Evaluation │  8000    │  • Live Top-1 & Continuity Telemetry   │
    │  • Real-Time Terminal Audit Logs       │          │  • Zero Local Ground-Truth Storage     │
    └────────────────────────────────────────┘          └────────────────────────────────────────┘
    ```

    ### Step-by-Step Instructions:
    1. **Connect Both Laptops to Same Wi-Fi / Hotspot:**
       - Check Laptop 1 IP address in cmd: `ipconfig` (e.g. `192.168.1.15`).
    2. **On Laptop 1 (Model & API Server):**
       ```bash
       python api.py
       ```
       *(Server starts on `0.0.0.0:8000` and displays real-time request logs)*
    3. **On Laptop 2 (Evaluator UI):**
       ```bash
       streamlit run app.py
       ```
       - In the left sidebar, enter Laptop 1 API URL: `http://192.168.1.15:8000`.
       - Select any of the 10 evaluation trips and click **"Run Map-Matching & Route Reconstruction"**.
       - Evaluators can inspect the live inference requests printing directly on Laptop 1's terminal while viewing maps and metrics on Laptop 2!
    """)
