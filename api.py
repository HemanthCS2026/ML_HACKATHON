import os
import sys
import time
import json
import pickle
import datetime
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import list_trips, load_trip
from src.candidate_generator import CandidateGenerator
from src.hmm_decoder import HMMRouteReconstructor
from train_and_evaluate import calculate_route_continuity

# Initialize FastAPI App
app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Distributed Backend for Map-Matching, Classifier Inference, and HMM/Viterbi Route Reconstruction (UE24CS352A)",
    version="2.0.0"
)

# Enable CORS for Cross-Laptop UI Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for models and scalers
MODELS = {}
SCALER = None
HMM_DECODER = HMMRouteReconstructor(beta=8.0, sigma=6.0, alpha=0.55)
CANDIDATE_GEN = CandidateGenerator(top_k=5)

MODEL_FILE_MAP = {
    "Random Forest": "model_random_forest.pkl",
    "LightGBM": "model_lightgbm.pkl",
    "Tuned LightGBM": "model_tuned_lightgbm.pkl",
    "Decision Tree": "model_decision_tree.pkl",
    "Logistic Regression": "model_logistic_regression.pkl",
    "Multi-Layer Perceptron": "model_multi_layer_perceptron.pkl"
}

def load_all_artifacts():
    global SCALER, MODELS
    artifacts_dir = "artifacts"
    
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            SCALER = pickle.load(f)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loaded feature scaler from {scaler_path}", flush=True)

    for display_name, fname in MODEL_FILE_MAP.items():
        m_path = os.path.join(artifacts_dir, fname)
        if os.path.exists(m_path):
            with open(m_path, "rb") as f:
                MODELS[display_name] = pickle.load(f)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loaded model '{display_name}' ({fname})", flush=True)

load_all_artifacts()

class PredictionRequest(BaseModel):
    trip_id: str
    split: str = "student_test"
    model_name: str = "Random Forest"
    use_hmm: bool = True

class BatchPredictionResponse(BaseModel):
    status: str
    trip_id: str
    split: str
    model_used: str
    use_hmm: bool
    num_observations: int
    candidate_miss_rate: float
    observation_top1_accuracy: Optional[float]
    route_continuity_pct: float
    inference_time_ms: float
    metrics: Optional[dict] = None
    predicted_route: Optional[List[int]] = None
    predicted_arc_ids: List[int]
    ground_truth_arc_ids: Optional[List[int]]
    gps_coordinates: List[List[float]]
    predicted_route_coordinates: List[List[List[float]]]
    ground_truth_coordinates: Optional[List[List[List[float]]]]
    timestamp: str

@app.get("/")
def root():
    return {
        "service": "PathSnap Map-Matching API",
        "status": "online",
        "architecture": "Two-Laptop Deployment (Laptop 1: Model & API, Laptop 2: Evaluator UI)",
        "loaded_models": list(MODELS.keys()),
        "has_scaler": SCALER is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/health")
def health():
    return {"status": "ok", "available_models": len(MODELS)}

@app.get("/trips")
def get_trips(split: str = Query("student_test", description="Dataset partition: student_test or train")):
    try:
        trips = list_trips(split=split)
        return {
            "split": split,
            "count": len(trips),
            "trip_ids": trips
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/models")
def get_models():
    return {
        "available_models": list(MODELS.keys()),
        "default_model": "Random Forest" if "Random Forest" in MODELS else list(MODELS.keys())[0] if MODELS else None
    }

@app.post("/predict", response_model=BatchPredictionResponse)
def predict_trip(req: PredictionRequest):
    req_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=======================================================", flush=True)
    print(f"[{req_time}] INCOMING EVALUATION REQUEST FROM LAPTOP 2", flush=True)
    print(f"  Trip ID:    {req.trip_id} ({req.split})", flush=True)
    print(f"  Model:      {req.model_name}", flush=True)
    print(f"  HMM Viterbi: {req.use_hmm}", flush=True)
    print(f"-------------------------------------------------------", flush=True)

    if SCALER is None or len(MODELS) == 0:
        load_all_artifacts()
        if SCALER is None:
            raise HTTPException(status_code=500, detail="Feature scaler not loaded. Run train_and_evaluate.py first.")

    # Select model
    model = MODELS.get(req.model_name)
    if model is None:
        # Fallback to any available model
        if len(MODELS) > 0:
            fallback_name = list(MODELS.keys())[0]
            model = MODELS[fallback_name]
            print(f"  Warning: Model '{req.model_name}' not found, falling back to '{fallback_name}'")
        else:
            raise HTTPException(status_code=400, detail=f"No models available in artifacts/.")

    # Load trip data
    try:
        trip = load_trip(req.trip_id, split=req.split)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load trip '{req.trip_id}': {e}")

    t0 = time.time()

    # Generate candidates & extract features
    X_trip, y_trip, meta, miss_count, n_obs = CANDIDATE_GEN.process_trip(trip, include_gt=True)
    X_scaled = SCALER.transform(X_trip)

    # Classifier inference
    t_inf_start = time.time()
    probs = model.predict_proba(X_scaled)[:, 1]
    raw_inf_time = time.time() - t_inf_start

    # Route reconstruction: HMM Viterbi vs Classifier Argmax
    if req.use_hmm:
        t_hmm_start = time.time()
        pred_arcs, sel_meta = HMM_DECODER.reconstruct_route(trip, meta, probs, top_k=5)
        hmm_time = time.time() - t_hmm_start
    else:
        hmm_time = 0.0
        pred_arcs = []
        sel_meta = []
        for i in range(n_obs):
            start = i * 5
            cand_probs = probs[start:start + 5]
            best_k = int(np.argmax(cand_probs))
            pred_arcs.append(meta[start + best_k]["candidate_arc_id"])
            sel_meta.append(meta[start + best_k])

    total_proc_time = (time.time() - t0) * 1000.0

    # Calculate metrics server-side (ground-truth reference never exposed to client beforehand)
    continuity = calculate_route_continuity(pred_arcs, trip["arcs"]) * 100.0
    miss_rate = (miss_count / n_obs * 100.0) if n_obs > 0 else 0.0

    top1_acc = None
    if trip["route"] is not None:
        correct_count = sum(1 for m in sel_meta if m.get("is_gt", 0) == 1)
        valid_count = sum(1 for i in range(n_obs) if np.sum(y_trip[i*5:(i+1)*5]) > 0)
        top1_acc = round((correct_count / valid_count * 100.0) if valid_count > 0 else 0.0, 2)

    # Extract coordinates for Folium visualization on Laptop 2
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    gps_coords = [[float(pt[1]), float(pt[0])] for pt in track]

    # Predicted route segments coordinates
    pred_segments_coords = []
    for aid in pred_arcs:
        u, v = arcs[aid]
        lat1, lon1 = float(nodes[u, 1]), float(nodes[u, 0])
        lat2, lon2 = float(nodes[v, 1]), float(nodes[v, 0])
        pred_segments_coords.append([[lat1, lon1], [lat2, lon2]])

    # Ground-truth route segments coordinates (if available)
    gt_segments_coords = None
    gt_arc_list = None
    if trip["route"] is not None:
        gt_arc_list = [int(a) for a in trip["route"]]
        gt_segments_coords = []
        for aid in trip["route"]:
            u, v = arcs[aid]
            lat1, lon1 = float(nodes[u, 1]), float(nodes[u, 0])
            lat2, lon2 = float(nodes[v, 1]), float(nodes[v, 0])
            gt_segments_coords.append([[lat1, lon1], [lat2, lon2]])

    # Console logging for Laptop 1
    print(f"  Processed {n_obs} GPS points in {total_proc_time:.1f} ms", flush=True)
    print(f"  [METRICS] Observation Top-1 Accuracy: {top1_acc if top1_acc is not None else 'N/A'}%", flush=True)
    print(f"  [METRICS] Route Continuity:         {continuity:.2f}%", flush=True)
    print(f"  [METRICS] Candidate Miss Rate:      {miss_rate:.2f}%", flush=True)
    print(f"  Status: 200 OK | Response dispatched to Laptop 2 UI", flush=True)
    print(f"=======================================================\n", flush=True)

    return BatchPredictionResponse(
        status="success",
        trip_id=req.trip_id,
        split=req.split,
        model_used=req.model_name,
        use_hmm=req.use_hmm,
        num_observations=n_obs,
        candidate_miss_rate=round(miss_rate, 2),
        observation_top1_accuracy=top1_acc,
        route_continuity_pct=round(continuity, 2),
        inference_time_ms=round(total_proc_time, 2),
        metrics={
            "top1_accuracy": top1_acc,
            "route_continuity": round(continuity, 2),
            "candidate_miss_rate": round(miss_rate, 2),
            "latency_ms": round(total_proc_time, 2)
        },
        predicted_route=[int(a) for a in pred_arcs],
        predicted_arc_ids=[int(a) for a in pred_arcs],
        ground_truth_arc_ids=gt_arc_list,
        gps_coordinates=gps_coords,
        predicted_route_coordinates=pred_segments_coords,
        ground_truth_coordinates=gt_segments_coords,
        timestamp=req_time
    )

if __name__ == "__main__":
    import uvicorn
    print("\nStarting PathSnap Model & API Server on Laptop 1 (Host: 0.0.0.0, Port: 8000)...")
    print("Accessible locally via http://localhost:8000 and across LAN via your Wi-Fi IP.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
