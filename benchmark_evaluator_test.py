import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import list_trips, load_trip
from src.candidate_generator import CandidateGenerator
from src.hmm_decoder import HMMRouteReconstructor
from train_and_evaluate import calculate_route_continuity

def benchmark_evaluator_test(model_name="Random Forest", model_file="artifacts/model_random_forest.pkl"):
    print("=========================================================")
    print("  PathSnap: Benchmark on New EVALUATOR-TEST Trips (10 Trips)")
    print(f"  Model: {model_name}")
    print("=========================================================")
    os.makedirs("outputs", exist_ok=True)

    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    eval_trips = list_trips(split="evaluator_test")
    generator = CandidateGenerator(top_k=5)
    hmm = HMMRouteReconstructor(beta=8.0, sigma=6.0, alpha=0.55)

    records = []

    for tid in eval_trips:
        t_start = time.time()
        trip = load_trip(tid, split="evaluator_test")
        X_trip, y_trip, meta, miss_count, n_obs = generator.process_trip(trip, include_gt=True)
        X_scaled = scaler.transform(X_trip)

        probs = model.predict_proba(X_scaled)[:, 1]

        # Classifier Only
        cls_preds = []
        cls_correct = 0
        valid_obs = 0
        for obs_idx in range(n_obs):
            best_cand = int(np.argmax(probs[obs_idx * 5:(obs_idx + 1) * 5]))
            cand = meta[obs_idx * 5 + best_cand]
            cls_preds.append(cand["candidate_arc_id"])
            if any(meta[obs_idx * 5 + k]["is_gt"] == 1 for k in range(5)):
                valid_obs += 1
                if cand["is_gt"] == 1:
                    cls_correct += 1

        cls_cont = calculate_route_continuity(cls_preds, trip["arcs"])
        cls_acc = (cls_correct / valid_obs) if valid_obs > 0 else 0.0

        # HMM Viterbi
        viterbi_preds, viterbi_meta = hmm.reconstruct_route(trip, meta, probs, top_k=5)
        hmm_correct = sum(1 for m in viterbi_meta if m["is_gt"] == 1)
        hmm_acc = (hmm_correct / valid_obs) if valid_obs > 0 else 0.0
        hmm_cont = calculate_route_continuity(viterbi_preds, trip["arcs"])

        miss_rate = (miss_count / n_obs) * 100.0
        trip_time = time.time() - t_start

        records.append({
            "Trip ID": tid,
            "Observations": n_obs,
            "Road Arcs": len(trip["arcs"]),
            "Miss Rate": f"{miss_rate:.2f}%",
            "Classifier Top-1 Acc": round(cls_acc * 100.0, 2),
            "Classifier Continuity": round(cls_cont * 100.0, 2),
            "HMM Top-1 Acc": round(hmm_acc * 100.0, 2),
            "HMM Continuity": round(hmm_cont * 100.0, 2),
            "Continuity Gain": round((hmm_cont - cls_cont) * 100.0, 2),
            "Latency (s)": round(trip_time, 2)
        })
        print(f"  Trip {tid}: {n_obs} obs | Miss: {miss_rate:.2f}% | Cls Acc: {cls_acc*100:.1f}%, Cont: {cls_cont*100:.1f}% -> HMM Acc: {hmm_acc*100:.1f}%, Cont: {hmm_cont*100:.1f}% ({trip_time:.1f}s)")

    df = pd.DataFrame(records)

    summary = {
        "Trip ID": "AVERAGE",
        "Observations": int(df["Observations"].mean()),
        "Road Arcs": int(df["Road Arcs"].mean()),
        "Miss Rate": f"{df['Miss Rate'].str.rstrip('%').astype(float).mean():.2f}%",
        "Classifier Top-1 Acc": round(df["Classifier Top-1 Acc"].mean(), 2),
        "Classifier Continuity": round(df["Classifier Continuity"].mean(), 2),
        "HMM Top-1 Acc": round(df["HMM Top-1 Acc"].mean(), 2),
        "HMM Continuity": round(df["HMM Continuity"].mean(), 2),
        "Continuity Gain": round(df["Continuity Gain"].mean(), 2),
        "Latency (s)": round(df["Latency (s)"].mean(), 2)
    }
    df_summary = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    csv_path = "outputs/evaluator_test_results.csv"
    df_summary.to_csv(csv_path, index=False)

    md_path = "outputs/evaluator_test_results.md"
    with open(md_path, "w") as f:
        f.write("# PathSnap: Performance Benchmark on Faculty Evaluator-Test Partition\n\n")
        headers = list(df_summary.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in df_summary.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        f.write("\n")

    print("\n=========================================================")
    print("         FINAL FACULTY EVALUATOR-TEST LEADERBOARD        ")
    print("=========================================================")
    print(df_summary.to_string(index=False))
    print("=========================================================")
    print(f"Saved benchmark results to {csv_path} and {md_path}")

if __name__ == "__main__":
    benchmark_evaluator_test()
