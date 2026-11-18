import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import list_trips, load_trip
from src.candidate_generator import CandidateGenerator
from src.hmm_decoder import HMMRouteReconstructor
from train_and_evaluate import calculate_route_continuity

def evaluate_hmm_benchmark():
    print("==================================================================", flush=True)
    print("  PathSnap: Full HMM + Viterbi vs Classifier-Only Benchmark      ", flush=True)
    print("==================================================================", flush=True)

    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # We evaluate using the top performing models
    model_files = {
        "Random Forest": "artifacts/model_random_forest.pkl",
        "LightGBM": "artifacts/model_lightgbm.pkl",
        "Decision Tree": "artifacts/model_decision_tree.pkl",
        "Logistic Regression": "artifacts/model_logistic_regression.pkl",
        "MLP": "artifacts/model_multi_layer_perceptron.pkl"
    }

    # If tuned LightGBM exists, add it too
    if os.path.exists("artifacts/model_tuned_lightgbm.pkl"):
        model_files["Tuned LightGBM"] = "artifacts/model_tuned_lightgbm.pkl"

    test_trips = list_trips(split="student_test")
    gen = CandidateGenerator(top_k=5)
    hmm = HMMRouteReconstructor(beta=8.0, sigma=6.0, alpha=0.55)

    print(f"Pre-extracting features for {len(test_trips)} student-test trips...", flush=True)
    trips_data = []
    for tid in test_trips:
        trip = load_trip(tid, split="student_test")
        X, y, meta, miss, n_obs = gen.process_trip(trip, include_gt=True)
        X_scaled = scaler.transform(X)
        trips_data.append({
            "trip_id": tid,
            "trip": trip,
            "X": X_scaled,
            "y": y,
            "meta": meta,
            "n_obs": n_obs,
            "miss": miss
        })

    trip_results = []
    summary_results = []

    for m_name, m_path in model_files.items():
        if not os.path.exists(m_path):
            continue
        with open(m_path, "rb") as f:
            model = pickle.load(f)

        print(f"\nEvaluating {m_name} across all 10 student-test trips...", flush=True)

        class_correct_total = 0
        hmm_correct_total = 0
        valid_obs_total = 0

        class_continuity_list = []
        hmm_continuity_list = []
        class_time_total = 0.0
        hmm_time_total = 0.0

        for td in trips_data:
            t0 = time.time()
            probs = model.predict_proba(td["X"])[:, 1]
            t_class = time.time() - t0
            class_time_total += t_class

            # 1. Classifier-Only Prediction (Greedy Argmax per observation)
            class_arcs = []
            class_correct = 0
            valid_obs = 0
            for i in range(td["n_obs"]):
                start = i * 5
                cand_probs = probs[start:start+5]
                best_k = int(np.argmax(cand_probs))
                class_arcs.append(td["meta"][start + best_k]["candidate_arc_id"])
                
                if np.sum(td["y"][start:start+5]) > 0:
                    valid_obs += 1
                    if td["y"][start + best_k] == 1:
                        class_correct += 1

            cont_class = calculate_route_continuity(class_arcs, td["trip"]["arcs"])
            class_continuity_list.append(cont_class)
            class_correct_total += class_correct
            valid_obs_total += valid_obs

            # 2. Classifier + HMM Viterbi Decoding
            t0 = time.time()
            hmm_arcs, hmm_meta = hmm.reconstruct_route(td["trip"], td["meta"], probs, top_k=5)
            t_hmm = time.time() - t0
            hmm_time_total += t_hmm

            hmm_correct = sum(1 for m in hmm_meta if m.get("is_gt", 0) == 1)
            cont_hmm = calculate_route_continuity(hmm_arcs, td["trip"]["arcs"])
            hmm_continuity_list.append(cont_hmm)
            hmm_correct_total += hmm_correct

            trip_results.append({
                "Model": m_name,
                "Trip ID": td["trip_id"],
                "GPS Points": td["n_obs"],
                "Classifier Top-1 Acc": round(class_correct / valid_obs if valid_obs > 0 else 0, 4),
                "HMM Top-1 Acc": round(hmm_correct / valid_obs if valid_obs > 0 else 0, 4),
                "Classifier Continuity": f"{cont_class*100:.1f}%",
                "HMM Continuity": f"{cont_hmm*100:.1f}%",
                "Continuity Improvement": f"{(cont_hmm - cont_class)*100:+.1f}%"
            })

        overall_class_acc = class_correct_total / valid_obs_total if valid_obs_total > 0 else 0
        overall_hmm_acc = hmm_correct_total / valid_obs_total if valid_obs_total > 0 else 0
        overall_class_cont = float(np.mean(class_continuity_list))
        overall_hmm_cont = float(np.mean(hmm_continuity_list))
        
        summary_results.append({
            "Model": m_name,
            "Classifier-Only Top-1 Acc": round(overall_class_acc, 4),
            "Classifier + HMM Top-1 Acc": round(overall_hmm_acc, 4),
            "Classifier Route Continuity": f"{overall_class_cont*100:.1f}%",
            "Classifier + HMM Route Continuity": f"{overall_hmm_cont*100:.1f}%",
            "Continuity Gain": f"{(overall_hmm_cont - overall_class_cont)*100:+.1f}%",
            "Avg HMM Latency (ms/trip)": round(hmm_time_total / len(trips_data) * 1000.0, 1)
        })

    # Save summary table
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv("outputs/hmm_comparison.csv", index=False)

    df_trips = pd.DataFrame(trip_results)
    df_trips.to_csv("outputs/hmm_per_trip_results.csv", index=False)

    md_path = "outputs/hmm_comparison.md"
    with open(md_path, "w") as f:
        f.write("# HMM & Viterbi Route Reconstruction Comparison\n\n")
        f.write("### Overall Model Summary (Classifier-Only vs Classifier + HMM Viterbi):\n\n")
        headers = list(df_summary.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in df_summary.iterrows():
            f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")
        f.write("\n\n### Per-Trip Evaluation on Student-Testing Dataset (Random Forest):\n\n")
        df_rf = df_trips[df_trips["Model"] == "Random Forest"]
        rf_headers = list(df_rf.columns)
        f.write("| " + " | ".join(rf_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(rf_headers)) + " |\n")
        for _, row in df_rf.iterrows():
            f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")

    print("\n==================================================================", flush=True)
    print("      SUMMARY: CLASSIFIER-ONLY vs CLASSIFIER + HMM VITERBI        ", flush=True)
    print("==================================================================", flush=True)
    print(df_summary.to_string(index=False), flush=True)
    print("==================================================================", flush=True)

    # Plot Comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df_summary))
    width = 0.35

    class_cont_vals = [float(v.replace("%", "")) for v in df_summary["Classifier Route Continuity"]]
    hmm_cont_vals = [float(v.replace("%", "")) for v in df_summary["Classifier + HMM Route Continuity"]]

    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, class_cont_vals, width, label="Classifier-Only", color="#94A3B8")
    plt.bar(x + width/2, hmm_cont_vals, width, label="Classifier + HMM Viterbi", color="#10B981")
    plt.xticks(x, df_summary["Model"], rotation=25, ha="right")
    plt.ylabel("Route Continuity (%)", fontweight="bold")
    plt.title("Route Continuity: Classifier vs HMM", fontweight="bold")
    plt.ylim(75, 102)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, df_summary["Classifier-Only Top-1 Acc"] * 100, width, label="Classifier-Only Acc", color="#3B82F6")
    plt.bar(x + width/2, df_summary["Classifier + HMM Top-1 Acc"] * 100, width, label="Classifier + HMM Acc", color="#8B5CF6")
    plt.xticks(x, df_summary["Model"], rotation=25, ha="right")
    plt.ylabel("Top-1 Accuracy (%)", fontweight="bold")
    plt.title("Top-1 Accuracy: Classifier vs HMM", fontweight="bold")
    plt.ylim(70, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "outputs/hmm_comparison_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved HMM comparison plot to {plot_path}", flush=True)

if __name__ == "__main__":
    evaluate_hmm_benchmark()
