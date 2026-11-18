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

def evaluate_all_test_trips(model_name="Random Forest", model_file="artifacts/model_random_forest.pkl"):
    print("=========================================================")
    print(f"  PathSnap: HMM vs Classifier-Only Benchmark on Test Data")
    print(f"  Model: {model_name}")
    print("=========================================================")
    os.makedirs("outputs", exist_ok=True)

    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    test_trips = list_trips(split="student_test")
    generator = CandidateGenerator(top_k=5)
    hmm = HMMRouteReconstructor(beta=8.0, sigma=6.0, alpha=0.55)

    records = []

    for tid in test_trips:
        trip = load_trip(tid, split="student_test")
        X_trip, y_trip, meta, miss_count, n_obs = generator.process_trip(trip, include_gt=True)
        X_scaled = scaler.transform(X_trip)
        
        t0 = time.time()
        probs = model.predict_proba(X_scaled)[:, 1]
        classifier_time = time.time() - t0

        # Classifier only
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
        t0 = time.time()
        viterbi_preds, viterbi_meta = hmm.reconstruct_route(trip, meta, probs, top_k=5)
        hmm_time = time.time() - t0

        hmm_correct = sum(1 for m in viterbi_meta if m["is_gt"] == 1)
        hmm_acc = (hmm_correct / valid_obs) if valid_obs > 0 else 0.0
        hmm_cont = calculate_route_continuity(viterbi_preds, trip["arcs"])

        miss_rate = (miss_count / n_obs) * 100.0

        records.append({
            "Trip ID": tid,
            "Observations": n_obs,
            "Miss Rate": f"{miss_rate:.2f}%",
            "Classifier Top-1 Acc": round(cls_acc * 100.0, 2),
            "Classifier Continuity": round(cls_cont * 100.0, 2),
            "HMM Top-1 Acc": round(hmm_acc * 100.0, 2),
            "HMM Continuity": round(hmm_cont * 100.0, 2),
            "Acc Delta": round((hmm_acc - cls_acc) * 100.0, 2),
            "Continuity Delta": round((hmm_cont - cls_cont) * 100.0, 2),
            "Classifier Latency (ms)": round(classifier_time * 1000.0, 1),
            "HMM Latency (ms)": round(hmm_time * 1000.0, 1)
        })

    df = pd.DataFrame(records)

    # Summary Row
    summary = {
        "Trip ID": "AVERAGE",
        "Observations": int(df["Observations"].mean()),
        "Miss Rate": f"{df['Miss Rate'].str.rstrip('%').astype(float).mean():.2f}%",
        "Classifier Top-1 Acc": round(df["Classifier Top-1 Acc"].mean(), 2),
        "Classifier Continuity": round(df["Classifier Continuity"].mean(), 2),
        "HMM Top-1 Acc": round(df["HMM Top-1 Acc"].mean(), 2),
        "HMM Continuity": round(df["HMM Continuity"].mean(), 2),
        "Acc Delta": round(df["Acc Delta"].mean(), 2),
        "Continuity Delta": round(df["Continuity Delta"].mean(), 2),
        "Classifier Latency (ms)": round(df["Classifier Latency (ms)"].mean(), 1),
        "HMM Latency (ms)": round(df["HMM Latency (ms)"].mean(), 1)
    }
    df_with_avg = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    # Save to CSV and Markdown
    csv_path = "outputs/hmm_comparison.csv"
    df_with_avg.to_csv(csv_path, index=False)

    md_path = "outputs/hmm_comparison.md"
    with open(md_path, "w") as f:
        f.write(f"# PathSnap: HMM vs Classifier-Only Ablation Study ({model_name})\n\n")
        headers = list(df_with_avg.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in df_with_avg.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        f.write("\n")

    print("\n" + df_with_avg.to_string(index=False))
    print(f"\nSaved HMM ablation study to {csv_path} and {md_path}")

    # Plot Comparison Charts
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["Classifier Top-1 Acc"], width, label="Classifier Only", color="#60A5FA")
    plt.bar(x + width/2, df["HMM Top-1 Acc"], width, label="Classifier + HMM Viterbi", color="#2563EB")
    plt.xticks(x, [f"T{i}" for i in range(len(df))], rotation=0)
    plt.ylabel("Top-1 Accuracy (%)", fontweight="bold")
    plt.title("Observation Top-1 Accuracy: Classifier vs HMM", fontweight="bold")
    plt.ylim(60, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, df["Classifier Continuity"], width, label="Classifier Only", color="#F59E0B")
    plt.bar(x + width/2, df["HMM Continuity"], width, label="Classifier + HMM Viterbi", color="#10B981")
    plt.xticks(x, [f"T{i}" for i in range(len(df))], rotation=0)
    plt.ylabel("Route Continuity (%)", fontweight="bold")
    plt.title("Route Continuity (%): Classifier vs HMM", fontweight="bold")
    plt.ylim(70, 102)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "outputs/hmm_comparison_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved HMM comparison chart to {plot_path}")

if __name__ == "__main__":
    evaluate_all_test_trips()
