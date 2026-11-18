import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, log_loss

from src.data_loader import list_trips, load_trip
from src.candidate_generator import CandidateGenerator, FEATURE_NAMES
from src.models import get_model_definitions, evaluate_predictions

def calculate_route_continuity(pred_arcs, arcs):
    """
    Computes route continuity: fraction of consecutive predicted arcs
    that are either identical or directly connected in the road graph.
    """
    if len(pred_arcs) < 2:
        return 1.0
    connected = 0
    total = len(pred_arcs) - 1
    for i in range(total):
        a1 = pred_arcs[i]
        a2 = pred_arcs[i + 1]
        if a1 == a2:
            connected += 1
        else:
            if arcs[a1, 1] == arcs[a2, 0]:
                connected += 1
    return connected / total

def build_dataset(trip_ids, split="train", max_trips=20, top_k=5):
    """
    Builds tabular candidate feature matrix across multiple trips.
    """
    generator = CandidateGenerator(top_k=top_k)
    X_list = []
    y_list = []
    trips_processed = 0
    total_misses = 0
    total_obs = 0

    print(f"Extracting candidates & features from {min(max_trips, len(trip_ids))} trips ({split})...", flush=True)
    for tid in trip_ids[:max_trips]:
        trip = load_trip(tid, split=split)
        X_trip, y_trip, meta, miss, n_obs = generator.process_trip(trip, include_gt=True)
        X_list.append(X_trip)
        y_list.append(y_trip)
        total_misses += miss
        total_obs += n_obs
        trips_processed += 1

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    miss_rate = total_misses / total_obs if total_obs > 0 else 0.0
    print(f"Dataset created: {X.shape[0]} candidate samples across {total_obs} observations. Candidate Miss Rate = {miss_rate*100:.2f}%", flush=True)
    return X, y, miss_rate

def main():
    print("=========================================================", flush=True)
    print("  PathSnap: 5-Model Training, Cross-Validation & Benchmark", flush=True)
    print("=========================================================", flush=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_trips = list_trips(split="train")
    test_trips = list_trips(split="student_test")

    # 1. Prepare Training Data
    X_train, y_train, train_miss_rate = build_dataset(train_trips, split="train", max_trips=20, top_k=5)

    # 2. Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    with open("artifacts/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved feature scaler to artifacts/scaler.pkl", flush=True)

    # 3. Model Definitions
    model_defs = get_model_definitions()
    results = []

    # 4. Cross-Validation on Train Data (3-Fold)
    print("\n--- Performing 3-Fold Cross-Validation on Training Set ---", flush=True)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    cv_scores = {}
    for name, info in model_defs.items():
        print(f"Evaluating {name} with CV...", flush=True)
        f1_list = []
        acc_list = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            m = info["model"]
            m.fit(X_train_scaled[train_idx], y_train[train_idx])
            val_probs = m.predict_proba(X_train_scaled[val_idx])[:, 1]
            eval_metrics = evaluate_predictions(y_train[val_idx], val_probs, top_k=5)
            f1_list.append(eval_metrics["macro_f1"])
            acc_list.append(eval_metrics["top1_acc"])
        cv_scores[name] = {
            "cv_macro_f1": float(np.mean(f1_list)),
            "cv_top1_acc": float(np.mean(acc_list))
        }
        print(f"  {name} -> CV Macro F1: {cv_scores[name]['cv_macro_f1']:.4f}, Top-1 Acc: {cv_scores[name]['cv_top1_acc']:.4f}", flush=True)

    # 5. Full Retrain on Training Set & Save Artifacts
    print("\n--- Fitting Full Models on Complete Training Dataset ---", flush=True)
    trained_models = {}
    for name, info in model_defs.items():
        print(f"Fitting {name} on full dataset...", flush=True)
        t0 = time.time()
        m = info["model"]
        m.fit(X_train_scaled, y_train)
        train_time = time.time() - t0
        trained_models[name] = m
        info["train_time"] = train_time
        
        # Save model artifact
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        with open(f"artifacts/model_{clean_name}.pkl", "wb") as f:
            pickle.dump(m, f)
    print("All 5 models saved to artifacts/", flush=True)

    # 6. Evaluation on ALL 10 Student-Test Trips
    print("\n--- Evaluating Models on All 10 Student-Test Trips ---", flush=True)
    generator = CandidateGenerator(top_k=5)
    
    test_trips_data = []
    for tid in test_trips:
        trip = load_trip(tid, split="student_test")
        X_test_trip, y_test_trip, meta_test, miss, n_obs = generator.process_trip(trip, include_gt=True)
        X_test_scaled = scaler.transform(X_test_trip)
        test_trips_data.append({
            "trip": trip,
            "X": X_test_scaled,
            "y": y_test_trip,
            "meta": meta_test,
            "n_obs": n_obs,
            "miss": miss
        })

    for name, info in model_defs.items():
        m = trained_models[name]
        total_top1_correct = 0
        total_valid_obs = 0
        all_y_true = []
        all_y_prob = []
        continuity_list = []
        total_inf_time = 0.0

        for tdata in test_trips_data:
            t0 = time.time()
            probs = m.predict_proba(tdata["X"])[:, 1]
            inf_time = time.time() - t0
            total_inf_time += inf_time

            all_y_true.append(tdata["y"])
            all_y_prob.append(probs)

            # Observation level predictions and route continuity
            pred_arcs = []
            for obs_idx in range(tdata["n_obs"]):
                start = obs_idx * 5
                end = start + 5
                obs_probs = probs[start:end]
                best_cand_idx = int(np.argmax(obs_probs))
                pred_arc_id = tdata["meta"][start + best_cand_idx]["candidate_arc_id"]
                pred_arcs.append(pred_arc_id)
                
                obs_true = tdata["y"][start:end]
                if np.sum(obs_true) > 0:
                    total_valid_obs += 1
                    if obs_true[best_cand_idx] == 1:
                        total_top1_correct += 1

            # Compute route continuity
            cont = calculate_route_continuity(pred_arcs, tdata["trip"]["arcs"])
            continuity_list.append(cont)

        all_y_true = np.concatenate(all_y_true)
        all_y_prob = np.concatenate(all_y_prob)

        # Macro F1 & Log Loss
        y_pred = (all_y_prob >= 0.5).astype(int)
        macro_f1 = f1_score(all_y_true, y_pred, average="macro", zero_division=0)
        eps = 1e-7
        loss = log_loss(all_y_true, np.clip(all_y_prob, eps, 1 - eps))
        top1_acc = total_top1_correct / total_valid_obs if total_valid_obs > 0 else 0.0
        avg_continuity = float(np.mean(continuity_list))
        inf_latency_ms = (total_inf_time / (len(all_y_true) / 5)) * 1000.0

        results.append({
            "Model": name,
            "Algorithm Family": info["family"],
            "Key Configuration": info["key_config"],
            "CV Macro F1": round(cv_scores[name]["cv_macro_f1"], 4),
            "CV Top-1 Accuracy": round(cv_scores[name]["cv_top1_acc"], 4),
            "Student-Test Top-1 Accuracy": round(top1_acc, 4),
            "Student-Test Macro F1": round(macro_f1, 4),
            "Log Loss": round(loss, 4),
            "Route Continuity": f"{avg_continuity*100:.1f}%",
            "Inference Latency (ms/obs)": round(inf_latency_ms, 3)
        })

    # Save Comparison Table
    df_results = pd.DataFrame(results)
    csv_out = "outputs/model_comparison.csv"
    df_results.to_csv(csv_out, index=False)
    
    md_out = "outputs/model_comparison.md"
    with open(md_out, "w") as f:
        f.write("# PathSnap: 5-Model Benchmark Comparison Table\n\n")
        # Format markdown table safely
        headers = list(df_results.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in df_results.iterrows():
            f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")
        f.write("\n")

    print("\n=========================================================", flush=True)
    print("               FINAL 5-MODEL BENCHMARK TABLE             ", flush=True)
    print("=========================================================", flush=True)
    print(df_results.to_string(index=False), flush=True)
    print("=========================================================", flush=True)
    print(f"Saved comparison results to {csv_out} and {md_out}", flush=True)

    # 7. Generate Comparison Visualizations
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    x = np.arange(len(df_results))
    width = 0.35
    plt.bar(x - width/2, df_results["Student-Test Top-1 Accuracy"], width, label="Top-1 Accuracy", color="#2563EB")
    plt.bar(x + width/2, df_results["Student-Test Macro F1"], width, label="Macro F1-Score", color="#10B981")
    plt.xticks(x, df_results["Model"], rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Student-Test Performance Comparison", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, df_results["CV Macro F1"], width, label="Cross-Validation F1", color="#7C3AED")
    plt.bar(x + width/2, df_results["Student-Test Macro F1"], width, label="Student-Test F1", color="#F59E0B")
    plt.xticks(x, df_results["Model"], rotation=25, ha="right")
    plt.ylabel("Macro F1-Score")
    plt.title("Cross-Validation vs Test Generalization", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    comp_plot_path = "outputs/model_comparison_plots.png"
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to {comp_plot_path}", flush=True)

if __name__ == "__main__":
    main()
