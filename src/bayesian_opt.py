import os
import sys
import time
import pickle
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, log_loss
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import list_trips, load_trip
from src.candidate_generator import CandidateGenerator
from train_and_evaluate import build_dataset, calculate_route_continuity

# Suppress optuna logging clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization(n_trials=25, seed=42):
    print("=========================================================")
    print("  PathSnap: Bayesian Hyperparameter Optimisation (Optuna) ")
    print("=========================================================")
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load Training Data
    train_trips = list_trips(split="train")
    X_train, y_train, miss_rate = build_dataset(train_trips, split="train", max_trips=20, top_k=5)
    
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_train_scaled = scaler.transform(X_train)

    kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    trials_records = []

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "random_state": seed,
            "verbose": -1,
            "n_jobs": -1
        }

        cv_f1_scores = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            clf = LGBMClassifier(**params)
            clf.fit(X_train_scaled[train_idx], y_train[train_idx])
            val_preds = clf.predict(X_train_scaled[val_idx])
            score = f1_score(y_train[val_idx], val_preds, average="macro", zero_division=0)
            cv_f1_scores.append(score)

        mean_f1 = float(np.mean(cv_f1_scores))
        
        # Save record
        record = {"trial_number": trial.number, "cv_macro_f1": round(mean_f1, 4)}
        record.update(params)
        trials_records.append(record)
        
        print(f"Trial {trial.number:02d} | CV Macro F1: {mean_f1:.4f} | n_est: {params['n_estimators']}, lr: {params['learning_rate']:.3f}, leaves: {params['num_leaves']}, depth: {params['max_depth']}")
        return mean_f1

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Bayesian Optimisation Completed ---")
    best_params = study.best_params
    best_score = study.best_value
    print(f"Best Trial #{study.best_trial.number} with CV Macro F1: {best_score:.4f}")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Save search records
    df_trials = pd.DataFrame(trials_records)
    df_trials.to_csv("outputs/bayesian_optimization_trials.csv", index=False)
    print("Saved trial records to outputs/bayesian_optimization_trials.csv")

    # 2. Refit Tuned Model on Full Training Dataset
    print("\nFitting Best Tuned Model on Full Dataset...")
    full_params = best_params.copy()
    full_params.update({"class_weight": "balanced", "random_state": seed, "verbose": -1, "n_jobs": -1})
    tuned_model = LGBMClassifier(**full_params)
    t0 = time.time()
    tuned_model.fit(X_train_scaled, y_train)
    tuned_train_time = time.time() - t0
    
    with open("artifacts/model_tuned_lightgbm.pkl", "wb") as f:
        pickle.dump(tuned_model, f)
    print("Saved tuned model to artifacts/model_tuned_lightgbm.pkl")

    # 3. Compare Untuned vs Tuned Model on Student-Test Trips
    print("\nEvaluating Untuned vs Tuned Model on 10 Student-Test Trips...")
    with open("artifacts/model_lightgbm.pkl", "rb") as f:
        untuned_model = pickle.load(f)

    test_trips = list_trips(split="student_test")
    generator = CandidateGenerator(top_k=5)

    test_data_list = []
    for tid in test_trips:
        trip = load_trip(tid, split="student_test")
        X_test_trip, y_test_trip, meta_test, miss, n_obs = generator.process_trip(trip, include_gt=True)
        X_test_scaled = scaler.transform(X_test_trip)
        test_data_list.append({
            "trip": trip,
            "X": X_test_scaled,
            "y": y_test_trip,
            "meta": meta_test,
            "n_obs": n_obs
        })

    def eval_model(m, m_name):
        total_top1_correct = 0
        total_valid_obs = 0
        all_y_true = []
        all_y_prob = []
        continuity_list = []
        total_inf_time = 0.0

        for tdata in test_data_list:
            t0 = time.time()
            probs = m.predict_proba(tdata["X"])[:, 1]
            inf_time = time.time() - t0
            total_inf_time += inf_time

            all_y_true.append(tdata["y"])
            all_y_prob.append(probs)

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

            cont = calculate_route_continuity(pred_arcs, tdata["trip"]["arcs"])
            continuity_list.append(cont)

        all_y_true = np.concatenate(all_y_true)
        all_y_prob = np.concatenate(all_y_prob)

        y_pred = (all_y_prob >= 0.5).astype(int)
        macro_f1 = f1_score(all_y_true, y_pred, average="macro", zero_division=0)
        eps = 1e-7
        loss = log_loss(all_y_true, np.clip(all_y_prob, eps, 1 - eps))
        top1_acc = total_top1_correct / total_valid_obs if total_valid_obs > 0 else 0.0
        avg_cont = float(np.mean(continuity_list))
        inf_latency_ms = (total_inf_time / (len(all_y_true) / 5)) * 1000.0

        return {
            "Model Variant": m_name,
            "Student-Test Top-1 Accuracy": round(top1_acc, 4),
            "Student-Test Macro F1": round(macro_f1, 4),
            "Log Loss": round(loss, 4),
            "Route Continuity": f"{avg_cont*100:.1f}%",
            "Inference Latency (ms/obs)": round(inf_latency_ms, 3)
        }

    res_untuned = eval_model(untuned_model, "Untuned LightGBM (Default)")
    res_untuned["CV Macro F1"] = 0.8391  # from official 5-model benchmark
    res_tuned = eval_model(tuned_model, "Tuned LightGBM (Bayesian TPE)")
    res_tuned["CV Macro F1"] = round(best_score, 4)

    df_compare = pd.DataFrame([res_untuned, res_tuned])
    # Reorder columns
    cols = ["Model Variant", "CV Macro F1", "Student-Test Top-1 Accuracy", "Student-Test Macro F1", "Log Loss", "Route Continuity", "Inference Latency (ms/obs)"]
    df_compare = df_compare[cols]

    csv_compare_path = "outputs/tuned_vs_untuned_comparison.csv"
    df_compare.to_csv(csv_compare_path, index=False)
    
    md_compare_path = "outputs/tuned_vs_untuned_comparison.md"
    with open(md_compare_path, "w") as f:
        f.write("# PathSnap: Tuned vs Untuned Model Comparison\n\n")
        f.write("| " + " | ".join(df_compare.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df_compare.columns)) + " |\n")
        for _, row in df_compare.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        f.write("\n")

    print("\n=========================================================")
    print("         TUNED VS UNTUNED MODEL BENCHMARK TABLE          ")
    print("=========================================================")
    print(df_compare.to_string(index=False))
    print("=========================================================")

    # Plot Optimization History
    plt.figure(figsize=(10, 5))
    trials_df = pd.DataFrame(trials_records)
    plt.plot(trials_df["trial_number"], trials_df["cv_macro_f1"], marker="o", color="#2563EB", linewidth=2, label="Trial CV Score")
    plt.plot(trials_df["trial_number"], trials_df["cv_macro_f1"].cummax(), color="#DC2626", linestyle="--", linewidth=2, label="Best-So-Far Objective")
    plt.title("Bayesian Hyperparameter Optimisation History (LightGBM)", fontsize=13, fontweight="bold")
    plt.xlabel("Trial Number", fontsize=11)
    plt.ylabel("3-Fold CV Macro F1 Score", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/bayesian_optimization_history.png", dpi=300)
    plt.close()
    print("Saved optimization history plot to outputs/bayesian_optimization_history.png")

if __name__ == "__main__":
    run_bayesian_optimization(n_trials=25)
