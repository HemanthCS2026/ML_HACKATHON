import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import list_trips, load_trip
from eda.visualize_map import create_folium_map

def run_eda():
    print("=== Starting Comprehensive PathSnap EDA ===")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

    train_trips = list_trips(split="train")
    test_trips = list_trips(split="student_test")
    print(f"Discovered {len(train_trips)} train trips and {len(test_trips)} student-test trips.")

    # 1. Collect summary statistics
    train_stats = []
    sample_intervals = []
    speeds = []
    
    # Process a representative subset of training trips for detailed point-level stats
    # and all trips for aggregate stats
    for i, tid in enumerate(train_trips):
        trip = load_trip(tid, split="train")
        track = trip["track"]
        nodes = trip["nodes"]
        arcs = trip["arcs"]
        route = trip["route"]

        n_track = len(track)
        n_nodes = len(nodes)
        n_arcs = len(arcs)
        n_route = len(route) if route is not None else 0

        # Trajectory length & duration
        duration_s = track[-1, 2] - track[0, 2] if n_track > 1 else 0.0

        # Sample intervals
        if n_track > 1:
            dt = np.diff(track[:, 2])
            sample_intervals.extend(dt[dt < 20.0]) # filter huge gaps for distribution plot

            # Speeds (m/s)
            dx = np.diff(trip["track_xy"][:, 0])
            dy = np.diff(trip["track_xy"][:, 1])
            dist = np.hypot(dx, dy)
            valid_dt = np.where(dt > 0, dt, 1.0)
            spd = dist / valid_dt
            speeds.extend(spd[spd < 60.0]) # filter extreme noise > 216 km/h

        train_stats.append({
            "trip_id": tid,
            "split": "train",
            "track_pts": n_track,
            "nodes": n_nodes,
            "arcs": n_arcs,
            "route_arcs": n_route,
            "duration_s": duration_s
        })

    test_stats = []
    for tid in test_trips:
        trip = load_trip(tid, split="student_test")
        test_stats.append({
            "trip_id": tid,
            "split": "student_test",
            "track_pts": len(trip["track"]),
            "nodes": len(trip["nodes"]),
            "arcs": len(trip["arcs"]),
            "route_arcs": len(trip["route"]) if trip["route"] is not None else 0,
            "duration_s": trip["track"][-1, 2] - trip["track"][0, 2] if len(trip["track"]) > 1 else 0.0
        })

    df_train = pd.DataFrame(train_stats)
    df_test = pd.DataFrame(test_stats)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    csv_path = "outputs/dataset_inventory.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"Saved dataset inventory to {csv_path}")

    # Print summary tables
    print("\n--- Training Set Summary ---")
    print(df_train[["track_pts", "nodes", "arcs", "route_arcs", "duration_s"]].describe())

    print("\n--- Student Test Set Summary ---")
    print(df_test[["track_pts", "nodes", "arcs", "route_arcs", "duration_s"]].describe())

    # 2. Plotting EDA Distributions
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("PathSnap Dataset Exploratory Data Analysis & Empirical Distributions", fontsize=16, fontweight="bold")

    # Plot 1: GPS Track Points Distribution
    sns.histplot(df_train["track_pts"], bins=20, ax=axes[0, 0], color="#2563EB", kde=True)
    axes[0, 0].set_title("Trajectory Length Distribution (Observations / Trip)", fontweight="bold")
    axes[0, 0].set_xlabel("Number of GPS Observations")
    axes[0, 0].set_ylabel("Count of Trips")

    # Plot 2: Sampling Interval Distribution (~1 Hz verification)
    sample_intervals = np.array(sample_intervals)
    exact_1s = np.mean((sample_intervals >= 0.95) & (sample_intervals <= 1.05)) * 100
    sns.histplot(sample_intervals[sample_intervals <= 5.0], bins=30, ax=axes[0, 1], color="#059669", kde=False)
    axes[0, 1].set_title(f"GPS Sampling Interval $\\Delta t$ ({exact_1s:.1f}% exactly 1.0s)", fontweight="bold")
    axes[0, 1].set_xlabel("Interval $\\Delta t$ (seconds)")
    axes[0, 1].set_ylabel("Frequency")

    # Plot 3: Instantaneous Speed Profile
    speeds = np.array(speeds)
    sns.histplot(speeds, bins=40, ax=axes[0, 2], color="#D97706", kde=True)
    axes[0, 2].set_title(f"Vehicle Speed Profile (Median: {np.median(speeds):.1f} m/s = {np.median(speeds)*3.6:.1f} km/h)", fontweight="bold")
    axes[0, 2].set_xlabel("Speed (m/s)")
    axes[0, 2].set_ylabel("Frequency")

    # Plot 4: Road Network Size (Nodes vs Arcs Correlation)
    axes[1, 0].scatter(df_train["nodes"], df_train["arcs"], color="#7C3AED", alpha=0.7, edgecolors="none", s=50, label="Train")
    axes[1, 0].scatter(df_test["nodes"], df_test["arcs"], color="#DC2626", alpha=0.9, marker="^", s=70, label="Student-Test")
    axes[1, 0].set_title("Road Network Scale: Nodes vs Arcs", fontweight="bold")
    axes[1, 0].set_xlabel("Number of Road Nodes")
    axes[1, 0].set_ylabel("Number of Road Arcs")
    axes[1, 0].legend()

    # Plot 5: Route Length vs Number of GPS Points
    axes[1, 1].scatter(df_train["track_pts"], df_train["route_arcs"], color="#0891B2", alpha=0.7, edgecolors="none", s=50)
    axes[1, 1].set_title("GPS Track Points vs Ground-Truth Route Arcs", fontweight="bold")
    axes[1, 1].set_xlabel("GPS Points")
    axes[1, 1].set_ylabel("Traversed Route Arcs")

    # Plot 6: Data Quality and Partition Comparison
    partition_summary = pd.DataFrame({
        "Partition": ["Train (80 trips)", "Student-Test (10 trips)"],
        "Avg Points": [df_train["track_pts"].mean(), df_test["track_pts"].mean()],
        "Avg Arcs": [df_train["route_arcs"].mean(), df_test["route_arcs"].mean()]
    })
    x = np.arange(len(partition_summary))
    width = 0.35
    axes[1, 2].bar(x - width/2, partition_summary["Avg Points"] / 10, width, label="Avg Points (x10)", color="#2563EB")
    axes[1, 2].bar(x + width/2, partition_summary["Avg Arcs"], width, label="Avg Route Arcs", color="#10B981")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(partition_summary["Partition"])
    axes[1, 2].set_title("Partition Scale Comparison", fontweight="bold")
    axes[1, 2].legend()

    plt.tight_layout()
    plot_path = "outputs/eda_analysis_plots.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved EDA visualization plots to: {plot_path}")

    # 3. Generate Representative Interactive Folium Maps
    print("\nGenerating Folium interactive HTML maps...")
    sample_train = load_trip(train_trips[0], split="train")
    create_folium_map(sample_train, output_html="maps/trip_train_00000001.html")

    sample_test = load_trip(test_trips[0], split="student_test")
    create_folium_map(sample_test, output_html="maps/trip_test_00000000.html")

    print("\n=== EDA Completed Successfully! ===")

if __name__ == "__main__":
    run_eda()
