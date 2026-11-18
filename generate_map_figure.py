import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_trip

def generate_static_map_visual(trip_id="00000000", split="student_test", out_path="outputs/folium_map_screenshot.png"):
    os.makedirs("outputs", exist_ok=True)
    trip = load_trip(trip_id, split=split)

    track_xy = trip["track_xy"]
    nodes_xy = trip["nodes_xy"]
    arcs = trip["arcs"]
    route = trip["route"]

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    ax.set_facecolor("#F8FAFC")

    # Bounding box of trajectory
    pad = 120.0
    min_x, max_x = track_xy[:, 0].min() - pad, track_xy[:, 0].max() + pad
    min_y, max_y = track_xy[:, 1].min() - pad, track_xy[:, 1].max() + pad

    # 1. Road network arcs
    u_idx = arcs[:, 0]
    v_idx = arcs[:, 1]
    p_u = nodes_xy[u_idx]
    p_v = nodes_xy[v_idx]

    in_box = (
        (p_u[:, 0] >= min_x) & (p_u[:, 0] <= max_x) &
        (p_u[:, 1] >= min_y) & (p_u[:, 1] <= max_y) &
        (p_v[:, 0] >= min_x) & (p_v[:, 0] <= max_x) &
        (p_v[:, 1] >= min_y) & (p_v[:, 1] <= max_y)
    )
    box_arcs = np.where(in_box)[0]

    for a in box_arcs:
        u, v = arcs[a]
        ax.plot([nodes_xy[u, 0], nodes_xy[v, 0]], [nodes_xy[u, 1], nodes_xy[v, 1]],
                color="#CBD5E1", linewidth=1.2, zorder=1)

    # 2. Ground-truth reference route
    if route is not None:
        for a in route:
            u, v = arcs[a]
            ax.plot([nodes_xy[u, 0], nodes_xy[v, 0]], [nodes_xy[u, 1], nodes_xy[v, 1]],
                    color="#10B981", linewidth=4.5, alpha=0.85, zorder=2)

    # 3. Raw GPS Trajectory
    ax.plot(track_xy[:, 0], track_xy[:, 1], color="#3B82F6", linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)
    step = max(1, len(track_xy) // 80)
    ax.scatter(track_xy[::step, 0], track_xy[::step, 1], color="#1D4ED8", edgecolor="white", s=28, zorder=4)

    # Start and End Markers
    ax.scatter([track_xy[0, 0]], [track_xy[0, 1]], color="#16A34A", s=130, marker="o", edgecolor="black", linewidth=1.5, zorder=5, label="Trip Origin")
    ax.scatter([track_xy[-1, 0]], [track_xy[-1, 1]], color="#DC2626", s=130, marker="s", edgecolor="black", linewidth=1.5, zorder=5, label="Trip Destination")

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.set_xlabel("Relative Metric Easting (meters)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Relative Metric Northing (meters)", fontsize=11, fontweight="bold")
    ax.set_title(f"PathSnap Map-Matching Geographical Reconstruction (Trip #{trip_id})", fontsize=13, fontweight="bold", pad=12)

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color="#CBD5E1", lw=2, label="Road Network Graph Arcs"),
        Line2D([0], [0], color="#10B981", lw=4, label="Ground-Truth Traversed Route"),
        Line2D([0], [0], color="#3B82F6", lw=1.5, linestyle="--", marker="o", markersize=6, markerfacecolor="#1D4ED8", markeredgecolor="white", label="Raw Noisy GPS Observations (~1 Hz)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#16A34A", markeredgecolor="black", markersize=10, label="Trajectory Origin"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#DC2626", markeredgecolor="black", markersize=10, label="Trajectory Destination")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#E2E8F0", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5, color="#94A3B8")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved high-resolution map figure to {out_path}")

if __name__ == "__main__":
    generate_static_map_visual()
