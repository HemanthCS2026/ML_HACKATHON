import os
import math
import numpy as np

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Student folder", "Student folder")

def get_base_dir():
    return DEFAULT_BASE_DIR

def list_trips(split="train", base_dir=None):
    if base_dir is None:
        base_dir = get_base_dir()
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(os.listdir(split_dir))

def load_trip(trip_id, split="train", base_dir=None):
    """
    Loads trip data for trip_id.
    Returns:
        dict containing:
            trip_id: str
            track: np.ndarray (N, 3) [lon, lat, timestamp]
            nodes: np.ndarray (M, 2) [lon, lat]
            arcs: np.ndarray (K, 2) [from_node, to_node]
            route: np.ndarray (R,) [ground-truth arc_ids] (or None if not present)
            nodes_xy: np.ndarray (M, 2) [x, y in meters relative to trip center]
            track_xy: np.ndarray (N, 2) [x, y in meters relative to trip center]
            m_per_deg_lon: float
            m_per_deg_lat: float
            center_lon: float
            center_lat: float
    """
    if base_dir is None:
        base_dir = get_base_dir()
    tdir = os.path.join(base_dir, split, trip_id)
    
    # Track: lon, lat, timestamp
    track_path = os.path.join(tdir, f"{trip_id}.track")
    track_pts = []
    with open(track_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                track_pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
    track = np.array(track_pts, dtype=np.float64)

    # Nodes: lon, lat
    nodes_path = os.path.join(tdir, f"{trip_id}.nodes")
    node_pts = []
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node_pts.append((float(parts[0]), float(parts[1])))
    nodes = np.array(node_pts, dtype=np.float64)

    # Arcs: from_node, to_node
    arcs_path = os.path.join(tdir, f"{trip_id}.arcs")
    arc_pairs = []
    with open(arcs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                arc_pairs.append((int(parts[0]), int(parts[1])))
    arcs = np.array(arc_pairs, dtype=np.int32)

    # Route: ground truth traversed arc_ids
    route_path = os.path.join(tdir, f"{trip_id}.route")
    route = None
    if os.path.exists(route_path):
        route_arcs = []
        with open(route_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    route_arcs.append(int(line_str))
        route = np.array(route_arcs, dtype=np.int32)

    # Geodesic metric projection around center of trajectory
    center_lon = np.median(track[:, 0]) if len(track) > 0 else 0.0
    center_lat = np.median(track[:, 1]) if len(track) > 0 else 0.0
    m_per_deg_lat = 111139.0
    m_per_deg_lon = 111139.0 * math.cos(math.radians(center_lat))

    nodes_xy = np.column_stack([
        (nodes[:, 0] - center_lon) * m_per_deg_lon,
        (nodes[:, 1] - center_lat) * m_per_deg_lat
    ])

    track_xy = np.column_stack([
        (track[:, 0] - center_lon) * m_per_deg_lon,
        (track[:, 1] - center_lat) * m_per_deg_lat
    ])

    return {
        "trip_id": trip_id,
        "split": split,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
        "nodes_xy": nodes_xy,
        "track_xy": track_xy,
        "m_per_deg_lon": m_per_deg_lon,
        "m_per_deg_lat": m_per_deg_lat,
        "center_lon": center_lon,
        "center_lat": center_lat
    }
