import math
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree

def extract_ground_truth_arcs(trip_data):
    """
    Associates each GPS track point with its reference route arc by finding
    the closest route segment in metric space.
    """
    route = trip_data["route"]
    if route is None or len(route) == 0:
        return None

    arcs = trip_data["arcs"]
    nodes_xy = trip_data["nodes_xy"]
    track_xy = trip_data["track_xy"]

    route_u = arcs[route, 0]
    route_v = arcs[route, 1]
    pu = nodes_xy[route_u]
    pv = nodes_xy[route_v]
    vec = pv - pu
    len2 = np.sum(vec**2, axis=1) + 1e-6

    gt_arcs = np.zeros(len(track_xy), dtype=np.int32)
    min_dists = np.zeros(len(track_xy), dtype=np.float32)

    for i, pt in enumerate(track_xy):
        diff = pt - pu
        t = np.clip(np.sum(diff * vec, axis=1) / len2, 0.0, 1.0)
        proj = pu + t[:, None] * vec
        dist = np.linalg.norm(pt - proj, axis=1)
        best_idx = np.argmin(dist)
        gt_arcs[i] = route[best_idx]
        min_dists[i] = dist[best_idx]

    return gt_arcs, min_dists

class CandidateGenerator:
    def __init__(self, top_k=5, search_k_nodes=20, sigma=6.0):
        self.top_k = top_k
        self.search_k_nodes = search_k_nodes
        self.sigma = sigma

    def process_trip(self, trip_data, include_gt=True):
        """
        Generates candidate arcs and feature vectors for each GPS point in trip_data.
        Returns:
            features: np.ndarray (N * top_k, num_features)
            labels: np.ndarray (N * top_k,) [1 if candidate is ground truth, 0 otherwise]
            meta: list of dicts with observation metadata (point_idx, candidate_arc_id, rank, is_gt)
            miss_count: int (observations where true arc was not in top_k)
        """
        arcs = trip_data["arcs"]
        nodes_xy = trip_data["nodes_xy"]
        track_xy = trip_data["track_xy"]
        track = trip_data["track"]
        num_obs = len(track_xy)

        # Ground truth if available
        gt_arcs = None
        if include_gt and trip_data["route"] is not None:
            gt_arcs, _ = extract_ground_truth_arcs(trip_data)

        # Arc vectors and lengths
        u_idx = arcs[:, 0]
        v_idx = arcs[:, 1]
        pu_all = nodes_xy[u_idx]
        pv_all = nodes_xy[v_idx]
        arc_vec_all = pv_all - pu_all
        arc_len2_all = np.sum(arc_vec_all**2, axis=1) + 1e-6
        arc_lens_all = np.sqrt(arc_len2_all)
        arc_headings_all = np.arctan2(arc_vec_all[:, 1], arc_vec_all[:, 0])

        # Node index mapping & KDTree
        node_to_arcs = defaultdict(list)
        for arc_id, (u, v) in enumerate(arcs):
            node_to_arcs[u].append(arc_id)
            node_to_arcs[v].append(arc_id)
        node_tree = KDTree(nodes_xy)

        # Instantaneous velocity and heading for GPS points
        # Using centered difference where possible
        gps_headings = np.zeros(num_obs, dtype=np.float32)
        gps_speeds = np.zeros(num_obs, dtype=np.float32)

        for i in range(num_obs):
            if num_obs == 1:
                break
            prev_i = max(0, i - 1)
            next_i = min(num_obs - 1, i + 1)
            dt = track[next_i, 2] - track[prev_i, 2]
            if dt <= 0:
                dt = 1.0
            dx = track_xy[next_i, 0] - track_xy[prev_i, 0]
            dy = track_xy[next_i, 1] - track_xy[prev_i, 1]
            dist = math.hypot(dx, dy)
            gps_speeds[i] = dist / dt
            gps_headings[i] = math.atan2(dy, dx)

        features_list = []
        labels_list = []
        meta_list = []
        miss_count = 0

        # Query candidates for each GPS observation
        for i in range(num_obs):
            pt = track_xy[i]
            true_arc = gt_arcs[i] if gt_arcs is not None else -1

            # Query nearby nodes
            k_query = min(self.search_k_nodes, len(nodes_xy))
            _, near_nodes = node_tree.query(pt, k=k_query)
            if np.isscalar(near_nodes):
                near_nodes = [near_nodes]

            cand_arcs_set = set()
            for nid in near_nodes:
                cand_arcs_set.update(node_to_arcs[nid])

            cand_arcs = list(cand_arcs_set)
            if len(cand_arcs) == 0:
                # Fallback: query arc midpoint if isolated
                cand_arcs = [0]

            # Vectorized point-to-segment distance for candidate arcs
            c_pu = pu_all[cand_arcs]
            c_vec = arc_vec_all[cand_arcs]
            c_len2 = arc_len2_all[cand_arcs]
            c_lens = arc_lens_all[cand_arcs]
            c_headings = arc_headings_all[cand_arcs]

            diff = pt - c_pu
            t_proj = np.clip(np.sum(diff * c_vec, axis=1) / c_len2, 0.0, 1.0)
            proj = c_pu + t_proj[:, None] * c_vec
            dists = np.linalg.norm(pt - proj, axis=1)

            # Sort and take top_k
            sort_order = np.argsort(dists)[:self.top_k]
            top_cand_arcs = [cand_arcs[idx] for idx in sort_order]
            top_dists = dists[sort_order]
            top_t_proj = t_proj[sort_order]
            top_lens = c_lens[sort_order]
            top_headings = c_headings[sort_order]

            # Check if true arc is retrieved
            if true_arc != -1:
                if true_arc not in top_cand_arcs:
                    miss_count += 1

            for rank in range(len(top_cand_arcs)):
                arc_id = top_cand_arcs[rank]
                dist_m = float(top_dists[rank])
                proj_r = float(top_t_proj[rank])
                arc_len = float(top_lens[rank])
                arc_hdg = float(top_headings[rank])

                # Angular heading difference
                d_theta = abs(gps_headings[i] - arc_hdg)
                d_theta = min(d_theta, 2 * math.pi - d_theta)
                heading_deg = math.degrees(d_theta)
                heading_align = math.cos(d_theta)

                # Prior / Gaussian emission
                spatial_prior = math.exp(-(dist_m**2) / (2 * (self.sigma**2)))
                speed = float(gps_speeds[i])

                feat = [
                    dist_m,                # Point-to-arc perpendicular distance (m)
                    proj_r,                # Projection ratio along arc [0, 1]
                    heading_deg,           # Heading difference (degrees)
                    heading_align,         # Cosine heading alignment [-1, 1]
                    arc_len,               # Arc physical length (m)
                    speed,                 # GPS speed (m/s)
                    spatial_prior,         # Gaussian spatial emission prior
                    float(rank)            # Distance ranking (0 to top_k-1)
                ]
                is_label = 1 if (true_arc != -1 and arc_id == true_arc) else 0

                features_list.append(feat)
                labels_list.append(is_label)
                meta_list.append({
                    "obs_idx": i,
                    "candidate_arc_id": arc_id,
                    "rank": rank,
                    "dist_m": dist_m,
                    "proj_r": proj_r,
                    "is_gt": is_label
                })

        return (
            np.array(features_list, dtype=np.float32),
            np.array(labels_list, dtype=np.int32),
            meta_list,
            miss_count,
            num_obs
        )

FEATURE_NAMES = [
    "dist_m",
    "proj_ratio",
    "heading_diff_deg",
    "heading_alignment",
    "arc_len",
    "speed_mps",
    "spatial_prior",
    "distance_rank"
]
