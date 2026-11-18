import math
import numpy as np
import networkx as nx

def build_road_graph(trip_data):
    """
    Builds a directed NetworkX graph of the road network with edge weights
    equal to the arc lengths in meters.
    """
    arcs = trip_data["arcs"]
    nodes_xy = trip_data["nodes_xy"]
    
    G = nx.DiGraph()
    for arc_id, (u, v) in enumerate(arcs):
        p1 = nodes_xy[u]
        p2 = nodes_xy[v]
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        G.add_edge(u, v, weight=max(length, 0.1), arc_id=arc_id)
        
    return G

class HMMRouteReconstructor:
    """
    Hidden Markov Model (HMM) with Viterbi decoding for map-matching route reconstruction.
    Combines classifier emission probabilities with road-network topological transition probabilities.
    Based on the Newson & Krumm (2009) formulation with robust handling of GPS gaps and network disconnections.
    """
    def __init__(self, beta=8.0, sigma=6.0, alpha=0.55, gap_threshold=10.0, max_cutoff=250.0):
        self.beta = beta                  # Scale parameter for transition probability exponential
        self.sigma = sigma                # Standard deviation for GPS sensor error (meters)
        self.alpha = alpha                # Weight balancing model probability vs spatial distance prior
        self.gap_threshold = gap_threshold  # Time gap (seconds) beyond which transition constraints relax
        self.max_cutoff = max_cutoff      # Maximum search radius for Dijkstra shortest path (meters)

    def reconstruct_route(self, trip_data, cand_meta, model_probs, top_k=5):
        """
        Runs Viterbi dynamic programming to find the optimal sequence of candidate arcs.
        Args:
            trip_data: dict containing 'arcs', 'nodes_xy', 'track_xy', 'track'
            cand_meta: list of candidate metadata dicts (from CandidateGenerator)
            model_probs: np.ndarray of shape (N * top_k,) containing model positive class probabilities
            top_k: number of candidates per observation
        Returns:
            viterbi_arcs: list of arc IDs (length N)
            viterbi_meta: list of metadata dicts for the selected candidates
        """
        track_xy = trip_data["track_xy"]
        track = trip_data["track"]
        arcs = trip_data["arcs"]
        nodes_xy = trip_data["nodes_xy"]
        num_obs = len(track_xy)

        if num_obs == 0:
            return [], []

        # Precalculate arc lengths
        u_idx = arcs[:, 0]
        v_idx = arcs[:, 1]
        arc_lens = np.hypot(nodes_xy[v_idx, 0] - nodes_xy[u_idx, 0],
                            nodes_xy[v_idx, 1] - nodes_xy[u_idx, 1])

        # Build graph
        G = build_road_graph(trip_data)

        # Precompute Dijkstra reachability cache for speed
        dijkstra_cache = {}

        def get_shortest_path_dist(source_node, target_node, cutoff):
            if source_node == target_node:
                return 0.0
            if source_node not in dijkstra_cache:
                try:
                    dijkstra_cache[source_node] = nx.single_source_dijkstra_path_length(G, source_node, cutoff=cutoff)
                except Exception:
                    dijkstra_cache[source_node] = {}
            dist_map = dijkstra_cache[source_node]
            return dist_map.get(target_node, None)

        # Trellis tables
        # V[t, k] = log-probability of best path ending in candidate k at step t
        V = np.full((num_obs, top_k), -np.inf, dtype=np.float64)
        backpointers = np.zeros((num_obs, top_k), dtype=np.int32)

        eps = 1e-7

        # Step 0: Initialize priors
        for k in range(top_k):
            idx = k
            dist_m = cand_meta[idx]["dist_m"]
            prob = float(model_probs[idx])
            
            # Log emission = alpha * log(prob) + (1 - alpha) * (-dist^2 / (2 * sigma^2))
            spatial_log = -(dist_m ** 2) / (2.0 * (self.sigma ** 2))
            model_log = math.log(max(prob, eps))
            log_emission = self.alpha * model_log + (1.0 - self.alpha) * spatial_log
            V[0, k] = log_emission

        # Steps 1 to T-1: Viterbi induction
        for t in range(1, num_obs):
            prev_base = (t - 1) * top_k
            curr_base = t * top_k

            # GPS displacement and time delta
            dx = track_xy[t, 0] - track_xy[t - 1, 0]
            dy = track_xy[t, 1] - track_xy[t - 1, 1]
            dgps = math.hypot(dx, dy)
            dt = track[t, 2] - track[t - 1, 2]

            is_gap = dt > self.gap_threshold
            cutoff = min(self.max_cutoff, max(50.0, dgps * 2.5 + 40.0))

            for j in range(top_k):
                curr_idx = curr_base + j
                curr_meta = cand_meta[curr_idx]
                curr_arc = curr_meta["candidate_arc_id"]
                curr_dist = curr_meta["dist_m"]
                curr_prob = float(model_probs[curr_idx])

                # Current emission log-prob
                spatial_log = -(curr_dist ** 2) / (2.0 * (self.sigma ** 2))
                model_log = math.log(max(curr_prob, eps))
                log_emission = self.alpha * model_log + (1.0 - self.alpha) * spatial_log

                u_j = arcs[curr_arc, 0]
                v_j = arcs[curr_arc, 1]
                L_j = arc_lens[curr_arc]
                t_proj_j = np.clip(curr_meta.get("proj_r", 0.5), 0.0, 1.0)

                best_prev_val = -np.inf
                best_prev_k = 0

                for k in range(top_k):
                    prev_idx = prev_base + k
                    prev_meta = cand_meta[prev_idx]
                    prev_arc = prev_meta["candidate_arc_id"]
                    u_k = arcs[prev_arc, 0]
                    v_k = arcs[prev_arc, 1]
                    L_k = arc_lens[prev_arc]
                    t_proj_k = np.clip(prev_meta.get("proj_r", 0.5), 0.0, 1.0)

                    # Compute network distance between candidate k at t-1 and candidate j at t
                    if is_gap:
                        # Relaxation during large GPS dropouts
                        log_trans = -math.log(self.beta * 2.0) - (abs(dgps) / (self.beta * 3.0))
                    elif prev_arc == curr_arc:
                        # Remained on the same arc
                        d_net = abs(t_proj_j - t_proj_k) * L_k
                        diff = abs(d_net - dgps)
                        log_trans = -math.log(self.beta) - (diff / self.beta)
                    elif v_k == u_j:
                        # Direct connected topological successor
                        d_net = (1.0 - t_proj_k) * L_k + t_proj_j * L_j
                        diff = abs(d_net - dgps)
                        # Bonus for direct connectivity
                        log_trans = -math.log(self.beta) - (diff / self.beta) + 0.5
                    else:
                        # Query shortest path from v_k to u_j
                        sp_len = get_shortest_path_dist(v_k, u_j, cutoff)
                        if sp_len is not None:
                            d_net = (1.0 - t_proj_k) * L_k + sp_len + t_proj_j * L_j
                            diff = abs(d_net - dgps)
                            log_trans = -math.log(self.beta) - (diff / self.beta)
                        else:
                            # Disconnected within search radius
                            penalty_dist = max(100.0, dgps * 3.0)
                            log_trans = -math.log(self.beta) - (penalty_dist / self.beta) - 4.0

                    total_score = V[t - 1, k] + log_trans
                    if total_score > best_prev_val:
                        best_prev_val = total_score
                        best_prev_k = k

                V[t, j] = best_prev_val + log_emission
                backpointers[t, j] = best_prev_k

        # Termination: choose candidate with highest log-prob at step T-1
        best_end_k = int(np.argmax(V[num_obs - 1]))

        # Traceback
        viterbi_cands = [0] * num_obs
        curr_k = best_end_k
        viterbi_cands[num_obs - 1] = curr_k

        for t in range(num_obs - 1, 0, -1):
            curr_k = int(backpointers[t, curr_k])
            viterbi_cands[t - 1] = curr_k

        # Extract arc IDs and metadata
        viterbi_arcs = []
        viterbi_meta = []
        for t in range(num_obs):
            selected_cand_idx = t * top_k + viterbi_cands[t]
            selected_meta = cand_meta[selected_cand_idx]
            viterbi_arcs.append(selected_meta["candidate_arc_id"])
            viterbi_meta.append(selected_meta)

        return viterbi_arcs, viterbi_meta
