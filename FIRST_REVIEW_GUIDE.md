# PathSnap: First Review (Phase 1) Evaluator Guide & Viva Cheat-Sheet

**Event:** UE24CS352A Machine Learning Hackathon  
**Topic:** PathSnap: Distributed Map-Matching & Route Reconstruction  
**Phase:** First Phase Evaluation (Before Lunch Break)

---

## 1. Quick System Demonstration Steps

When the evaluators approach the desk, run:
```bash
streamlit run app.py
```
This opens the **Interactive Evaluator Dashboard**:
1. **Live Route Reconstruction Tab**: Pick any of the 10 student-test trips (`00000000`, `00000007`, `00000040`, etc.), choose a model (e.g., **Random Forest** or **LightGBM**), and click **"Run Map-Matching Prediction"**.
   - Show the live **Top-1 Observation Accuracy (~84.4%)**, **Candidate Miss Rate (~3.5%)**, and **Route Continuity (~90.6%)**.
   - Show the interactive **Folium Map** rendering:
     - **Blue markers & dashed polyline:** Raw noisy GPS observations.
     - **Gray lines:** Local road network subgraph.
     - **Green bold polyline:** Ground-truth traversed route.
     - **Orange dashed polyline:** Predicted route.
2. **5-Model Benchmark Leaderboard Tab**: Displays the side-by-side comparison across all 5 models.
3. **Dataset EDA & Quality Tab**: Displays the empirical distributions of trajectory lengths, sampling frequency (~1 Hz), vehicle velocity profiles, and road network scale.

---

## 2. Dataset Architecture & Preprocessing

* **File Formats per Trip:**
  - `.track`: GPS observations `(longitude, latitude, timestamp_sec)` at ~1 Hz (97% of intervals are exactly 1.0s).
  - `.nodes`: Road network vertices `(longitude, latitude)`. 0-based row index represents the unique `node_id`.
  - `.arcs`: Directed road network edges `(from_node_id, to_node_id)`. 0-based row index represents `arc_id`.
  - `.route`: Ground-truth chronological sequence of traversed `arc_id` values.
* **Geodesic Metric Projection:**
  - Lat/Lon are angular coordinates; Euclidean distance in degrees is invalid.
  - We project coordinates into local metric coordinates $(x, y)$ in meters centered at the trajectory median:
    $$x = (\text{lon} - \overline{\text{lon}}) \cdot 111139 \cdot \cos(\overline{\text{lat}})$$
    $$y = (\text{lat} - \overline{\text{lat}}) \cdot 111139$$
* **Spatial Candidate Generation:**
  - We build a spatial `KDTree` on road network nodes.
  - For each GPS observation, we query the $K=20$ nearest nodes, gather all incident directed arcs, and compute exact point-to-segment perpendicular distance.
  - We select the top-5 candidate arcs.
  - **Empirical Candidate Miss Rate is only 3.49%** (i.e., **96.51% Top-5 Candidate Recall**).

---

## 3. Feature Engineering Rationale

| Feature Name | Formulation | Physical & Behavioral Justification |
| :--- | :--- | :--- |
| `dist_m` | Point-to-segment Euclidean distance (m) | Vehicles travel on road surfaces; proximity to segment center is the primary spatial indicator. |
| `proj_ratio` | $t_{\text{proj}} \in [0, 1]$ | Measures where along the segment the vehicle projects; penalizes projecting beyond endpoints. |
| `heading_diff_deg` | $\|\theta_{\text{gps}} - \theta_{\text{arc}}\|$ (0° to 180°) | Resolves parallel opposite-direction roads (e.g. dual carriageways) by checking movement direction. |
| `heading_alignment` | $\cos(\Delta \theta) \in [-1, 1]$ | Smooth continuous feature (+1 for forward alignment, -1 for opposite direction). |
| `arc_len` | $\sqrt{\Delta x^2 + \Delta y^2}$ | Normalizes road segment lengths; longer links have wider spatial capture basins. |
| `speed_mps` | Instantaneous velocity $\Delta s / \Delta t$ | Distinguishes intersections/stopping from highway cruising. |
| `spatial_prior` | $\exp(-d^2 / (2\sigma^2))$ with $\sigma=6$m | Standard Gaussian GPS measurement error model (Newson & Krumm, 2009). |
| `distance_rank` | Ordinal rank 0 to 4 | Relative distance priority among the 5 local candidates. |

---

## 4. Official 5-Model Benchmark Results

| Model | Algorithm Family | Key Configuration | CV Macro F1 | CV Top-1 Acc | Student-Test Top-1 Acc | Student-Test Macro F1 | Log Loss | Route Continuity | Inference Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | $C=1.0$, balanced | 0.7975 | 0.8315 | 0.8397 | 0.8185 | 0.2987 | 88.7% | 0.002 ms/obs |
| **Decision Tree** | Tree-Based | max_depth=10, leaf=20 | 0.8237 | 0.8609 | 0.8408 | 0.8406 | 0.2502 | 90.1% | 0.002 ms/obs |
| **Random Forest** | Ensemble (Bagging) | 100 trees, depth=12 | 0.8361 | 0.8656 | **0.8443** | 0.8483 | 0.2249 | **90.6%** | 0.056 ms/obs |
| **LightGBM** | Gradient Boosting | 150 trees, lr=0.05 | 0.8391 | **0.8764** | 0.8385 | 0.8447 | 0.2276 | **90.6%** | 0.039 ms/obs |
| **Multi-Layer Perceptron** | Neural Networks | layers=(64,32), ReLU | **0.8489** | 0.8578 | 0.8352 | **0.8866** | **0.1744** | 89.4% | 0.006 ms/obs |

---

## 5. Answers to Compulsory Viva Questions

### Q1: Why did you consider each algorithm suitable?
* **Logistic Regression:** Serves as a transparent linear baseline providing well-calibrated probabilistic confidence scores.
* **Decision Tree:** Provides interpretable hierarchical decision rules (e.g., thresholding distance and heading alignment).
* **Random Forest:** Ensemble bagging reduces variance, prevents overfitting to GPS multipath anomalies, and captures non-linear feature interactions.
* **LightGBM:** Sequential boosting prioritizes hard-to-classify edge cases (e.g., overlapping freeway ramps and parallel service roads) with extreme computational efficiency.
* **Multi-Layer Perceptron:** Learns continuous non-linear representations combining spatial distance with directional velocity.

### Q2: Did you observe overfitting or underfitting?
* **Logistic Regression:** Exhibits slight **underfitting** (CV Top-1 83.15%), as a linear boundary cannot fully capture non-linear combinations of heading and distance.
* **Decision Tree:** Single tree prone to local variance, but controlled via `min_samples_leaf=20`.
* **Random Forest & LightGBM:** Demonstrate **strong generalization**; CV Top-1 (86.5% - 87.6%) and Test Top-1 (83.8% - 84.4%) remain tightly bounded without degradation.
* **MLP:** Achieves the lowest Log Loss (0.1744) and highest Macro F1 (0.8866), aided by early stopping.

### Q3: Why is Observation-Level Accuracy alone insufficient? What is Route Continuity?
* Independent point-by-point classification ignores road network topology. Two consecutive GPS pings might be assigned to disjoint road segments that do not physically connect, resulting in unrealistic "teleportation" jumps.
* Our baseline models already achieve **90.6% Route Continuity**. In Phase 2, sequential **HMM + Viterbi decoding** will explicitly enforce graph transition constraints to reach near-100% path coherence.
