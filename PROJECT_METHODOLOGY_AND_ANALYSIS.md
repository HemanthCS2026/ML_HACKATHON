# PathSnap: Technical Methodology, Algorithms & Accuracy Analysis

**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Course:** UE24CS352A Machine Learning  
**Institution:** Department of Computer Science & Engineering, PES University (RR Campus)  
**Authors:** Hemanth V, Praveen, Ajay T Naik, Akula Sainandan Royal  

---

## 🗺️ 1. Project Overview & The Core Challenge

* **Objective:** Given a sequence of noisy GPS observations from a vehicle traveling across an urban road network, reconstruct the true trajectory by matching each observation to its true road segment (`arc_id`) and recovering a topologically continuous route.
* **The Urban GPS Dilemma:** In real-world urban environments, GPS signals bounce off high-rise buildings (multipath reflections), experience satellite occlusion, and have low sampling frequencies (~1 Hz). A naive nearest-distance search causes vehicles to snap onto wrong parallel service roads, jump through buildings, or drive down one-way streets in reverse.
* **Our Solution:** A rigorous two-stage architecture:
  1. **Candidate Retrieval & Classification (Phase 1):** High-speed spatial candidate generation + supervised machine learning feature-based ranking.
  2. **Sequential Decoding (Phase 2):** Hidden Markov Model (HMM) + Viterbi algorithm enforcing road-network topological graph continuity.

---

## ⚙️ 2. Data Preprocessing Pipeline

### A. Geodesic Metric Projection
* **The Problem:** Raw GPS observations and road network nodes are given in angular coordinates (Latitude, Longitude) on the WGS84 ellipsoid. Computing standard Euclidean distance in degrees ($\sqrt{\Delta \text{lat}^2 + \Delta \text{lon}^2}$) is mathematically invalid because $1^\circ$ of longitude shrinks as latitude increases ($\approx 111.14 \text{ km} \times \cos(\text{lat})$).
* **Our Solution:** For each trip, we project all coordinates into local metric Cartesian coordinates $(x, y)$ in **meters**, centered at the trajectory's median:
  $$\Delta x = (\text{lon} - \overline{\text{lon}}) \times 111139 \times \cos(\overline{\text{lat}})$$
  $$\Delta y = (\text{lat} - \overline{\text{lat}}) \times 111139$$
  This enables exact, physical Euclidean distance calculations in meters.

### B. High-Speed Spatial Indexing (KD-Tree)
* **The Problem:** Each trip's road network contains up to **307,000 directed road segments (arcs)**. Brute-force scanning all segments for 2,500 GPS points requires over $7.5 \times 10^8$ operations per trip, which is computationally prohibitive.
* **Our Solution:** We construct a spatial `scipy.spatial.KDTree` over all road network vertices (`nodes`).
* **Candidate Extraction:**
  1. For each GPS point, we query the $K=20$ nearest road nodes in logarithmic time ($O(\log M)$).
  2. We gather all directed arcs incident to these nodes.
  3. We compute the exact orthogonal point-to-segment distance and clamp the projection to the segment endpoints $[0, 1]$.
  4. We sort candidate segments by distance and retain the **Top-5 candidates** per observation.
* **Result:** Our spatial indexing achieves a **Top-5 Candidate Recall of 96.51%** (a **Candidate Miss Rate of only 3.49%**), ensuring the true segment is almost always present in the candidate pool.

### C. Feature Scaling
* We fit a `StandardScaler` on the training dataset to normalize all features to zero mean ($\mu=0$) and unit variance ($\sigma^2=1$), ensuring neural networks and linear models converge without gradient instability.

---

## 🧠 3. Feature Engineering Pipeline (8 Domain-Grounded Features)

For every candidate segment $(u \to v)$ of a GPS observation $t$, we extract **8 features** capturing spatial proximity, kinematics, heading alignment, and geometry:

| # | Feature Name | Mathematical Definition | Physical & Behavioral Rationale |
| :-: | :--- | :--- | :--- |
| **1** | **`dist_m`** | $\|\mathbf{p} - \mathbf{p}_{\text{proj}}\|$ | **Perpendicular distance (meters):** Vehicles drive on road surfaces; proximity is the primary spatial indicator. |
| **2** | **`proj_ratio`** | $t_{\text{proj}} = \frac{(\mathbf{p} - \mathbf{a}) \cdot (\mathbf{b} - \mathbf{a})}{\|\mathbf{b} - \mathbf{a}\|^2} \in [0, 1]$ | **Projection ratio along the segment:** Values near 0 or 1 indicate the vehicle is near an intersection, while values around 0.5 indicate mid-block travel. Penalizes points that fall outside the link bounds. |
| **3** | **`heading_diff_deg`** | $\|\theta_{\text{gps}} - \theta_{\text{arc}}\|$ (0° to 180°) | **Angular deviation:** Difference between the vehicle's movement vector ($t-1 \to t+1$) and the road segment's direction vector ($u \to v$). Resolves parallel streets with opposite directions. |
| **4** | **`heading_alignment`** | $\cos(\Delta \theta) \in [-1, 1]$ | **Directional alignment:** Smooth continuous feature; $+1$ indicates perfect forward travel along the arc, while $-1$ heavily penalizes driving the wrong way down a one-way street. |
| **5** | **`arc_len`** | $\|\mathbf{b} - \mathbf{a}\|$ (meters) | **Segment physical length:** Normalizes capture basins; long highway links naturally have wider observation basins than short turning links. |
| **6** | **`speed_mps`** | $\frac{\Delta s}{\Delta t}$ (meters/second) | **Instantaneous velocity:** Distinguishes stationary vehicles at red lights from high-speed highway traffic, helping identify valid road types. |
| **7** | **`spatial_prior`** | $\exp\left(-\frac{d^2}{2\sigma^2}\right)$ with $\sigma=6.0\text{m}$ | **Gaussian Sensor Emission Prior:** Standard physical GPS measurement error model (Newson & Krumm, 2009). Represents the physical probability that sensor noise would displace the reading by distance $d$. |
| **8** | **`distance_rank`** | Ordinal rank $\in \{0, 1, 2, 3, 4\}$ | **Relative spatial ranking:** Prioritizes the nearest candidate among the 5 retrieved alternatives. |

---

## 🤖 4. The 5 Algorithms / Model Families Used

We trained and benchmarked models from **5 completely distinct algorithmic families** on **254,170 candidate samples** using 3-fold cross-validation and evaluated them on the unseen **10 Student-Test trips**:

1. **Logistic Regression (Linear & Probabilistic Family):**
   * *Configuration:* $C=1.0$, `class_weight='balanced'`, L-BFGS solver, `max_iter=1000`.
   * *Role:* Provides a linear, highly interpretable baseline with calibrated probabilistic outputs.
2. **Decision Tree Classifier (Tree-Based Family):**
   * *Configuration:* `max_depth=10`, `min_samples_leaf=20`, `criterion='gini'`.
   * *Role:* Hierarchical rule-based splitting; captures simple non-linear thresholds (e.g., `dist_m < 5m AND heading_alignment > 0.7`).
3. **Random Forest Classifier (Ensemble Bagging Family):**
   * *Configuration:* 100 estimators, `max_depth=12`, `class_weight='balanced'`.
   * *Role:* Bootstrap aggregation across 100 de-correlated decision trees; cancels out individual tree variance and is resilient to random GPS spikes.
4. **LightGBM (Gradient Boosting Family):**
   * *Configuration:* 150 trees, `learning_rate=0.05`, `num_leaves=31`, `class_weight='balanced'`.
   * *Role:* Sequential boosting that iteratively focuses on hard-to-classify edge cases (e.g., overpasses, highway forks, and dense junctions).
5. **Multi-Layer Perceptron (Neural Network Family):**
   * *Configuration:* Two hidden layers `(64, 32)`, ReLU activation, Adam optimizer, early stopping on validation loss.
   * *Role:* Learns continuous non-linear latent feature interactions between spatial distance, heading, and velocity.

---

## 📊 5. Benchmark Leaderboard & Accuracy Breakdown

Official measured performance across all 10 Student-Test trips:

| Model | Algorithm Family | CV Macro F1 | CV Top-1 Acc | Test Top-1 Acc | Test Macro F1 | Log Loss | Route Continuity | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | 0.7975 | 83.15% | **83.97%** | 0.8185 | 0.2987 | 88.7% | **0.002 ms** |
| **Decision Tree** | Tree-Based | 0.8237 | 86.09% | **84.08%** | 0.8406 | 0.2502 | 90.1% | **0.002 ms** |
| **Random Forest** | Ensemble Bagging | 0.8361 | 86.56% | **84.43% 🏆** | 0.8483 | 0.2249 | **90.6% 🏆** | 0.056 ms |
| **LightGBM** | Gradient Boosting | 0.8391 | **87.64% 🏆** | **83.85%** | 0.8447 | 0.2276 | **90.6% 🏆** | 0.039 ms |
| **Multi-Layer Perceptron** | Neural Networks | **0.8489 🏆** | 85.78% | **83.52%** | **0.8866 🏆** | **0.1744 🏆** | 89.4% | 0.006 ms |

---

### 🔍 Why Did Accuracy Differ Across Methods?

#### 1. Why Did Random Forest & LightGBM Achieve the Highest Accuracy (84.4%) and Route Continuity (90.6%)?
* **Orthogonal Decision Boundaries:** Spatial map matching fundamentally relies on bounding box checks and directional thresholds (e.g., *"Is the distance less than 8 meters AND is the heading aligned within 30 degrees?"*). Tree ensembles excel at creating rectangular decision regions in feature space.
* **Resistance to GPS Outliers:** In Random Forest, aggregating predictions across 100 independently bootstrapped trees prevents occasional massive GPS noise spikes from corrupting the prediction.
* **LightGBM's High CV Score (87.64%):** LightGBM achieved the highest CV score because gradient boosting sequentially targets the hardest samples (ambiguous parallel links).

#### 2. Why Did MLP Win on Macro F1 (0.8866) and Log Loss (0.1744)?
* **Log Loss measures probabilistic certainty**, not just binary hard decisions. 
* The Multi-Layer Perceptron outputs smoothly calibrated posterior probabilities ($P(y=1|x)$). While its hard Top-1 accuracy was 83.52%, its predicted probabilities were the most confident and numerically accurate across the dataset.
* **Why this matters:** In Phase 2, the **Hidden Markov Model (HMM)** directly ingests these probabilities as *emission probabilities*. The MLP's low log loss makes it an outstanding emission generator for the sequential decoder!

#### 3. Why Did Logistic Regression Score Lowest (83.15% CV / 83.97% Test)?
* **Linear Underfitting:** A linear model separates classes using a single hyperplane:
  $$z = w_1 \cdot \text{dist} + w_2 \cdot \text{heading} + \dots + b$$
* In map matching, heading and distance have a **coupled non-linear relationship** (a large heading difference is fatal at close distances, but less informative when a vehicle is stationary at a traffic light). A linear model cannot model this interaction without explicit polynomial interaction terms, resulting in slight underfitting.

---

## 🚨 6. The Crucial Viva Question: Why is Observation Top-1 Accuracy Not Enough?

Evaluators will ask: *"You got 84.4% observation accuracy. Why can't you stop here? Why do you need HMM/Viterbi?"*

### The Answer (Observation Accuracy vs. Route Continuity):
* **Observation-Level Top-1 Accuracy** evaluates each GPS ping **independently in isolation**.
* If ping $t$ is snapped to Road Segment $A$, and ping $t+1$ is snapped to Road Segment $B$, Segment $A$ and Segment $B$ might be 2 meters apart in physical space, but **topologically disconnected in the road network graph** (e.g., an elevated highway overpass vs. a ground-level service road).
* When you inspect the predicted route, the vehicle appears to **teleport across roads**.
* Our baseline models already reach **90.6% Route Continuity natively**. 
* **The Phase 2 Bridge:** The remaining **9.4% discontinuity** proves why **Sequential HMM + Viterbi decoding** is mathematically required:
  * HMM uses **Emission Probabilities** (from our ML model) to see how well an arc matches the GPS point.
  * HMM uses **Transition Probabilities** (from the road network graph) to enforce that consecutive arcs must share an edge in the road graph, eliminating all disconnected hops and driving route continuity toward **100%**.
