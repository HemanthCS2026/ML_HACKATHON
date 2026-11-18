# PathSnap: Methodology, Algorithms & Results

**Course:** UE24CS352A Machine Learning  
**Department:** Computer Science & Engineering, PES University (RR Campus)  
**Authors:** Hemanth V, Praveen, Ajay T Naik, Akula Sainandan Royal  

---

## 1. Problem Context

GPS receivers in urban areas produce noisy coordinates. Tall buildings reflect signals (multipath error), overhead bridges cause blackouts, and cheap smartphone receivers have low sampling frequency (usually 1 Hz). 

If you just find the closest road segment by Euclidean distance:
1. Parallel roads get confused (service lanes vs main carriage roads)
2. Direction is ignored (the car gets snapped to a one-way street going backward)
3. Consecutive points jump randomly between disconnected streets, making the vehicle look like it is teleporting

To fix this, we designed a two-step approach:
- **Step 1 (Candidate Generation + ML Ranking):** Find candidate roads near each GPS ping and use a trained classifier to score them based on distance, heading, and speed.
- **Step 2 (HMM Viterbi Route Reconstruction):** Take the sequence of candidates and find the most probable path through the road graph, enforcing that consecutive roads must actually connect.

---

## 2. Preprocessing & Candidate Retrieval

### 2.1 Coordinate Projection
Raw files give longitude and latitude in degrees. Since the Earth is curved, 1 degree of longitude is not the same physical distance as 1 degree of latitude (it shrinks by `cos(latitude)`). Calculating Euclidean distance directly in degrees gives stretched, inaccurate distances.

For each trip, we project coordinates into meters centered around the median position of the trip:
- `dx = (lon - median_lon) * 111139.0 * cos(median_lat)`
- `dy = (lat - median_lat) * 111139.0`

This lets us work with standard meters for road lengths, vehicle speeds, and perpendicular offsets.

### 2.2 KDTree Candidate Retrieval
Some trips have over 30,000 road segments (`.arcs`). Checking every road segment for 2,500 GPS observations would mean 75 million distance checks per trip.

Instead, we:
1. Build a `scipy.spatial.KDTree` on all road vertices (`.nodes`) in metric space.
2. For each GPS ping, query the 20 nearest nodes in `O(log N)` time.
3. Collect all directed segments that touch these nodes.
4. Calculate perpendicular distance to each segment, clamping the projection to `[0, 1]` so points don't project past segment endpoints.
5. Keep the closest 5 candidates per GPS point.

Across the entire dataset, our Top-5 recall is **96.51%** (Candidate Miss Rate is only **3.49%**).

---

## 3. Feature Engineering

For each candidate road segment, we compute 8 features:

| Feature | Description | Why it helps |
| :--- | :--- | :--- |
| `dist_m` | Distance from GPS point to the segment (meters) | Primary check: cars drive on roads, so true segment is usually close. |
| `proj_ratio` | Value in `[0, 1]` showing where along the segment the point lands | Values near 0 or 1 mean the car is near an intersection; values outside `[0, 1]` are penalized. |
| `heading_diff_deg` | Angle between car movement and road direction (0 to 180 degrees) | Tells apart parallel roads going in opposite directions. |
| `heading_align` | Cosine of heading difference (`cos(theta)` between -1 and +1) | Smooth indicator: +1 means travelling along the road, -1 means wrong-way. |
| `arc_len` | Length of the road segment in meters | Normalizes capture zone: longer highway links catch points differently than short intersection links. |
| `speed_mps` | Instantaneous GPS speed (meters per second) | Helps separate stopped cars (red lights) from moving traffic. |
| `spatial_prior` | Gaussian density `exp(-dist^2 / (2 * sigma^2))` with `sigma = 6m` | Sensor noise model from Newson & Krumm (2009). |
| `distance_rank` | Rank among the 5 candidates (0 for closest, 4 for furthest) | Relative local ordering. |

Features are standardized with `StandardScaler` fitted on the training split only (no data leakage).

---

## 4. Model Training & Comparison

We implemented 5 models from 5 different algorithm families to compare their behavior on this data:

1. **Logistic Regression (Linear / Probabilistic):** L-BFGS solver, `class_weight='balanced'`. Fast baseline, gives probabilistic outputs.
2. **Decision Tree (Tree-based):** `max_depth=10`, `min_samples_leaf=20`. Simple if-else decision boundaries on distance and heading.
3. **Random Forest (Ensemble / Bagging):** 100 trees, `max_depth=12`. Averages multiple trees, resistant to random GPS spikes.
4. **LightGBM (Boosting):** 150 estimators, `learning_rate=0.05`, `num_leaves=31`. Builds trees sequentially to fix previous errors.
5. **Multi-Layer Perceptron (Neural Network):** Hidden layers `(64, 32)`, ReLU, Adam optimizer, early stopping. Captures smooth non-linear feature combinations.

### Benchmark Results (10 Student-Test Trips)

| Model | CV Macro F1 | CV Top-1 Acc | Test Top-1 Acc | Test Macro F1 | Log Loss | Route Continuity | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7975 | 83.15% | 83.97% | 0.8185 | 0.2987 | 88.7% | 0.002 ms |
| Decision Tree | 0.8237 | 86.09% | 84.08% | 0.8406 | 0.2502 | 90.1% | 0.002 ms |
| Random Forest | 0.8361 | 86.56% | **84.43%** | 0.8483 | 0.2249 | **90.6%** | 0.056 ms |
| LightGBM | 0.8391 | **87.64%** | 83.85% | 0.8447 | 0.2276 | **90.6%** | 0.039 ms |
| Multi-Layer Perceptron | **0.8489** | 85.78% | 83.52% | **0.8866** | **0.1744** | 89.4% | 0.006 ms |

### Observations:
- **Random Forest & LightGBM** achieved the best Top-1 test accuracy (84.4%) and highest route continuity (90.6%). Tree splits fit spatial threshold rules well (e.g., `dist < 7m` AND `heading_align > 0.6`).
- **MLP** had the lowest log loss (0.1744) and highest Macro F1 (0.8866). Its probability outputs are smoothly calibrated, making it a strong candidate for HMM emission probabilities.
- **Logistic Regression** slightly underperformed because distance and heading interact non-linearly, which a single linear hyperplane cannot completely separate.

---

## 5. Bayesian Hyperparameter Tuning

We ran Bayesian optimization using **Optuna (TPE sampler)** over 25 trials on LightGBM:
- Objective: 3-fold CV Macro F1 on the training split.
- Parameters tuned: `n_estimators`, `learning_rate` (log-uniform), `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`.

### Result:
- Untuned LightGBM: CV F1 = 0.8391, Test Top-1 Acc = 83.85%
- Tuned LightGBM: **CV F1 = 0.8590**, Test Top-1 Acc = 82.52%

The tuned model achieved higher CV F1 (+0.02) and higher test Macro F1 (0.8488 vs 0.8447), but slightly lower Top-1 accuracy due to slight overfitting to training folds. We retained both models in `artifacts/`.

---

## 6. HMM & Viterbi Sequential Decoding

Pointwise classifiers make decisions for each second in isolation. Even at 84.4% accuracy, ~10% of transitions jump between roads that do not connect in the road network.

To resolve this, we implemented a Hidden Markov Model:
- **Hidden states:** The candidate road segments at each time step.
- **Emission probability:** Combines the classifier's predicted probability with the Gaussian sensor distance prior:
  `log P(emission) = alpha * log(P_model) + (1 - alpha) * (-dist^2 / (2 * sigma^2))`
- **Transition probability:** Compares the actual shortest path distance in the road network graph against the straight-line GPS movement:
  `P(transition) = (1 / beta) * exp(-|d_network - d_gps| / beta)`
  If two candidates share a node or are on the same road, transition probability is high. If they are disconnected, a penalty is applied.
- **GPS Gaps:** When timestamps show a gap > 10 seconds, the transition penalty is relaxed so the model does not force an impossible connection.
- **Viterbi Algorithm:** Runs dynamic programming to find the sequence of road segments that maximizes total log probability over the entire trip.

### HMM Test Results (All 10 Student-Test Trips):
- **Classifier-Only Avg Continuity:** 90.62%
- **Classifier + HMM Avg Continuity:** **97.23% (+6.61% gain)**
- **Classifier-Only Avg Top-1 Accuracy:** 86.54%
- **Classifier + HMM Avg Top-1 Accuracy:** **89.66% (+3.12% gain)**

The HMM eliminates almost all disconnected hops while simultaneously correcting ambiguous candidate predictions near intersections.
