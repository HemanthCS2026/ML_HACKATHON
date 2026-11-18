# PathSnap: Presentation Speaker Script

**Event:** UE24CS352A Machine Learning Hackathon  
**Institution:** Computer Science & Engineering, PES University (RR Campus)  
**Topic:** PathSnap: Distributed Map-Matching & Route Reconstruction  

---

## Speaking Roles
- **Speaker 1: Hemanth V (PES1UG25CS817)** — Introduction, Problem Overview, Dataset EDA
- **Speaker 2: Praveen (PES1UG25CS832)** — Coordinate Projection, KDTree Candidate Search, Feature Engineering
- **Speaker 3: Ajay T Naik (PES1UG24CS552)** — 5-Model Benchmark, Results Analysis, Viva Defense
- **Speaker 4: Akula Sainandan Royal (PES1UG24CS553)** — Bayesian Tuning, HMM Viterbi Decoder, Two-Laptop Demo

---

## Slide-by-Slide Script

### Slide 1: Introduction
**Speaker: Hemanth V**

> "Good morning evaluators. We are Team PathSnap, presenting our project on Distributed Map-Matching and Route Reconstruction for the Machine Learning Hackathon.
>
> Our team members are Hemanth, Praveen, Ajay, and Akula from the CSE Department at RR Campus.
>
> Today we'll walk you through our end-to-end pipeline: how we handled the raw GPS and road network data, engineered spatial features, benchmarked five different model families, tuned them with Bayesian optimization, and reconstructed continuous routes using an HMM Viterbi decoder. Let me start with the problem we're solving."

---

### Slide 2: Problem Statement
**Speaker: Hemanth V**

> "In urban areas, GPS signals bounce off high buildings, get blocked under bridges, and suffer from noise. As a result, raw vehicle GPS points often look like they're jumping across buildings or drifting onto parallel roads.
>
> If you just snap each GPS point to the closest road segment by distance, you run into three problems:
> 1. You snap onto the wrong parallel road or service lane
> 2. You ignore direction, sometimes placing the vehicle on a one-way street going the wrong way
> 3. Consecutive points jump between roads that don't actually connect, making the route look disconnected
>
> PathSnap fixes this with a two-stage approach: first, machine learning scores the best candidate roads for each GPS point using distance, heading, and speed. Second, an HMM with Viterbi decoding enforces road network connectivity so the reconstructed route is continuous."

---

### Slide 3: Dataset & EDA
**Speaker: Hemanth V**

> "We analyzed all 90 provided trips — 80 for training and 10 for student testing.
>
> Each trip folder has four files:
> - `.track` with timestamped GPS points
> - `.nodes` with road junction coordinates
> - `.arcs` with directed road connections
> - `.route` with the ground-truth sequence of roads
>
> From our EDA, we found that about 97% of GPS readings are spaced exactly 1 second apart, confirming a 1 Hz sampling rate. GPS noise typically caused deviations of 3 to 18 meters from the road centerline. The road networks varied from small neighborhoods of 650 links to large regional networks of over 300,000 links.
>
> Now Praveen will explain how we preprocessed the data and extracted candidate road segments."

---

### Slide 4: Coordinate Projection & Candidate Retrieval
**Speaker: Praveen**

> "Thanks, Hemanth. First, we had to fix the coordinate system. Longitude and latitude are angles on a sphere, so 1 degree of longitude shrinks as you move away from the equator. Taking Euclidean distance directly in degrees gives warped results. We projected coordinates into local meters centered at each trip's median position so all distance and speed calculations are in actual meters.
>
> Next was candidate retrieval. Some road networks have tens of thousands of links, so checking all of them for thousands of GPS points would be too slow. We built a spatial KDTree on the road nodes. For each GPS point, we query the 20 nearest nodes in O(log N) time and only check segments touching those nodes. We calculate perpendicular distance to each segment and keep the top 5 candidates.
>
> This gives us a 96.51% candidate recall, meaning the true road segment is in our top 5 candidates almost 97% of the time, with a miss rate of only 3.49%."

---

### Slide 5: Feature Engineering
**Speaker: Praveen**

> "For every candidate road segment, we compute eight domain-specific features:
> 1. `dist_m`: Perpendicular distance in meters from the GPS point to the road segment.
> 2. `proj_ratio`: Where along the segment the point projects, between 0 and 1. Values outside this range mean the point is beyond the segment's endpoints.
> 3. `heading_diff_deg` and `heading_align`: We calculate the car's heading from consecutive GPS points and compare it with the road's direction. The cosine alignment gives +1 for moving in the right direction and -1 for wrong-way travel.
> 4. `arc_len`: Length of the road segment in meters, which accounts for differing segment lengths.
> 5. `speed_mps`: Instantaneous speed, helping distinguish stopped vehicles at signals from free-flowing traffic.
> 6. `spatial_prior`: A Gaussian probability based on GPS sensor noise, following the standard Newson and Krumm model.
> 7. `distance_rank`: Rank from 0 to 4 among the 5 candidates.
>
> Features are scaled using StandardScaler fitted only on the training set. Now Ajay will cover our model benchmark results."

---

### Slide 6: 5-Model Benchmark Results
**Speaker: Ajay T Naik**

> "Thanks, Praveen. As required by the guidelines, we trained and compared models from five distinct algorithm families using 3-fold cross-validation on 80 training trips, and tested them on all 10 student-test trips:
>
> 1. **Logistic Regression (Linear):** 83.97% test accuracy, 88.7% continuity. Fast baseline, but slightly underfits because heading and distance interact non-linearly.
> 2. **Decision Tree (Tree-based):** 84.08% test accuracy, 90.1% continuity. Captures simple if-else threshold rules.
> 3. **Random Forest (Ensemble Bagging):** Achieved our best test accuracy at 84.43% and 90.6% continuity. Averages 100 trees to cancel out noise spikes.
> 4. **LightGBM (Boosting):** 83.85% test accuracy, 87.64% CV accuracy, 90.6% continuity. Fastest inference at 0.039 ms per observation.
> 5. **Multi-Layer Perceptron (Neural Net):** 83.52% test accuracy, but achieved the best Log Loss at 0.1744 and best Macro F1 at 0.8866. Its calibrated probabilities make it very effective for generating HMM emission scores."

---

### Slide 7: Model Comparison & Generalization
**Speaker: Ajay T Naik**

> "Looking at generalization: the gap between cross-validation and unseen test accuracy was under 2% across all models, showing that none of them suffered from severe overfitting. 
>
> For tree models, we limited tree depth and set minimum leaf samples to prevent them from memorizing specific road layouts. 
>
> Why is 84.4% observation accuracy not enough on its own? Because each point is classified independently. Even with 84.4% accuracy, about 10% of transitions jump between roads that don't connect in the road network graph. That's why the second stage — HMM decoding — is necessary."

---

### Slide 8: Bayesian Optimization & HMM Decoding
**Speaker: Akula Sainandan Royal**

> "Thanks, Ajay. For Phase 2, we took our top model, LightGBM, and ran Bayesian hyperparameter optimization using Optuna's TPE sampler over 25 trials. Tuning improved our cross-validation F1 from 0.8391 to 0.8590.
>
> Then, we implemented our Hidden Markov Model with Viterbi decoding:
> - **States:** The candidate road segments at each second.
> - **Emission probability:** Combines the classifier's predicted probability with the Gaussian spatial distance prior.
> - **Transition probability:** Compares the shortest path distance through the road graph against straight-line GPS movement. If two roads are connected, the transition probability is high; if they don't connect, a penalty is applied.
> - **Large gaps:** If timestamps show a GPS dropout over 10 seconds, the transition penalty is relaxed so the path doesn't break.
>
> **The results speak for themselves:** Across all 10 student-test trips, HMM decoding boosted average route continuity from 90.62% to **97.23%**, and average Top-1 accuracy improved from 86.54% to **89.66%**."

---

### Slide 9: Two-Laptop Architecture & Live Demo
**Speaker: Akula Sainandan Royal**

> "To meet the deployment requirements, we built a distributed two-laptop setup:
> - **Laptop 1 (Server):** Runs our FastAPI backend on port 8000. It hosts the trained models, scaler, road graphs, and server-side ground truth. Every inference request logs live to the terminal.
> - **Laptop 2 (Client):** Runs our Streamlit UI. The evaluator picks any of the 10 evaluation trips, selects a model, toggles HMM on or off, and clicks run.
>
> Laptop 2 sends the trip ID to Laptop 1 over the local network, and Laptop 1 computes candidate generation, model inference, and HMM decoding, then returns the predicted route and verified metrics. Ground truth stays on Laptop 1 and is never exposed to the client.
>
> The UI renders an interactive Folium map showing the blue GPS points, gray road network, green ground truth, and orange predicted route.
>
> We have the system running right now on both laptops. We invite the evaluators to test any trip on the dashboard and inspect the server logs. Thank you!"
