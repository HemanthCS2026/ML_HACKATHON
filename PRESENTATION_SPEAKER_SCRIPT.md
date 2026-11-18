# PathSnap: Presentation Speaker Script (Phase 1 Review)

**Event:** UE24CS352A Machine Learning Hackathon  
**Institution:** Department of Computer Science & Engineering, PES University (RR Campus)  
**Project:** PathSnap: Distributed Map-Matching & Route Reconstruction  
**Phase:** Review 1 (Phase 1 Evaluation)

---

## 👥 Team Members & Speaking Roles
* **Speaker 1 (HEMANTH V - PES1UG25CS817):** Slides 1 to 3 *(Introduction, Problem Statement, Dataset & EDA)*
* **Speaker 2 (PRAVEEN - PES1UG25CS832):** Slides 4 & 5 *(Preprocessing, Spatial Candidate Generation, Feature Engineering)*
* **Speaker 3 (AJAY T NAIK - PES1UG24CS552):** Slides 6 to 8 *(5-Model Benchmark, Visual Diagnostics, Viva Technical Defense)*
* **Speaker 4 (AKULA SAINANDAN ROYAL - PES1UG24CS553):** Slides 9 to 11 *(Live UI Demo, Phase 2 Roadmap, Conclusion & Q&A)*

---

## 🎤 Word-for-Word Slide Scripts

### 🟢 SLIDE 1: Title & Team Introduction
* **Speaker:** HEMANTH V (PES1UG25CS817)
* **On Screen:** Title Slide with PES University details and 4 team member cards.

> "Good morning respected evaluators and professors. We are Team PathSnap, presenting our Phase 1 implementation for the UE24CS352A Machine Learning Hackathon on **'Distributed Map-Matching and Route Reconstruction'**."
>
> "Our team consists of: Hemanth V, Praveen, Ajay T Naik, and Akula Sainandan Royal from the Department of Computer Science and Engineering, RR Campus."
>
> "Today, we will walk you through our complete Phase 1 milestone—covering our empirical exploratory data analysis, our high-recall spatial candidate generator, our feature engineering pipeline, our 5-model benchmark evaluation across 5 distinct algorithm families, and our live interactive evaluator dashboard. I will now explain the core problem we are tackling."

---

### 🟢 SLIDE 2: Problem Statement & Real-World Challenges
* **Speaker:** HEMANTH V (PES1UG25CS817)
* **On Screen:** 3 structured cards detailing GPS noise, naive matching pitfalls, and PathSnap architecture.

> "To understand the core problem: In real-world urban environments, GPS tracking signals suffer heavily from multipath effects, satellite occlusion, and urban canyon reflections caused by high-rise buildings. As a result, raw vehicle GPS observations jump across parallel streets, cut through buildings, and exhibit intermittent dropouts."
>
> "If we simply use naive nearest-distance matching, the system frequently snaps the vehicle onto an adjacent service road or even an opposite-direction one-way street. Furthermore, independent point-by-point snapping produces physically impossible 'teleportation' jumps that break the road network's graph topology."
>
> "PathSnap solves this through a rigorous two-stage architecture: First, high-speed spatial indexing and supervised machine learning classify the most probable candidate road segment for each observation. Second, sequential hidden Markov models and Viterbi decoding enforce network graph continuity to reconstruct the true traversed trajectory."

---

### 🟢 SLIDE 3: Dataset Architecture & Empirical EDA
* **Speaker:** HEMANTH V (PES1UG25CS817)
* **On Screen:** Dataset structure details on the left, high-resolution 6-panel distribution plots on the right.

> "Before writing any model code, we conducted an empirical exploratory data analysis across all 90 provided trips—consisting of 80 training trips and 10 student-test trips."
>
> "Each trip directory provides four core files: `.track` containing longitude, latitude, and timestamps; `.nodes` providing road vertices; `.arcs` defining directed network edges; and `.route` containing the chronological sequence of ground-truth traversed arc IDs."
>
> "From our empirical distributions on screen:
> 1. We verified that **97.4% of GPS sampling intervals are exactly 1.0 second**, validating the nominal 1 Hz sampling frequency.
> 2. The road network subgraphs are localized per trip, varying from small networks of 650 links to large regional subgraphs of over 300,000 links.
> 3. Reference routes in `.route` were verified to possess **100% topological connectivity**.
> 4. The median vehicle velocity is 13.8 meters per second—approximately 50 km/h—with highway bursts exceeding 25 m/s."
>
> "I will now pass the floor to Praveen to explain how we preprocessed this data and generated spatial candidates."

---

### 🟢 SLIDE 4: Data Preprocessing & Spatial Candidate Generation
* **Speaker:** PRAVEEN (PES1UG25CS832)
* **On Screen:** 3 chronological cards detailing Geodesic Projection, KD-Tree Search, and Candidate Recall.

> "Thank you, Hemanth. Moving to our data preprocessing and candidate generation pipeline."
>
> "First, because latitude and longitude are angular coordinates on an ellipsoid, computing Euclidean distance in degrees introduces severe distortion. We project all coordinates into local metric Cartesian coordinates $(x, y)$ in meters centered at the trajectory median using geodesic scaling factors."
>
> "Second, scanning all 48,000 arcs for every GPS point would require billions of operations. To achieve real-time performance, we construct a spatial `KDTree` on all road network nodes. For each GPS point, we query the $K=20$ nearest vertices in logarithmic time, retrieve their incident directed edges, and compute exact perpendicular point-to-segment distances."
>
> "Third, we rank these candidate segments by physical proximity and retain the top-5 candidates per observation. On evaluating this candidate generator across the dataset, our **Candidate Miss Rate is only 3.49%**—meaning our spatial indexing successfully captures the true road segment in **96.51% of cases** in the top-5 candidate set."

---

### 🟢 SLIDE 5: Feature Engineering Pipeline
* **Speaker:** PRAVEEN (PES1UG25CS832)
* **On Screen:** 8 structured cards detailing the physical and behavioral justification of each engineered feature.

> "To allow machine learning models to effectively discriminate between candidate road segments, we engineered eight physically and behaviorally grounded features:
>
> 1. **`dist_m`:** The perpendicular Euclidean distance in meters from the GPS observation to the road segment.
> 2. **`proj_ratio`:** The projection parameter $t \in [0, 1]$, which indicates where along the segment the vehicle falls and penalizes points projecting beyond link intersections.
> 3. **`heading_diff_deg` and `heading_alignment`:** We compute the vehicle's heading vector from consecutive GPS points and compare it with the candidate arc's direction vector. The cosine alignment $\cos(\Delta \theta)$ smoothly rewards forward travel (+1) and heavily penalizes wrong-way driving on dual carriageways (-1).
> 4. **`arc_len`:** The segment length, which normalizes capture basins between short turning lanes and long arterial roads.
> 5. **`speed_mps`:** Instantaneous vehicle velocity, distinguishing stationary stops from high-speed highway cruising.
> 6. **`spatial_prior`:** A Gaussian emission density $\exp(-d^2 / 2\sigma^2)$ with $\sigma=6.0$ meters, reflecting physical GPS sensor noise characteristics.
> 7. **`distance_rank`:** The ordinal rank among the five retrieved local candidates."
>
> "I will now hand over to Ajay to present our 5-model benchmark and experimental findings."

---

### 🟢 SLIDE 6: 5-Model Benchmark Leaderboard & Comparison
* **Speaker:** AJAY T NAIK (PES1UG24CS552)
* **On Screen:** Official 5-Model benchmark leaderboard table with highlighted metrics and bullet findings.

> "Thank you, Praveen. In accordance with the hackathon guidelines, we trained and evaluated five distinct machine learning models from five completely different algorithm families on 250,000 candidate observations using 3-fold cross-validation and tested them out-of-sample on all 10 student-test trips."
>
> "Looking at our leaderboard on screen:
> 1. **Logistic Regression (Linear & Probabilistic):** Achieved an 83.15% CV accuracy and 83.97% test accuracy with 88.7% route continuity.
> 2. **Decision Tree (Tree-Based):** Improved CV accuracy to 86.09% and reached 84.08% test accuracy with 90.1% continuity.
> 3. **Random Forest (Ensemble Bagging):** Achieved our highest student-test Top-1 Accuracy at **84.43%**, an out-of-sample Macro F1 of **0.8483**, and the highest Route Continuity at **90.6%**.
> 4. **LightGBM (Gradient Boosting):** Achieved the highest Cross-Validation Top-1 Accuracy at **87.64%** and tied Random Forest with **90.6% Route Continuity**.
> 5. **Multi-Layer Perceptron (Neural Networks):** Achieved our lowest Log Loss at **0.1744** and highest Macro F1 at **0.8866**, demonstrating superior probabilistic calibration."
>
> "All five models infer in under 0.06 milliseconds per observation, proving their capability for real-time edge vehicle tracking."

---

### 🟢 SLIDE 7: Model Comparison Diagnostic Visualizations
* **Speaker:** AJAY T NAIK (PES1UG24CS552)
* **On Screen:** Side-by-side diagnostic bar charts and generalization stability cards.

> "On this slide, we examine model generalization and stability across our cross-validation folds and student-test evaluation."
>
> "As seen in the left plot, the Student-Test Top-1 Accuracy and Macro F1-score closely match the cross-validation scores for every single model family. There is zero catastrophic overfitting. The gap between cross-validation and unseen student-test accuracy remains bounded within 2%."
>
> "For our tree-based and ensemble models, we controlled variance by capping tree depth and setting a minimum leaf sample threshold, preventing the trees from memorizing specific road geometries. Next, let us address the compulsory viva questions."

---

### 🟢 SLIDE 8: Compulsory Viva Defense & Questions
* **Speaker:** AJAY T NAIK (PES1UG24CS552)
* **On Screen:** 3 Viva question cards: Bias-Variance, Observation Accuracy vs Continuity, and Model Selection.

> "We have prepared rigorous answers for the compulsory evaluation questions:
>
> "First, regarding **Bias-Variance and Overfitting**: Logistic Regression exhibits slight underfitting because linear decision boundaries cannot isolate conical multi-dimensional combinations of heading alignment and distance. Meanwhile, Random Forest and LightGBM balance bias and variance optimally, while our Multi-Layer Perceptron uses validation-based early stopping to prevent over-parameterization."
>
> "Second, regarding **Observation-Level Accuracy versus Route Continuity**: Why is 84.4% accuracy not enough on its own? Because independent candidate classification evaluates each GPS point in isolation. If two consecutive points are snapped to nearby road segments that do not share a connecting node in the graph, the vehicle appears to 'teleport'. Our baseline models already achieve 90.6% continuity, but the remaining 9.4% gap proves why sequential HMM decoding is mathematically necessary."
>
> "Third, for **Model Selection**: We have selected Random Forest and LightGBM as our primary candidates for Bayesian optimization in Phase 2 due to their superior accuracy, 90.6% continuity, and low latency."
>
> "I now invite Akula Sainandan Royal to demonstrate our live evaluator system and outline Phase 2."

---

### 🟢 SLIDE 9: System Demo & Evaluator UI
* **Speaker:** AKULA SAINANDAN ROYAL (PES1UG24CS553)
* **On Screen:** Streamlit UI overview, feature highlights, and live execution URLs.

> "Thank you, Ajay. To make our system fully demonstrable to the evaluators, we developed a production-ready interactive dashboard using Streamlit, which is running live at `localhost:8501`."
>
> "The dashboard allows evaluators to:
> 1. Select any of the 10 reserved student-test trips from an interactive dropdown.
> 2. Switch between any of our 5 trained machine learning models dynamically.
> 3. Click 'Run Map-Matching Prediction' to extract candidates, scale features, and compute inference in real time.
> 4. Inspect an interactive Folium map rendering four distinct layers: raw blue GPS observations, the gray road network graph, the green ground-truth route, and the orange dashed predicted trajectory."
>
> "Evaluators can also connect directly over the local network via `10.20.203.229:8501`, which directly prepares our setup for the two-laptop deployment requirement."

---

### 🟢 SLIDE 10: Roadmap for Phase 2 (After Lunch Break)
* **Speaker:** AKULA SAINANDAN ROYAL (PES1UG24CS553)
* **On Screen:** 3 chronological milestone cards for Bayesian Tuning, HMM/Viterbi, and Distributed Deployment.

> "Having achieved all Phase 1 milestones, our roadmap for Phase 2 immediately after the lunch break includes three critical technical tasks:
>
> 1. **Bayesian Hyperparameter Optimization using Optuna:** We will optimize tree depth, learning rate, and leaf configurations for our chosen ensemble models using group-aware cross-validation, logging the trial histories.
> 2. **Hidden Markov Model & Viterbi Decoding:** We will formulate road segments as hidden states, using classifier posterior probabilities as emission likelihoods and network shortest-path distances as transition probabilities. The Viterbi recurrence will decode the globally optimal sequence, eliminating disjoint jumps and pushing route continuity toward 100%.
> 3. **FastAPI Distributed Deployment:** We will deploy our dedicated FastAPI backend on Laptop 1 and connect Laptop 2's evaluator interface across the network, culminating in the final technical report."

---

### 🟢 SLIDE 11: Conclusion & Q&A
* **Speaker:** AKULA SAINANDAN ROYAL (PES1UG24CS553)
* **On Screen:** Thank You slide with Phase 1 summary, GitHub repo link, and live demo invitation.

> "To conclude: For Phase 1, our team has delivered a complete, fully reproducible pipeline—from empirical data exploration and 96.5% recall candidate indexing to benchmarking five distinct model families and launching a live interactive evaluator dashboard. All code, model weights, and documentation are committed and pushed to our public GitHub repository."
>
> "We would now like to invite the evaluators to test our live dashboard and are happy to answer any questions. Thank you!"
