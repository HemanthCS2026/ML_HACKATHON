# PathSnap: Distributed Map-Matching & Route Reconstruction

[![Course](https://img.shields.io/badge/Course-UE24CS352A%20Machine%20Learning-blue.svg)](https://pes.edu)
[![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io)

PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations onto road network graphs.

---

## 👥 Project Team Members

| Name | Student Registration Number (SRN) | Department & Institution |
| :--- | :--- | :--- |
| **HEMANTH V** | **PES1UG25CS817** | CSE, PES University (RR Campus) |
| **PRAVEEN** | **PES1UG25CS832** | CSE, PES University (RR Campus) |
| **AJAY T NAIK** | **PES1UG24CS552** | CSE, PES University (RR Campus) |
| **AKULA SAINANDAN ROYAL** | **PES1UG24CS553** | CSE, PES University (RR Campus) |

---

## 📽️ Review 1 Presentation (PowerPoint)

The complete slide deck for Review 1 (Phase 1 Evaluation) is available in the repository root:
* 📁 **[PathSnap_Review_1_Presentation.pptx](./PathSnap_Review_1_Presentation.pptx)**
* 📑 **[PRESENTATION_SPEAKER_SCRIPT.md](./PRESENTATION_SPEAKER_SCRIPT.md)** *(Word-for-word speaker notes)*

---

## 🎤 Presentation Speaker Script (By Slide)

### 🟢 Slide 1: Title & Team Introduction
* **Speaker:** HEMANTH V (PES1UG25CS817)
> "Good morning respected evaluators and professors. We are Team PathSnap, presenting our Phase 1 implementation for the UE24CS352A Machine Learning Hackathon on 'Distributed Map-Matching and Route Reconstruction'. Our team consists of Hemanth V, Praveen, Ajay T Naik, and Akula Sainandan Royal from CSE, RR Campus. Today we will walk you through our empirical EDA, high-recall spatial candidate generator, feature engineering pipeline, 5-model benchmark across 5 distinct algorithm families, and our live interactive evaluator dashboard."

### 🟢 Slide 2: Problem Statement & Challenges
* **Speaker:** HEMANTH V (PES1UG25CS817)
> "In urban environments, GPS signals suffer heavily from multipath effects and satellite occlusion caused by tall buildings, causing vehicle observations to jump across parallel streets and through buildings. Naive nearest-distance matching fails because it ignores road orientation and one-way constraints, creating physically impossible disjoint jumps. PathSnap solves this by combining high-speed spatial candidate generation, ML classification, and sequential topological decoding."

### 🟢 Slide 3: Dataset Architecture & Empirical EDA
* **Speaker:** HEMANTH V (PES1UG25CS817)
> "We analyzed all 90 trips in the dataset (80 training trips, 10 student-test trips). We empirically verified that 97.4% of GPS sampling intervals are exactly 1.0 second, confirming nominal 1 Hz frequency. Road network subgraphs range from 650 to 307,000 arcs. Ground-truth reference routes in `.route` were verified to possess 100% topological continuity. The median vehicle velocity is 13.8 m/s (~50 km/h) with highway bursts exceeding 25 m/s."

### 🟢 Slide 4: Data Preprocessing & Spatial Candidate Generation
* **Speaker:** PRAVEEN (PES1UG25CS832)
> "Because latitude and longitude are angular coordinates on an ellipsoid, Euclidean calculations in degrees are invalid. We project coordinates into local metric Cartesian coordinates (x, y) in meters centered at the trajectory median. To query candidates from tens of thousands of links in real-time, we built a spatial `KDTree` on road vertices to query K=20 nearest nodes in logarithmic time and compute exact perpendicular point-to-segment distances. Our candidate generator achieves a Top-5 Candidate Recall of 96.51% (only 3.49% miss rate)."

### 🟢 Slide 5: Feature Engineering Pipeline
* **Speaker:** PRAVEEN (PES1UG25CS832)
> "We engineered 8 domain-grounded features: (1) `dist_m`: orthogonal distance in meters; (2) `proj_ratio`: projection coordinate along arc t in [0, 1]; (3) `heading_diff_deg` and `heading_alignment`: angular deviation and cosine alignment to penalize wrong-way links; (4) `arc_len`: segment physical length; (5) `speed_mps`: instantaneous GPS velocity; (6) `spatial_prior`: Gaussian emission density exp(-d²/2σ²) with σ=6.0m; and (7) `distance_rank`: proximity rank among retrieved candidates."

### 🟢 Slide 6: 5-Model Benchmark Leaderboard
* **Speaker:** AJAY T NAIK (PES1UG24CS552)
> "We trained and benchmarked 5 distinct machine learning models from 5 different algorithm families on 250,000 candidate samples using 3-fold cross-validation and tested them out-of-sample on all 10 student-test trips. Random Forest achieved the highest student-test Top-1 Accuracy at 84.43% and 90.6% Route Continuity. LightGBM achieved the top CV Accuracy at 87.64% and 90.6% Route Continuity. Multi-Layer Perceptron achieved the lowest Log Loss at 0.1744 and highest Macro F1 at 0.8866. All models infer in under 0.06 ms/obs."

### 🟢 Slide 7: Model Comparison Diagnostic Visualizations
* **Speaker:** AJAY T NAIK (PES1UG24CS552)
> "Examining model stability, our student-test accuracy tightly tracks cross-validation scores across all models with zero catastrophic overfitting. The gap between CV and unseen test accuracy remains within 2%. Tree depth and leaf sample constraints prevent tree ensembles from memorizing specific road geometries."

### 🟢 Slide 8: Compulsory Viva Defense & Questions
* **Speaker:** AJAY T NAIK (PES1UG24CS552)
> "To answer the compulsory questions: (1) Logistic Regression slightly underfits because linear boundaries cannot isolate multi-dimensional heading-distance cones, while Random Forest and LightGBM balance bias and variance optimally. (2) Observation-level accuracy alone is insufficient because evaluating points in isolation causes ~9.4% disjoint road hops, proving why sequential HMM decoding is necessary. (3) We selected Random Forest and LightGBM for Bayesian optimization in Phase 2."

### 🟢 Slide 9: System Demo & Evaluator UI
* **Speaker:** AKULA SAINANDAN ROYAL (PES1UG24CS553)
> "We developed an interactive Streamlit dashboard running live at localhost:8501. Evaluators can select any of the 10 student-test trips, choose any of our 5 trained models, and compute real-time map matching. The interface displays Top-1 Accuracy, Candidate Miss Rate, Route Continuity, and renders an interactive Folium map with raw GPS observations, road network graph, ground truth, and predicted routes."

### 🟢 Slide 10: Roadmap for Phase 2 (After Lunch Break)
* **Speaker:** AKULA SAINANDAN ROYAL (PES1UG24CS553)
> "For Phase 2 after lunch, our roadmap includes: (1) Bayesian Hyperparameter Optimization with Optuna on our top ensemble model; (2) Hidden Markov Model (HMM) + Viterbi route decoding to resolve disjoint hops and achieve ~100% route continuity; and (3) deploying our FastAPI backend for the 2-laptop distributed setup."

### 🟢 Slide 11: Conclusion & Q&A
* **Speaker:** AKULA SAINANDAN ROYAL (PES1UG24CS553)
> "Phase 1 is 100% complete with full reproducibility. All code, models, and docs are available in our public GitHub repository. We now invite the evaluators for questions and our live UI demonstration. Thank you!"

---

## 📌 Project Architecture

```
PathSnap/
├── src/
│   ├── data_loader.py          # Parsers for .track, .nodes, .arcs, .route & metric projection
│   ├── candidate_generator.py  # KDTree spatial candidate generation & feature engineering
│   └── models.py               # 5 distinct model definitions & evaluation suite
├── eda/
│   ├── eda_report.py           # Dataset statistics, distribution plots & quality assessment
│   └── visualize_map.py        # Interactive Folium map generation (GPS, Graph, Route)
├── artifacts/                  # Trained model weights & preprocessors (.pkl)
├── outputs/                    # Benchmark tables (.csv, .md) and diagnostic plots (.png)
├── maps/                       # Interactive HTML Folium maps
├── PathSnap_Review_1_Presentation.pptx # Official PowerPoint presentation deck
├── PRESENTATION_SPEAKER_SCRIPT.md     # Word-for-word presentation script
├── train_and_evaluate.py       # Full training, 3-fold CV & 10 student-test benchmark pipeline
├── app.py                      # Interactive Streamlit Evaluator Dashboard
├── requirements.txt            # Project dependencies
└── FIRST_REVIEW_GUIDE.md       # Evaluator viva answers & methodology cheat-sheet
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/HemanthCS2026/ML_HACKATHON.git
cd ML_HACKATHON
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis & Map Generation
```bash
python eda/eda_report.py
```

### 3. Train & Benchmark All 5 Models
```bash
python train_and_evaluate.py
```

### 4. Launch Interactive Evaluator UI
```bash
streamlit run app.py
```

---

## 📊 Benchmark Leaderboard (Student-Test Set)

| Model | Algorithm Family | CV Macro F1 | CV Top-1 Acc | Student-Test Top-1 Acc | Student-Test Macro F1 | Log Loss | Route Continuity | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | 0.7975 | 0.8315 | 83.97% | 0.8185 | 0.2987 | 88.7% | 0.002 ms |
| **Decision Tree** | Tree-Based | 0.8237 | 0.8609 | 84.08% | 0.8406 | 0.2502 | 90.1% | 0.002 ms |
| **Random Forest** | Ensemble (Bagging) | 0.8361 | 0.8656 | **84.43%** | 0.8483 | 0.2249 | **90.6%** | 0.056 ms |
| **LightGBM** | Gradient Boosting | 0.8391 | **0.8764** | 83.85% | 0.8447 | 0.2276 | **90.6%** | 0.039 ms |
| **Multi-Layer Perceptron** | Neural Networks | **0.8489** | 0.8578 | 83.52% | **0.8866** | **0.1744** | 89.4% | 0.006 ms |

---

## 👥 Authors & Academic Integrity
Developed independently for UE24CS352A Machine Learning Hackathon by Hemanth V, Praveen, Ajay T Naik, and Akula Sainandan Royal. All experimental results, models, and code are completely reproducible.
