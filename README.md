# PathSnap: Distributed Map-Matching & Route Reconstruction

[![Course](https://img.shields.io/badge/Course-UE24CS352A%20Machine%20Learning-blue.svg)](https://pes.edu)
[![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io)

PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations onto road network graphs.

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
├── train_and_evaluate.py       # Full training, 3-fold CV & 10 student-test benchmark pipeline
├── app.py                      # Interactive Streamlit Evaluator Dashboard
├── requirements.txt            # Project dependencies
└── FIRST_REVIEW_GUIDE.md       # Evaluator viva answers & methodology cheat-sheet
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone <your-repo-url>
cd HACKATHON
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis & Map Generation
```bash
python eda/eda_report.py
```
Outputs generated:
- `outputs/dataset_inventory.csv`
- `outputs/eda_analysis_plots.png`
- `maps/trip_train_00000001.html`
- `maps/trip_test_00000000.html`

### 3. Train & Benchmark All 5 Models
```bash
python train_and_evaluate.py
```
Trains and benchmarks:
1. **Logistic Regression** (Linear & Probabilistic)
2. **Decision Tree** (Tree-Based)
3. **Random Forest** (Ensemble Bagging)
4. **LightGBM** (Gradient Boosting)
5. **Multi-Layer Perceptron** (Neural Networks)

Saves comparison metrics to `outputs/model_comparison.csv` and plots to `outputs/model_comparison_plots.png`.

### 4. Launch Interactive Evaluator UI
```bash
streamlit run app.py
```
Provides:
- Real-time observation-level prediction and candidate evaluation.
- Live interactive Folium map visualization showing raw GPS, road network, reference route, and predicted path.
- Side-by-side benchmark comparison leaderboard.

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
Developed independently for UE24CS352A Machine Learning Hackathon. All experimental results, models, and code are completely reproducible.
