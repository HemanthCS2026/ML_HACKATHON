# PathSnap: Distributed Map-Matching & Route Reconstruction

[![Course](https://img.shields.io/badge/Course-UE24CS352A%20Machine%20Learning-blue.svg)](https://pes.edu)
[![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/Academic-PES%20University-red.svg)](https://pes.edu)

**PathSnap** is an end-to-end, distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations onto road network graphs. It integrates spatial KDTree candidate pruning, machine learning classification across five distinct model families, Bayesian hyperparameter optimisation via Optuna (TPE), sequential Hidden Markov Model (HMM) Viterbi decoding, a high-throughput FastAPI backend for Laptop 1, and an interactive Streamlit UI for Laptop 2.

---

## 👥 Project Team Members

| Name | Student Registration Number (SRN) | Department & Institution |
| :--- | :--- | :--- |
| **HEMANTH V** | **PES1UG25CS817** | CSE, PES University (RR Campus) |
| **PRAVEEN** | **PES1UG25CS832** | CSE, PES University (RR Campus) |
| **AJAY T NAIK** | **PES1UG24CS552** | CSE, PES University (RR Campus) |
| **AKULA SAINANDAN ROYAL** | **PES1UG24CS553** | CSE, PES University (RR Campus) |

---

## 📑 Official Deliverables

1. **Deliverable 1 (GitHub Repository):** Complete source code, preprocessing, models, hyperparameter search, HMM decoding, FastAPI backend, Streamlit UI, Folium maps, and test scripts.
2. **Deliverable 2 (Consolidated Technical PDF):** 📄 **[PathSnap_Technical_Report.pdf](./PathSnap_Technical_Report.pdf)** (7-page comprehensive report covering Sections 2.2 to 8 as specified in the hackathon guidelines).
3. **Academic Integrity Compliance:** 🛡️ **[ACADEMIC_INTEGRITY_COMPLIANCE.md](./ACADEMIC_INTEGRITY_COMPLIANCE.md)** (Full audit verifying 100% compliance with the PES Assessment & Plagiarism Policy, including zero data leakage, server-side truth, and no hardcoded outputs).
4. **Presentation Deck & Script:** 📁 **[PathSnap_Review_1_Presentation.pptx](./PathSnap_Review_1_Presentation.pptx)** and 🎤 **[PRESENTATION_SPEAKER_SCRIPT.md](./PRESENTATION_SPEAKER_SCRIPT.md)**.

---

## 🏆 Benchmark Results Summary

### 1. Official 5-Model Leaderboard (Student-Test Set)
Evaluated across all 10 student-test trips using 3-fold cross-validation on 80 training trips:

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Student-Test Top-1 Acc | Student-Test Macro F1 | Log Loss | Route Continuity | Inference Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | C=1.0, balanced | 0.7975 | 83.97% | 0.8185 | 0.2987 | 88.7% | 0.002 ms/obs |
| **Decision Tree** | Tree-Based | max_depth=10, min_samples=20 | 0.8237 | 84.08% | 0.8406 | 0.2502 | 90.1% | 0.002 ms/obs |
| **Random Forest** | Ensemble (Bagging) | 100 trees, depth=12 | 0.8361 | **84.43%** | 0.8483 | 0.2249 | **90.6%** | 0.056 ms/obs |
| **LightGBM** | Gradient Boosting | 150 trees, lr=0.05, leaves=31 | 0.8391 | 83.85% | 0.8447 | 0.2276 | **90.6%** | **0.039 ms/obs** |
| **Multi-Layer Perceptron** | Neural Networks | layers=(64,32), ReLU, Adam | **0.8489** | 83.52% | **0.8866** | **0.1744** | 89.4% | 0.006 ms/obs |

---

### 2. Bayesian Hyperparameter Optimisation (Optuna TPE)
Optimised LightGBM over 25 trials on 3-fold CV Macro F1 (`outputs/bayesian_optimization_trials.csv`):

| Model Variant | CV Macro F1 | Student-Test Top-1 Acc | Student-Test Macro F1 | Log Loss | Route Continuity | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned LightGBM (Default)** | 0.8391 | 83.85% | 0.8447 | 0.2276 | 90.6% | 0.050 ms/obs |
| **Tuned LightGBM (Optuna TPE)** | **0.8590** | 82.52% | **0.8488** | 0.2280 | 89.4% | **0.047 ms/obs** |

* **Best Parameters:** `n_estimators=225`, `learning_rate=0.1721`, `num_leaves=50`, `max_depth=11`, `min_child_samples=11`, `subsample=0.7082`, `colsample_bytree=0.8911`.

---

### 3. HMM + Viterbi Route Reconstruction Ablation
Comparing Classifier-Only vs Classifier + HMM Viterbi on all 10 student-test trips (`outputs/hmm_comparison.csv`):

| Trip ID | Observations | Miss Rate | Classifier Top-1 Acc | Classifier Continuity | HMM Top-1 Acc | HMM Continuity | Continuity Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 2503 | 1.04% | 94.71% | 96.08% | **96.08%** | **99.24%** | **+3.16%** |
| `00000007` | 1354 | 0.22% | 90.82% | 99.33% | 86.45% | **100.00%** | **+0.67%** |
| `00000009` | 1049 | 3.24% | 87.68% | 85.69% | **92.71%** | **93.80%** | **+8.11%** |
| `00000040` | 2995 | 0.57% | 93.18% | 95.62% | **97.11%** | **99.30%** | **+3.67%** |
| `00000052` | 262 | 1.15% | 94.21% | 91.19% | **99.61%** | **95.79%** | **+4.60%** |
| `00000056` | 3958 | 2.80% | 81.15% | 86.05% | 79.83% | **96.03%** | **+9.98%** |
| `00000065` | 768 | 2.99% | 78.52% | 88.66% | **79.73%** | **97.65%** | **+9.00%** |
| `00000076` | 1486 | 5.85% | 88.42% | 85.99% | **90.92%** | **94.41%** | **+8.42%** |
| `00000077` | 3911 | 0.95% | 69.82% | 91.23% | **82.32%** | **98.70%** | **+7.47%** |
| `00000091` | 874 | 1.72% | 86.85% | 86.37% | **91.85%** | **97.37%** | **+11.00%** |
| **AVERAGE** | **1916** | **2.05%** | **86.54%** | **90.62%** | **89.66%** | **97.23%** | **+6.61%** |

---

## 💻 Two-Laptop Deployment Architecture

The solution fulfills the hackathon mandate for distributed deployment across two separate machines connected over LAN:

```
┌────────────────────────────────────────────────────────┐          ┌────────────────────────────────────────────────────────┐
│                   LAPTOP 1 (SERVER)                    │          │                   LAPTOP 2 (CLIENT)                    │
│                                                        │          │                                                        │
│  • Component: FastAPI Backend (`api.py`)               │  HTTP    │  • Component: Streamlit Evaluator UI (`app.py`)        │
│  • Port: 8000 (LAN bind: 0.0.0.0:8000)                 │ ◄──────► │  • LAN Connection: http://<Laptop-1-IP>:8000           │
│  • Preloaded Models & Scalers in `artifacts/`          │  JSON    │  • 10 Reserved Evaluator Trip Options                  │
│  • Graph Engine & HMM Viterbi Sequential Decoder       │          │  • Interactive Folium Multi-Layer Map View             │
│  • Server-Side Ground Truth Evaluation Engine          │          │  • Real-Time Accuracy & Continuity Metrics             │
│  • Real-Time Terminal Audit Logging                    │          │  • Zero Local Ground-Truth Storage (Leakage Free)      │
└────────────────────────────────────────────────────────┘          └────────────────────────────────────────────────────────┘
```

### Execution Steps for Evaluation:
1. **Connect both laptops to the same Wi-Fi / Hotspot.**
2. **On Laptop 1 (Model and API Server):**
   ```bash
   python api.py
   ```
   *Terminal displays live startup telemetry and waits for incoming inference requests.*
3. **On Laptop 2 (User Interface):**
   ```bash
   streamlit run app.py
   ```
   *In the left sidebar, enter Laptop 1's IP address (e.g. `http://192.168.1.15:8000`). Select any of the 10 evaluation trips, pick a model, toggle HMM on/off, and click **"Run Map-Matching & Route Reconstruction"**.*
   *Evaluators will observe live request logs streaming on Laptop 1's CMD while interactive maps and verified metrics display on Laptop 2.*

---

## 📁 Repository Directory Structure

```
PathSnap/
├── src/
│   ├── data_loader.py          # Parsers for .track, .nodes, .arcs, .route & metric projection
│   ├── candidate_generator.py  # Spatial KDTree pruning & 8-feature engineering
│   ├── models.py               # 5 distinct model definitions & evaluation suite
│   ├── hmm_decoder.py          # HMM & Viterbi sequential decoder with gap handling
│   └── bayesian_opt.py         # Optuna TPE Bayesian hyperparameter optimization
├── eda/
│   ├── eda_report.py           # Dataset statistics, distribution plots & quality audit
│   └── visualize_map.py        # Interactive Folium map generation (GPS, Network, Routes)
├── artifacts/                  # Trained model weights & preprocessors (.pkl)
│   ├── model_random_forest.pkl
│   ├── model_lightgbm.pkl
│   ├── model_tuned_lightgbm.pkl
│   ├── model_multi_layer_perceptron.pkl
│   ├── model_decision_tree.pkl
│   ├── model_logistic_regression.pkl
│   └── scaler.pkl
├── outputs/                    # Empirical results, trial logs & plots
│   ├── dataset_inventory.csv
│   ├── model_comparison.csv
│   ├── model_comparison.md
│   ├── model_comparison_plots.png
│   ├── bayesian_optimization_trials.csv
│   ├── bayesian_optimization_history.png
│   ├── tuned_vs_untuned_comparison.csv
│   ├── tuned_vs_untuned_comparison.md
│   ├── hmm_comparison.csv
│   ├── hmm_comparison.md
│   └── hmm_comparison_plot.png
├── maps/                       # Generated interactive Folium HTML maps
├── api.py                      # FastAPI server for Laptop 1
├── app.py                      # Streamlit evaluator interface for Laptop 2
├── train_and_evaluate.py       # Full training, 3-fold CV & benchmark pipeline
├── evaluate_hmm.py             # 10-trip HMM ablation benchmarking script
├── generate_report_pdf.py      # Automated compiler for Deliverable 2 PDF
├── PathSnap_Technical_Report.pdf # Deliverable 2: Consolidated Technical Report PDF
├── ACADEMIC_INTEGRITY_COMPLIANCE.md # Verification against plagiarism & integrity policy
├── FIRST_REVIEW_GUIDE.md       # Evaluator viva answers & quick demonstration cheat-sheet
├── PRESENTATION_SPEAKER_SCRIPT.md # Word-for-word presentation speech
├── PathSnap_Review_1_Presentation.pptx # Review 1 PowerPoint slide deck
└── requirements.txt            # System dependencies
```

---

## 🛡️ Academic Integrity & Anti-Plagiarism Compliance

This project complies strictly with the **PES University UE24CS352A Assessment and Academic Integrity Policy**:
* **Independent Implementation:** All candidate generation, metric projection, HMM decoding, and API logic were written from first principles.
* **No Code Sharing:** Zero source code, model weights, or hyperparameter configs were shared between teams.
* **No Fabricated Data:** Every single metric, table, and graph presented in the report is directly generated from executing the Python scripts on the dataset.
* **Zero Data Leakage:** The 10 evaluator trips are completely untouched. Evaluator ground-truth remains strictly server-side.
* **No Hardcoding:** All predictions are produced through real-time mathematical inference.
* Detailed compliance records and viva answers are documented in **[ACADEMIC_INTEGRITY_COMPLIANCE.md](./ACADEMIC_INTEGRITY_COMPLIANCE.md)**.
