# PathSnap - Map Matching & Route Reconstruction

**Course:** UE24CS352A Machine Learning  
**College:** PES University, RR Campus  
**Team:** Hemanth V (PES1UG25CS817), Praveen (PES1UG25CS832), Ajay T Naik (PES1UG24CS552), Akula Sainandan Royal (PES1UG24CS553)

---

## What is PathSnap?

PathSnap takes noisy GPS pings from a moving vehicle and figures out which roads it actually drove on. GPS signals in cities are messy — they bounce off buildings, drop out in tunnels, and sometimes show the car on the wrong side of the road. We built a system that cleans up these errors using ML models and graph-based decoding.

The system works in two stages:
1. For each GPS point, we find the 5 closest road segments and use a trained classifier to pick the best match
2. We then run HMM Viterbi decoding on the full sequence to make sure the predicted route actually makes sense on the road network (no teleporting between disconnected roads)

---

## Team

| Name | SRN | Branch |
| :--- | :--- | :--- |
| Hemanth V | PES1UG25CS817 | CSE, PESU RR |
| Praveen | PES1UG25CS832 | CSE, PESU RR |
| Ajay T Naik | PES1UG24CS552 | CSE, PESU RR |
| Akula Sainandan Royal | PES1UG24CS553 | CSE, PESU RR |

---

## Deliverables

- **GitHub Repo:** This repository (all code, models, results)
- **Technical Report PDF:** `PathSnap_Technical_Report.pdf`
- **Presentation:** `PathSnap_Review_1_Presentation.pptx`
- **Integrity Audit:** `ACADEMIC_INTEGRITY_COMPLIANCE.md`

---

## Results

### 5-Model Comparison (tested on 10 student-test trips)

| Model | Family | CV F1 | Test Top-1 Acc | Test F1 | Log Loss | Continuity | Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | Linear | 0.7975 | 83.97% | 0.8185 | 0.2987 | 88.7% | 0.002 ms |
| Decision Tree | Tree | 0.8237 | 84.08% | 0.8406 | 0.2502 | 90.1% | 0.002 ms |
| Random Forest | Bagging | 0.8361 | **84.43%** | 0.8483 | 0.2249 | **90.6%** | 0.056 ms |
| LightGBM | Boosting | 0.8391 | 83.85% | 0.8447 | 0.2276 | **90.6%** | 0.039 ms |
| MLP | Neural Net | **0.8489** | 83.52% | **0.8866** | **0.1744** | 89.4% | 0.006 ms |

Random Forest gave us the best test accuracy and route continuity. MLP had the best F1 and log loss because its probability outputs are well-calibrated.

### Bayesian Tuning (Optuna, 25 trials on LightGBM)

| Variant | CV F1 | Test Acc | Test F1 | Log Loss |
| :--- | :---: | :---: | :---: | :---: |
| Default LightGBM | 0.8391 | 83.85% | 0.8447 | 0.2276 |
| Tuned LightGBM | **0.8590** | 82.52% | **0.8488** | 0.2280 |

CV score improved but test accuracy dipped a bit — slight overfit to CV folds. We kept both models for comparison.

### HMM Viterbi Decoding (average over 10 test trips)

| Method | Avg Top-1 Acc | Avg Continuity |
| :--- | :---: | :---: |
| Classifier only | 86.54% | 90.62% |
| Classifier + HMM | **89.66%** | **97.23%** |

The HMM bumps continuity from ~90% to ~97% by penalising jumps between disconnected road segments.

---

## How to Run

### Two-Laptop Setup (for evaluation)

Both laptops need to be on the same Wi-Fi.

**Laptop 1 (server):**
```bash
python api.py
```
Starts FastAPI on port 8000. Shows live request logs in the terminal.

**Laptop 2 (UI):**
```bash
streamlit run app.py
```
Enter Laptop 1's IP in the sidebar (e.g. `http://192.168.1.15:8000`), pick a trip, and hit run.

### Single Laptop (standalone)

```bash
# Train models (if not already done)
python train_and_evaluate.py

# Run Bayesian optimisation
python src/bayesian_opt.py

# Run HMM comparison
python evaluate_hmm.py

# Launch the dashboard
streamlit run app.py
```
Switch to "Local Standalone" mode in the sidebar.

---

## Folder Structure

```
├── src/
│   ├── data_loader.py          # reads .track, .nodes, .arcs, .route files
│   ├── candidate_generator.py  # KDTree spatial search + feature extraction
│   ├── models.py               # defines all 5 models
│   ├── hmm_decoder.py          # HMM Viterbi route reconstruction
│   └── bayesian_opt.py         # Optuna hyperparameter search
├── eda/
│   ├── eda_report.py           # dataset stats and plots
│   └── visualize_map.py        # Folium map generation
├── artifacts/                  # saved model .pkl files and scaler
├── outputs/                    # CSV results, comparison tables, plots
├── maps/                       # generated Folium HTML maps
├── api.py                      # FastAPI backend (Laptop 1)
├── app.py                      # Streamlit frontend (Laptop 2)
├── train_and_evaluate.py       # training + 3-fold CV + benchmark
├── evaluate_hmm.py             # HMM ablation study
└── requirements.txt
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Main packages: scikit-learn, lightgbm, streamlit, fastapi, uvicorn, folium, scipy, numpy, pandas, matplotlib, optuna, networkx
