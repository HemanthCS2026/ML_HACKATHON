# PathSnap: 5-Model Benchmark Comparison Table

| Model | Algorithm Family | Key Configuration | CV Macro F1 | CV Top-1 Accuracy | Student-Test Top-1 Accuracy | Student-Test Macro F1 | Log Loss | Route Continuity | Inference Latency (ms/obs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Linear & Probabilistic | max_iter=1000, class_weight=balanced, C=1.0 | 0.7975 | 0.8315 | 0.8397 | 0.8185 | 0.2987 | 88.7% | 0.002 |
| Decision Tree | Tree-Based | max_depth=10, min_samples_leaf=20, criterion=gini | 0.8237 | 0.8609 | 0.8408 | 0.8406 | 0.2502 | 90.1% | 0.002 |
| Random Forest | Ensemble (Bagging) | n_estimators=100, max_depth=12, class_weight=balanced | 0.8361 | 0.8656 | 0.8443 | 0.8483 | 0.2249 | 90.6% | 0.056 |
| LightGBM | Gradient Boosting | n_estimators=150, lr=0.05, num_leaves=31, balanced | 0.8391 | 0.8764 | 0.8385 | 0.8447 | 0.2276 | 90.6% | 0.039 |
| Multi-Layer Perceptron | Neural Networks | layers=(64, 32), act=relu, solver=adam, early_stopping | 0.8489 | 0.8578 | 0.8352 | 0.8866 | 0.1744 | 89.4% | 0.006 |

