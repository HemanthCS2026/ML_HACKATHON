import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

def get_model_definitions():
    """
    Returns a dictionary of 5 models from 5 distinct algorithm families:
    1. Linear & Probabilistic: Logistic Regression
    2. Tree-based: Decision Tree Classifier
    3. Ensemble (Bagging): Random Forest Classifier
    4. Gradient Boosting: LightGBM Classifier
    5. Neural Networks: Multi-Layer Perceptron (MLP)
    """
    models = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "key_config": "max_iter=1000, class_weight=balanced, C=1.0"
        },
        "Decision Tree": {
            "family": "Tree-Based",
            "model": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=42),
            "key_config": "max_depth=10, min_samples_leaf=20, criterion=gini"
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
            "key_config": "n_estimators=100, max_depth=12, class_weight=balanced"
        },
        "LightGBM": {
            "family": "Gradient Boosting",
            "model": LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=42, verbose=-1),
            "key_config": "n_estimators=150, lr=0.05, num_leaves=31, balanced"
        },
        "Multi-Layer Perceptron": {
            "family": "Neural Networks",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, activation="relu", random_state=42, early_stopping=True),
            "key_config": "layers=(64, 32), act=relu, solver=adam, early_stopping"
        }
    }
    return models

def evaluate_predictions(y_true, y_prob, top_k=5):
    """
    Evaluates candidate predictions at both candidate-level and observation-level.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Clip probabilities for numerical stability in log loss
    eps = 1e-7
    probs_clipped = np.clip(y_prob, eps, 1 - eps)
    loss = log_loss(y_true, probs_clipped)
    
    # Observation-level Top-1 accuracy
    num_obs = len(y_true) // top_k
    correct_obs = 0
    valid_obs = 0
    for obs_idx in range(num_obs):
        start = obs_idx * top_k
        end = start + top_k
        obs_probs = y_prob[start:end]
        obs_true = y_true[start:end]
        
        # Check if ground truth was present in candidate set
        if np.sum(obs_true) > 0:
            valid_obs += 1
            best_cand = np.argmax(obs_probs)
            if obs_true[best_cand] == 1:
                correct_obs += 1
                
    top1_acc = (correct_obs / valid_obs) if valid_obs > 0 else 0.0
    return {
        "macro_f1": float(macro_f1),
        "log_loss": float(loss),
        "top1_acc": float(top1_acc),
        "valid_obs": int(valid_obs),
        "correct_obs": int(correct_obs)
    }
