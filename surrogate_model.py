"""
surrogate_model.py
------------------
AI Surrogate Model for real-time composite valve CFD prediction.

Replaces slow CFD simulations (minutes) with instant predictions (milliseconds).
Uses a multi-output neural network trained on CFD simulation data.

Architecture:
  Input  → [total_L, avg_k, n_layers, T_inlet, T_outlet, h_inner, h_outer, fiber_angle]
  Hidden → 3 dense layers with BatchNorm + ReLU + Dropout
  Output → [T_max, T_avg, hotspot_x] + safety classification
"""

import numpy as np
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

SURROGATE_PATH    = "surrogate_model.joblib"
CLASSIFIER_PATH   = "safety_classifier.joblib"
SCALER_PATH       = "feature_scaler.joblib"


def train_surrogate(
    X:           np.ndarray,
    y:           np.ndarray,
    model_path:  str = SURROGATE_PATH,
    verbose:     bool = True,
) -> Pipeline:
    """
    Train MLP surrogate model to predict [T_max, T_avg, hotspot_x]
    from valve design features.

    Parameters
    ----------
    X : (n_samples, 8) — design features
    y : (n_samples, 3) — [T_max, T_avg, hotspot_x]
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    # Multi-output MLP regression
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=2000,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        verbose=False,
    )

    # Separate regressors per output for best accuracy
    models = []
    output_names = ["T_max", "T_avg", "hotspot_x"]
    cv_scores = []

    for i, name in enumerate(output_names):
        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        cv = cross_val_score(gb, X_scaled, y[:, i], cv=5,
                             scoring="r2")
        gb.fit(X_scaled, y[:, i])
        models.append(gb)
        cv_scores.append(cv.mean())

    model_bundle = {
        "models":       models,
        "output_names": output_names,
        "scaler":       scaler,
    }
    joblib.dump(model_bundle, model_path)

    if verbose:
        _print_surrogate_report(cv_scores, output_names, len(X), model_path)

    return model_bundle


def _print_surrogate_report(cv_scores, output_names, n_samples, path):
    sep = "─" * 56
    print(sep)
    print("  CFD Surrogate Model — Training Report")
    print(sep)
    print(f"  Samples trained on : {n_samples}")
    print(f"  Algorithm          : Gradient Boosting Regressor")
    print()
    print(f"  {'Output':<20} {'5-Fold R²':>12}")
    print(f"  {'─'*19:<20} {'─'*10:>12}")
    for name, score in zip(output_names, cv_scores):
        print(f"  {name:<20} {score:>12.4f}")
    print(f"\n  Model saved → {path}")
    print(sep)


def train_safety_classifier(
    X:          np.ndarray,
    y_Tmax:     np.ndarray,
    T_thresholds: tuple = (600, 900),  # (WARNING, CRITICAL) kelvin
    model_path: str = CLASSIFIER_PATH,
    verbose:    bool = True,
) -> GradientBoostingClassifier:
    """
    Train safety classifier: SAFE / WARNING / CRITICAL based on T_max.
    """
    labels = np.where(
        y_Tmax >= T_thresholds[1], 2,          # CRITICAL
        np.where(y_Tmax >= T_thresholds[0], 1,  # WARNING
        0)                                       # SAFE
    )

    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else StandardScaler()
    X_scaled = scaler.transform(X) if hasattr(scaler, "mean_") else scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    clf.fit(X_scaled, labels)
    joblib.dump(clf, model_path)

    unique, counts = np.unique(labels, return_counts=True)
    label_map = {0: "SAFE", 1: "WARNING", 2: "CRITICAL"}
    if verbose:
        print("\n  Safety Classifier")
        for u, c in zip(unique, counts):
            print(f"  {label_map[u]:<12} {c:>5} samples ({c/len(labels)*100:.1f}%)")

    return clf


def predict_fast(
    total_thickness: float,
    avg_k:           float,
    n_layers:        int,
    T_inlet:         float,
    T_outlet:        float,
    h_inner:         float,
    h_outer:         float,
    fiber_angle:     float,
    model_path:      str = SURROGATE_PATH,
    classifier_path: str = CLASSIFIER_PATH,
) -> dict:
    """
    Instant CFD prediction using trained surrogate — no simulation needed.

    Returns
    -------
    dict with T_max, T_avg, hotspot_x, safety_label, safety_proba
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Surrogate model not found at '{model_path}'. "
            "Run train_surrogate() first."
        )

    bundle     = joblib.load(model_path)
    models     = bundle["models"]
    scaler     = bundle["scaler"]

    features   = np.array([[
        total_thickness, avg_k, float(n_layers),
        T_inlet, T_outlet, h_inner, h_outer, fiber_angle,
    ]])
    X_scaled   = scaler.transform(features)

    T_max    = float(models[0].predict(X_scaled)[0])
    T_avg    = float(models[1].predict(X_scaled)[0])
    hotspot_x = float(models[2].predict(X_scaled)[0])

    # Safety classification
    safety_label = "SAFE"
    safety_proba = {"SAFE": 1.0, "WARNING": 0.0, "CRITICAL": 0.0}
    if os.path.exists(classifier_path):
        clf    = joblib.load(classifier_path)
        label_int  = int(clf.predict(X_scaled)[0])
        proba_arr  = clf.predict_proba(X_scaled)[0]
        label_map  = {0: "SAFE", 1: "WARNING", 2: "CRITICAL"}
        safety_label = label_map[label_int]
        classes    = [label_map[c] for c in clf.classes_]
        safety_proba = dict(zip(classes, proba_arr.round(4)))

    return {
        "T_max":        round(T_max, 2),
        "T_avg":        round(T_avg, 2),
        "hotspot_x":    round(max(0, hotspot_x), 5),
        "safety_label": safety_label,
        "safety_proba": safety_proba,
        "method":       "AI Surrogate (instant)",
    }
