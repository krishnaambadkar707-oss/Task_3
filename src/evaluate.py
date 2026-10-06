"""
evaluate.py - Model evaluation metrics, threshold tuning, and feature importance analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def evaluate_model(model, X_test, y_test, threshold=0.5) -> dict:
    """
    Evaluate fitted pipeline model on test set.
    Returns comprehensive metrics dictionary including accuracy, recall, precision, f1, roc_auc, pr_auc, and confusion matrix.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    # Custom threshold prediction
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "accuracy": round(float(acc), 4),
        "recall": round(float(rec), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "y_prob": y_prob.tolist(),
        "y_true": y_test.tolist()
    }


def evaluate_threshold_range(model, X_test, y_test, thresholds=None) -> pd.DataFrame:
    """Evaluate performance across range of decision probability thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    results = []
    for th in thresholds:
        metrics = evaluate_model(model, X_test, y_test, threshold=th)
        results.append({
            "Threshold": round(th, 2),
            "Accuracy": metrics["accuracy"],
            "Recall": metrics["recall"],
            "Precision": metrics["precision"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "TN": metrics["tn"]
        })
    return pd.DataFrame(results)


def extract_feature_importance(model, feature_names: list = None) -> pd.DataFrame:
    """
    Extract feature importances or coefficients from a fitted Pipeline model.
    """
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    # Get transformed feature names
    if hasattr(preprocessor, 'get_feature_names_out'):
        transformed_names = list(preprocessor.get_feature_names_out())
    else:
        transformed_names = [f"feature_{i}" for i in range(100)]

    # Clean feature names (remove num__ or cat__ prefix)
    clean_names = [name.replace('num__', '').replace('cat__', '') for name in transformed_names]

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        df_imp = pd.DataFrame({'Feature': clean_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    elif hasattr(classifier, 'coef_'):
        coefs = classifier.coef_[0]
        odds_ratio = np.exp(coefs)
        df_imp = pd.DataFrame({
            'Feature': clean_names,
            'Coefficient': coefs,
            'OddsRatio': odds_ratio,
            'AbsImportance': np.abs(coefs)
        })
        df_imp = df_imp.sort_values(by='AbsImportance', ascending=False).reset_index(drop=True)
    else:
        df_imp = pd.DataFrame({'Feature': clean_names, 'Importance': 0.0})

    return df_imp
