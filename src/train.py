"""
src/train.py
End-to-End Machine Learning Model Training, Benchmark Evaluation & Artifact Serialization Pipeline.

Models Evaluated:
1. Logistic Regression (L2 Regularized, Balanced)
2. Random Forest Classifier (100 Estimators)
3. XGBoost / Gradient Boosting Classifier (Gradient Boosted Decision Trees)

Secondary Models:
- Multi-class Classifier for Phishing Type Identification
- Unsupervised Isolation Forest for Zero-Day Threat Anomaly Detection
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import CybersecurityFeaturePipeline
from src.anomaly_detection import ZeroDayAnomalyDetector

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ModelTrainer")


def evaluate_classifier(name: str, model: Any, X_test, y_test) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for a binary classifier.
    Highlights Recall as the critical metric for cybersecurity defense.
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        roc_auc = float(roc_auc_score(y_test, y_prob))
    else:
        y_prob = y_pred
        roc_auc = float(roc_auc_score(y_test, y_prob))

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # False Negative Rate (FNR = FN / (FN + TP)) - Critical threat metric
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC curve points for dashboard visualization
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fpr_list = [round(float(v), 4) for v in fpr]
        tpr_list = [round(float(v), 4) for v in tpr]
    except Exception:
        fpr_list, tpr_list = [0.0, 1.0], [0.0, 1.0]

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "roc_fpr": fpr_list,
        "roc_tpr": tpr_list,
    }


def train_models():
    """Execute complete training, evaluation, and serialization pipeline."""
    logger.info("=== STEP 1: Loading Dataset ===")
    df = load_dataset()

    logger.info("=== STEP 2: Preprocessing and Cleaning Text ===")
    df_clean = preprocess_dataframe(df, drop_duplicates=True)
    logger.info(f"Cleaned dataset contains {len(df_clean)} records.")

    # 80/20 Stratified Split on binary label
    logger.info("=== STEP 3: Stratified 80/20 Train-Test Split ===")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df_clean,
        df_clean["label"],
        test_size=0.20,
        random_state=42,
        stratify=df_clean["label"]
    )
    logger.info(f"Training Set: {len(X_train_df)} samples | Testing Set: {len(X_test_df)} samples")

    # Feature Extraction
    logger.info("=== STEP 4: TF-IDF & Structural Feature Engineering ===")
    feature_pipeline = CybersecurityFeaturePipeline(
        max_features=3000,
        ngram_range=(1, 2),
        include_structural=True
    )
    X_train_vec = feature_pipeline.fit_transform(
        raw_texts=X_train_df["text"],
        cleaned_texts=X_train_df["cleaned_text"]
    )
    X_test_vec = feature_pipeline.transform(
        raw_texts=X_test_df["text"],
        cleaned_texts=X_test_df["cleaned_text"]
    )
    logger.info(f"Feature matrix shape: {X_train_vec.shape} (Features: {len(feature_pipeline.feature_names)})")

    # Train 3 Binary Classifiers
    logger.info("=== STEP 5: Training and Comparing 3 Classifiers ===")
    
    # Try importing XGBoost, fallback to GradientBoostingClassifier if needed
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss"
        )
        xgb_name = "XGBoost Classifier"
    except Exception as e:
        logger.warning(f"XGBoost unavailable ({e}), using GradientBoostingClassifier.")
        xgb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        xgb_name = "Gradient Boosting Classifier"

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=25,
            random_state=42,
            n_jobs=-1
        ),
        xgb_name: xgb_model
    }

    metrics_results = []
    trained_binary_models = {}

    for name, clf in candidate_models.items():
        logger.info(f"Training {name}...")
        clf.fit(X_train_vec, y_train)
        eval_metrics = evaluate_classifier(name, clf, X_test_vec, y_test)
        metrics_results.append(eval_metrics)
        trained_binary_models[name] = clf
        logger.info(
            f"[{name}] Acc: {eval_metrics['accuracy']:.4f} | "
            f"Recall: {eval_metrics['recall']:.4f} (CRITICAL) | "
            f"Precision: {eval_metrics['precision']:.4f} | "
            f"F1: {eval_metrics['f1_score']:.4f} | "
            f"ROC-AUC: {eval_metrics['roc_auc']:.4f} | "
            f"False Negatives: {eval_metrics['false_negatives']}"
        )

    # Select Best Model based on F1-Score & Recall
    best_result = max(metrics_results, key=lambda x: (x["f1_score"] + x["recall"]))
    best_model_name = best_result["model_name"]
    best_binary_model = trained_binary_models[best_model_name]
    logger.info(f"\n--> Selected Best Binary Classifier: '{best_model_name}' (F1: {best_result['f1_score']}, Recall: {best_result['recall']})")

    # Print Comparative Benchmark Table
    print_benchmark_table(metrics_results)

    # STEP 6: Fine-Grained Multi-Class Phishing Type Classifier
    logger.info("=== STEP 6: Training Fine-Grained Phishing Type Multi-Class Model ===")
    label_encoder = LabelEncoder()
    y_train_cat = label_encoder.fit_transform(X_train_df["phishing_type"])
    y_test_cat = label_encoder.transform(X_test_df["phishing_type"])

    category_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    )
    category_model.fit(X_train_vec, y_train_cat)
    cat_pred = category_model.predict(X_test_vec)
    cat_acc = accuracy_score(y_test_cat, cat_pred)
    logger.info(f"Phishing Type Multi-Class Accuracy: {cat_acc:.4f} across {len(label_encoder.classes_)} classes.")

    # STEP 7: Isolation Forest for Zero-Day Emerging Patterns
    logger.info("=== STEP 7: Training Isolation Forest for Zero-Day Threat Discovery ===")
    anomaly_detector = ZeroDayAnomalyDetector(contamination=0.06, random_state=42)
    anomaly_detector.fit(X_train_vec)
    logger.info("Isolation Forest fitted successfully.")

    # STEP 8: Export All Model Artifacts
    logger.info("=== STEP 8: Exporting Serialized Model Artifacts to models/ ===")
    joblib.dump(best_binary_model, MODELS_DIR / "binary_phishing_model.pkl")
    joblib.dump(category_model, MODELS_DIR / "category_model.pkl")
    joblib.dump(feature_pipeline, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(anomaly_detector, MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    joblib.dump(feature_pipeline.feature_names, MODELS_DIR / "feature_names.pkl")

    # Save metrics summary
    summary_data = {
        "best_model_name": best_model_name,
        "models_evaluated": metrics_results,
        "multiclass_accuracy": round(float(cat_acc), 4),
        "total_training_samples": len(X_train_df),
        "total_testing_samples": len(X_test_df),
        "feature_count": len(feature_pipeline.feature_names),
        "phishing_categories": list(label_encoder.classes_)
    }
    with open(MODELS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)

    logger.info("All artifacts successfully exported to models/ directory.")
    return summary_data


def print_benchmark_table(metrics: list):
    """Print clean formatted comparison table highlighting Recall."""
    print("\n" + "=" * 90)
    print("                    CYBERSECURITY MODEL COMPARISON BENCHMARK                  ")
    print("=" * 90)
    header = f"{'Model':<30} | {'Accuracy':<8} | {'Recall (Key)':<12} | {'Precision':<9} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'FN Count':<8}"
    print(header)
    print("-" * 90)
    for m in metrics:
        row = (
            f"{m['model_name']:<30} | "
            f"{m['accuracy']:<8.4f} | "
            f"{m['recall']:<12.4f} | "
            f"{m['precision']:<9.4f} | "
            f"{m['f1_score']:<8.4f} | "
            f"{m['roc_auc']:<8.4f} | "
            f"{m['false_negatives']:<8}"
        )
        print(row)
    print("=" * 90)
    print("CRITICAL CYBERSECURITY NOTE:")
    print("In Phishing Detection, RECALL is the foremost operational metric.")
    print("A False Negative (FN) means an attacker's malicious email enters an employee's inbox,")
    print("leading to potential breach, whereas False Positives can be reviewed in quarantine.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    train_models()
