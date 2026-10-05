"""
src/explainability.py
Explainable AI (XAI) Engine using SHAP and Feature Contribution Breakdown.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def explain_prediction_keywords(
    model: Any,
    vectorizer: Any,
    feature_names: List[str],
    email_text: str,
    feature_vector: Any,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Extract localized token and structural feature contributions for an email prediction.
    Supports Logistic Regression weights, Tree/Ensemble feature contributions, and linear approximations.

    Args:
        model: Trained scikit-learn or XGBoost classifier.
        vectorizer: Fitted CybersecurityFeaturePipeline or TfidfVectorizer.
        feature_names: List of all feature names (words + structural columns).
        email_text: Raw or clean email text.
        feature_vector: Transformed sparse feature vector (1, n_features).
        top_n: Number of top influential features to return.

    Returns:
        Dict containing:
            - 'top_phishing_indicators': List[Dict(feature, importance, direction)]
            - 'top_legitimate_indicators': List[Dict(feature, importance, direction)]
            - 'all_contributions': pd.DataFrame of active non-zero features
    """
    if not hasattr(feature_vector, "tocoo"):
        from scipy.sparse import csr_matrix
        feature_vector = csr_matrix(feature_vector)

    coo = feature_vector.tocoo()
    active_indices = coo.col
    active_values = coo.data

    contributions = []

    # Case 1: Linear Model (Logistic Regression) with explicit coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        for idx, val in zip(active_indices, active_values):
            if idx < len(feature_names):
                feat_name = feature_names[idx]
                weight = coefs[idx]
                impact = val * weight
                contributions.append({
                    "feature": feat_name,
                    "value": float(val),
                    "coefficient": float(weight),
                    "impact": float(impact),
                    "direction": "Phishing" if impact > 0 else "Legitimate"
                })

    # Case 2: Tree-Based Model (Random Forest, Gradient Boosting, XGBoost)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for idx, val in zip(active_indices, active_values):
            if idx < len(feature_names):
                feat_name = feature_names[idx]
                weight = importances[idx]
                # In tree models, present keywords in phishing context
                impact = val * weight
                contributions.append({
                    "feature": feat_name,
                    "value": float(val),
                    "coefficient": float(weight),
                    "impact": float(impact),
                    "direction": "Suspicious / Phishing" if impact > 0 else "Neutral"
                })

    # Case 3: Fallback surrogate
    else:
        for idx, val in zip(active_indices, active_values):
            if idx < len(feature_names):
                feat_name = feature_names[idx]
                contributions.append({
                    "feature": feat_name,
                    "value": float(val),
                    "coefficient": 1.0,
                    "impact": float(val),
                    "direction": "Present Token"
                })

    if not contributions:
        return {
            "top_phishing_indicators": [],
            "top_legitimate_indicators": [],
            "contributions_df": pd.DataFrame()
        }

    df_contrib = pd.DataFrame(contributions)
    df_contrib = df_contrib.sort_values(by="impact", ascending=False)

    # Top phishing drivers (highest positive impact)
    phishing_drivers = df_contrib[df_contrib["impact"] > 0].head(top_n).to_dict(orient="records")
    # Top legitimate drivers (lowest negative impact)
    legit_drivers = df_contrib[df_contrib["impact"] < 0].sort_values(by="impact", ascending=True).head(top_n).to_dict(orient="records")

    return {
        "top_phishing_indicators": phishing_drivers,
        "top_legitimate_indicators": legit_drivers,
        "contributions_df": df_contrib
    }


def compute_global_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract global top influential features from a trained model.

    Args:
        model: Trained classifier.
        feature_names: List of all vocabulary and structural feature names.
        top_n: Number of top features to extract.

    Returns:
        pd.DataFrame with columns ['feature', 'importance', 'direction']
    """
    records = []

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        for name, coef in zip(feature_names, coefs):
            records.append({
                "feature": name,
                "importance": abs(float(coef)),
                "raw_weight": float(coef),
                "direction": "Phishing Indicator" if coef > 0 else "Legitimate Indicator"
            })
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for name, imp in zip(feature_names, importances):
            records.append({
                "feature": name,
                "importance": float(imp),
                "raw_weight": float(imp),
                "direction": "High Threat Indicator"
            })
    else:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(by="importance", ascending=False).head(top_n).reset_index(drop=True)
    return df
