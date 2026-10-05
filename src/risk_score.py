"""
src/risk_score.py
Calibrated Threat Scoring Engine (0-100 Scale) & Risk Level Categorization.
"""

from typing import Dict, Any, Tuple


def calculate_risk_score(
    ml_probability: float,
    structural_features: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate a calibrated cybersecurity threat score (0 to 100) combining
    probabilistic ML model inference and rule-based heuristic indicator penalties.

    Formula:
        Base Score = ML Probability * 70
        Heuristic Penalty Points (up to +30):
            + 8 pts: Credential harvesting terms (password, login, SSO, OTP)
            + 7 pts: Urgency indicators (suspended, 24 hours, immediate action)
            + 5 pts: Financial/wire transfer keywords (invoice, bitcoin, routing)
            + 4 pts: Presence of URLs / IP-based links
            + 3 pts: Excessive uppercase letter ratio (> 15%)
            + 3 pts: Multiple exclamation marks (>= 2)
        Final Risk Score = min(100, max(0, round(Base Score + Heuristic Penalty)))

    Args:
        ml_probability: Probability of phishing from binary ML classifier (0.0 to 1.0).
        structural_features: Dictionary of structural indicators extracted from email.

    Returns:
        Dict with keys:
            - 'risk_score': integer (0-100)
            - 'risk_level': str ('Low', 'Medium', 'High', 'Critical')
            - 'risk_color': str (HEX color code for UI)
            - 'heuristic_penalties': Dict[str, float] (breakdown)
            - 'base_score': float
    """
    # Clamp probability
    prob = max(0.0, min(1.0, float(ml_probability)))
    base_score = prob * 70.0

    penalties = {}
    heuristic_total = 0.0

    # 1. Credential harvesting penalty
    if structural_features.get("feat_has_credential", 0) > 0:
        penalties["Credential Harvesting Flag"] = 8.0
        heuristic_total += 8.0

    # 2. Urgency pressure penalty
    if structural_features.get("feat_has_urgent", 0) > 0:
        penalties["Urgent Time Pressure Flag"] = 7.0
        heuristic_total += 7.0

    # 3. Financial/wire fraud penalty
    if structural_features.get("feat_has_financial", 0) > 0:
        penalties["Financial / Transaction Flag"] = 5.0
        heuristic_total += 5.0

    # 4. URL / IP links penalty
    num_urls = structural_features.get("feat_num_urls", 0)
    has_ip_url = structural_features.get("feat_has_ip_url", 0)
    if has_ip_url > 0:
        penalties["Direct IP-Based URL Flag"] = 5.0
        heuristic_total += 5.0
    elif num_urls > 0:
        penalties["Embedded URL Detected"] = min(4.0, num_urls * 2.0)
        heuristic_total += penalties["Embedded URL Detected"]

    # 5. Excessive uppercase ratio (> 12%)
    caps_ratio = structural_features.get("feat_uppercase_ratio", 0)
    if caps_ratio > 0.12:
        penalties["High Uppercase Letter Density"] = 3.0
        heuristic_total += 3.0

    # 6. Exclamation mark density
    num_excl = structural_features.get("feat_num_exclamation", 0)
    if num_excl >= 2:
        penalties["Excessive Exclamation Punctuation"] = 3.0
        heuristic_total += 3.0

    # Total score calculation
    raw_total = base_score + heuristic_total
    final_score = int(round(max(0.0, min(100.0, raw_total))))

    # Risk level thresholding
    if final_score <= 30:
        level = "Low"
        color = "#28a745"  # Green
        description = "Routine communication with negligible indicators of compromise."
    elif final_score <= 60:
        level = "Medium"
        color = "#ffc107"  # Amber / Yellow
        description = "Suspicious attributes detected. Exercise caution before interacting."
    elif final_score <= 80:
        level = "High"
        color = "#fd7e14"  # Orange
        description = "High probability of social engineering or malicious intent."
    else:
        level = "Critical"
        color = "#dc3545"  # Red
        description = "Severe threat detected. Highly likely to be active phishing or credential theft."

    return {
        "risk_score": final_score,
        "risk_level": level,
        "risk_color": color,
        "description": description,
        "base_score": round(base_score, 2),
        "heuristic_total": round(heuristic_total, 2),
        "heuristic_penalties": penalties,
        "ml_probability": round(prob, 4)
    }
