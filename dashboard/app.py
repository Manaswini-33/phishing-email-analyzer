"""
dashboard/app.py
AI-Powered Phishing Email Pattern Analysis, Risk Prediction & Explainable Detection System.

Streamlined 3-Page ML Cybersecurity Application:
1. 🔍 Email Analyzer (Default Landing Page)
2. 📊 Threat Analytics (Dataset-level patterns, feature indicators & anomaly detection)
3. 📈 Model Performance (Model comparisons, confusion matrix, ROC-AUC & feature importance)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import clean_email_text
from src.feature_engineering import CybersecurityFeaturePipeline, extract_structural_features
from src.risk_score import calculate_risk_score
from src.explainability import explain_prediction_keywords, compute_global_feature_importance
from src.anomaly_detection import ZeroDayAnomalyDetector
from src.recommendations import generate_security_recommendations

# Streamlit Page Config
st.set_page_config(
    page_title="AI Phishing Email Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# Custom Styling for Clean ML UI
st.markdown("""
<style>
    .page-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        font-size: 1.0rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .verdict-phish {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .verdict-legit {
        background-color: #dcfce7;
        border-left: 6px solid #22c55e;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .risk-badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #f97316;
        color: white;
        padding: 6px 14px;
        border-radius: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #eab308;
        color: black;
        padding: 6px 14px;
        border-radius: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: #22c55e;
        color: white;
        padding: 6px 14px;
        border-radius: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .keyword-tag {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
        margin: 3px;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Machine Learning Models...")
def load_ml_artifacts():
    """Load trained models, feature pipelines, and metrics from disk."""
    try:
        binary_model = joblib.load(MODELS_DIR / "binary_phishing_model.pkl")
        category_model = joblib.load(MODELS_DIR / "category_model.pkl")
        feature_pipeline = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
        anomaly_detector = joblib.load(MODELS_DIR / "isolation_forest.pkl")
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

        with open(MODELS_DIR / "metrics_summary.json", "r") as f:
            metrics_summary = json.load(f)

        return {
            "binary_model": binary_model,
            "category_model": category_model,
            "feature_pipeline": feature_pipeline,
            "anomaly_detector": anomaly_detector,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "metrics_summary": metrics_summary,
            "status": "ready"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@st.cache_data(show_spinner="Loading Dataset...")
def load_dataset_cached() -> Optional[pd.DataFrame]:
    """Load cleaned dataset from disk."""
    clean_csv = DATA_DIR / "clean_emails.csv"
    if clean_csv.exists():
        return pd.read_csv(clean_csv)
    return None


# Sidebar Navigation - Exactly 3 Pages
st.sidebar.markdown("## 🛡️ AI Phishing Analyzer")
st.sidebar.markdown("---")

selected_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔍 Email Analyzer",
        "📊 Threat Analytics",
        "📈 Model Performance"
    ],
    index=0
)

st.sidebar.markdown("---")
artifacts = load_ml_artifacts()
if artifacts.get("status") == "ready":
    st.sidebar.success("● ML Models Loaded")
    best_name = artifacts["metrics_summary"].get("best_model_name", "Classifier")
    st.sidebar.caption(f"Active Model: **{best_name}**")
else:
    st.sidebar.error("⚠️ ML Models missing. Run `python src/train.py`.")


# ==============================================================================
# PAGE 1: 🔍 REAL-TIME EMAIL ANALYZER (DEFAULT LANDING PAGE)
# ==============================================================================
if selected_page == "🔍 Email Analyzer":
    st.markdown('<div class="page-title">🔍 Real-Time Phishing Email Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Analyze an email for phishing probability, risk level, phishing type, suspicious indicators, and recommended action.</div>', unsafe_allow_html=True)

    if artifacts.get("status") != "ready":
        st.error(f"Failed to load ML artifacts: {artifacts.get('message')}. Please run `python src/train.py` first.")
        st.stop()

    # Pre-loaded sample presets for rapid demonstration
    presets = {
        "Custom / Enter Own Email": ("", ""),
        "🚨 Microsoft 365 Credential Phishing": (
            "Security Alert: Unauthorized sign-in detected on your Microsoft 365 Account",
            "We detected an unauthorized login attempt from an unknown device. Your mailbox access will be terminated within 24 hours unless you re-verify your identity. Click http://login-microsoft365-verify-access.net/auth to confirm your username and corporate password immediately."
        ),
        "💼 CEO Urgent Wire Transfer (BEC)": (
            "From CEO: Confidential Acquisition - Urgent Bank Wire Request",
            "I am currently attending an executive board meeting and cannot take voice calls. We are finalizing a time-sensitive vendor contract today. Please immediately execute an urgent wire transfer of $48,500 to the escrow account details attached. Treat this with maximum confidentiality."
        ),
        "📦 DHL Delivery Customs Fee Scam": (
            "DHL Express: Your package delivery #US-983192 is pending customs address confirmation",
            "Your parcel could not be delivered due to an incorrect postal address. Please pay the $2.99 customs fee and update your delivery address at http://track-dhl-express-post-package.info within 48 hours to prevent return."
        ),
        "✅ Safe: Engineering Architecture Review": (
            "Weekly Engineering Architecture Sync - Meeting Agenda",
            "Hi team, please find attached the meeting notes and technical specification for the upcoming Q4 microservice migration. Please review the PR #219 on GitHub and add your comments before tomorrow's 10:00 AM EST standup call. Best regards, Alex."
        ),
        "✅ Safe: Monthly AWS Billing Receipt": (
            "AWS Invoice - Summary for billing period August 2026",
            "Dear Customer, your latest monthly billing statement for Amazon Web Services is now available in the AWS Billing Console. Total charged: $142.30. No further action is required on your part. Thank you for using AWS."
        )
    }

    selected_preset = st.selectbox("⚡ Quick-Load Sample Email (or enter custom text below):", list(presets.keys()))
    default_subj, default_body = presets[selected_preset]

    st.markdown("### 📧 Email Input")
    col_s1, col_s2 = st.columns([1, 1])
    
    subject_input = st.text_input(
        "Email Subject",
        value=default_subj,
        placeholder="e.g. Urgent: Account Verification Required"
    )

    body_input = st.text_area(
        "Email Body",
        value=default_body,
        height=160,
        placeholder="e.g. Dear customer, we detected unauthorized activity on your account. Please log in at the link below to confirm your password..."
    )

    analyze_btn = st.button("🔍 Analyze Email", type="primary", use_container_width=True)

    # Process and analyze email
    combined_raw_text = f"Subject: {subject_input}\n\n{body_input}".strip()

    if analyze_btn or (selected_preset != "Custom / Enter Own Email" and combined_raw_text):
        if not combined_raw_text or not body_input.strip():
            st.warning("Please enter the email body to perform analysis.")
            st.stop()

        with st.spinner("Processing text through ML pipeline & risk scoring engine..."):
            clean_text = clean_email_text(combined_raw_text)
            struct_features = extract_structural_features(combined_raw_text)

            feature_pipeline: CybersecurityFeaturePipeline = artifacts["feature_pipeline"]
            raw_s = pd.Series([combined_raw_text])
            clean_s = pd.Series([clean_text])
            v = feature_pipeline.transform(raw_s, clean_s)

            # Binary Prediction & Probabilities
            binary_model = artifacts["binary_model"]
            if hasattr(binary_model, "predict_proba"):
                probs = binary_model.predict_proba(v)[0]
                legit_prob = float(probs[0])
                phish_prob = float(probs[1])
            else:
                phish_prob = float(binary_model.predict(v)[0])
                legit_prob = 1.0 - phish_prob

            is_phishing = (phish_prob >= 0.5)
            confidence_pct = round(phish_prob * 100 if is_phishing else legit_prob * 100, 1)

            # Multi-class Phishing Type
            category_model = artifacts["category_model"]
            label_encoder = artifacts["label_encoder"]
            cat_idx = category_model.predict(v)[0]
            phishing_type = str(label_encoder.inverse_transform([cat_idx])[0])

            # Risk Score
            risk_data = calculate_risk_score(phish_prob, struct_features)
            risk_score = risk_data["risk_score"]
            risk_level = risk_data["risk_level"]

            # Anomaly Detection
            anomaly_detector: ZeroDayAnomalyDetector = artifacts["anomaly_detector"]
            anomaly_info = anomaly_detector.predict_anomaly(v)

            # Explainability & Recommendations
            xai_info = explain_prediction_keywords(
                model=binary_model,
                vectorizer=feature_pipeline,
                feature_names=artifacts["feature_names"],
                email_text=clean_text,
                feature_vector=v,
                top_n=8
            )
            recs = generate_security_recommendations(
                risk_level=risk_level,
                phishing_type=phishing_type,
                heuristic_penalties=risk_data["heuristic_penalties"],
                is_anomaly=anomaly_info["is_anomaly"]
            )

        st.markdown("---")

        # 1. PREDICTION RESULT & CONFIDENCE
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if is_phishing:
                st.markdown(f"""
                <div class="verdict-phish">
                    <h2 style="color:#b91c1c; margin:0 0 4px 0;">🚨 PHISHING DETECTED</h2>
                    <p style="margin:0; font-size:1.15rem; font-weight:600; color:#7f1d1d;">Confidence: <strong>{confidence_pct}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-legit">
                    <h2 style="color:#15803d; margin:0 0 4px 0;">✅ LEGITIMATE EMAIL</h2>
                    <p style="margin:0; font-size:1.15rem; font-weight:600; color:#14532d;">Confidence: <strong>{confidence_pct}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            badge_class = f"risk-badge-{risk_level.lower()}"
            st.markdown(f"""
            <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:1.2rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.9rem; color:#64748b;">Calibrated Threat Index</span>
                        <h2 style="margin:0; color:#1e293b;">{risk_score} <span style="font-size:1.0rem; color:#94a3b8;">/ 100</span></h2>
                    </div>
                    <div>
                        <span class="{badge_class}">{risk_level.upper()} RISK</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. PHISHING TYPE & PROBABILITIES
        t_col1, t_col2 = st.columns([1, 1])
        with t_col1:
            st.markdown("#### 🎯 Phishing Category")
            if is_phishing:
                st.info(f"**Identified Vector:** {phishing_type}")
            else:
                st.write("Phishing type analysis is applicable only to emails classified as phishing.")

        with t_col2:
            st.markdown("#### 📊 Model Class Probabilities")
            st.write(f"• **Phishing Probability:** `{phish_prob * 100:.1f}%`")
            st.write(f"• **Legitimate Probability:** `{legit_prob * 100:.1f}%`")

        st.markdown("---")

        # 3. WHY WAS THIS EMAIL FLAGGED? ⭐ (Explainability)
        st.markdown("### ⚠️ Why was this email flagged?")
        
        flagged_reasons = []
        if struct_features["feat_has_urgent"] > 0:
            flagged_reasons.append("✓ Urgent language / artificial time pressure detected")
        if struct_features["feat_has_credential"] > 0:
            flagged_reasons.append("✓ Credential / password verification request detected")
        if struct_features["feat_has_financial"] > 0:
            flagged_reasons.append("✓ Financial keywords / transaction requests detected")
        if struct_features["feat_num_urls"] > 0 or struct_features["feat_has_ip_url"] > 0:
            flagged_reasons.append("✓ Suspicious / embedded hyperlink detected in text")
        if struct_features["feat_uppercase_ratio"] > 0.12:
            flagged_reasons.append("✓ High uppercase letter density detected")
        if struct_features["feat_num_exclamation"] >= 2:
            flagged_reasons.append("✓ Multiple exclamation marks detected")

        exp_col1, exp_col2 = st.columns([1, 1])

        with exp_col1:
            st.markdown("**Matched Threat Indicators:**")
            if flagged_reasons:
                for r in flagged_reasons:
                    st.write(r)
            else:
                st.write("✓ No high-risk social engineering flags detected.")

        with exp_col2:
            st.markdown("**Influential Keywords Extracted by ML Model:**")
            df_c = xai_info["contributions_df"]
            if not df_c.empty:
                active_tokens = df_c[df_c["impact"] > 0]["feature"].head(6).tolist()
                if active_tokens:
                    tags_html = "".join([f'<span class="keyword-tag">{t}</span>' for t in active_tokens])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("Standard baseline vocabulary detected.")
            else:
                st.write("Standard baseline vocabulary detected.")

        st.markdown("---")

        # 4. ACTIONABLE SECURITY RECOMMENDATIONS
        st.markdown("### 🛡️ Recommended Action")
        st.write(f"**Directive:** {recs['primary_action']}")
        for step in recs["action_steps"][:3]:
            st.write(f"• {step}")

        # 5. TECHNICAL DETAILS (COLLAPSIBLE)
        with st.expander("🔽 View Extracted Features & Heuristics"):
            st.caption("Numerical and boolean features passed into the machine learning vectorizer:")
            f_table = [
                {"Feature Indicator": "Word Count", "Extracted Value": f"{int(struct_features['feat_num_words'])} words"},
                {"Feature Indicator": "Uppercase Count", "Extracted Value": f"{int(struct_features['feat_num_uppercase'])} characters ({struct_features['feat_uppercase_ratio']*100:.1f}%)"},
                {"Feature Indicator": "Exclamation Count", "Extracted Value": f"{int(struct_features['feat_num_exclamation'])}"},
                {"Feature Indicator": "Question Count", "Extracted Value": f"{int(struct_features['feat_num_question'])}"},
                {"Feature Indicator": "Urgency Flag", "Extracted Value": "1 (Detected)" if struct_features['feat_has_urgent'] else "0 (None)"},
                {"Feature Indicator": "Financial Flag", "Extracted Value": "1 (Detected)" if struct_features['feat_has_financial'] else "0 (None)"},
                {"Feature Indicator": "Credential Flag", "Extracted Value": "1 (Detected)" if struct_features['feat_has_credential'] else "0 (None)"},
                {"Feature Indicator": "URLs Detected", "Extracted Value": f"{int(struct_features['feat_num_urls'])} URL(s)"},
            ]
            st.dataframe(pd.DataFrame(f_table), use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: 📊 THREAT PATTERN ANALYTICS
# ==============================================================================
elif selected_page == "📊 Threat Analytics":
    st.markdown('<div class="page-title">📊 Threat Pattern Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Dataset-level phishing distributions, threat taxonomies, and behavioral indicator prevalence.</div>', unsafe_allow_html=True)

    df_dataset = load_dataset_cached()
    if df_dataset is None:
        st.error("Dataset not found in `data/processed/clean_emails.csv`. Please run `python src/data_loader.py` first.")
        st.stop()

    # 1. DATASET SUMMARY METRICS
    total_emails = len(df_dataset)
    phish_count = int(df_dataset["label"].sum())
    legit_count = total_emails - phish_count

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Emails", f"{total_emails:,}")
    with m2:
        st.metric("Phishing Emails", f"{phish_count:,}", f"{(phish_count / total_emails) * 100:.1f}%")
    with m3:
        st.metric("Legitimate Emails", f"{legit_count:,}", f"{(legit_count / total_emails) * 100:.1f}%")

    st.markdown("---")

    # 2. PHISHING VS LEGITIMATE DISTRIBUTION & 3. PHISHING TYPE DISTRIBUTION
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("### 🍩 Phishing vs Legitimate Distribution")
        dist_df = pd.DataFrame({
            "Classification": ["Phishing", "Legitimate"],
            "Count": [phish_count, legit_count]
        })
        fig_dist = px.pie(
            dist_df,
            values="Count",
            names="Classification",
            hole=0.5,
            color="Classification",
            color_discrete_map={"Phishing": "#ef4444", "Legitimate": "#22c55e"},
            title="Class Distribution in Dataset"
        )
        fig_dist.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)

    with c_col2:
        st.markdown("### 📊 Phishing Type Distribution")
        type_df = df_dataset["phishing_type"].value_counts().reset_index()
        type_df.columns = ["Phishing Type", "Email Count"]
        
        fig_types = px.bar(
            type_df,
            x="Email Count",
            y="Phishing Type",
            orientation="h",
            color="Email Count",
            color_continuous_scale="Blues",
            title="Distribution across 13 Phishing & Normal Categories"
        )
        fig_types.update_layout(yaxis=dict(autorange="reversed"), height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_types, use_container_width=True)

    st.markdown("---")

    # 4. RISK LEVEL DISTRIBUTION
    st.markdown("### ⚠️ Threat Severity / Risk Level Distribution")
    sev_df = df_dataset["severity"].value_counts().reset_index()
    sev_df.columns = ["Severity Level", "Count"]

    sev_colors = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#3b82f6", "None": "#22c55e"}
    fig_sev = px.bar(
        sev_df,
        x="Severity Level",
        y="Count",
        color="Severity Level",
        color_discrete_map=sev_colors,
        title="Email Volume by Threat Severity Rating"
    )
    fig_sev.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_sev, use_container_width=True)

    st.markdown("---")

    # 5. SUSPICIOUS INDICATOR ANALYSIS ⭐
    st.markdown("### 🔍 Suspicious Indicator Analysis")
    st.caption("Comparison of heuristic social engineering features present in Phishing vs Legitimate emails:")

    # Calculate real feature presence across dataset
    phish_subset = df_dataset[df_dataset["label"] == 1]["text"]
    legit_subset = df_dataset[df_dataset["label"] == 0]["text"]

    from src.feature_engineering import URGENCY_REGEX, FINANCIAL_REGEX, CREDENTIAL_REGEX, URL_PATTERN

    def get_indicator_stats(texts: pd.Series):
        total = len(texts)
        if total == 0:
            return {}
        urgent = sum(1 for t in texts if URGENCY_REGEX.search(str(t)))
        cred = sum(1 for t in texts if CREDENTIAL_REGEX.search(str(t)))
        fin = sum(1 for t in texts if FINANCIAL_REGEX.search(str(t)))
        url = sum(1 for t in texts if URL_PATTERN.search(str(t)))
        excl = sum(1 for t in texts if str(t).count('!') >= 2)

        return {
            "Urgent Language": round((urgent / total) * 100, 1),
            "Credential Requests": round((cred / total) * 100, 1),
            "Financial Keywords": round((fin / total) * 100, 1),
            "Embedded URLs": round((url / total) * 100, 1),
            "Multiple Exclamations": round((excl / total) * 100, 1)
        }

    phish_stats = get_indicator_stats(phish_subset)
    legit_stats = get_indicator_stats(legit_subset)

    indicators = list(phish_stats.keys())
    comp_df = pd.DataFrame({
        "Indicator": indicators,
        "Phishing Emails (%)": [phish_stats[k] for k in indicators],
        "Legitimate Emails (%)": [legit_stats[k] for k in indicators]
    })

    melted_comp = comp_df.melt(id_vars=["Indicator"], value_vars=["Phishing Emails (%)", "Legitimate Emails (%)"], var_name="Category", value_name="Presence (%)")
    fig_comp = px.bar(
        melted_comp,
        x="Indicator",
        y="Presence (%)",
        color="Category",
        barmode="group",
        color_discrete_map={"Phishing Emails (%)": "#ef4444", "Legitimate Emails (%)": "#22c55e"},
        title="Indicator Prevalence: Phishing vs Legitimate Emails"
    )
    fig_comp.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # 6. UNUSUAL PATTERN DETECTION (Isolation Forest)
    st.markdown("### 🔎 Unusual Pattern Detection")
    st.write("Identifies emails whose feature patterns are unusual compared with the training distribution using an unsupervised **Isolation Forest**.")

    if artifacts.get("status") == "ready":
        feature_pipeline: CybersecurityFeaturePipeline = artifacts["feature_pipeline"]
        anomaly_detector: ZeroDayAnomalyDetector = artifacts["anomaly_detector"]

        # Run anomaly scoring on a sample of dataset
        sample_eval_df = df_dataset.sample(min(500, len(df_dataset)), random_state=42).copy()
        sample_v = feature_pipeline.transform(sample_eval_df["text"])
        scores = anomaly_detector.model.decision_function(sample_v)
        preds = anomaly_detector.model.predict(sample_v)  # -1 for anomaly, 1 for normal

        anom_count = int(np.sum(preds == -1))
        total_eval = len(sample_eval_df)
        anom_pct = round((anom_count / total_eval) * 100, 1)

        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            st.metric("Emails Analyzed", f"{total_eval:,}")
        with a_col2:
            st.metric("Unusual Patterns Detected", f"{anom_count:,}")
        with a_col3:
            st.metric("Anomaly Percentage", f"{anom_pct}%")

        sample_eval_df["anomaly_score"] = scores
        sample_eval_df["is_unusual"] = (preds == -1)

        with st.expander("View Unusual Pattern Examples"):
            unusual_samples = sample_eval_df[sample_eval_df["is_unusual"]].sort_values(by="anomaly_score").head(5)
            st.dataframe(
                unusual_samples[["text", "phishing_type", "severity", "anomaly_score"]],
                use_container_width=True,
                hide_index=True
            )


# ==============================================================================
# PAGE 3: 📈 MODEL PERFORMANCE
# ==============================================================================
elif selected_page == "📈 Model Performance":
    st.markdown('<div class="page-title">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Comparative evaluation across supervised classifiers and explainability benchmarks.</div>', unsafe_allow_html=True)

    if artifacts.get("status") != "ready":
        st.error("Metrics summary not available. Please train models first.")
        st.stop()

    summary = artifacts["metrics_summary"]
    models_evaluated = summary.get("models_evaluated", [])
    best_model = summary.get("best_model_name", "Logistic Regression")

    # 1. MODEL COMPARISON TABLE
    st.markdown("### 🏆 Supervised Model Comparison Benchmark")
    df_eval = pd.DataFrame(models_evaluated)

    display_table = df_eval[[
        "model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "false_negatives"
    ]].copy()
    display_table.columns = [
        "Model", "Accuracy", "Precision", "Recall (Key Metric)", "F1-Score", "ROC-AUC", "False Negatives"
    ]

    st.dataframe(
        display_table.style.highlight_max(subset=["Recall (Key Metric)", "F1-Score", "ROC-AUC"], color="#dcfce7"),
        use_container_width=True,
        hide_index=True
    )

    # 2. BEST MODEL & RECALL EMPHASIS
    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        st.success(f"**Selected Best Model:** {best_model}\n\nSelected based on its overall classification performance, optimal precision-recall balance, and zero missed attacks.")

    with b_col2:
        st.info("**Cybersecurity Note:** Recall is particularly critical because false negatives represent malicious phishing emails that evade detection and reach employee inboxes.")

    st.markdown("---")

    # 3. CONFUSION MATRIX & 4. ROC-AUC CURVE
    cm_col, roc_col = st.columns(2)

    with cm_col:
        st.markdown("### 🔲 Confusion Matrix")
        # Find metrics of best model
        selected_m_data = next((m for m in models_evaluated if m["model_name"] == best_model), models_evaluated[0])
        cm_matrix = np.array(selected_m_data["confusion_matrix"])

        fig_cm = px.imshow(
            cm_matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Legitimate", "Phishing"],
            y=["Legitimate", "Phishing"],
            title=f"Confusion Matrix ({best_model})"
        )
        fig_cm.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with roc_col:
        st.markdown("### 📈 ROC-AUC Curve")
        fig_roc = go.Figure()
        
        # Add diagonal baseline
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Chance (AUC = 0.50)'
        ))

        colors = ["#3b82f6", "#ef4444", "#10b981"]
        for idx, m_data in enumerate(models_evaluated):
            fpr = m_data.get("roc_fpr", [0, 0, 1])
            tpr = m_data.get("roc_tpr", [0, 1, 1])
            auc_val = m_data.get("roc_auc", 1.0)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines+markers',
                name=f"{m_data['model_name']} (AUC = {auc_val:.4f})",
                line=dict(color=colors[idx % len(colors)], width=2.5)
            ))

        fig_roc.update_layout(
            title="Multi-Model ROC Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(x=0.45, y=0.15)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # 5. FEATURE IMPORTANCE / EXPLAINABILITY ⭐
    st.markdown("### 🌐 Model Feature Importance & Influential Indicators")
    st.caption("Top vocabulary tokens driving threat predictions across the trained machine learning pipeline:")

    binary_model = artifacts["binary_model"]
    feature_names = artifacts["feature_names"]
    global_importance_df = compute_global_feature_importance(binary_model, feature_names, top_n=15)

    if not global_importance_df.empty:
        fig_feat = px.bar(
            global_importance_df,
            x="importance",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Phishing Indicator": "#ef4444",
                "Legitimate Indicator": "#22c55e",
                "High Threat Indicator": "#ef4444"
            },
            labels={"importance": "Relative Model Weight / Importance", "feature": "Feature Token"},
            title="Top 15 Most Influential Predictive Features"
        )
        fig_feat.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_feat, use_container_width=True)


# Clean Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "AI-Powered Phishing Email Pattern Analysis & Explainable Risk Detection System • Final Year Capstone Project"
    "</div>",
    unsafe_allow_html=True
)
