"""
src/feature_engineering.py
Cybersecurity Feature Extraction, Structural Indicators & TF-IDF Vectorization Pipeline.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import (
    URL_PATTERN,
    IP_URL_PATTERN,
    clean_email_text
)

# Regex dictionaries for heuristic behavioral cybersecurity flags
URGENCY_REGEX = re.compile(
    r'\b(urgent|urgently|immediate|immediately|action required|suspended|suspension|'
    r'expires|expired|expiration|within 24 hours|within 48 hours|final notice|'
    r'account closure|deactivation|security alert|warning|unauthorized|blocked|'
    r'attention required|act now|critical notice|temporarily locked)\b',
    re.IGNORECASE
)

FINANCIAL_REGEX = re.compile(
    r'\b(invoice|invoices|wire transfer|payment|credit card|debit card|bank|'
    r'banking|refund|direct deposit|payroll|dollars|\$|bitcoin|crypto|'
    r'billing|overdue|balance|remittance|statement|escrow|transaction)\b',
    re.IGNORECASE
)

CREDENTIAL_REGEX = re.compile(
    r'\b(password|passcode|pin|login|log in|sign in|username|credentials|'
    r'sso|otp|one-time code|two-factor|2fa|mfa|reset password|'
    r'verify identity|confirm credentials|update details|security questions)\b',
    re.IGNORECASE
)


FEATURE_COLUMNS = [
    "feat_num_words",
    "feat_num_uppercase",
    "feat_uppercase_ratio",
    "feat_num_exclamation",
    "feat_num_question",
    "feat_num_urls",
    "feat_has_urgent",
    "feat_has_financial",
    "feat_has_credential",
    "feat_has_ip_url",
]


def extract_structural_features(raw_text: str) -> Dict[str, float]:
    """
    Extract structural and NLP heuristic indicators from raw email text.

    Args:
        raw_text: Uncleaned original email text containing headers and formatting.

    Returns:
        Dict[str, float]: Dictionary of numerical and boolean features.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {col: 0.0 for col in FEATURE_COLUMNS}

    total_chars = len(raw_text)
    words = raw_text.split()
    num_words = len(words)

    # Uppercase letter count and ratio
    uppercase_chars = sum(1 for c in raw_text if c.isupper())
    uppercase_ratio = uppercase_chars / max(total_chars, 1)

    # Punctuation counts
    num_exclamation = raw_text.count('!')
    num_question = raw_text.count('?')

    # URL counts
    urls_found = URL_PATTERN.findall(raw_text)
    num_urls = len(urls_found)
    has_ip_url = 1.0 if IP_URL_PATTERN.search(raw_text) else 0.0

    # Keyword heuristics
    has_urgent = 1.0 if URGENCY_REGEX.search(raw_text) else 0.0
    has_financial = 1.0 if FINANCIAL_REGEX.search(raw_text) else 0.0
    has_credential = 1.0 if CREDENTIAL_REGEX.search(raw_text) else 0.0

    return {
        "feat_num_words": float(num_words),
        "feat_num_uppercase": float(uppercase_chars),
        "feat_uppercase_ratio": float(uppercase_ratio),
        "feat_num_exclamation": float(num_exclamation),
        "feat_num_question": float(num_question),
        "feat_num_urls": float(num_urls),
        "feat_has_urgent": float(has_urgent),
        "feat_has_financial": float(has_financial),
        "feat_has_credential": float(has_credential),
        "feat_has_ip_url": float(has_ip_url),
    }


def extract_feature_dataframe(raw_texts: pd.Series) -> pd.DataFrame:
    """
    Extract structural features for a Series of raw email texts.

    Args:
        raw_texts: pd.Series of raw email strings.

    Returns:
        pd.DataFrame: DataFrame containing all numerical FEATURE_COLUMNS.
    """
    feature_list = [extract_structural_features(text) for text in raw_texts]
    return pd.DataFrame(feature_list, columns=FEATURE_COLUMNS)


class CybersecurityFeaturePipeline:
    """
    Combined TF-IDF and Structural Feature Extraction Pipeline for Phishing Analysis.
    """

    def __init__(
        self,
        max_features: int = 3000,
        ngram_range: Tuple[int, int] = (1, 2),
        include_structural: bool = True
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.include_structural = include_structural
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            token_pattern=r'(?u)\b\w+\b|\[\w+_PLACEHOLDER\]'
        )
        self.feature_names: List[str] = []

    def fit(self, raw_texts: pd.Series, cleaned_texts: Optional[pd.Series] = None):
        """Fit the TF-IDF vectorizer and compile feature names."""
        if cleaned_texts is None:
            cleaned_texts = raw_texts.apply(clean_email_text)

        self.vectorizer.fit(cleaned_texts)
        tfidf_features = list(self.vectorizer.get_feature_names_out())

        if self.include_structural:
            self.feature_names = tfidf_features + FEATURE_COLUMNS
        else:
            self.feature_names = tfidf_features
        return self

    def transform(self, raw_texts: pd.Series, cleaned_texts: Optional[pd.Series] = None) -> csr_matrix:
        """Transform raw and cleaned texts into a sparse feature matrix."""
        if cleaned_texts is None:
            cleaned_texts = raw_texts.apply(clean_email_text)

        tfidf_matrix = self.vectorizer.transform(cleaned_texts)

        if self.include_structural:
            struct_df = extract_feature_dataframe(raw_texts)
            struct_matrix = csr_matrix(struct_df[FEATURE_COLUMNS].values)
            combined_matrix = hstack([tfidf_matrix, struct_matrix]).tocsr()
            return combined_matrix
        return tfidf_matrix

    def fit_transform(self, raw_texts: pd.Series, cleaned_texts: Optional[pd.Series] = None) -> csr_matrix:
        """Fit and transform in a single call."""
        self.fit(raw_texts, cleaned_texts)
        return self.transform(raw_texts, cleaned_texts)
