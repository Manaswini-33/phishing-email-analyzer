"""
src/preprocessing.py
Robust Text Cleaning, Normalization, and Regex Tokenization Pipeline.
"""

import re
import html
from typing import List, Union
import pandas as pd


# Regex Patterns for Cybersecurity Text Sanitization
URL_PATTERN = re.compile(
    r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'https?:\/\/[^\s]+|'
    r'http:\/\/[^\s]+)',
    re.IGNORECASE
)

IP_URL_PATTERN = re.compile(
    r'https?:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?(?:\/[^\s]*)?',
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    re.IGNORECASE
)

CURRENCY_PATTERN = re.compile(
    r'(\$[\d,]+(?:\.\d+)?|\€[\d,]+(?:\.\d+)?|\£[\d,]+(?:\.\d+)?|\d+\s*(?:usd|eur|gbp|dollars|bitcoins?|btc))',
    re.IGNORECASE
)

HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_email_text(
    text: str,
    replace_urls: bool = True,
    replace_emails: bool = True,
    replace_currency: bool = True,
    to_lower: bool = True
) -> str:
    """
    Clean and sanitize raw email text for machine learning modeling.

    Args:
        text: Raw email subject + body text string.
        replace_urls: Whether to replace URLs with [URL_PLACEHOLDER].
        replace_emails: Whether to replace emails with [EMAIL_PLACEHOLDER].
        replace_currency: Whether to replace currency figures with [CURRENCY_PLACEHOLDER].
        to_lower: Whether to convert text to lowercase.

    Returns:
        str: Sanitized and normalized email text string.
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
    cleaned = html.unescape(text)

    # Strip HTML tags
    cleaned = HTML_TAG_PATTERN.sub(' ', cleaned)

    # Handle IP-based URLs first (more specific)
    cleaned = IP_URL_PATTERN.sub(' [IP_URL_PLACEHOLDER] ', cleaned)

    # Replace regular URLs with token placeholder
    if replace_urls:
        cleaned = URL_PATTERN.sub(' [URL_PLACEHOLDER] ', cleaned)

    # Replace email addresses with token placeholder
    if replace_emails:
        cleaned = EMAIL_PATTERN.sub(' [EMAIL_PLACEHOLDER] ', cleaned)

    # Replace currency amounts
    if replace_currency:
        cleaned = CURRENCY_PATTERN.sub(' [CURRENCY_PLACEHOLDER] ', cleaned)

    # Normalize whitespace
    cleaned = WHITESPACE_PATTERN.sub(' ', cleaned).strip()

    # Case normalization
    if to_lower:
        cleaned = cleaned.lower()

    return cleaned


def preprocess_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Preprocess entire email DataFrame:
    1. Drop missing records in text/label.
    2. Remove duplicate text entries.
    3. Retain raw text for structural feature extraction and add cleaned_text column.

    Args:
        df: Input pandas DataFrame with columns 'text', 'label', etc.
        drop_duplicates: Whether to drop duplicate email text entries.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'cleaned_text' column added.
    """
    processed = df.copy()

    # Drop null rows in critical columns
    processed = processed.dropna(subset=["text", "label"]).reset_index(drop=True)

    # Deduplicate
    if drop_duplicates:
        initial_count = len(processed)
        processed = processed.drop_duplicates(subset=["text"]).reset_index(drop=True)
        dropped_count = initial_count - len(processed)
        if dropped_count > 0:
            print(f"[Preprocessing] Removed {dropped_count} duplicate email entries.")

    # Apply text cleaning
    processed["cleaned_text"] = processed["text"].apply(clean_email_text)

    return processed
