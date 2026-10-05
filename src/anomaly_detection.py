"""
src/anomaly_detection.py
Unsupervised Anomaly Detection using Isolation Forest for Zero-Day & Novel Attack Vectors.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD


class ZeroDayAnomalyDetector:
    """
    Isolation Forest Anomaly Detector for identifying novel or emerging phishing vectors
    that deviate substantially from standard baseline email distribution.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.reducer = TruncatedSVD(n_components=2, random_state=random_state)
        self.is_fitted = False

    def fit(self, feature_matrix):
        """Fit the Isolation Forest model and 2D TruncatedSVD projection."""
        self.model.fit(feature_matrix)
        self.reducer.fit(feature_matrix)
        self.is_fitted = True
        return self

    def predict_anomaly(self, sample_vector) -> Dict[str, Any]:
        """
        Assess whether an incoming email represents an anomalous / emerging threat pattern.

        Args:
            sample_vector: 1xD sparse or dense vector.

        Returns:
            Dict containing:
                - 'is_anomaly': bool (True if anomalous/unseen, False otherwise)
                - 'anomaly_score': float (-1.0 to 1.0, lower means more anomalous)
                - 'status_label': str ('Emerging Novel Pattern / Anomaly' or 'Standard Pattern')
                - 'coordinates_2d': Tuple[float, float]
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector must be fitted before running inference.")

        # IsolationForest decision_function returns anomaly score (negative = anomaly)
        raw_score = float(self.model.decision_function(sample_vector)[0])
        pred = int(self.model.predict(sample_vector)[0])  # -1 for anomaly, 1 for normal

        is_anomaly = (pred == -1)
        status_label = "Emerging / Zero-Day Pattern Anomaly" if is_anomaly else "Familiar Baseline Pattern"

        # 2D projection for visualization
        coords = self.reducer.transform(sample_vector)[0]

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(raw_score, 4),
            "status_label": status_label,
            "coords_2d": (round(float(coords[0]), 4), round(float(coords[1]), 4))
        }

    def transform_coordinates(self, feature_matrix) -> np.ndarray:
        """Project high-dimensional matrix into 2D plane for scatter plotting."""
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transforming.")
        return self.reducer.transform(feature_matrix)
