"""
ML-based predictors for session resumption probability.

Three tiers of prediction models for comparison:
  1. HeuristicPredictor  - Weighted formula baseline (already exists in predictive.py)
  2. LogisticPredictor   - Interpretable logistic regression
  3. GBTPredictor        - Gradient boosted trees (highest accuracy)

All predictors share a common interface: fit() and predict_resume_probability().
"""

import logging
import pickle
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .features import SessionFeatures

logger = logging.getLogger(__name__)


class ResumePredictor(ABC):
    """Abstract interface for session resumption predictors."""

    @abstractmethod
    def fit(self, features: List[SessionFeatures], labels: List[int]) -> dict:
        """
        Train the model.

        Args:
            features: List of SessionFeatures, one per training sample.
            labels: List of binary labels (1 = session was resumed, 0 = not resumed).

        Returns:
            Dictionary of training metrics (e.g. accuracy, AUC).
        """
        pass

    @abstractmethod
    def predict_resume_probability(self, features: SessionFeatures) -> float:
        """
        Predict the probability that a session will be resumed.

        Args:
            features: SessionFeatures for the cache entry to evaluate.

        Returns:
            Float in [0, 1] representing P(session resumes).
        """
        pass

    @abstractmethod
    def predict_batch(self, features_list: List[SessionFeatures]) -> np.ndarray:
        """Predict probabilities for a batch of sessions."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def save(self, path: str) -> None:
        """Persist the trained model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.model_name} to {path}")

    @staticmethod
    def load(path: str) -> "ResumePredictor":
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded {model.model_name} from {path}")
        return model


def _features_to_matrix(features_list: List[SessionFeatures]) -> np.ndarray:
    """Convert a list of SessionFeatures into a 2D numpy array."""
    return np.vstack([f.to_array() for f in features_list])


# ---------------------------------------------------------------------------
# 1. Logistic Regression Predictor
# ---------------------------------------------------------------------------

class LogisticPredictor(ResumePredictor):
    """
    Logistic Regression predictor for session resumption.

    This model is interpretable: coefficients directly reveal which features
    drive cache retention decisions. For example, a large positive coefficient
    on `user_historical_return_rate` means the model learned that users with
    high historical return rates are much more likely to resume.
    """

    def __init__(self, resume_window_minutes: int = 60):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.resume_window = resume_window_minutes
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        return "logistic_predictor"

    def fit(self, features: List[SessionFeatures], labels: List[int]) -> dict:
        X = _features_to_matrix(features)
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        # Training metrics
        from sklearn.metrics import accuracy_score, roc_auc_score

        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "auc_roc": float(roc_auc_score(y, y_prob)) if len(set(y)) > 1 else 0.0,
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        }

        # Log feature importances (interpretable coefficients)
        coefs = self.model.coef_[0]
        feature_names = SessionFeatures.feature_names()
        logger.info("Logistic Regression coefficients:")
        for name, coef in zip(feature_names, coefs):
            logger.info(f"  {name:40s} = {coef:+.4f}")

        return metrics

    def predict_resume_probability(self, features: SessionFeatures) -> float:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict_proba(X_scaled)[0][1])

    def predict_batch(self, features_list: List[SessionFeatures]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = _features_to_matrix(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_coefficients(self) -> dict:
        """Return a dict mapping feature name -> learned coefficient."""
        if not self._is_fitted:
            return {}
        return dict(zip(SessionFeatures.feature_names(), self.model.coef_[0].tolist()))


# ---------------------------------------------------------------------------
# 2. Gradient Boosted Trees Predictor
# ---------------------------------------------------------------------------

class GBTPredictor(ResumePredictor):
    """
    Gradient Boosted Trees predictor for session resumption.

    Uses sklearn's HistGradientBoostingClassifier for fast training.
    Higher accuracy than logistic regression but less interpretable.
    Feature importances are available via permutation importance.
    """

    def __init__(self, resume_window_minutes: int = 60,
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 learning_rate: float = 0.1):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.resume_window = resume_window_minutes
        self.model = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weight="balanced",
            random_state=42,
        )
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        return "gbt_predictor"

    def fit(self, features: List[SessionFeatures], labels: List[int]) -> dict:
        X = _features_to_matrix(features)
        y = np.array(labels)

        self.model.fit(X, y)
        self._is_fitted = True

        from sklearn.metrics import accuracy_score, roc_auc_score

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "auc_roc": float(roc_auc_score(y, y_prob)) if len(set(y)) > 1 else 0.0,
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        }

        return metrics

    def predict_resume_probability(self, features: SessionFeatures) -> float:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = features.to_array().reshape(1, -1)
        return float(self.model.predict_proba(X)[0][1])

    def predict_batch(self, features_list: List[SessionFeatures]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = _features_to_matrix(features_list)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> dict:
        """Return a dict mapping feature name -> importance (for tree models)."""
        if not self._is_fitted:
            return {}
        # HistGradientBoosting doesn't expose feature_importances_ directly in
        # all sklearn versions, so we fall back gracefully.
        try:
            importances = self.model.feature_importances_
        except AttributeError:
            return {}
        return dict(zip(SessionFeatures.feature_names(), importances.tolist()))
