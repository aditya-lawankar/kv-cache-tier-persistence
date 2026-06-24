"""
Tests for ML-based predictive eviction models.
"""

import pytest
from src.kv_cache_tier.eviction.features import SessionFeatures
from src.kv_cache_tier.eviction.predictors import LogisticPredictor, GBTPredictor
from benchmarks.workload_simulator import WorkloadSimulator
from src.kv_cache_tier.eviction.train_predictors import extract_training_data


@pytest.fixture
def training_data():
    """Generate a small training dataset from a short enterprise trace."""
    sim = WorkloadSimulator("enterprise", duration_days=0.5, seed=123)
    events = sim.generate()
    features, labels = extract_training_data(events, resume_window_minutes=60.0)
    return features, labels


def test_session_features_to_array():
    """SessionFeatures.to_array() produces a vector of the right shape."""
    feat = SessionFeatures(
        session_age_minutes=10.0,
        token_count=2048,
        revisit_count=3,
        time_since_last_access_minutes=2.5,
        hour_of_day=14,
        day_of_week=2,
        user_historical_return_rate=0.65,
        is_business_hours=1,
        avg_session_tokens=1800.0,
    )
    arr = feat.to_array()
    assert arr.shape == (9,)
    assert arr[0] == 10.0      # session_age_minutes
    assert arr[1] == 2048      # token_count
    assert arr[7] == 1         # is_business_hours


def test_logistic_predictor_fit_and_predict(training_data):
    """LogisticPredictor can fit and produce probabilities in [0, 1]."""
    features, labels = training_data
    predictor = LogisticPredictor(resume_window_minutes=60)
    metrics = predictor.fit(features, labels)

    assert "accuracy" in metrics
    assert "auc_roc" in metrics
    assert 0 < metrics["accuracy"] < 1.0

    # Predict on a single sample
    prob = predictor.predict_resume_probability(features[0])
    assert 0.0 <= prob <= 1.0

    # Predict on a batch
    probs = predictor.predict_batch(features[:10])
    assert probs.shape == (10,)
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Coefficients are interpretable
    coefs = predictor.get_coefficients()
    assert "user_historical_return_rate" in coefs


def test_gbt_predictor_fit_and_predict(training_data):
    """GBTPredictor can fit and produce probabilities in [0, 1]."""
    features, labels = training_data
    predictor = GBTPredictor(resume_window_minutes=60, n_estimators=20, max_depth=3)
    metrics = predictor.fit(features, labels)

    assert "accuracy" in metrics
    assert 0 < metrics["accuracy"] < 1.0

    # Predict on a single sample
    prob = predictor.predict_resume_probability(features[0])
    assert 0.0 <= prob <= 1.0

    # Predict on a batch
    probs = predictor.predict_batch(features[:10])
    assert probs.shape == (10,)


def test_predictor_save_load(training_data, tmp_path):
    """Predictors can be saved and loaded from disk."""
    features, labels = training_data
    predictor = LogisticPredictor(resume_window_minutes=60)
    predictor.fit(features, labels)

    save_path = str(tmp_path / "test_model.pkl")
    predictor.save(save_path)

    loaded = LogisticPredictor.load(save_path)
    assert loaded.model_name == "logistic_predictor"

    # Predictions should be identical
    original_prob = predictor.predict_resume_probability(features[0])
    loaded_prob = loaded.predict_resume_probability(features[0])
    assert abs(original_prob - loaded_prob) < 1e-6


def test_extract_training_data_produces_labels():
    """extract_training_data produces both positive and negative labels for enterprise profiles."""
    sim = WorkloadSimulator("enterprise", duration_days=0.25, seed=99)
    events = sim.generate()
    features, labels = extract_training_data(events, resume_window_minutes=60.0)

    assert len(features) > 0
    assert len(features) == len(labels)
    assert sum(labels) > 0        # at least some resumes
    assert sum(labels) < len(labels)  # at least some non-resumes
