"""
Training data generation and model training pipeline.

This script:
  1. Generates workload traces using WorkloadSimulator for all three profiles.
  2. Extracts SessionFeatures and binary labels (resumed=1, not resumed=0)
     from the traces.
  3. Trains LogisticPredictor and GBTPredictor on the combined dataset.
  4. Evaluates both models and saves them to disk for later use.

Usage:
    python -m src.kv_cache_tier.eviction.train_predictors
    
    Or from the project root:
    python src/kv_cache_tier/eviction/train_predictors.py
"""

import json
import logging
import math
import os
import sys
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

# Ensure the project root is on the path when running as a script
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.kv_cache_tier.eviction.features import SessionFeatures
from src.kv_cache_tier.eviction.predictors import LogisticPredictor, GBTPredictor
from benchmarks.workload_simulator import WorkloadSimulator, WorkloadEvent, PROFILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_training_data(
    events: List[WorkloadEvent],
    resume_window_minutes: float = 60.0,
) -> Tuple[List[SessionFeatures], List[int]]:
    """
    Walk through a workload trace and extract (features, label) pairs.

    For each 'end' event we ask: was this session resumed within
    `resume_window_minutes`?  That gives us a binary label.

    The features are computed from everything we know at the moment
    the session ends.
    """
    features_list: List[SessionFeatures] = []
    labels: List[int] = []

    # Pre-index: session_id -> list of events, sorted by time
    session_events: Dict[str, List[WorkloadEvent]] = defaultdict(list)
    for e in events:
        session_events[e.session_id].append(e)

    # Track per-user statistics CAUSALLY (accumulated as we scan in time order)
    user_total_sessions: Dict[str, int] = defaultdict(int)
    user_total_resumes: Dict[str, int] = defaultdict(int)
    user_total_tokens: Dict[str, float] = defaultdict(float)

    # Sort all events by timestamp
    sorted_events = sorted(events, key=lambda e: e.timestamp)

    # Scan through events in time order, accumulating user stats causally
    # and extracting features only at 'end' events
    for evt in sorted_events:
        # Accumulate user stats BEFORE extracting features (causal)
        if evt.action == "start":
            user_total_sessions[evt.user_id] += 1
            user_total_tokens[evt.user_id] += evt.token_count
            continue
        elif evt.action == "resume":
            user_total_resumes[evt.user_id] += 1
            continue

        if evt.action != "end":
            continue

        sid = evt.session_id
        uid = evt.user_id
        end_ts = evt.timestamp

        # --- Compute features ---
        sess_evts = session_events[sid]

        # Session age: time from first 'start' to this 'end'
        first_start = min(e.timestamp for e in sess_evts if e.action == "start")
        session_age_minutes = (end_ts - first_start) / 60.0

        # Revisit count so far for this session
        revisit_count = sum(1 for e in sess_evts if e.action == "resume" and e.timestamp <= end_ts)

        # Time since last access (the most recent start/resume before this end)
        prior_accesses = [e.timestamp for e in sess_evts
                          if e.action in ("start", "resume") and e.timestamp <= end_ts]
        last_access_ts = max(prior_accesses) if prior_accesses else first_start
        time_since_last_access_minutes = (end_ts - last_access_ts) / 60.0

        # Time-of-day and day-of-week (simulate from timestamp offset)
        # We treat timestamp 0 as Monday 00:00
        total_seconds = end_ts
        hour_of_day = int((total_seconds / 3600) % 24)
        day_of_week = int((total_seconds / 86400) % 7)

        # User historical return rate (causal — only past data)
        total_sess = user_total_sessions.get(uid, 1)
        total_resumes_count = user_total_resumes.get(uid, 0)
        user_return_rate = min(total_resumes_count / max(total_sess, 1), 1.0)

        # Is business hours? (Mon-Fri 9am-5pm)
        is_business_hours = 1 if (9 <= hour_of_day < 17 and day_of_week < 5) else 0

        # Average session tokens for this user (causal)
        avg_tokens = user_total_tokens.get(uid, evt.token_count) / max(total_sess, 1)

        feat = SessionFeatures(
            session_age_minutes=session_age_minutes,
            token_count=evt.token_count,
            revisit_count=revisit_count,
            time_since_last_access_minutes=time_since_last_access_minutes,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            user_historical_return_rate=user_return_rate,
            is_business_hours=is_business_hours,
            avg_session_tokens=avg_tokens,
        )

        # --- Compute label ---
        # Did a 'resume' event for this session occur within resume_window_minutes
        # after this end event?
        window_seconds = resume_window_minutes * 60.0
        resumed = any(
            e.action == "resume" and 0 < (e.timestamp - end_ts) <= window_seconds
            for e in sess_evts
        )
        label = 1 if resumed else 0

        features_list.append(feat)
        labels.append(label)

    return features_list, labels


def train_and_evaluate(
    output_dir: str = "models",
    resume_window_minutes: float = 60.0,
    duration_days: float = 7.0,
    seed: int = 42,
) -> dict:
    """
    Full training pipeline:
      1. Generate traces for all three workload profiles.
      2. Extract features + labels.
      3. Train LogisticPredictor and GBTPredictor.
      4. Evaluate on a held-out split.
      5. Save trained models and results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Generate traces for all profiles ----
    all_features: List[SessionFeatures] = []
    all_labels: List[int] = []

    for profile_name in PROFILES:
        logger.info(f"Generating {duration_days}-day trace for profile: {profile_name}")
        sim = WorkloadSimulator(profile_name, duration_days=duration_days, seed=seed)
        events = sim.generate()
        summary = sim.summary(events)
        logger.info(f"  -> {summary['total_events']} events, "
                     f"{summary['unique_sessions']} sessions, "
                     f"{summary['total_resumes']} resumes")

        feats, lbls = extract_training_data(events, resume_window_minutes)
        logger.info(f"  -> {len(feats)} training samples, "
                     f"{sum(lbls)} positive, {len(lbls) - sum(lbls)} negative")

        all_features.extend(feats)
        all_labels.extend(lbls)

    logger.info(f"\nTotal training data: {len(all_features)} samples "
                 f"({sum(all_labels)} positive, {len(all_labels) - sum(all_labels)} negative)")

    # ---- 2. Train/test split ----
    from sklearn.model_selection import train_test_split

    indices = list(range(len(all_features)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed,
                                            stratify=all_labels)

    train_features = [all_features[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_features = [all_features[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    logger.info(f"Train: {len(train_features)} samples | Test: {len(test_features)} samples")

    results = {}

    # ---- 3. Train Logistic Predictor ----
    logger.info("\n--- Training Logistic Predictor ---")
    logistic = LogisticPredictor(resume_window_minutes=int(resume_window_minutes))
    train_metrics_lr = logistic.fit(train_features, train_labels)
    logger.info(f"  Train -> accuracy={train_metrics_lr['accuracy']:.4f}, "
                 f"AUC={train_metrics_lr['auc_roc']:.4f}")

    # Evaluate on test set
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    test_probs_lr = logistic.predict_batch(test_features)
    test_preds_lr = (test_probs_lr >= 0.5).astype(int)
    test_acc_lr = accuracy_score(test_labels, test_preds_lr)
    test_auc_lr = roc_auc_score(test_labels, test_probs_lr) if len(set(test_labels)) > 1 else 0.0

    logger.info(f"  Test  -> accuracy={test_acc_lr:.4f}, AUC={test_auc_lr:.4f}")
    logger.info(f"  Coefficients: {logistic.get_coefficients()}")

    results["logistic"] = {
        "train_accuracy": train_metrics_lr["accuracy"],
        "train_auc": train_metrics_lr["auc_roc"],
        "test_accuracy": test_acc_lr,
        "test_auc": test_auc_lr,
        "coefficients": logistic.get_coefficients(),
    }

    logistic.save(os.path.join(output_dir, "logistic_predictor.pkl"))

    # ---- 4. Train GBT Predictor ----
    logger.info("\n--- Training GBT Predictor ---")
    gbt = GBTPredictor(resume_window_minutes=int(resume_window_minutes))
    train_metrics_gbt = gbt.fit(train_features, train_labels)
    logger.info(f"  Train -> accuracy={train_metrics_gbt['accuracy']:.4f}, "
                 f"AUC={train_metrics_gbt['auc_roc']:.4f}")

    test_probs_gbt = gbt.predict_batch(test_features)
    test_preds_gbt = (test_probs_gbt >= 0.5).astype(int)
    test_acc_gbt = accuracy_score(test_labels, test_preds_gbt)
    test_auc_gbt = roc_auc_score(test_labels, test_probs_gbt) if len(set(test_labels)) > 1 else 0.0

    logger.info(f"  Test  -> accuracy={test_acc_gbt:.4f}, AUC={test_auc_gbt:.4f}")
    logger.info(f"  Feature importances: {gbt.get_feature_importances()}")

    results["gbt"] = {
        "train_accuracy": train_metrics_gbt["accuracy"],
        "train_auc": train_metrics_gbt["auc_roc"],
        "test_accuracy": test_acc_gbt,
        "test_auc": test_auc_gbt,
        "feature_importances": gbt.get_feature_importances(),
    }

    gbt.save(os.path.join(output_dir, "gbt_predictor.pkl"))

    # ---- 5. Save results summary ----
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # ---- 6. Print comparison table ----
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON — Session Resumption Prediction")
    print("=" * 70)
    print(f"  {'Model':<20} {'Train Acc':>10} {'Test Acc':>10} {'Train AUC':>10} {'Test AUC':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Logistic Reg.':<20} {train_metrics_lr['accuracy']:>10.4f} {test_acc_lr:>10.4f} "
          f"{train_metrics_lr['auc_roc']:>10.4f} {test_auc_lr:>10.4f}")
    print(f"  {'Gradient Boosted':<20} {train_metrics_gbt['accuracy']:>10.4f} {test_acc_gbt:>10.4f} "
          f"{train_metrics_gbt['auc_roc']:>10.4f} {test_auc_gbt:>10.4f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    train_and_evaluate(
        output_dir="models",
        resume_window_minutes=60.0,
        duration_days=7.0,
        seed=42,
    )
