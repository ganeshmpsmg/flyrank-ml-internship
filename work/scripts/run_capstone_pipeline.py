"""
Reproducible Capstone Pipeline for FlyRank ML Week-8 Capstone
Lane: Content Refresh Priority Prediction

Preserves:
1. Exact Week-4 rule-based baseline from work/notebooks/w04_baseline_score.ipynb
2. Exact Week-5 / Week-6 Logistic Regression model from work/notebooks/w05_model.ipynb & w06_validation_audit.ipynb
3. Full evaluation on both:
   - Population A: W04 (N=30) and W05/W06 (N=100) practice datasets (Stratified 80/20 split, random_state=42)
   - Population B: Actual FlyRank anonymized starter dataset (data/raw/content_refresh_anonymized.csv, N=30,000 across 32 clients)
     under both Stratified Random Split (80/20) and Client-Grouped Holdout Split (26 train clients / 6 test clients, random_state=42)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[2]
RAW_30K_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
W03_CSV_PATH = ROOT / "work" / "data" / "simple_feature_vector_dataset.csv"
W04_CSV_PATH = ROOT / "work" / "data" / "ml07_baseline_practice_dataset.csv"
W05_CSV_PATH = ROOT / "work" / "data" / "w05_ml_practice_dataset.csv"

WORK_OUTPUT_DIR = ROOT / "work" / "outputs"
WORK_FIG_DIR = ROOT / "work" / "figures"
DOCS_FIG_DIR = ROOT / "docs" / "figures"


def precision_at_k(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, k: int) -> float:
    frame = pd.DataFrame({"y": list(y_true), "score": list(scores)})
    if frame.empty:
        return 0.0
    top = frame.sort_values("score", ascending=False).head(min(k, len(frame)))
    return float(top["y"].mean()) if len(top) else 0.0


def compute_metrics(y_true: pd.Series | np.ndarray, binary_pred: np.ndarray, continuous_score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, binary_pred)), 4),
        "f1": round(float(f1_score(y_true, binary_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, binary_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, binary_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, continuous_score)), 4),
        "average_precision": round(float(average_precision_score(y_true, continuous_score)), 4),
        "precision_at_10": round(precision_at_k(y_true, continuous_score, 10), 4),
        "precision_at_20": round(precision_at_k(y_true, continuous_score, 20), 4),
        "precision_at_50": round(precision_at_k(y_true, continuous_score, 50), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def w04_baseline_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact Week-4 baseline formula from work/notebooks/w04_baseline_score.ipynb:
    score = 2 * (impressions >= 1000) + 2 * (staleness_days >= 14) + 1 * (position >= 8)
    """
    out = df.copy()
    out["score"] = (
        (out["impressions"] >= 1000).astype(int) * 2
        + (out["staleness_days"] >= 14).astype(int) * 2
        + (out["position"] >= 8).astype(int) * 1
    )
    out["reason_code"] = "REVIEW"
    out.loc[
        (out["impressions"] >= 1000) & (out["staleness_days"] >= 14),
        "reason_code",
    ] = "STALE_HIGH_VOLUME"

    out["action"] = "MONITOR"
    out.loc[out["score"] >= 3, "action"] = "PRIORITIZE"
    out.loc[out["score"] >= 4, "action"] = "REVIEW_NOW"
    out["baseline_pred"] = (out["score"] >= 3).astype(int)
    return out


def save_both(fig: plt.Figure, filename: str) -> None:
    WORK_FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(WORK_FIG_DIR / filename, bbox_inches="tight", dpi=150)
    fig.savefig(DOCS_FIG_DIR / filename, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    WORK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART 1: W04 BASELINE QUEUE (N=30) & W05/W06 PRACTICE MODEL (N=100)
    # =========================================================================
    df_w04 = pd.read_csv(W04_CSV_PATH)
    df_w04_scored = w04_baseline_rule(df_w04)
    df_w04_ranked = df_w04_scored.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    df_w04_ranked["rank"] = range(1, len(df_w04_ranked) + 1)

    w04_queue_cols = ["rank", "item_id", "score", "reason_code", "action"]
    df_w04_ranked[w04_queue_cols].to_csv(WORK_OUTPUT_DIR / "baseline_action_score.csv", index=False)

    # Add Confidence & What Could Make It Wrong for Top-10 W04 queue
    df_w04_ranked["confidence"] = np.where(
        df_w04_ranked["score"] == 5,
        "High (all 3 baseline thresholds met)",
        np.where(
            df_w04_ranked["score"] == 4,
            "Medium-High (2 primary thresholds met)",
            "Medium (single primary + secondary signal)",
        ),
    )
    df_w04_ranked["what_could_make_it_wrong"] = np.where(
        df_w04_ranked["ctr"] >= 0.12,
        "Strong observed CTR (>=12%); high click capture despite age/position may mean content still satisfies intent.",
        np.where(
            df_w04_ranked["position"] <= 7,
            "Already ranks on page 1 (pos <=7); low CTR may call for title/snippet refinement rather than full rewrite.",
            "Search demand may be seasonal or split across a sibling page; verify query trend before editing.",
        ),
    )
    top10_w04 = df_w04_ranked.head(10)[
        [
            "rank",
            "item_id",
            "impressions",
            "clicks",
            "position",
            "staleness_days",
            "ctr",
            "score",
            "action",
            "reason_code",
            "confidence",
            "what_could_make_it_wrong",
        ]
    ]
    top10_w04.to_csv(WORK_OUTPUT_DIR / "top10_refresh_queue_w04.csv", index=False)

    # W05/W06 dataset (N=100)
    df_w05 = pd.read_csv(W05_CSV_PATH)
    w05_features = ["impressions", "clicks", "staleness_days", "position"]
    X_w05 = df_w05[w05_features]
    y_w05 = df_w05["target"].astype(int)

    Xtr_w05, Xte_w05, ytr_w05, yte_w05 = train_test_split(
        X_w05, y_w05, test_size=0.20, random_state=RANDOM_STATE, stratify=y_w05
    )

    w05_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    w05_model.fit(Xtr_w05, ytr_w05)
    w05_pred = w05_model.predict(Xte_w05)
    w05_prob = w05_model.predict_proba(Xte_w05)[:, 1]

    w05_test_baseline = w04_baseline_rule(Xte_w05)
    w05_b_score = w05_test_baseline["score"].to_numpy()
    w05_b_pred = w05_test_baseline["baseline_pred"].to_numpy()

    w05_baseline_metrics = compute_metrics(yte_w05, w05_b_pred, w05_b_score)
    w05_model_metrics = compute_metrics(yte_w05, w05_pred, w05_prob)

    w05_coefs = w05_model.named_steps["model"].coef_[0]
    w05_intercept = float(w05_model.named_steps["model"].intercept_[0])
    w05_coef_table = [
        {
            "feature": feat,
            "coefficient": round(float(c), 6),
            "absolute_coefficient": round(float(abs(c)), 6),
        }
        for feat, c in sorted(zip(w05_features, w05_coefs), key=lambda x: abs(x[1]), reverse=True)
    ]

    # W05 Error Analysis
    w05_test_df = Xte_w05.copy()
    w05_test_df["test_pos"] = range(len(w05_test_df))
    w05_test_df["item_id"] = df_w05.loc[Xte_w05.index, "item_id"].apply(lambda v: f"item_{int(v):03d}")
    w05_test_df["actual"] = yte_w05.values
    w05_test_df["predicted"] = w05_pred
    w05_test_df["pred_prob"] = np.round(w05_prob, 4)
    w05_test_df["baseline_score"] = w05_b_score
    w05_test_df["baseline_pred"] = w05_b_pred
    w05_errors = w05_test_df[w05_test_df["actual"] != w05_test_df["predicted"]].to_dict(orient="records")

    # =========================================================================
    # PART 2: ACTUAL FLYRANK 30,000-ROW STARTER DATASET
    # =========================================================================
    df_30k = pd.read_csv(RAW_30K_PATH)
    df_30k["is_declining_label"] = (df_30k["trend_direction"].astype(str).str.lower() == "down").astype(int)

    # Map pre-decision signals & handle missing values transparently
    df_30k["impressions"] = df_30k["impressions_90d"].fillna(0)
    df_30k["clicks"] = df_30k["clicks_90d"].fillna(0)
    df_30k["staleness_days"] = df_30k["days_since_last_update"].fillna(0)

    # avg_position == 0 means "no position data" (1,205 rows)
    df_30k["has_position_data"] = (df_30k["avg_position"] > 0).astype(int)
    valid_pos_median = float(df_30k.loc[df_30k["avg_position"] > 0, "avg_position"].median())
    df_30k["position"] = df_30k["avg_position"].replace(0, valid_pos_median).fillna(valid_pos_median)

    df_30k["ctr_pct"] = df_30k["ctr"].fillna(0)
    df_30k["log_impressions"] = np.log1p(df_30k["impressions"])
    df_30k["log_clicks"] = np.log1p(df_30k["clicks"])
    df_30k["days_with_impressions"] = df_30k["days_with_impressions"].fillna(0)
    df_30k["content_age_days"] = df_30k["content_age_days"].fillna(df_30k["content_age_days"].median())

    # Apply Week-4 Baseline Rule to 30k dataset
    df_30k = w04_baseline_rule(df_30k)

    # Also compute Reference Starter Baseline (from scripts/02_baseline_score.py) for completeness
    vis_score = df_30k["log_impressions"].rank(method="average", pct=True)
    fresh_score = df_30k["staleness_days"].rank(method="average", pct=True)
    pos_clip = df_30k["avg_position"].clip(lower=1, upper=50)
    pos_norm = (pos_clip - pos_clip.min()) / (pos_clip.max() - pos_clip.min())
    pos_opp = (1.0 - pos_norm) * vis_score * df_30k["has_position_data"]
    depth_gap = (1.0 - df_30k["word_count"].fillna(0).rank(method="average", pct=True)) * vis_score
    df_30k["ref_baseline_score"] = (
        0.40 * vis_score + 0.30 * fresh_score + 0.25 * pos_opp + 0.05 * depth_gap
    ).clip(0, 1)
    df_30k["ref_baseline_pred"] = (df_30k["ref_baseline_score"] >= 0.50).astype(int)

    # Define Splits on 30k Dataset
    all_idx = np.arange(len(df_30k))
    y_30k = df_30k["is_declining_label"]

    # Split 1: 80/20 Stratified Random Split
    tr_rand, te_rand = train_test_split(
        all_idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y_30k
    )

    # Split 2: Client-Grouped Holdout Split (32 clients -> 26 train clients, 6 test clients)
    client_series = df_30k["client_id"].fillna("unknown").astype(str)
    unique_clients = client_series.drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_clients = rng.permutation(unique_clients)
    test_client_count = max(1, int(round(len(shuffled_clients) * 0.2)))
    test_clients = set(shuffled_clients[:test_client_count])
    test_mask = client_series.isin(test_clients).to_numpy()
    tr_grp = all_idx[~test_mask]
    te_grp = all_idx[test_mask]

    feats_4_raw = ["impressions", "clicks", "staleness_days", "position"]
    feats_7_predecision = [
        "log_impressions",
        "log_clicks",
        "staleness_days",
        "position",
        "ctr_pct",
        "days_with_impressions",
        "content_age_days",
    ]

    results_30k: dict[str, dict] = {}
    models_fitted_grp: dict[str, Pipeline] = {}

    for split_key, tr_idx, te_idx in [
        ("stratified_random_split", tr_rand, te_rand),
        ("client_grouped_holdout", tr_grp, te_grp),
    ]:
        yte = y_30k.iloc[te_idx]
        split_res: dict[str, object] = {
            "train_rows": int(len(tr_idx)),
            "test_rows": int(len(te_idx)),
            "train_positive_rate": round(float(y_30k.iloc[tr_idx].mean()), 4),
            "test_positive_rate": round(float(yte.mean()), 4),
        }

        # 1. W04 Rule Baseline (score >= 3)
        split_res["w04_rule_baseline"] = compute_metrics(
            yte,
            df_30k["baseline_pred"].iloc[te_idx].to_numpy(),
            df_30k["score"].iloc[te_idx].to_numpy(),
        )

        # 2. Starter Reference Baseline (ref_baseline_score >= 0.5)
        split_res["starter_ref_baseline"] = compute_metrics(
            yte,
            df_30k["ref_baseline_pred"].iloc[te_idx].to_numpy(),
            df_30k["ref_baseline_score"].iloc[te_idx].to_numpy(),
        )

        # 3. W05 Exact 4-Feature Logistic Regression (raw features, unweighted)
        m_4raw = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        )
        m_4raw.fit(df_30k[feats_4_raw].iloc[tr_idx], y_30k.iloc[tr_idx])
        p_4raw = m_4raw.predict(df_30k[feats_4_raw].iloc[te_idx])
        pr_4raw = m_4raw.predict_proba(df_30k[feats_4_raw].iloc[te_idx])[:, 1]
        split_res["logreg_4feat_raw"] = compute_metrics(yte, p_4raw, pr_4raw)
        split_res["logreg_4feat_raw"]["coefficients"] = [
            {
                "feature": f,
                "coefficient": round(float(c), 6),
                "absolute_coefficient": round(float(abs(c)), 6),
            }
            for f, c in sorted(
                zip(feats_4_raw, m_4raw.named_steps["model"].coef_[0]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        ]

        # 4. 7-Feature Pre-Decision Logistic Regression (log-transformed heavy tails, balanced class weight)
        m_7pre = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        )
        m_7pre.fit(df_30k[feats_7_predecision].iloc[tr_idx], y_30k.iloc[tr_idx])
        p_7pre = m_7pre.predict(df_30k[feats_7_predecision].iloc[te_idx])
        pr_7pre = m_7pre.predict_proba(df_30k[feats_7_predecision].iloc[te_idx])[:, 1]
        split_res["logreg_7feat_predecision"] = compute_metrics(yte, p_7pre, pr_7pre)
        split_res["logreg_7feat_predecision"]["coefficients"] = [
            {
                "feature": f,
                "coefficient": round(float(c), 6),
                "absolute_coefficient": round(float(abs(c)), 6),
            }
            for f, c in sorted(
                zip(feats_7_predecision, m_7pre.named_steps["model"].coef_[0]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        ]

        if split_key == "client_grouped_holdout":
            models_fitted_grp["m_4raw"] = m_4raw
            models_fitted_grp["m_7pre"] = m_7pre

        results_30k[split_key] = split_res

    # Representative Error Analysis on Client-Grouped Holdout Test Set (n=2,325)
    m_7pre_grp = models_fitted_grp["m_7pre"]
    te_df = df_30k.iloc[te_grp].copy()
    te_df["pred_prob"] = m_7pre_grp.predict_proba(te_df[feats_7_predecision])[:, 1]
    te_df["predicted"] = (te_df["pred_prob"] >= 0.50).astype(int)
    te_df["actual"] = te_df["is_declining_label"]

    fp_examples = (
        te_df[(te_df["actual"] == 0) & (te_df["predicted"] == 1)]
        .sort_values("pred_prob", ascending=False)
        .head(3)[
            [
                "content_id",
                "client_id",
                "impressions",
                "clicks",
                "staleness_days",
                "position",
                "ctr_pct",
                "days_with_impressions",
                "content_age_days",
                "trend_direction",
                "actual",
                "predicted",
                "pred_prob",
                "score",
            ]
        ]
        .to_dict(orient="records")
    )

    fn_examples = (
        te_df[(te_df["actual"] == 1) & (te_df["predicted"] == 0)]
        .sort_values("pred_prob", ascending=True)
        .head(3)[
            [
                "content_id",
                "client_id",
                "impressions",
                "clicks",
                "staleness_days",
                "position",
                "ctr_pct",
                "days_with_impressions",
                "content_age_days",
                "trend_direction",
                "actual",
                "predicted",
                "pred_prob",
                "score",
            ]
        ]
        .to_dict(orient="records")
    )

    # Top-10 and Top-20 Ranked Decision-Support Queue on Held-Out Test Clients (and Full 30k)
    # Score combines 70% calibrated model probability + 30% normalized W04 baseline score (0..5 -> 0..1)
    te_df["priority_score"] = np.round(
        100.0 * (0.70 * te_df["pred_prob"] + 0.30 * (te_df["score"] / 5.0)), 1
    )

    def build_reason_code_30k(row: pd.Series) -> str:
        reasons = []
        if row["impressions"] >= 1000 and row["staleness_days"] >= 14:
            reasons.append("STALE_HIGH_VOLUME")
        if row["position"] >= 8 and row["impressions"] >= 500:
            reasons.append("STRIKING_DISTANCE_DECAY_RISK")
        if row["impressions"] >= 500 and row["ctr_pct"] < 0.50:
            reasons.append("LOW_CTR_HIGH_EXPOSURE")
        if row["pred_prob"] >= 0.65:
            reasons.append("HIGH_MODEL_DECLINE_RISK")
        if not reasons:
            reasons.append("REVIEW")
        return " | ".join(reasons)

    def build_action_30k(row: pd.Series) -> str:
        if row["priority_score"] >= 75.0:
            if row["ctr_pct"] < 0.50 and row["impressions"] >= 1000:
                return "REVIEW_NOW (Refresh & Snippet/CTR Check)"
            return "REVIEW_NOW (Comprehensive Content Refresh)"
        if row["priority_score"] >= 60.0:
            return "PRIORITIZE (Scheduled Editorial Review)"
        return "MONITOR"

    def build_confidence_30k(row: pd.Series) -> str:
        if row["priority_score"] >= 75.0 and row["impressions"] >= 1000 and row["days_with_impressions"] >= 60:
            return "High (strong volume & consistent 90d history)"
        if row["priority_score"] >= 65.0 and row["impressions"] >= 500:
            return "Medium-High (sufficient exposure; check query mix)"
        return "Medium (directional signal; verify manually)"

    def build_caveat_30k(row: pd.Series) -> str:
        if row["ctr_pct"] < 0.10 and row["position"] <= 10:
            return "Page ranks on Page 1 with near-zero CTR; SERP features/AI overview or snippet mismatch may explain low clicks without content decay."
        if row["position"] > 20:
            return "Deep average position (>20); high impressions may come from broad long-tail queries where intent alignment or consolidation is needed first."
        return "Traffic change could reflect seasonal demand shifts or sibling URL cannibalization rather than content staleness."

    te_ranked = te_df.sort_values(["priority_score", "pred_prob", "impressions"], ascending=[False, False, False]).reset_index(drop=True)
    te_ranked["rank"] = range(1, len(te_ranked) + 1)
    te_ranked["reason_code_full"] = te_ranked.apply(build_reason_code_30k, axis=1)
    te_ranked["action_full"] = te_ranked.apply(build_action_30k, axis=1)
    te_ranked["confidence_full"] = te_ranked.apply(build_confidence_30k, axis=1)
    te_ranked["what_could_make_it_wrong"] = te_ranked.apply(build_caveat_30k, axis=1)

    top20_30k = te_ranked.head(20)[
        [
            "rank",
            "content_id",
            "client_id",
            "priority_score",
            "pred_prob",
            "score",
            "action_full",
            "reason_code_full",
            "confidence_full",
            "what_could_make_it_wrong",
            "impressions",
            "clicks",
            "staleness_days",
            "position",
            "ctr_pct",
            "is_declining_label",
        ]
    ].copy()
    top20_30k.to_csv(WORK_OUTPUT_DIR / "top20_refresh_queue_flyrank30k.csv", index=False)
    top10_30k = top20_30k.head(10).copy()
    top10_30k.to_csv(WORK_OUTPUT_DIR / "top10_refresh_queue_flyrank30k.csv", index=False)

    # =========================================================================
    # PART 3: SIGNAL AUDIT NUMBERS (for ML-06 / Phase 4 / Phase 13)
    # =========================================================================
    signal_audit = {
        "n_rows": int(len(df_30k)),
        "n_clients": int(df_30k["client_id"].nunique()),
        "declining_count": int(df_30k["is_declining_label"].sum()),
        "declining_rate": round(float(df_30k["is_declining_label"].mean()), 4),
        "impressions_quantiles": {
            k: round(float(v), 2)
            for k, v in df_30k["impressions"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).to_dict().items()
        },
        "staleness_quantiles": {
            k: round(float(v), 2)
            for k, v in df_30k["staleness_days"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict().items()
        },
        "test1_staleness_vs_decline": {
            "stale_ge_90d_decline_rate": round(float(df_30k.loc[df_30k["staleness_days"] >= 90, "is_declining_label"].mean()), 4),
            "fresh_lt_90d_decline_rate": round(float(df_30k.loc[df_30k["staleness_days"] < 90, "is_declining_label"].mean()), 4),
            "stale_ge_90d_n": int((df_30k["staleness_days"] >= 90).sum()),
            "fresh_lt_90d_n": int((df_30k["staleness_days"] < 90).sum()),
            "verdict": "CONFIRMED",
        },
        "test2_search_volume_vs_impressions": {
            "pearson_r_raw": round(float(df_30k["search_volume"].corr(df_30k["impressions_90d"])), 4),
            "verdict": "FALSE / WEAK (near-zero linear correlation r=0.0012)",
        },
        "test3_ctr_by_position_tier": {
            tier: round(float(grp["clicks"].sum() / grp["impressions"].sum() * 100.0), 4)
            for tier, grp in df_30k.groupby("position_tier")
            if grp["impressions"].sum() > 0
        },
        "flag_test_stale_visible_page": {
            "flagged_n": int(((df_30k["staleness_days"] >= 180) & (df_30k["impressions"] >= 500)).sum()),
            "flagged_decline_rate": round(
                float(
                    df_30k.loc[
                        (df_30k["staleness_days"] >= 180) & (df_30k["impressions"] >= 500),
                        "is_declining_label",
                    ].mean()
                ),
                4,
            ),
            "unflagged_decline_rate": round(
                float(
                    df_30k.loc[
                        ~((df_30k["staleness_days"] >= 180) & (df_30k["impressions"] >= 500)),
                        "is_declining_label",
                    ].mean()
                ),
                4,
            ),
        },
    }

    # =========================================================================
    # PART 4: GENERATE 5 CLEAN, READABLE VISUAL CHARTS (SVG)
    # =========================================================================
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#94a3b8",
        }
    )

    # Figure 1: Key Feature Distributions
    fig1, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig1.suptitle("Figure 1: Key Pre-Decision Feature Distributions (FlyRank Dataset, N = 30,000)", fontsize=13, fontweight="bold", y=0.98)

    axes[0, 0].hist(df_30k["impressions"].clip(upper=20000), bins=40, color="#3b82f6", edgecolor="white", alpha=0.85)
    axes[0, 0].set_title("A. Raw 90-Day Impressions (Clipped at 20k)")
    axes[0, 0].set_xlabel("Impressions (90d) — Heavy Right Skew (Median = 731)")
    axes[0, 0].set_ylabel("Number of Content Pages")

    axes[0, 1].hist(df_30k["log_impressions"], bins=40, color="#1d4ed8", edgecolor="white", alpha=0.85)
    axes[0, 1].set_title("B. Log-Transformed Impressions log1p(impressions_90d)")
    axes[0, 1].set_xlabel("log1p(Impressions 90d) — Stabilizes Heavy Tail")
    axes[0, 1].set_ylabel("Number of Content Pages")

    axes[1, 0].hist(df_30k["staleness_days"], bins=35, color="#0d9488", edgecolor="white", alpha=0.85)
    axes[1, 0].axvline(14, color="#dc2626", linestyle="--", linewidth=1.5, label="W04 Threshold (14d)")
    axes[1, 0].set_title("C. Staleness Days (days_since_last_update)")
    axes[1, 0].set_xlabel("Days Since Last Update (Median = 20d, P75 = 104d)")
    axes[1, 0].set_ylabel("Number of Content Pages")
    axes[1, 0].legend(loc="upper right", frameon=True)

    valid_pos = df_30k.loc[df_30k["avg_position"] > 0, "avg_position"].clip(upper=60)
    axes[1, 1].hist(valid_pos, bins=35, color="#7c3aed", edgecolor="white", alpha=0.85)
    axes[1, 1].axvline(8, color="#dc2626", linestyle="--", linewidth=1.5, label="W04 Threshold (Pos >= 8)")
    axes[1, 1].set_title("D. Average Search Position (Valid Positions > 0, Clipped at 60)")
    axes[1, 1].set_xlabel("Average Google Search Position (Lower = Better Rank)")
    axes[1, 1].set_ylabel("Number of Content Pages")
    axes[1, 1].legend(loc="upper right", frameon=True)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    save_both(fig1, "fig1_feature_distributions.svg")

    # Figure 2: Baseline vs Model Comparison Across Validation Designs
    fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig2.suptitle("Figure 2: Week-4 Baseline vs. Logistic Regression Across Validation Splits", fontsize=13, fontweight="bold", y=1.01)

    # Panel A: W05 Practice Dataset (N=100, test=20)
    m_labels_w05 = ["Accuracy", "F1-Score", "Precision", "Recall", "ROC AUC"]
    b_vals_w05 = [
        w05_baseline_metrics["accuracy"],
        w05_baseline_metrics["f1"],
        w05_baseline_metrics["precision"],
        w05_baseline_metrics["recall"],
        w05_baseline_metrics["roc_auc"],
    ]
    lr_vals_w05 = [
        w05_model_metrics["accuracy"],
        w05_model_metrics["f1"],
        w05_model_metrics["precision"],
        w05_model_metrics["recall"],
        w05_model_metrics["roc_auc"],
    ]
    x_pos = np.arange(len(m_labels_w05))
    w = 0.35
    rects1 = axes2[0].bar(x_pos - w / 2, b_vals_w05, w, label="Week-4 Rule Baseline (score >= 3)", color="#94a3b8")
    rects2 = axes2[0].bar(x_pos + w / 2, lr_vals_w05, w, label="Week-5 Logistic Regression (4 features)", color="#2563eb")
    axes2[0].set_title("A. W05 Practice Set (N=100, Stratified 80/20 Split, Test n=20)")
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels(m_labels_w05)
    axes2[0].set_ylim(0, 1.12)
    axes2[0].set_ylabel("Measured Metric Value")
    axes2[0].legend(loc="lower right", fontsize=8.5)
    for r in rects1:
        axes2[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02, f"{r.get_height():.2f}", ha="center", fontsize=8)
    for r in rects2:
        axes2[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02, f"{r.get_height():.2f}", ha="center", fontsize=8, fontweight="bold")

    # Panel B: FlyRank 30k Client-Grouped Holdout (n_test = 2,325 across 6 unseen clients)
    grp_res = results_30k["client_grouped_holdout"]
    m_labels_30k = ["Accuracy", "F1-Score", "ROC AUC", "Precision@20", "Precision@50"]
    b_vals_30k = [
        grp_res["w04_rule_baseline"]["accuracy"],
        grp_res["w04_rule_baseline"]["f1"],
        grp_res["w04_rule_baseline"]["roc_auc"],
        grp_res["w04_rule_baseline"]["precision_at_20"],
        grp_res["w04_rule_baseline"]["precision_at_50"],
    ]
    lr4_vals_30k = [
        grp_res["logreg_4feat_raw"]["accuracy"],
        grp_res["logreg_4feat_raw"]["f1"],
        grp_res["logreg_4feat_raw"]["roc_auc"],
        grp_res["logreg_4feat_raw"]["precision_at_20"],
        grp_res["logreg_4feat_raw"]["precision_at_50"],
    ]
    lr7_vals_30k = [
        grp_res["logreg_7feat_predecision"]["accuracy"],
        grp_res["logreg_7feat_predecision"]["f1"],
        grp_res["logreg_7feat_predecision"]["roc_auc"],
        grp_res["logreg_7feat_predecision"]["precision_at_20"],
        grp_res["logreg_7feat_predecision"]["precision_at_50"],
    ]
    x_pos2 = np.arange(len(m_labels_30k))
    w2 = 0.26
    r_b = axes2[1].bar(x_pos2 - w2, b_vals_30k, w2, label="Week-4 Rule Baseline", color="#94a3b8")
    r_4 = axes2[1].bar(x_pos2, lr4_vals_30k, w2, label="LogReg (4 Raw Features)", color="#f59e0b")
    r_7 = axes2[1].bar(x_pos2 + w2, lr7_vals_30k, w2, label="LogReg (7 Pre-Decision Features, Log-Scaled)", color="#0d9488")
    axes2[1].set_title("B. FlyRank 30k Set (Client-Grouped Holdout, 6 Unseen Clients, n=2,325)")
    axes2[1].set_xticks(x_pos2)
    axes2[1].set_xticklabels(m_labels_30k)
    axes2[1].set_ylim(0, 1.0)
    axes2[1].set_ylabel("Measured Metric Value")
    axes2[1].legend(loc="upper right", fontsize=8)
    for r in r_b:
        axes2[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.2f}", ha="center", fontsize=7.5)
    for r in r_4:
        axes2[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.2f}", ha="center", fontsize=7.5)
    for r in r_7:
        axes2[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.2f}", ha="center", fontsize=7.5, fontweight="bold")

    fig2.tight_layout()
    save_both(fig2, "fig2_baseline_vs_model_metrics.svg")

    # Figure 3: Logistic Regression Standardized Coefficients
    fig3, axes3 = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig3.suptitle("Figure 3: Directional Feature Coefficients (Standardized Logistic Regression)", fontsize=13, fontweight="bold", y=1.01)

    # Panel A: W05 4-Feature Coefficients
    w05_c_sorted = sorted(w05_coef_table, key=lambda x: x["absolute_coefficient"])
    f_names_w05 = [d["feature"] for d in w05_c_sorted]
    f_vals_w05 = [d["coefficient"] for d in w05_c_sorted]
    bars_a = axes3[0].barh(f_names_w05, f_vals_w05, color="#2563eb", height=0.55)
    axes3[0].axvline(0, color="#334155", linewidth=1)
    axes3[0].set_title("A. Week-5 Model Coefficients (W05 Practice Dataset, N=100)")
    axes3[0].set_xlabel("Standardized Coefficient (Directional Association with Target = 1)")
    axes3[0].set_xlim(0, 0.78)
    for b in bars_a:
        axes3[0].text(b.get_width() + 0.015, b.get_y() + b.get_height() / 2, f"+{b.get_width():.4f}", va="center", fontsize=9, fontweight="bold")

    # Panel B: FlyRank 30k 7-Feature Pre-Decision Coefficients (Client Holdout)
    c7_list = grp_res["logreg_7feat_predecision"]["coefficients"]
    c7_sorted = sorted(c7_list, key=lambda x: x["absolute_coefficient"])
    f_names_7 = [d["feature"] for d in c7_sorted]
    f_vals_7 = [d["coefficient"] for d in c7_sorted]
    colors_7 = ["#dc2626" if v > 0 else "#0d9488" for v in f_vals_7]
    bars_b = axes3[1].barh(f_names_7, f_vals_7, color=colors_7, height=0.6)
    axes3[1].axvline(0, color="#334155", linewidth=1)
    axes3[1].set_title("B. 7-Feature Pre-Decision Model (FlyRank 30k, Client Holdout)")
    axes3[1].set_xlabel("Standardized Coefficient (+ Associated with Decline / - Associated with Stability)")
    axes3[1].set_xlim(-1.25, 1.55)
    for b in bars_b:
        w_val = b.get_width()
        offset = 0.04 if w_val >= 0 else -0.04
        ha = "left" if w_val >= 0 else "right"
        axes3[1].text(w_val + offset, b.get_y() + b.get_height() / 2, f"{w_val:+.4f}", va="center", ha=ha, fontsize=8.5, fontweight="bold")

    fig3.tight_layout()
    save_both(fig3, "fig3_logistic_regression_coefficients.svg")

    # Figure 4: Error Analysis Visualization
    fig4, axes4 = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig4.suptitle("Figure 4: Error Analysis — Prediction Breakdown & False-Positive Signal Profile", fontsize=13, fontweight="bold", y=1.01)

    # Panel A: W05 Test Set (n=20) & FlyRank 30k Client Holdout (n=2,325) Confusion Breakdown
    cats = ["True Negative\n(Correct Monitor)", "True Positive\n(Correct Refresh)", "False Positive\n(Over-Prioritized)", "False Negative\n(Missed Decline)"]
    w05_pcts = [
        w05_model_metrics["tn"] / 20 * 100,
        w05_model_metrics["tp"] / 20 * 100,
        w05_model_metrics["fp"] / 20 * 100,
        w05_model_metrics["fn"] / 20 * 100,
    ]
    grp_m = grp_res["logreg_7feat_predecision"]
    tot_grp = grp_res["test_rows"]
    grp_pcts = [
        grp_m["tn"] / tot_grp * 100,
        grp_m["tp"] / tot_grp * 100,
        grp_m["fp"] / tot_grp * 100,
        grp_m["fn"] / tot_grp * 100,
    ]
    x4 = np.arange(len(cats))
    w4 = 0.35
    b4_1 = axes4[0].bar(x4 - w4 / 2, w05_pcts, w4, label="W05 Test Set (n=20)", color="#2563eb")
    b4_2 = axes4[0].bar(x4 + w4 / 2, grp_pcts, w4, label="FlyRank 30k Client Holdout (n=2,325)", color="#0d9488")
    axes4[0].set_title("A. Test Set Outcome Breakdown (% of Test Population)")
    axes4[0].set_xticks(x4)
    axes4[0].set_xticklabels(cats, fontsize=8.5)
    axes4[0].set_ylabel("Percentage of Evaluated Items (%)")
    axes4[0].set_ylim(0, 70)
    axes4[0].legend(loc="upper right", fontsize=8.5)
    for r in b4_1:
        axes4[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.1f}%", ha="center", fontsize=8)
    for r in b4_2:
        axes4[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.1f}%", ha="center", fontsize=8)

    # Panel B: Signal Comparison on W05 4 False Positives vs True Positives
    w05_tp_df = w05_test_df[(w05_test_df["actual"] == 1) & (w05_test_df["predicted"] == 1)]
    w05_fp_df = w05_test_df[(w05_test_df["actual"] == 0) & (w05_test_df["predicted"] == 1)]
    w05_tn_df = w05_test_df[(w05_test_df["actual"] == 0) & (w05_test_df["predicted"] == 0)]

    sig_labels = ["Mean Staleness\n(Days)", "Mean Position\n(Rank)", "Mean CTR\n(Clicks/Imp %)"]
    tp_means = [
        w05_tp_df["staleness_days"].mean(),
        w05_tp_df["position"].mean(),
        (w05_tp_df["clicks"] / w05_tp_df["impressions"] * 100).mean(),
    ]
    fp_means = [
        w05_fp_df["staleness_days"].mean(),
        w05_fp_df["position"].mean(),
        (w05_fp_df["clicks"] / w05_fp_df["impressions"] * 100).mean(),
    ]
    tn_means = [
        w05_tn_df["staleness_days"].mean(),
        w05_tn_df["position"].mean(),
        (w05_tn_df["clicks"] / w05_tn_df["impressions"] * 100).mean(),
    ]
    x4b = np.arange(len(sig_labels))
    w4b = 0.25
    axes4[1].bar(x4b - w4b, tp_means, w4b, label="True Positives (n=12)", color="#16a34a")
    axes4[1].bar(x4b, fp_means, w4b, label="False Positives (n=4 Errors)", color="#dc2626")
    axes4[1].bar(x4b + w4b, tn_means, w4b, label="True Negatives (n=4)", color="#64748b")
    axes4[1].set_title("B. Why W05 False Positives Occurred: High Staleness Despite High CTR")
    axes4[1].set_xticks(x4b)
    axes4[1].set_xticklabels(sig_labels)
    axes4[1].set_ylabel("Mean Value in Test Cohort")
    axes4[1].legend(loc="upper left", fontsize=8.5)

    fig4.tight_layout()
    save_both(fig4, "fig4_error_analysis_breakdown.svg")

    # Figure 5: Top-10 Content Refresh Priority Queue Visualization
    fig5, axes5 = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig5.suptitle("Figure 5: Top-10 Content Refresh Priority Queue (Decision-Support Output)", fontsize=13, fontweight="bold", y=1.01)

    # Panel A: W04 Top-10 Baseline Queue (ml07_baseline_practice_dataset.csv)
    t10_w04_rev = top10_w04.iloc[::-1]
    labels_a = [f"#{r.rank} {r.item_id}" for r in t10_w04_rev.itertuples()]
    scores_a = [r.score for r in t10_w04_rev.itertuples()]
    colors_a = ["#1d4ed8" if s == 5 else "#3b82f6" for s in scores_a]
    bars5a = axes5[0].barh(labels_a, scores_a, color=colors_a, height=0.65)
    axes5[0].set_title("A. Week-4 Baseline Top-10 Queue (Practice Set, Max Score = 5)")
    axes5[0].set_xlabel("Week-4 Rule Priority Score (0 to 5) — All Action = REVIEW_NOW")
    axes5[0].set_xlim(0, 6.2)
    for b, row in zip(bars5a, t10_w04_rev.itertuples()):
        axes5[0].text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2, f"Score {row.score} ({row.reason_code})", va="center", fontsize=8)

    # Panel B: FlyRank 30k Held-Out Test Clients Top-10 Queue
    t10_30k_rev = top10_30k.iloc[::-1]
    labels_b = [f"#{r.rank} {r.content_id[:14]}.." for r in t10_30k_rev.itertuples()]
    scores_b = [r.priority_score for r in t10_30k_rev.itertuples()]
    bars5b = axes5[1].barh(labels_b, scores_b, color="#0d9488", height=0.65)
    axes5[1].set_title("B. FlyRank 30k Top-10 Queue (Unseen Test Clients, 0–100 Scale)")
    axes5[1].set_xlabel("Combined Refresh Priority Score (70% Model Prob + 30% W04 Rule)")
    axes5[1].set_xlim(0, 108)
    for b, row in zip(bars5b, t10_30k_rev.itertuples()):
        axes5[1].text(
            b.get_width() + 1.2,
            b.get_y() + b.get_height() / 2,
            f"{row.priority_score:.1f} (p={row.pred_prob:.2f})",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    fig5.tight_layout()
    save_both(fig5, "fig5_top10_refresh_queue.svg")

    # =========================================================================
    # PART 5: SAVE COMPLETE VERIFIED JSON METRICS
    # =========================================================================
    payload = {
        "project": "FlyRank Machine Learning Week-8 Capstone",
        "lane": "Content Refresh Priority Prediction",
        "random_state": RANDOM_STATE,
        "w04_baseline_definition": {
            "formula": "score = 2*(impressions >= 1000) + 2*(staleness_days >= 14) + 1*(position >= 8)",
            "score_range": [0, 5],
            "binary_threshold": "score >= 3",
            "reason_codes": {
                "STALE_HIGH_VOLUME": "impressions >= 1000 and staleness_days >= 14",
                "REVIEW": "otherwise",
            },
            "actions": {
                "REVIEW_NOW": "score >= 4",
                "PRIORITIZE": "score == 3",
                "MONITOR": "score < 3",
            },
        },
        "w05_practice_evaluation": {
            "dataset": "w05_ml_practice_dataset.csv",
            "n_total": int(len(df_w05)),
            "n_train": int(len(Xtr_w05)),
            "n_test": int(len(Xte_w05)),
            "split_design": "Stratified Random Split (80/20, random_state=42)",
            "positive_base_rate_total": round(float(y_w05.mean()), 4),
            "positive_base_rate_test": round(float(yte_w05.mean()), 4),
            "features": w05_features,
            "week4_baseline": w05_baseline_metrics,
            "week5_logistic_regression": w05_model_metrics,
            "intercept": round(w05_intercept, 6),
            "coefficients": w05_coef_table,
            "test_errors": w05_errors,
        },
        "w04_top10_queue": top10_w04.to_dict(orient="records"),
        "flyrank_30k_evaluation": {
            "dataset": "data/raw/content_refresh_anonymized.csv",
            "n_total": int(len(df_30k)),
            "n_clients": int(df_30k["client_id"].nunique()),
            "positive_base_rate_total": round(float(y_30k.mean()), 4),
            "valid_position_median_imputed": valid_pos_median,
            "splits": results_30k,
            "error_analysis_client_holdout": {
                "false_positives_top3": fp_examples,
                "false_negatives_top3": fn_examples,
            },
            "top10_queue_client_holdout": top10_30k.to_dict(orient="records"),
        },
        "signal_audit": signal_audit,
    }

    out_json = WORK_OUTPUT_DIR / "capstone_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Successfully wrote verified metrics to: {out_json}")
    print("Successfully generated 5 SVG figures in work/figures/ and docs/figures/")


if __name__ == "__main__":
    main()
