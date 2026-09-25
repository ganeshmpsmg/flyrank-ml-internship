# Prioritizing Content Refresh Reviews Using Pre-Decision Search Signals: A Decision-Support Study

- **Author:** Ganesh MP (`ganeshmpsmg`)
- **Capstone Track:** FlyRank Machine Learning Week-8 Capstone — Google Search Ranking & Discoverability
- **Lane:** Content Refresh Priority Prediction (Lane 2 — Refresh / Content Opportunity Scoring)
- **Repository:** [`https://github.com/ganeshmpsmg/flyrank-ml-internship`](https://github.com/ganeshmpsmg/flyrank-ml-internship)
- **Date:** September 2026

---

## 1. Title

**Prioritizing Content Refresh Reviews Using Pre-Decision Search Signals: A Reproducible Baseline-vs-Logistic-Regression Decision-Support Study**

---

## 2. Abstract

- **Question:** Using only information available before the decision moment, which content items should be reviewed first for a possible refresh, and what signals explain that priority?
- **Data:** We evaluate two actual datasets from the project workflow without mixing their populations: (1) the Week-4/Week-5 practice datasets (`ml07_baseline_practice_dataset.csv`, $N = 30$; `w05_ml_practice_dataset.csv`, $N = 100$) used in our earlier weekly milestones, and (2) the bundled FlyRank pseudonymized starter dataset (`data/raw/content_refresh_anonymized.csv`, $N = 30,000$ content items across $32$ pseudonymized clients over a trailing 90-day window, with reference to the upstream `FlyRank/internship-warehouse` release `v20260703`).
- **Method:** We preserve the transparent Week-4 rule-based baseline score ($\text{score} = 2\cdot\mathbb{I}(\text{impressions}\ge 1000) + 2\cdot\mathbb{I}(\text{staleness\_days}\ge 14) + 1\cdot\mathbb{I}(\text{position}\ge 8)$, predicting priority when $\text{score} \ge 3$) and compare it against standardized Logistic Regression models trained strictly on pre-decision search and content signals.
- **Validation:** On the 100-row Week-5 practice dataset (which contains no client or date identifier), we use an 80/20 stratified random split ($n_{\text{train}} = 80$, $n_{\text{test}} = 20$, `random_state=42`). On the 30,000-row FlyRank dataset, we evaluate both an 80/20 stratified random split ($n_{\text{train}} = 24,000$, $n_{\text{test}} = 6,000$) and an honest **Client-Grouped Holdout Split** ($26$ training clients with $n_{\text{train}} = 27,675$ rows; $6$ unseen test clients with $n_{\text{test}} = 2,325$ rows, `random_state=42`).
- **Measured Result:** On the Week-5 practice test set (positive base rate $0.6000$), Week-5 Logistic Regression achieved a measured accuracy of **$0.8000$** ($\text{F1} = 0.8571$, $\text{ROC AUC} = 0.9375$) compared to **$0.7000$** ($\text{F1} = 0.7857$, $\text{ROC AUC} = 0.8542$) for the Week-4 rule baseline. On the 30,000-row FlyRank Client-Grouped Holdout split (unseen test client base rate $0.3910$ vs. training base rate $0.5548$), un-logged 4-feature Logistic Regression degraded to **$0.3897$** accuracy ($\text{ROC AUC} = 0.4191$, $\text{Precision@50} = 0.2800$) due to extreme heavy tails in raw impressions and cross-client base-rate shift, whereas a 7-feature pre-decision Logistic Regression using log-scaled traffic counts (`log1p`) and balanced class weights achieved **$0.6688$** accuracy, **$0.5595$** F1, **$0.7144$** ROC AUC, and **$0.6200$** Precision@50—outperforming both the Week-4 rule baseline ($\text{Accuracy} = 0.4817$, $\text{ROC AUC} = 0.4821$, $\text{Precision@50} = 0.5000$) and the repository's reference baseline ($\text{Accuracy} = 0.6086$, $\text{ROC AUC} = 0.6269$, $\text{Precision@50} = 0.2400$).
- **Cautious Conclusion:** Observed historical search and staleness signals provide useful directional discrimination for ranking content review candidates, but predictions remain decision-support aids rather than causal proof that refreshing a page will improve search performance.

---

## 3. Introduction & Problem Statement

Content portfolios that earn organic search visibility naturally experience performance decay over time as competing pages update, search intent shifts, or topics lose freshness. Editorial and SEO teams have limited review capacity: when a portfolio spans thousands of URLs, reviewing every page manually is impossible, while relying on ad-hoc intuition risks overlooking high-exposure pages that are quietly slipping.

- **Unit of Analysis:** One content item (page), represented by a pseudonymized identifier (`item_id` in the Week-4/Week-5 practice tables; `content_id` paired with `client_id` in the 30,000-row FlyRank starter dataset).
- **Output:** A continuous priority score, a ranked review queue, human-readable reason codes, and a recommended triage action (`REVIEW_NOW`, `PRIORITIZE`, or `MONITOR`).
- **Human Action Supported:** Content editors and SEO analysts use the ranked queue to decide **which pages to inspect first** during weekly content audits.
- **Cost of a Wrong Call:**
  - *False Positive (over-prioritizing a stable page):* Wastes editorial review hours and risks making unnecessary changes to a page that is already performing adequately or whose low click capture stems from SERP layout changes rather than stale text.
  - *False Negative (missing a declining high-value page):* Allows a high-impression asset in striking distance to continue losing search visibility before an editor intervenes.
- **Scope Discipline (What This Project Is Not):** This project is strictly a **decision-support** triage system. It is **not** a guaranteed SEO ranking predictor, **not** an automated publishing or content-rewriting tool, **not** a causal inference study, and **not** a promise that editing a flagged page will cause traffic recovery.

---

## 4. Research Question

> **"Using only information available before the decision moment, which content items should be reviewed first for a possible refresh, and what signals explain that priority?"**

---

## 5. Data & Data Contract

We document the exact datasets inspected and used in this repository without fabricating any rows, dates, or grouping fields:

### 5.1 Data Sources and Releases

1. **FlyRank Pseudonymized Starter Dataset (`data/raw/content_refresh_anonymized.csv`):**
   - **Population & Size:** $30,000$ rows $\times$ $44$ columns covering $32$ pseudonymized clients (`client_id`).
   - **Row Grain:** One row per pseudonymized content item (`content_id`). Verified zero duplicate `content_id` rows.
   - **Time Window & Decision Moment:** Aggregated over a trailing 90-day window ending at the snapshot export cutoff (`content_age_days >= 90` and `impressions_90d >= 1` for all rows). The decision moment is the end of the 90-day observation window, prior to any editorial refresh action.
   - **Target / Proxy Label (`is_declining_label`):** Binary indicator defined as `1` when `trend_direction == "down"` ($16,262$ rows, **$54.21\%$** positive base rate) and `0` otherwise (`stable`: $5,962$; `up`: $4,388$; `new`: $2,236$; `flat`: $1,152$). As documented in `docs/ml-intern-dataset-and-lane-guide.md`, `is_declining_label` is a contemporaneous proxy label derived from 30d-vs-prev-30d impression movement (`trend_pct < -20%`), which motivates strict exclusion of `trend_direction`, `trend_pct`, and the 30-day sub-window components when testing generalizable pre-decision signals.
2. **Upstream Warehouse Context (`hf://datasets/FlyRank/internship-warehouse`, build `v20260703`):**
   - Documented in `work/notebooks/w03_data_contract.ipynb`: `fact_content_daily_performance_sample.parquet` ($11,694,072$ rows; full table $78,835,655$ rows across $70$ active clients from `2025-01-27` to `2026-06-30` with a 3-day freshness cutoff from export date `2026-07-03`).
3. **Weekly Milestone Practice Datasets (Preserved from W03–W06):**
   - `simple_feature_vector_dataset.csv` ($12$ rows $\times$ $5$ columns: `views, clicks, position, device, category`) used in `w03_feature_leakage_check.ipynb`.
   - `ml07_baseline_practice_dataset.csv` ($30$ rows $\times$ $6$ columns: `item_id, impressions, clicks, position, staleness_days, ctr`) used in `w04_baseline_score.ipynb`.
   - `w05_ml_practice_dataset.csv` ($100$ rows $\times$ $6$ columns: `item_id, impressions, clicks, staleness_days, position, target`; positive base rate **$61.00\%$** overall, **$60.00\%$** in the 20-row stratified test split) used in `w05_model.ipynb` and `w06_validation_audit.ipynb`.

### 5.2 Field Classification (Included, Context, and Excluded)

| Bucket | Columns | Role & Justification |
|---|---|---|
| **Included Pre-Decision Features (W05 Core 4-Feature Set)** | `impressions` (`impressions_90d`), `clicks` (`clicks_90d`), `staleness_days` (`days_since_last_update`), `position` (`avg_position`) | Core search exposure, click capture, content recency, and ranking position signals available before the review decision. |
| **Included Pre-Decision Features (FlyRank 30k 7-Feature Set)** | `log_impressions` (`log1p(impressions_90d)`), `log_clicks` (`log1p(clicks_90d)`), `staleness_days` (`days_since_last_update`), `position` (`avg_position` imputed), `ctr_pct` (`ctr`), `days_with_impressions`, `content_age_days` | Adds log-scaling for heavy-tailed traffic counts plus pre-decision CTR, visibility consistency (0–90 days), and content maturity. |
| **Context / Grouping Keys (Never Model Features)** | `content_id`, `client_id`, `item_id` | Pseudonymous identifiers used solely for row identification, deduplication, and `client_id` grouped holdout splitting. |
| **Target / Label** | `target` (W05 dataset), `is_declining_label` (30k dataset) | Binary outcome variable (`0` or `1`) predicted by the baseline and Logistic Regression models. |
| **Excluded — Label-Derived / Window-Overlapping (Leakage Risk)** | `trend_direction`, `trend_pct`, `impressions_last_30d`, `impressions_prev_30d`, `clicks_last_30d`, `clicks_prev_30d`, `sessions_last_30d`, `sessions_prev_30d` | `is_declining_label` is deterministically computed from `trend_direction` and `trend_pct` via the 30d vs. prev-30d impression windows. Including them leaks the label directly. |
| **Excluded — Post-Decision / Product Flags** | `health_score`, `priority_score`, `action_type`, `refresh_tier` | Product rule outputs that encode existing system decisions; excluded to prevent circular predictions. |
| **Excluded — Metadata / Low-Signal / Non-Feature Fields** | `provider_used`, `model_used` | LLM generation metadata (`openai`, `google`); excluded per data dictionary rules (`Not a model feature`). |

---

## 6. Feature Design & Signal Audit

Rather than indiscriminately feeding all 44 columns into a black-box model, we use a small, defensible pre-decision feature set and audit the empirical distributions first (`work/notebooks/w04_signal_audit.ipynb`).

### 6.1 Feature Specifications

| Feature Name | Source Column (`30k` / `W05`) | Meaning | Missing-Value & Gotcha Handling | Available at Decision Time? | Reason for Inclusion |
|---|---|---|---|---|---|
| `impressions` / `log_impressions` | `impressions_90d` / `impressions` | Total Google Search Console impressions over the 90-day window (and `log1p` transform on the 30k dataset). | No missing values in either dataset (`min = 1` in 30k). Because 30k impressions span $1$ to $>73,500$ ($P_{99} = 73,505.8$, $\text{median} = 731.0$), `log1p` stabilizes extreme leverage points. | Yes | Measures search demand and visibility exposure; pages with meaningful impressions have more at stake if they decay. |
| `clicks` / `log_clicks` | `clicks_90d` / `clicks` | Total organic search clicks over the 90-day window (and `log1p` transform on the 30k dataset). | Filled with `0` if missing. Log-scaled on the 30k set to match `log_impressions`. | Yes | Captures realized user traffic; high impressions paired with low clicks indicates uncaptured demand or declining competitiveness. |
| `staleness_days` | `days_since_last_update` / `staleness_days` | Days elapsed since the content item was last updated prior to the snapshot cutoff. | Filled with `0` if missing ($\text{median} = 20.0\text{d}$, $P_{75} = 104.0\text{d}$ in 30k). | Yes | Directly measures editorial recency; older un-updated content is more likely to lose freshness relevance. |
| `position` | `avg_position` / `position` | Mean Google Search ranking position over the observation window (lower numbers indicate stronger rank). | **Critical gotcha handled:** In `content_refresh_anonymized.csv`, `avg_position == 0` means *"no position data"* ($1,205$ rows), **not** rank zero. We replace `0` with the valid-row median ($11.4$) and track `has_position_data`. | Yes | Identifies pages in page-1 ($1\text{–}10$) or striking distance ($11\text{–}20$) where refresh reviews have the clearest operational rationale. |
| `ctr_pct` | `ctr` | Click-through rate expressed as a $\times 100$ percentage (`0.76` = $0.76\%$). | Filled with `0.0` when missing. Verified as pre-decision 90-day aggregate rate. | Yes | Distinguishes pages that convert impressions into clicks from high-exposure pages suffering snippet or intent mismatch. |
| `days_with_impressions` | `days_with_impressions` | Count of days ($0\text{–}90$) in the trailing window with at least 1 search impression. | Filled with `0` if missing. | Yes | Measures visibility consistency across the 90-day window rather than one-off traffic spikes. |
| `content_age_days` | `content_age_days` | Total days since initial publication (`>= 90` for all rows in the 30k slice). | Filled with median if missing. | Yes | Controls for lifecycle stage separately from update recency (`staleness_days`). |

### 6.2 Signal Audit Verdicts on the 30,000-Row Dataset

1. **Signal Test 1 — Content Staleness vs. Observed Decline (`VERDICT: CONFIRMED`):** Pages with `staleness_days >= 90` ($n = 9,345$) exhibit an observed declining rate of **$60.85\%$**, compared to **$51.20\%$** for fresher pages updated within `< 90` days ($n = 20,655$)—a $+9.65$ percentage-point directional gap.
2. **Signal Test 2 — Stored Keyword Search Volume vs. Realized Page Impressions (`VERDICT: FALSE / WEAK`):** The linear Pearson correlation between `search_volume` and `impressions_90d` across the $30,000$ pages is **$r = 0.0012$** (virtually zero), confirming the FlyRank March 2026 research paper's finding that nominal keyword volume alone is a poor standalone proxy for page-level visibility.
3. **Signal Test 3 — Weighted Portfolio CTR by Position Tier (`VERDICT: CONFIRMED`):** Weighted CTR (`SUM(clicks_90d) / SUM(impressions_90d) * 100`) drops monotonically as rank deepens: **`top_3` = $0.4885\%$**, **`page_1` = $0.3503\%$**, **`striking` = $0.3469\%$**, **`page_3_5` = $0.1549\%$**, and **`deep` = $0.0414\%$**.
4. **Flag-Linked Test — `stale_visible_page` (`days_since_last_update >= 180` and `impressions_90d >= 500`):** Although only $n = 17$ pages in this specific 30k slice satisfy the strict $180$-day + $500$-impression threshold, **$94.12\%$** ($16/17$) of flagged pages are declining versus **$54.18\%$** of unflagged pages, supporting the directional validity of combining staleness with visibility while showing why a smoother learned score is needed to cover more than $17$ items.

---

## 7. Leakage & Privacy Audit

We executed a formal leakage and privacy audit across both code and outputs (`work/notebooks/w03_feature_leakage_check.ipynb` and `work/notebooks/w06_validation_audit.ipynb`):

1. **No Label-Derived Features:** Confirmed programmatically that `trend_direction`, `trend_pct`, and `is_declining_label` are absent from all feature matrices ($X$).
2. **No Window-Component Leakage:** Confirmed that `impressions_last_30d`, `impressions_prev_30d`, `clicks_last_30d`, `clicks_prev_30d`, `sessions_last_30d`, and `sessions_prev_30d` (from which `trend_pct` and `trend_direction` are directly calculated) are excluded from the feature vector.
3. **No Post-Decision Product Flags:** Confirmed no composite product flags (`health_score`, `priority_score`, `action_type`) are used as inputs.
4. **Preprocessing Inside Pipeline Only:** `StandardScaler()` is fit strictly inside `sklearn.pipeline.Pipeline` on `X_train` and applied to `X_test`, preventing train-test distribution leakage.
5. **No Fabricated Dates or Groups:** On the 100-row Week-5 practice dataset (which lacks date and client columns), we report the honest stratified random split and explicitly state that a time/group split cannot be performed without those columns. On the 30,000-row FlyRank dataset, we use the actual `client_id` column ($32$ clients) for grouped holdout validation.
6. **Privacy & Credential Check:** All identifiers (`item_001`–`item_100`, `content_3e79eaafc89d`, `client_f74efabef1`) are pseudonymized hashes. No client names, raw domains, raw URLs, private search queries, or API tokens/secrets appear anywhere in the repository or notebooks.

---

## 8. Week-4 Baseline (Preserved Exactly)

We recover and preserve the exact rule-based baseline constructed in `work/notebooks/w04_baseline_score.ipynb` and evaluated in `work/notebooks/w05_model.ipynb`:

### 8.1 Baseline Formula & Thresholds

$$\text{baseline\_score} = 2 \cdot \mathbb{I}(\text{impressions} \ge 1000) + 2 \cdot \mathbb{I}(\text{staleness\_days} \ge 14) + 1 \cdot \mathbb{I}(\text{position} \ge 8)$$

- **Score Range:** Integer values from $0$ to $5$.
- **Reason Codes (`w04_baseline_score.ipynb`):**
  - `STALE_HIGH_VOLUME`: assigned when `impressions >= 1000` and `staleness_days >= 14`.
  - `REVIEW`: assigned otherwise.
- **Action Tiers (`w04_baseline_score.ipynb`):**
  - `REVIEW_NOW`: when `score >= 4`
  - `PRIORITIZE`: when `score == 3`
  - `MONITOR`: when `score < 3`
- **Binary Classification Rule (`w05_model.ipynb`):**
  - `baseline_pred = 1` when `score >= 3`, else `0`.

In addition, when evaluating on the 30,000-row FlyRank dataset, we also report the repository's reference percentile baseline (`baseline_refresh_score = 0.40 * visibility_score + 0.30 * freshness_risk_score + 0.25 * position_opportunity_score + 0.05 * depth_gap_score` from `scripts/02_baseline_score.py`) so both baseline formulations are transparently benchmarked on the exact same holdout split.

---

## 9. Methodology & Model Choice

Following the principle of using the simplest defensible model (`work/notebooks/w05_model.ipynb`), we use **Logistic Regression** inside a scikit-learn `Pipeline`:

1. **Why Logistic Regression Fits This Lane:**
   - Content refresh prioritization requires transparent, monotonic, and auditable relationships between pre-decision signals and predicted risk so editors can understand *why* a page was surfaced.
   - Standardized Logistic Regression coefficients ($\beta_j$) directly quantify the direction and relative magnitude of each feature's association with the target, whereas complex ensembles obscure directionality.
2. **Preprocessing & Architecture:**
   - **Week-5 4-Feature Model (`logreg_4feat_raw`):**
     - Features: `["impressions", "clicks", "staleness_days", "position"]`
     - Pipeline: `Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])`
   - **FlyRank 30k 7-Feature Pre-Decision Model (`logreg_7feat_predecision`):**
     - Features: `["log_impressions", "log_clicks", "staleness_days", "position", "ctr_pct", "days_with_impressions", "content_age_days"]`
     - Justification for refinement on the 30k dataset: Raw 90-day impressions and clicks in the 30,000-row dataset exhibit extreme right skew ($P_{99}/P_{50} > 100\times$), and held-out clients exhibit different base rates ($39.10\%$ test vs. $55.48\%$ train). Applying `log1p` to impressions and clicks and setting `class_weight="balanced"` in `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)` keeps the model strictly linear and interpretable while preventing extreme-volume outliers and majority-class bias from dominating out-of-client predictions.

---

## 10. Honest Validation Design

We evaluate models using validation splits that match the actual structure of each dataset (`work/notebooks/w06_validation_audit.ipynb`):

1. **Population A — Week-5 Practice Dataset (`w05_ml_practice_dataset.csv`, $N = 100$):**
   - **Validation Design:** **80/20 Stratified Random Split** (`train_test_split(..., test_size=0.20, random_state=42, stratify=y)` $\to$ $n_{\text{train}} = 80$, $n_{\text{test}} = 20$; positive base rate $= 0.6100$ overall, $0.6000$ in test).
   - **Honesty Note:** Because `w05_ml_practice_dataset.csv` contains only `item_id, impressions, clicks, staleness_days, position, target` with no date or client group column, a time-aware or client-grouped split cannot be performed on this file, and we never fabricate synthetic dates or groups.
2. **Population B — FlyRank Anonymized Starter Dataset (`data/raw/content_refresh_anonymized.csv`, $N = 30,000$ across $32$ clients):**
   - **Validation Design 1 (Diagnostic Comparison): 80/20 Stratified Random Split** ($n_{\text{train}} = 24,000$, $n_{\text{test}} = 6,000$, `random_state=42`, test positive base rate $= 0.5420$). Because pages from the same client appear in both train and test, this split is optimistic when client-level baselines differ.
   - **Validation Design 2 (Primary Honest Split): Client-Grouped Holdout Split (`client_holdout`)** using `client_id` (`random_state=42`). Exactly $20\%$ of clients ($6$ whole clients, $n_{\text{test}} = 2,325$ pages, test positive base rate $= 0.3910$) are held out entirely for testing, while the remaining $26$ clients ($n_{\text{train}} = 27,675$ pages, train positive base rate $= 0.5548$) are used for training. This tests whether the model generalizes to **unseen client portfolios** with different underlying decline rates.

---

## 11. Results (Baseline vs. Model on Identical Splits)

All numbers below are exact measured outputs produced by `work/scripts/run_capstone_pipeline.py` and stored in `work/outputs/capstone_metrics.json`. Populations and validation splits are strictly separated.

### 11.1 Primary Comparison Table (Requested Format)

| Method | Validation Design | Metric | Result |
|---|---|---|---:|
| **Week-4 Baseline (`score >= 3`)** | W05 Practice Set — 80/20 Stratified Random Split ($n_{\text{test}}=20$, base rate $0.6000$) | Accuracy | **0.7000** |
| **ML Model (W05 Logistic Regression, 4 features)** | W05 Practice Set — 80/20 Stratified Random Split ($n_{\text{test}}=20$, base rate $0.6000$) | Accuracy | **0.8000** |
| **Week-4 Baseline (`score >= 3`)** | W05 Practice Set — 80/20 Stratified Random Split ($n_{\text{test}}=20$, base rate $0.6000$) | F1-Score | **0.7857** |
| **ML Model (W05 Logistic Regression, 4 features)** | W05 Practice Set — 80/20 Stratified Random Split ($n_{\text{test}}=20$, base rate $0.6000$) | F1-Score | **0.8571** |
| **Week-4 Baseline (`score >= 3`)** | FlyRank 30k — 80/20 Stratified Random Split ($n_{\text{test}}=6,000$, base rate $0.5420$) | Accuracy / F1 / ROC AUC | **0.5432 / 0.6557 / 0.5530** |
| **ML Model (W05 Raw 4-Feature Logistic Regression)** | FlyRank 30k — 80/20 Stratified Random Split ($n_{\text{test}}=6,000$, base rate $0.5420$) | Accuracy / F1 / ROC AUC | **0.5498 / 0.6685 / 0.5557** |
| **ML Model (7-Feature Pre-Decision Logistic Regression)** | FlyRank 30k — 80/20 Stratified Random Split ($n_{\text{test}}=6,000$, base rate $0.5420$) | Accuracy / F1 / ROC AUC | **0.6358 / 0.6575 / 0.6900** |
| **Week-4 Baseline (`score >= 3`)** | FlyRank 30k — Client-Grouped Holdout Split ($6$ unseen clients, $n_{\text{test}}=2,325$, base rate $0.3910$) | Accuracy / F1 / ROC AUC / P@50 | **0.4817 / 0.4356 / 0.4821 / 0.5000** |
| **Starter Reference Baseline (`scripts/02_baseline_score.py`)** | FlyRank 30k — Client-Grouped Holdout Split ($6$ unseen clients, $n_{\text{test}}=2,325$, base rate $0.3910$) | Accuracy / F1 / ROC AUC / P@50 | **0.6086 / 0.2743 / 0.6269 / 0.2400** |
| **ML Model (W05 Raw 4-Feature Logistic Regression)** | FlyRank 30k — Client-Grouped Holdout Split ($6$ unseen clients, $n_{\text{test}}=2,325$, base rate $0.3910$) | Accuracy / F1 / ROC AUC / P@50 | **0.3897 / 0.5256 / 0.4191 / 0.2800** |
| **ML Model (7-Feature Pre-Decision Logistic Regression)** | FlyRank 30k — Client-Grouped Holdout Split ($6$ unseen clients, $n_{\text{test}}=2,325$, base rate $0.3910$) | Accuracy / F1 / ROC AUC / P@50 | **0.6688 / 0.5595 / 0.7144 / 0.6200** |

### 11.2 Full Detailed Metrics Across Splits

| Population & Split | Method | Accuracy | F1-Score | Precision | Recall | ROC AUC | Avg Precision | Precision@20 | Precision@50 | Confusion Matrix (TP / FP / TN / FN) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **W05 Practice Set** (`N=100`, Stratified 80/20, `n_test=20`, Base Rate `0.6000`) | Week-4 Baseline (`score >= 3`) | 0.7000 | 0.7857 | 0.6875 | 0.9167 | 0.8542 | 0.8851 | 0.6000 | — | 11 / 5 / 3 / 1 |
| **W05 Practice Set** (`N=100`, Stratified 80/20, `n_test=20`, Base Rate `0.6000`) | Week-5 Logistic Regression (4 raw features) | **0.8000** | **0.8571** | **0.7500** | **1.0000** | **0.9375** | **0.9511** | 0.6000 | — | 12 / 4 / 4 / 0 |
| **FlyRank 30k** (Stratified 80/20, `n_test=6,000`, Base Rate `0.5420`) | Week-4 Baseline (`score >= 3`) | 0.5432 | 0.6557 | 0.5543 | 0.8026 | 0.5530 | 0.5780 | 0.7000 | 0.7400 | 2610 / 2099 / 649 / 642 |
| **FlyRank 30k** (Stratified 80/20, `n_test=6,000`, Base Rate `0.5420`) | Starter Reference Baseline (`score >= 0.50`) | 0.5372 | 0.5166 | 0.5953 | 0.4563 | 0.5787 | 0.5699 | 0.4000 | 0.4800 | 1484 / 1009 / 1739 / 1768 |
| **FlyRank 30k** (Stratified 80/20, `n_test=6,000`, Base Rate `0.5420`) | Logistic Regression (4 raw features) | 0.5498 | **0.6685** | 0.5563 | **0.8376** | 0.5557 | 0.5922 | 0.3500 | 0.4600 | 2724 / 2173 / 575 / 528 |
| **FlyRank 30k** (Stratified 80/20, `n_test=6,000`, Base Rate `0.5420`) | Logistic Regression (7 pre-decision features, log-scaled) | **0.6358** | 0.6575 | **0.6706** | 0.6448 | **0.6900** | **0.7079** | **0.9000** | **0.9000** | 2097 / 1030 / 1718 / 1155 |
| **FlyRank 30k** (**Client-Grouped Holdout**, `n_test=2,325`, Base Rate `0.3910`) | Week-4 Baseline (`score >= 3`) | 0.4817 | 0.4356 | 0.3793 | 0.5116 | 0.4821 | 0.4015 | 0.4000 | 0.5000 | 465 / 761 / 655 / 444 |
| **FlyRank 30k** (**Client-Grouped Holdout**, `n_test=2,325`, Base Rate `0.3910`) | Starter Reference Baseline (`score >= 0.50`) | 0.6086 | 0.2743 | 0.4986 | 0.1892 | 0.6269 | 0.4676 | 0.1500 | 0.2400 | 172 / 173 / 1243 / 737 |
| **FlyRank 30k** (**Client-Grouped Holdout**, `n_test=2,325`, Base Rate `0.3910`) | Logistic Regression (4 raw features, unweighted) | 0.3897 | 0.5256 | 0.3775 | **0.8647** | 0.4191 | 0.3351 | 0.3000 | 0.2800 | 786 / 1296 / 120 / 123 |
| **FlyRank 30k** (**Client-Grouped Holdout**, `n_test=2,325`, Base Rate `0.3910`) | Logistic Regression (7 pre-decision features, log-scaled) | **0.6688** | **0.5595** | **0.5828** | 0.5380 | **0.7144** | **0.5583** | **0.7000** | **0.6200** | 489 / 350 / 1066 / 420 |

**Key Honest Findings from the Results Table:**
1. **On the W05 Practice Set ($n_{\text{test}}=20$):** The 4-feature Logistic Regression model improves accuracy from $0.7000$ to $0.8000$ ($+10.0$ percentage points) and ROC AUC from $0.8542$ to $0.9375$ over the Week-4 rule baseline.
2. **Negative/Diagnostic Result on Un-Logged 4-Feature Logistic Regression under Client Holdout ($n_{\text{test}}=2,325$):** When applied directly to raw un-logged traffic counts on the 30,000-row dataset under Client-Grouped Holdout, the raw 4-feature Logistic Regression **does not beat the Week-4 baseline on Accuracy or ROC AUC** ($0.3897$ vs. $0.4817$ Accuracy; $0.4191$ vs. $0.4821$ ROC AUC). Why? Because the $26$ training clients have a $55.48\%$ positive rate while the $6$ held-out test clients have a $39.10\%$ positive rate, and extreme raw traffic outliers distort the linear decision boundary, causing the unweighted raw model to predict `1` on $2,082 / 2,325$ ($89.5\%$) of test rows.
3. **Positive Result with Log-Scaled Pre-Decision Features (`logreg_7feat_predecision`):** Once heavy-tailed impressions and clicks are log-transformed (`log1p`) alongside `staleness_days`, `position`, `ctr_pct`, `days_with_impressions`, and `content_age_days` with balanced class weights, Logistic Regression generalizes across unseen clients, achieving **$0.6688$ Accuracy**, **$0.5595$ F1**, **$0.7144$ ROC AUC**, and **$0.6200$ Precision@50** (vs. $0.5000$ for the Week-4 rule baseline and $0.2400$ for the reference baseline, against a $0.3910$ test base rate).

---

## 12. Model Interpretation

We inspect the standardized Logistic Regression coefficients ($\beta_j$) and absolute coefficients ($|\beta_j|$) to understand which signals drive the predictions. All interpretations are strictly **observational and directional**, not causal.

### 12.1 Week-5 Model Coefficients (W05 Practice Dataset, $N = 100$, Intercept $= +0.524358$)

| Rank | Feature | Standardized Coefficient ($\beta$) | Absolute Coefficient ($\|\beta\|$) | Directional Interpretation |
|---:|---|---:|---:|---|
| 1 | `staleness_days` | **+0.640502** | **0.640502** | Strongest positive association: items with more days since their last update are given higher refresh priority probability. |
| 2 | `impressions` | **+0.432939** | **0.432939** | Second strongest positive association: higher search exposure increases the predicted priority score. |
| 3 | `clicks` | **+0.113056** | **0.113056** | Mild positive association in the 100-row sample. |
| 4 | `position` | **+0.104730** | **0.104730** | Mild positive association: deeper (worse) numerical position numbers are associated with higher refresh need. |

### 12.2 7-Feature Pre-Decision Model Coefficients (FlyRank 30k Client-Grouped Holdout)

| Rank | Feature | Standardized Coefficient ($\beta$) | Absolute Coefficient ($\|\beta\|$) | Directional Interpretation |
|---:|---|---:|---:|---|
| 1 | `log_impressions` | **+1.236082** | **1.236082** | Holding clicks and other variables constant, high search impressions without proportional clicks are strongly associated with declining performance (`is_declining_label = 1`). |
| 2 | `log_clicks` | **-0.957725** | **0.957725** | Holding impressions constant, pages that actively capture organic clicks are strongly associated with stability/growth (`is_declining_label = 0`). Together, `+1.236 * log_impressions - 0.958 * log_clicks` acts as a learned log-CTR gap detector. |
| 3 | `content_age_days` | **-0.399013** | **0.399013** | Controlling for staleness and visibility, established older pages in the active sample show lower short-term 30d drop rates than recently matured cohorts hitting the mid-life plateau. |
| 4 | `position` | **-0.293176** | **0.293176** | Pages at stronger (lower numerical) positions that lack clicks or consistency face sharper measurable 30-day impression loss risk than already-deep pages. |
| 5 | `days_with_impressions` | **-0.210529** | **0.210529** | Consistent daily visibility across the 90-day window is associated with lower decline probability. |
| 6 | `staleness_days` | **+0.102167** | **0.102167** | Longer time since last update (`days_since_last_update`) is positively associated with content decline risk. |
| 7 | `ctr_pct` | **+0.006194** | **0.006194** | Near-zero marginal linear weight once `log_impressions` and `log_clicks` are both included in the linear model. |

---

## 13. Error Analysis

A metric table alone hides where a decision-support model can mislead a human reviewer. We inspect the actual misclassifications on both test sets.

### 13.1 Exact Test Errors on the Week-5 Practice Test Set ($n_{\text{test}} = 20$, Total Errors $= 4$)

On the 20-row test set in `w05_model.ipynb`, the model made **0 False Negatives** and **4 False Positives** (`actual = 0`, `predicted = 1`):

| Test Index | Item ID | Actual | Predicted | Model Prob | W04 Baseline Score (Pred) | Important Signals (`impressions`, `clicks`, `staleness_days`, `position`) | Possible Missing Information & Why Recommendation Could Be Wrong |
|---:|---|---:|---:|---:|---|---|---|
| 5 | `item_100` | 0 | 1 | 0.5336 | 2 (`0`) | `imp=904`, `clicks=740`, `stale=22d`, `pos=2` (CTR = $81.9\%$) | **Why wrong:** Ranked at position 2 with an exceptionally high CTR ($740/904$). Because `staleness_days=22` and `clicks` has a positive coefficient ($+0.113$), the model narrowly crosses $0.50$ ($p=0.5336$). Missing information: recent click trend is actually healthy; page does not need a rewrite. |
| 9 | `item_067` | 0 | 1 | 0.5379 | 4 (`1`) | `imp=1095`, `clicks=460`, `stale=24d`, `pos=2` (CTR = $42.0\%$) | **Why wrong:** Both the W04 rule (`score=4`) and Logistic Regression ($p=0.5379$) flag this item because `impressions >= 1000` and `staleness_days >= 14` ($24\text{d}$). However, it sits at position 2 with strong click capture ($42\%$). Missing information: competitor stability and conversion rate. |
| 10 | `item_079` | 0 | 1 | 0.5557 | 3 (`1`) | `imp=479`, `clicks=476`, `stale=25d`, `pos=10` (CTR = $99.4\%$) | **Why wrong:** High staleness ($25\text{d}$) and borderline position ($10$) push both the baseline (`score=3`) and model ($p=0.5557$) to predict refresh, ignoring that nearly every impression converts to a click (likely navigational/branded query intent). |
| 12 | `item_063` | 0 | 1 | 0.7891 | 4 (`1`) | `imp=4898`, `clicks=749`, `stale=19d`, `pos=7` (CTR = $15.3\%$) | **Why wrong:** Very high impressions ($4,898$) and moderate staleness ($19\text{d}$) produce a high priority probability ($p=0.7891$) even though the page ranks on Page 1 (`pos=7`) with $749$ clicks. Missing information: whether impressions are growing or seasonal. |

### 13.2 Representative Errors on the FlyRank 30k Client-Grouped Holdout Set ($n_{\text{test}} = 2,325$)

On the 6 unseen test clients, the 7-feature pre-decision model produced $1,066$ True Negatives, $489$ True Positives, $350$ False Positives, and $420$ False Negatives:

| Error Type | Item ID (`content_id`) | Actual (`trend_direction`) | Predicted (`pred_prob`) | Important Signals | Possible Missing Information & Why Recommendation Could Be Wrong |
|---|---|---|---|---|---|
| **False Positive** | `content_3e79eaafc89d` | `0` (`stable`) | `1` ($p = 0.9159$) | `imp=8,779`, `clicks=0`, `stale=20d`, `pos=11.6`, `days_vis=88` | High impressions ($8,779$) with $0$ clicks at position $11.6$ strongly resembles a decaying page, but its 30d impression trend was actually `stable`. Missing info: SERP zero-click features (AI Overviews / featured snippets) may be absorbing clicks while impression volume holds steady. |
| **False Positive** | `content_8fdbff16a886` | `0` (`new`) | `1` ($p = 0.9060$) | `imp=3,863`, `clicks=0`, `stale=20d`, `pos=12.0`, `days_vis=24` | Content item had $0$ impressions in days 31–60 (`impressions_prev_30d = 0`) and surged to $3,863$ impressions in the last 30 days (`trend_direction = "new"`). Because pre-decision 90d totals do not expose the 30d sub-windows (to prevent label leakage), the model sees high 90d impressions with 0 clicks and flags it for review. |
| **False Negative** | `content_86748254b6bf` | `1` (`down`) | `0` ($p = 0.0253$) | `imp=1`, `clicks=0`, `stale=20d`, `pos=90.0`, `age=495d` | The item's label is `down`, but it has only $1$ impression over 90 days at position $90.0$. From a decision-support standpoint, deprioritizing a 1-impression deep page is operationally sensible—its "decline" is low-volume noise rather than a high-value refresh opportunity. |

---

## 14. Top-10 Content Refresh Queue

Below are the ranked Top-10 decision-support tables generated from the actual data. All identifiers are safe, non-private pseudonyms.

### 14.1 Table A — Preserved Week-4 Baseline Top-10 Queue (`ml07_baseline_practice_dataset.csv`, $N = 30$)

| Rank | Item ID | Score | Action | Reason Code | Confidence | What Could Make It Wrong |
|---:|---|---:|---|---|---|---|
| 1 | `item_002` | 5 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | High (all 3 baseline thresholds met: `imp=3892`, `stale=25d`, `pos=11`) | Search demand may be seasonal or split across a sibling page; verify query trend before editing. |
| 2 | `item_011` | 5 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | High (all 3 baseline thresholds met: `imp=2679`, `stale=25d`, `pos=12`) | Search demand may be seasonal or split across a sibling page; verify query trend before editing. |
| 3 | `item_015` | 5 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | High (all 3 baseline thresholds met: `imp=3615`, `stale=24d`, `pos=10`) | Search demand may be seasonal or split across a sibling page; verify query trend before editing. |
| 4 | `item_019` | 5 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | High (all 3 baseline thresholds met: `imp=4214`, `stale=15d`, `pos=8`) | Search demand may be seasonal or split across a sibling page; verify query trend before editing. |
| 5 | `item_020` | 5 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | High (all 3 baseline thresholds met: `imp=2306`, `stale=22d`, `pos=10`) | Strong observed CTR ($16.91\%$); high click capture despite age/position may mean content still satisfies user intent. |
| 6 | `item_024` | 5 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | High (all 3 baseline thresholds met: `imp=4641`, `stale=14d`, `pos=12`) | Borderline staleness (`14d` cutoff); recent update may still be indexing or re-crawling. |
| 7 | `item_025` | 5 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | High (all 3 baseline thresholds met: `imp=3929`, `stale=16d`, `pos=9`) | Search demand may be seasonal or split across a sibling page; verify query trend before editing. |
| 8 | `item_026` | 5 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | High (all 3 baseline thresholds met: `imp=3254`, `stale=18d`, `pos=10`) | Very low CTR ($0.80\%$) at position 10; title/meta snippet mismatch may be the primary issue rather than body text staleness. |
| 9 | `item_003` | 4 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | Medium-High (2 primary thresholds met: `imp=3307`, `stale=26d`, `pos=6`) | Already ranks on Page 1 (`pos=6`); low CTR ($1.51\%$) calls for snippet/title review before altering page structure. |
| 10 | `item_005` | 4 | `REVIEW_NOW` | `STALE_HIGH_VOLUME` | Medium-High (2 primary thresholds met: `imp=2221`, `stale=27d`, `pos=6`) | Strong observed CTR ($19.99\%$) and Page-1 rank (`pos=6`); editing risks disrupting an already well-performing page. |

### 14.2 Table B — FlyRank 30k Held-Out Test Clients Top-10 Queue (`data/raw/content_refresh_anonymized.csv`, $n_{\text{test}} = 2,325$)

*Note: Priority Score ($0\text{–}100$) combines $70\%$ pre-decision Logistic Regression probability + $30\%$ normalized Week-4 baseline score (`score / 5`).*

| Rank | Item ID | Score | Action | Reason Code | Confidence | What Could Make It Wrong |
|---:|---|---:|---|---|---|---|
| 1 | `content_3e79eaafc89d` | 94.1 (`p=0.916`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=8,779`, `days_vis=88`, `pos=11.6`) | Traffic pattern is actually stable in 30d window; zero clicks at pos 11.6 may reflect SERP layout/AI overview click compression rather than content decay. |
| 2 | `content_8fdbff16a886` | 93.4 (`p=0.906`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | Medium-High (`imp=3,863`, `days_vis=24`, `pos=12.0`) | Newly surging impression trend (`24` visible days); page may still be climbing into Page 1 and needs monitoring rather than a rewrite. |
| 3 | `content_477f7892c1f1` | 93.0 (`p=0.900`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=5,147`, `days_vis=88`, `pos=8.4`) | Page ranks on Page 1 (`pos=8.4`) with zero clicks; snippet/title intent mismatch or SERP feature crowding could be the true bottleneck. |
| 4 | `content_ef731e95e774` | 93.0 (`p=0.899`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=5,226`, `days_vis=88`, `pos=10.7`) | Traffic change could reflect seasonal demand shifts or sibling URL cannibalization rather than content staleness. |
| 5 | `content_03582b12af32` | 92.6 (`p=0.895`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=4,918`, `days_vis=88`, `pos=10.0`) | Page ranks at the bottom of Page 1 (`pos=10.0`); check whether a sibling page on the same domain is splitting query demand. |
| 6 | `content_25ebfb5aa399` | 92.1 (`p=0.887`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=4,984`, `days_vis=88`, `pos=14.4`) | Observed 30d impressions are stable; striking-distance position (`14.4`) may benefit more from internal linking than text replacement. |
| 7 | `content_c2623272bc49` | 91.9 (`p=0.885`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=4,050`, `days_vis=88`, `pos=11.3`) | Traffic change could reflect seasonal demand shifts or sibling URL cannibalization rather than content staleness. |
| 8 | `content_1a0b27cdb6a1` | 91.9 (`p=0.885`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=3,969`, `days_vis=88`, `pos=9.9`) | Page ranks on Page 1 (`pos=9.9`) with near-zero CTR; SERP features/AI overview or snippet mismatch may explain low clicks. |
| 9 | `content_38b3f0575991` | 91.9 (`p=0.884`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=3,754`, `days_vis=88`, `pos=8.9`) | Page ranks on Page 1 (`pos=8.9`) with near-zero CTR; verify title/meta description alignment before full content rewrite. |
| 10 | `content_ccd35f0d9336` | 91.5 (`p=0.879`, `w04=5`) | `REVIEW_NOW (Refresh & Snippet/CTR Check)` | `STALE_HIGH_VOLUME \| STRIKING_DISTANCE_DECAY_RISK \| LOW_CTR_HIGH_EXPOSURE \| HIGH_MODEL_DECLINE_RISK` | High (`imp=3,583`, `days_vis=88`, `pos=12.1`) | Traffic change could reflect seasonal demand shifts or sibling URL cannibalization rather than content staleness. |

*(Observe that 7 of the Top 10 items on the unseen held-out clients—Ranks 3, 4, 5, 7, 8, 9, and 10—are true declining pages [`is_declining_label = 1`], giving a measured **Precision@10 of $0.7000$** against a held-out test base rate of $0.3910$.)*

---

## 15. Limitations

1. **Observational Data Only (No Causal Identification):** All findings are based on observational search and analytics logs. We cannot claim that refreshing a prioritized page will *cause* rankings or clicks to recover without a controlled experiment or matched difference-in-differences design.
2. **Contemporaneous Proxy Label (`is_declining_label`):** In the 30,000-row starter snapshot, `is_declining_label` is derived from `trend_direction == "down"` (comparing the most recent 30 days against days 31–60 within the 90-day window) rather than a strictly post-90-day forward window. While we excluded `trend_direction`, `trend_pct`, and all `*_last_30d` / `*_prev_30d` columns from our models, cumulative 90-day aggregates (`impressions_90d`, `clicks_90d`) still span the same 90-day calendar period. Building strictly non-overlapping prior-90d $\to$ next-30d windows on `fact_content_daily_performance` is the recommended next step for production deployment.
3. **Cross-Client Base-Rate Heterogeneity:** Across the 32 clients in the 30k dataset, the proportion of declining pages varies substantially ($55.48\%$ in the 26 training clients vs. $39.10\%$ in the 6 held-out test clients). Uncalibrated probability thresholds (`0.50`) can over-predict decline on healthier client portfolios unless recalibrated per client.
4. **Unobserved External Confounders:** Competitor publishing, Google core algorithm updates, SERP layout changes (such as AI Overviews compressing organic CTR), and site-internal URL cannibalization are not directly observed in page-level summary rows.

---

## 16. Practical Decision-Support Takeaways

1. **How an Editor Uses the Queue Tomorrow:** Start at Rank 1 of the high-confidence queue (`REVIEW_NOW`). Before rewriting any text, perform a 3-step human check:
   - *Step 1 (Query & Cannibalization Check):* Verify whether another URL on the same client domain gained the impressions this page lost. If so, consolidate/redirect rather than refreshing both.
   - *Step 2 (Snippet vs. Body Diagnosis):* If the page already ranks on Page 1 (`position <= 10`) with high impressions but near-zero CTR (e.g., `content_477f7892c1f1` at pos $8.4$), test a title and meta-description refinement first rather than rewriting the body copy.
   - *Step 3 (Depth & Freshness Update):* If the page sits in striking distance (`position 8–20`) with `staleness_days >= 14` and confirmed impression decay, update outdated statistics, examples, and subtopics.
2. **The No-Go List (What Must Never Be Automated):** Never connect model scores directly to automated LLM rewrites, automated URL pruning/deletion, or unattended publishing. High-traffic pages with high CTR (such as `item_005` and `item_063` in our error analysis) can receive high raw volume/staleness scores while already performing strongly.
3. **Monitoring & Retrain Triggers:** Re-evaluate model calibration whenever portfolio-level declining base rates shift by more than $\pm 10$ percentage points, after major search engine core updates, or when `Precision@20` on human-reviewed audit batches falls below $0.50$.

---

## 17. Conclusion

By preserving our transparent Week-4 rule baseline and testing it fairly against standardized Logistic Regression models across both our weekly practice data ($N = 100$) and the 30,000-row FlyRank portfolio dataset ($N = 30,000$, $32$ clients), we demonstrated two central lessons of applied machine learning for search discovery:

1. **Simple linear models beat hand-written rules when features are properly scaled for the domain:** On the Week-5 practice set, Logistic Regression improved test accuracy from **$0.7000$** to **$0.8000$** ($\text{ROC AUC}: 0.8542 \to 0.9375$). On the 30,000-row dataset under an honest **Client-Grouped Holdout Split**, raw un-logged traffic counts failed to generalize ($\text{Accuracy} = 0.3897$), whereas log-scaling heavy-tailed impressions and clicks (`log1p`) alongside pre-decision staleness, position, CTR, visibility consistency, and content age achieved **$0.6688$ Accuracy**, **$0.7144$ ROC AUC**, and **$0.6200$ Precision@50** (versus $0.5000$ for the Week-4 rule baseline and $0.2400$ for the reference baseline).
2. **Error analysis and honest claim discipline are essential:** Inspecting actual false positives revealed that linear combinations of volume and staleness can over-flag pages that already achieve high CTR on Page 1 or pages experiencing zero-click SERP compression. Pairing model probabilities with explicit reason codes and human review guardrails turns noisy search metrics into a practical, trustworthy decision-support workflow.

---

## 18. Reproducibility

Everything in this report and on the deployed research paper page can be reproduced from a fresh clone of [`https://github.com/ganeshmpsmg/flyrank-ml-internship`](https://github.com/ganeshmpsmg/flyrank-ml-internship):

### 18.1 Environment & Random Seeds

- **Python Version:** Python 3.11+ (verified on Python 3.14)
- **Random Seed:** Fixed at `RANDOM_STATE = 42` across all train/test splits, client holdout permutations (`np.random.default_rng(42)`), and `LogisticRegression(random_state=42)`.
- **Dependencies (`requirements.txt`):** `pandas>=2.2`, `numpy>=1.26`, `scikit-learn>=1.4`, `matplotlib>=3.8`, `reportlab>=4.0`, `duckdb>=1.0`, `huggingface_hub>=0.24`.

### 18.2 Exact Commands to Reproduce All Results, Figures, and Notebooks

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the capstone analysis pipeline (generates work/outputs/capstone_metrics.json and all 5 SVG figures in work/figures/ and docs/figures/)
python work/scripts/run_capstone_pipeline.py

# 3. Run the reference starter pipeline for comparison
python scripts/run_all.py
```

### 18.3 Visual Figures Generated (`work/figures/` and `docs/figures/`)

1. [`fig1_feature_distributions.svg`](figures/fig1_feature_distributions.svg) — Key Pre-Decision Feature Distributions (`impressions_90d`, `log1p(impressions_90d)`, `staleness_days`, and `avg_position`, $N=30,000$).
2. [`fig2_baseline_vs_model_metrics.svg`](figures/fig2_baseline_vs_model_metrics.svg) — Week-4 Baseline vs. Logistic Regression Metrics Across Validation Designs.
3. [`fig3_logistic_regression_coefficients.svg`](figures/fig3_logistic_regression_coefficients.svg) — Standardized Logistic Regression Coefficients & Absolute Magnitudes.
4. [`fig4_error_analysis_breakdown.svg`](figures/fig4_error_analysis_breakdown.svg) — Error Analysis Breakdown & False-Positive Signal Profile.
5. [`fig5_top10_refresh_queue.svg`](figures/fig5_top10_refresh_queue.svg) — Top-10 Content Refresh Priority Queue Visualization.
