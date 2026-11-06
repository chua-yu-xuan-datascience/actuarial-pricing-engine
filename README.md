# End-to-End Predictive Insurance Pricing Engine

A portfolio project demonstrating actuarial pricing methodology from raw data through to an interactive commercial tariff tool. Built using the French Motor Third-Party Liability (MTPL) open dataset.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [Stakeholders](#3-stakeholders)
4. [How It Works](#4-how-it-works)
5. [Actuarial Enhancements](#5-actuarial-enhancements)
6. [Model Testing & Validation](#6-model-testing--validation)
7. [Project Structure](#7-project-structure)
8. [How to Run](#8-how-to-run)
9. [How to Use the Website](#9-how-to-use-the-website)
10. [How to Modify the Code](#10-how-to-modify-the-code)
11. [Maintenance Guide](#11-maintenance-guide)
12. [CAS Exam Relevance](#12-cas-exam-relevance)

---

## 1. Purpose

This project builds a **data-driven insurance pricing engine** for motor third-party liability (MTPL) business. It demonstrates the full actuarial workflow:

- Sourcing and cleaning real claims data
- Fitting industry-standard Generalised Linear Models (GLMs)
- Converting statistical outputs into a transparent, regulator-ready tariff
- Deploying the tariff as a live interactive web application

It is designed as a **learning and portfolio project** to practise the pricing concepts tested in CAS Exams 5, 6, 8, and PA, and to simulate the kind of work a pricing actuary performs at an insurance company.

---

## 2. The Problem It Solves

### The Commercial Failure of Flat or Simplistic Pricing

If an insurer charges the same premium to every driver regardless of their risk profile, two destructive forces immediately emerge:

**Adverse Selection**
High-risk drivers — young drivers, those with poor claims history, high-powered vehicles — recognise they are getting a bargain and disproportionately flood the portfolio. Low-risk drivers recognise they are overpaying and defect to competitors offering better rates. Over time the insurer is left holding only the worst risks at the wrong price.

**Cross-Subsidisation**
Low-risk policyholders are forced to fund the losses of high-risk policyholders. This violates the principle of actuarial equity — that each policyholder should pay a premium commensurate with their own expected cost.

### What This Engine Does

This pricing engine answers the core actuarial question:

> *For a given policyholder profile, what is the expected annual cost of their claims, and what premium should we charge to cover that cost plus expenses and profit?*

It achieves this by decomposing risk into two components — **how often** a claim occurs (frequency) and **how much** each claim costs (severity) — and multiplying them to arrive at a risk-specific pure premium. Commercial loadings are then layered on top to produce a final indicated premium.

---

## 3. Stakeholders

| Stakeholder | What They Get from This Tool |
|---|---|
| **Pricing Actuary** | A reproducible, statistically defensible rating model with GLM outputs, lift charts, and Gini diagnostics |
| **Underwriter** | A live quote interface — enter a risk profile, instantly see frequency, severity, and final premium |
| **Product Manager / Commercial Actuary** | The What-If tab: explore how changing expense ratios or profit targets shifts premiums across the portfolio before a business decision is made |
| **Regulator / Chief Actuary** | The Model Diagnostics tab: GLM coefficient table, statistical significance flags, lift charts, and Lorenz curve demonstrate the model is transparent and explainable |
| **Student / Trainee Actuary** | The Actuarial Concepts tab maps every step of the model to CAS exam syllabus topics with worked explanations |

---

## 4. How It Works

The engine follows seven sequential steps.

### Step 1 — Data Acquisition and Pre-Processing

**Source:** French Motor MTPL dataset (`freMTPL2freq` + `freMTPL2sev`), fetched automatically via `sklearn.datasets.fetch_openml`. Contains ~678,000 policies with vehicle, driver, and geographic attributes plus claim counts and amounts.

**Key actions:**
- Merge claim severity data onto the policy table
- Clip exposure between 1 day and 1 year (no policy should exceed an annual window)
- Calculate the **pure premium** = total loss ÷ exposure
- Cap individual pure premiums at the 95th percentile; store the excess as a **large loss loading factor**
- Engineer rating bands for driver age, vehicle age, and bonus-malus score
- Create an interaction flag for young drivers (≤25) in high-powered vehicles (≥10 CV)

**Actuarial concept:** *Exposure adjustment* and *large loss capping*. Exposure normalises claim counts to a rate per policy-year. Capping protects systemic rating factors from distortion by random high-severity events.

### Step 2 — Train / Test Split (70 / 30)

Before any model is fitted, the dataset is split into a **70% training set** and a **30% held-out test set**. The split is stratified on whether a claim occurred so both sets have a similar claim rate. This is the foundation of honest model evaluation — every performance metric reported is computed on data the model has never seen.

The test set is sealed until Step 6. It is never used to fit, tune, or select the model.

**Actuarial concept:** *Out-of-sample validation*. A model that looks good on the data it was trained on tells you nothing about how it will perform on next quarter's new business. Only out-of-sample metrics are meaningful for production deployment.

### Step 3 — Frequency Model (Poisson GLM)

Fits a **Poisson GLM with a log link** to predict the expected number of claims per year. The critical implementation detail is the **offset term**: passing `log(Exposure)` as an offset forces the model to estimate a *rate*, not a raw count. Without this, a 1-month policy and a 12-month policy are treated identically.

The log link produces a **multiplicative rating structure** — each factor (age band, region, bonus-malus) multiplies into the base rate — which is the universal standard for insurance tariff manuals globally.

### Step 4 — Severity Model (Gamma GLM)

Fits a **Gamma GLM with a log link** to predict the expected cost *per claim*. Critically, this model is fitted only on records where a claim occurred — severity is a **conditional distribution** and mixing zero-loss records would bias the mean downwards. The Gamma family is appropriate because claim amounts are strictly positive, right-skewed, and exhibit variance proportional to the mean.

### Step 5 — Tariff Construction

**Pure premium** = Predicted Frequency × Predicted Severity

**Indicated premium** = Pure Premium × (1 + Large Loss Load) ÷ (1 − Expense Ratio − Profit Margin)

The denominator is the **permissible loss ratio** — the fraction of premium the insurer can pay in claims after covering expenses and meeting its profit target.

A **Bühlmann credibility blend** then adjusts predictions for policies with observed claim history, weighting the GLM estimate against actual observed losses proportionally to data volume.

Finally, a **minimum premium floor** is applied: no policy can ever be priced below a fixed amount regardless of how low the statistical risk score is, reflecting real-world fixed acquisition and administrative costs.

### Step 6 — Validation (test set only)

All validation metrics are computed exclusively on the 30% held-out test set. See [Section 6](#6-model-testing--validation) for the full list and how to interpret each one.

### Step 7 — Tariff Manual and Off-Balance Correction

GLM coefficients are converted to **relativities** (multiplicative factors relative to a base class), formatted as a regulator-ready tariff table. An **off-balance correction factor** is then calculated to ensure the sum of all indicated premiums exactly equals the sum of predicted losses at portfolio level — a mandatory step before any rate filing.

---

## 5. Actuarial Enhancements

The following features go beyond a standard academic GLM to reflect real commercial practice:

| Feature | Why It Matters |
|---|---|
| **Large loss capping + loading factor** | Prevents one catastrophic claim from distorting a demographic's rating factor for years |
| **Exposure offset in Poisson GLM** | Industry-standard; omitting it is a fundamental pricing error |
| **Gamma GLM for severity (claims-only)** | Correctly models the conditional distribution of claim costs |
| **Interaction term: young driver × high power** | Captures compounded risk that additive terms alone cannot express |
| **Bühlmann credibility blend** | Partial credibility for high-claim-count segments; tested on CAS Exam 6 |
| **Increased Limits Factors (ILF)** | Demonstrates understanding of limit-layer pricing; CAS Exam 8 topic |
| **Gini coefficient and Lorenz curve** | Model performance metric used in real model validation frameworks |
| **Off-balance correction** | Mandatory in commercial rate filings; often missed in academic projects |
| **Minimum premium floor** | Reflects operational reality that policies have a fixed cost floor |
| **Quasi-Poisson GLM** | Pearson phi = 2.66 confirmed — scales all standard errors by ×1.63 so p-values and confidence intervals are honest; identical point estimates to standard Poisson |
| **Isotonic smoothing (Pool Adjacent Violators)** | 3 inversions found in DrivAge bands and 2 in VehAge bands — corrected without refitting the GLM; corrections are computed from training data only |
| **3-metric train vs test evaluation** | Gini (ranking), Normalised Poisson deviance (distributional fit), E/A frequency ratio (level calibration) — each metric catches a different failure mode that the others miss |

---

## 6. Model Testing & Validation

### 6.1 The 70 / 30 Train-Test Split

The full dataset (~678,000 policies) is split **before any modelling begins**:

| Split | Size | Purpose |
|---|---|---|
| **Train (70%)** | ~475,000 policies | Fit both GLMs and compute normalisation statistics |
| **Test (30%)** | ~203,000 policies | Evaluate all performance metrics — never seen during training |

Stratification on the claim indicator (`ClaimNb > 0`) ensures both splits have the same underlying claim rate, preventing a lucky draw from concentrating all claimants in one set.

A fixed random seed (`RANDOM_SEED = 42`) makes the split reproducible. Change it in `pricing_engine.py` to stress-test stability.

**Why not cross-validation?** Insurance pricing GLMs are typically validated on a chronological hold-out (future quarters). A stratified random split is the right proxy when time-stamped data is unavailable, as with this dataset. Cross-validation is valuable for hyperparameter tuning but less meaningful here since GLMs have no tuneable regularisation hyperparameters by default.

**Data leakage fix:** numeric features are standardised using the training set's mean and standard deviation only. The same statistics are then applied to the test set. This prevents the test set from influencing the feature scaling, which would be a form of data leakage that inflates apparent performance.

---

### 6.2 Validation Tests (all on test set)

#### Test 1 — Lift Chart (Loss Ratio by Predicted Risk Decile)

Rank test policies into 10 deciles by predicted pure premium (safest to riskiest). For each decile, compute the actual loss ratio (actual losses ÷ indicated premium).

**What good looks like:** a steeply upward-sloping curve, with the bottom decile well below 1.0 and the top decile well above 1.0. A flat line means the model adds no value over charging everyone the same rate.

**Actuarial interpretation:** demonstrates the model's ability to *segment* risk — the fundamental goal of a tariff.

---

#### Test 2 — Double-Lift Chart (Predicted vs Actual Pure Premium per Decile)

Within each decile, plot the mean *predicted* pure premium alongside the mean *actual* pure premium.

**What good looks like:** bars of similar height in every decile. A consistent gap (model always higher or lower) indicates systematic bias. A gap only in extreme deciles suggests the model struggles to price tail risks.

---

#### Test 3 — Lorenz Curve and Gini Coefficient

Sort test policies by indicated premium. Plot the cumulative share of actual losses against the cumulative share of policies. The Gini coefficient = 2 × area between the model curve and the 45-degree diagonal.

**Interpretation guide:**

| Gini Range | Signal |
|---|---|
| < 0.15 | Poor — model barely outperforms random assignment |
| 0.15 – 0.30 | Acceptable |
| 0.30 – 0.50 | Good — typical for a well-specified motor GLM |
| > 0.50 | Excellent (verify for data leakage if > 0.60) |

---

#### Test 4 — RMSE vs Naive Benchmark

Compute the Root Mean Squared Error of predicted pure premium against actual pure premium on the test set. Compare to a **naive benchmark** that predicts the training mean pure premium for every policy.

```
RMSE_model  = sqrt( mean( (PredPP - ActualPP)² ) )
RMSE_naive  = sqrt( mean( (TrainMeanPP - ActualPP)² ) )
Improvement = (1 - RMSE_model / RMSE_naive) × 100%
```

A positive improvement % means the GLM adds predictive value beyond simply predicting the mean. If the improvement is near zero, the model's rating variables are not informative.

---

#### Test 5 — MAE vs Naive Benchmark

Mean Absolute Error is less sensitive to outliers than RMSE. It answers: on average, by how many euros is the predicted pure premium wrong?

```
MAE_model  = mean( |PredPP - ActualPP| )
MAE_naive  = mean( |TrainMeanPP - ActualPP| )
```

Comparing RMSE and MAE together is useful: if RMSE improvement >> MAE improvement, the model is doing well on extreme risks but not on the average policy (or vice versa).

---

#### Test 6 — Calibration by Driver Age Band

Group test policies by driver age band. Within each band, compare mean predicted pure premium to mean actual pure premium. Compute the ratio.

**What good looks like:** ratios close to 1.0 in every band (green). A ratio of 1.20 in the 18-25 band means the model is over-predicting risk for young drivers by 20% relative to what actually happened — a systematic bias that would cause competitive harm (young drivers are priced out).

**In the app:** ratios are colour-coded green (within ±15%), orange (±15–30%), and red (>30%) directly on the chart.

---

#### Test 7 — Pearson Residual Distribution

For the Poisson frequency model, the Pearson residual for each policy is:

```
r = (Actual ClaimNb - Predicted ClaimNb) / sqrt(Predicted ClaimNb)
```

Under a correctly specified Poisson model these should be approximately Normal(0,1). A histogram that is heavily skewed or fat-tailed compared to the N(0,1) reference curve signals model mis-specification — for example, the Poisson assumption may need to be relaxed to a Negative Binomial.

---

#### Test 8 — Premium Adequacy (Portfolio Loss Ratio)

At portfolio level on the test set:

```
Portfolio Loss Ratio = Total Actual Losses / Total Indicated Premium
Expected (permissible) LR = 1 - Expense Ratio - Profit Margin
```

If the portfolio loss ratio equals the permissible loss ratio, the model is perfectly calibrated at the aggregate level. A loss ratio significantly above permissible means the book is underpriced; below means it is overpriced.

---

### 6.3 Train vs Test Comparison Table

After scoring both splits, `compare_train_test()` builds a side-by-side table:

| Metric | Train (70%) | Test (30%) | Gap |
|---|---|---|---|
| Claim rate | … | … | … |
| Mean actual PP | … | … | … |
| Mean predicted PP | … | … | … |
| Pred / Actual PP ratio | … | … | … |
| Mean indicated premium | … | … | … |
| Portfolio loss ratio | … | … | … |
| RMSE | … | … | … |
| MAE | … | … | … |
| Gini | … | … | … |
| % at minimum premium | … | … | … |

**How to read the gap column:**
- A small gap (< 2%) means the model generalises well to unseen data.
- A large gap on Gini or RMSE (> 10%) is a red flag for overfitting — the model has over-fitted to noise in the training data and its discriminatory power collapses on new business.
- A gap in loss ratio means the train and test populations have meaningfully different risk profiles, which can happen even with stratified splitting on very heterogeneous datasets.

The table is saved to `train_test_comparison.csv` and displayed in the Model Diagnostics tab of the Streamlit app.

---

## 7. Project Structure  

```
edu info/
│
├── pricing_engine.py           # Core analytical pipeline (Steps 1–7) — GLMs, scoring, validation
├── app.py                      # Streamlit web application (5 tabs)
├── model_diagnostics.py        # Standalone statistical tests (overdispersion, monotonicity, CV Gini)
├── generate_report.py          # Generates the Word document report
├── data_investigation.py       # Exploratory data analysis scripts
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── scored_policies.parquet     # Generated on first run — all policies with predictions
├── tariff_relativities.csv     # Generated on first run — GLM coefficient table
├── lift_chart_data.csv         # Generated on first run — decile-level validation stats
├── train_test_comparison.csv   # Generated on first run — train vs test metric table
├── validation_charts.png       # Generated on first run — 6-panel validation chart
└── evaluation_metrics.png      # Generated on first run — 3-metric train vs test evaluation chart
```

`pricing_engine.py` is completely self-contained. `app.py` imports it as a module. You can run either independently.

---

## 8. How to Run

### Prerequisites

Python 3.10 or higher is recommended.

```bash
pip install -r requirements.txt
```

### Run the core pipeline only (no browser)

```bash
python pricing_engine.py
```

This trains both models, prints the tariff manual, and saves outputs to the same folder. Expect ~3–5 minutes on first run as the dataset downloads (~200 MB).

### Run the Streamlit web app

```bash
streamlit run app.py
```

A browser tab will open automatically at `http://localhost:8501`. The first load trains the models; Streamlit caches them so all subsequent interactions (slider moves, tab switches) are instant.

---

## 9. How to Use the Website

The app has **five tabs** accessible from the top navigation bar. The **sidebar** (visible on all tabs) controls the policy profile and commercial parameters — every change re-prices instantly.

### Sidebar controls

| Control | What it does |
|---------|-------------|
| **Driver Age** | Age of the primary driver (18–90) |
| **Vehicle Age** | Age of the vehicle in years (0–20) |
| **Bonus-Malus Score** | Claims history: 50 = perfect, 100 = neutral, 230 = worst |
| **Vehicle Power** | Engine power class (4 = smallest, 15 = most powerful) |
| **Population Density** | Persons per km² — proxy for urban vs rural exposure |
| **Region** | One of 22 French administrative regions (code + full name shown) |
| **Vehicle Brand** | One of 11 brand groups (B1 = Renault/Nissan/Dacia, B10 = BMW/Mercedes, etc.) |
| **Policy Duration** | Fraction of a year the policy is active (1.0 = full year) |
| **Expense Ratio** | Commissions + operating expenses + taxes as % of premium |
| **Target Profit Margin** | Required underwriting profit as % of premium |
| **Minimum Premium** | Absolute floor applied regardless of the GLM output |

The **permissible loss ratio** (= 1 − expense − profit) is shown live; this is the maximum claim spend per unit of premium.

---

### Tab 1 — Underwriter Quote

The primary quoting interface for a single policy.

**Four headline metrics** — predicted claim frequency (isotonic-corrected), predicted average severity, pure premium (risk cost only), and indicated premium (all loadings applied). A warning badge appears if the minimum premium floor was triggered.

**Premium build-up waterfall** — shows exactly how the indicated premium is constructed step by step: pure premium → large loss load → expense load → profit load → indicated premium.

**Risk flag analysis** — plain-language flags for elevated risk characteristics:
- Young driver (age ≤ 25)
- High Bonus-Malus (> 130 — poor claims history)
- High-power vehicle (class ≥ 10)
- Young + high-power interaction term triggered (compounded risk)
- Dense urban area (density > 10,000 persons/km²)

A progress bar shows where this policy sits in the full portfolio premium distribution.

**Isotonic corrections** (expandable) — shows the multiplicative correction factors applied to the driver age band and vehicle age band, with a plain-English explanation of why they exist.

**Compare with typical profiles** (expandable) — side-by-side table of four benchmark profiles (young high-risk, middle-aged clean, experienced low-BM, elderly urban) priced under the same sidebar parameters for easy relativity checking.

---

### Tab 2 — Batch Quote

For pricing multiple policies at once: renewals, new business lists, portfolio repricing exercises.

1. Click **Download template CSV** to get a file with the correct headers and four example rows.
2. Fill in one row per policy with these columns:

   | Column | Format | Example |
   |--------|--------|---------|
   | DrivAge | Integer | 35 |
   | VehAge | Integer | 3 |
   | BonusMalus | Integer 50–230 | 100 |
   | VehPower | Integer 4–15 | 7 |
   | Density | Raw persons/km² (not log-transformed) | 1000 |
   | Region | Code e.g. R82 | R82 |
   | VehBrand | Code e.g. B1 | B1 |
   | Exposure | Decimal 0–1 | 1.0 |

3. Upload the completed CSV. The app prices every row using the current sidebar parameters.
4. A portfolio summary (mean, median, min, max premium; % at floor) and a distribution histogram are shown automatically.
5. Click **Download priced CSV** — the output adds `PredFreq`, `PredSev`, `PurePremium`, `IndicatedPremium`, and `MinPremApplied` columns to your original file.

**Valid region codes:** R11 R21 R22 R23 R24 R25 R26 R31 R41 R42 R43 R52 R53 R54 R72 R73 R74 R82 R83 R91 R93 R94

**Valid brand codes:** B1 B2 B3 B4 B5 B6 B10 B11 B12 B13 B14

---

### Tab 3 — Model Diagnostics

For actuaries and data scientists validating that the model is well-calibrated and not overfitted.

| Panel | What it tells you |
|-------|------------------|
| **Lift chart** | Loss ratio per decile — steeply rising = good risk discrimination |
| **Double-lift** | Predicted vs actual PP per decile — close bars = well-calibrated |
| **Lorenz curve** | Cumulative loss concentration — greater bow = higher Gini |
| **3-metric evaluation** | Gini (ranking), Normalised Poisson deviance (distributional fit), E/A frequency ratio (level calibration) — train vs test with pass/fail verdicts |
| **Full train vs test table** | 11 metrics compared; gap column is green (< 2%) / orange (2–10%) / red (> 10%) |
| **Diagnostic KPIs** | Gini, RMSE, MAE, portfolio loss ratio vs their naive benchmarks |
| **Calibration chart** | Mean predicted vs actual PP per driver age band; labels go red if more than 15% off |

**Interpreting the 3-metric results:**
- Gini gap < 0.03 → no overfitting in risk ranking
- E/A within 2% on the test set → aggregate claim rate transfers to new business
- Deviance gap < 1pp → distributional fit generalises cleanly

---

### Tab 4 — What-If Scenario

For commercial sensitivity analysis without retraining.

- Adjust expense ratio, profit margin, and minimum premium sliders independently
- **Scatter plot** — 2,000 sampled policies shown as current premium (x) vs what-if premium (y), coloured by risk quintile. Points above the diagonal = premiums go up.
- **Histogram** — distribution of % premium change across the sample
- **Driver age sensitivity chart** — predicted frequency (left axis) and indicated premium (right axis) for ages 18–90, all other factors fixed at the sidebar values

Practical use: finance asks what happens if the expense ratio drops from 28% to 22% (digital distribution). Move the slider; immediately see the portfolio-wide impact and which risk segments are most affected.

---

### Tab 5 — Actuarial Concepts / Model Parameters

Two sub-tabs:

**Actuarial Concepts** — expandable study notes for each modelling step (pure premium, large loss capping, Quasi-Poisson GLM, Gamma GLM, isotonic smoothing, credibility blending, lift charts, off-balance correction) with CAS exam references. A mapping table at the bottom links each concept to its exam.

**Model Parameters** — full transparency into what the model learned:

| Section | Content |
|---------|---------|
| Frequency GLM (Quasi-Poisson) | Coefficient, relativity, quasi-SE, p-value, significance stars for every parameter. Green rows = significant at 5% |
| Severity GLM (Gamma) | Top 15 parameters by absolute coefficient |
| Isotonic corrections | Per-band multipliers for DrivAge and VehAge with UP↑ / DOWN↓ / unchanged labels |
| Model statistics | Deviance, pseudo-R², phi, SE inflation, large loss load, Gini/E/A train vs test |

All parameters were fixed from the 70% training set and applied unchanged to test data and new business.

---

## 10. How to Modify the Code

### Changing commercial assumptions (expense ratio, profit, minimum premium)

Open `pricing_engine.py` and edit the constants at the top of the file:

```python
LARGE_LOSS_PERCENTILE  = 0.95   # change to 0.90 for a more aggressive cap
EXPENSE_RATIO          = 0.28   # e.g. 0.22 for a leaner digital insurer
PROFIT_MARGIN          = 0.05   # e.g. 0.08 for a more profitable target
MINIMUM_PREMIUM        = 80.0   # e.g. 150.0 for a higher acquisition cost floor
CREDIBILITY_THRESHOLD  = 30     # lower = more weight on observed data sooner
```

These defaults are also overridable at runtime via the `quote_single_policy()` function arguments, which is how the Streamlit sidebar passes live values without retraining the models.

### Adding a new rating variable

1. In `pricing_engine.py`, add the variable to the `preprocess()` function. Create a banded version if it is continuous (follow the `DrivAge_band` pattern using `pd.cut`).
2. Add the column name to the list returned by `_get_glm_features()`.
3. Add it to the sidebar in `app.py` (slider or selectbox) and pass it through to `quote_single_policy()`.
4. Add it to the single-policy dictionary in `quote_single_policy()` so it is present when `_design_matrix()` encodes it.

### Switching to a different dataset

Replace the `load_data()` function in `pricing_engine.py`. The downstream functions only require that the returned DataFrame contains these columns:

| Column | Description |
|---|---|
| `ClaimNb` | Claim count per policy |
| `TotalLoss` | Total incurred loss per policy |
| `Exposure` | Policy duration in years |
| `DrivAge`, `VehAge`, `BonusMalus`, `VehPower`, `Density` | Numeric risk features |
| `Region`, `VehBrand` | Categorical risk features |

Rename your dataset's columns to match, or update `_get_glm_features()` to point to your own column names.

### Switching the frequency model family

The Poisson assumption (variance = mean) can be relaxed. In `build_freq_model()`, change:

```python
# Current: Poisson
family=sm.families.Poisson(link=sm.families.links.Log())

# Alternative: Negative Binomial (handles overdispersion)
family=sm.families.NegativeBinomial(link=sm.families.links.Log())
```

### Switching the severity model family

In `build_sev_model()`:

```python
# Current: Gamma
family=sm.families.Gamma(link=sm.families.links.Log())

# Alternative: Inverse Gaussian (heavier tail)
family=sm.families.InverseGaussian(link=sm.families.links.Log())
```

### Saving the trained models to disk (avoid retraining on every app restart)

Add the following after the models are built in `pricing_engine.py`:

```python
import pickle

with open("freq_model.pkl", "wb") as f:
    pickle.dump(freq_model, f)
with open("sev_model.pkl", "wb") as f:
    pickle.dump(sev_model, f)
```

Then in `app.py`, replace the training call inside `load_models()` with a pickle load. This reduces cold-start time from ~3 minutes to under 1 second.

### Adding a new tab to the Streamlit app

In `app.py`, find the tab declaration near the top of the main section:

```python
tab1, tab2, tab3, tab4, tab5 = st.tabs([...])
```

Add a sixth element to both the tuple and the list, then write its content inside a new `with tab6:` block below the existing tabs.

---

### Changing the train-test split ratio

In `pricing_engine.py`, change the `test_size` argument in `split_data()`:

```python
train, test = train_test_split(
    df, test_size=0.30,          # change to 0.20 for 80/20, 0.40 for 60/40
    random_state=RANDOM_SEED,
    stratify=df["_had_claim"]
)
```

Also update `RANDOM_SEED = 42` to any other integer to test a different random split.

---

## 11. Maintenance Guide

### Retraining the model

The model trains automatically on first run and is held in Streamlit's `@st.cache_resource` cache for the lifetime of the process. The cache does **not** persist to disk — every fresh `streamlit run` retrains from scratch (~2 minutes).

To force a retrain without restarting the server, open the app's top-right menu (**⋮**) and click **Clear cache**, then refresh the browser.

### Persisting models to disk (skip retraining on restart)

Add this to `pricing_engine.py` after both models are built, then load from pickle in `app.py`'s `load_models()`:

```python
import pickle

# Save (run once after training)
with open("freq_model.pkl", "wb") as f: pickle.dump(freq_model, f)
with open("sev_model.pkl", "wb") as f: pickle.dump(sev_model, f)
with open("iso_corr.pkl",   "wb") as f: pickle.dump(iso_corr, f)

# Load (replace training call in load_models())
with open("freq_model.pkl", "rb") as f: freq_model = pickle.load(f)
```

This cuts cold-start time from ~2 minutes to under 1 second.

### Updating default commercial parameters

Edit the constants at the top of `pricing_engine.py`:

```python
EXPENSE_RATIO   = 0.28  # commissions + opex + taxes
PROFIT_MARGIN   = 0.05  # target underwriting profit
MINIMUM_PREMIUM = 80.0  # absolute premium floor
LARGE_LOSS_PERCENTILE = 0.95  # individual loss cap quantile
CREDIBILITY_THRESHOLD = 30    # Bühlmann k parameter
```

The sidebar sliders initialise to these values on every app start.

### Adding or changing a rating factor

1. Engineer the new column in `preprocess()` in `pricing_engine.py`
2. Add its name to `_get_glm_features()`
3. Add a sidebar control in `app.py` and pass the value through to `quote_single_policy()`
4. Add the column to the single-row DataFrame built inside `quote_single_policy()`
5. Restart the app (clears cache and retrains)

### Adjusting or disabling isotonic corrections

Corrections are configured in `compute_isotonic_corrections()` in `pricing_engine.py`:

```python
config = [
    ("DrivAge_band", ["18-25","26-35","36-45","46-55","56-65","66+"], False),
    ("VehAge_band",  ["0-2","3-5","6-10","11+"],                      False),
]
```

`False` = non-increasing (younger/newer = higher risk). Change to `True` to enforce non-decreasing. Remove a row from `config` to skip that variable entirely.

### Updating the dataset

The app downloads `freMTPL2freq` + `freMTPL2sev` from OpenML on first run. If a new version is published, update the `version=` parameter in `load_data()`. To use a local file instead, replace the `fetch_openml` calls with `pd.read_csv("path/to/file.csv")` and ensure column names match.

### Common errors

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `AttributeError: '_io.StringIO' has no attribute 'reconfigure'` | `pricing_engine` imported after stdout redirect | Ensure `import pricing_engine as pe` is at module level in `app.py`, before `load_models()` |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Corporate proxy | The `ssl._create_default_https_context` override in `pricing_engine.py` should handle this; if not, download freMTPL2 manually and switch to `pd.read_csv` |
| `KeyError` on a Region/VehBrand in batch upload | Code not in training data | Unknown codes default to the reference category (R11 / B1); check the valid code lists in section 9 |
| App slow after restarting | Cache cleared — retraining | Normal; wait ~2 min on first load per session |
| Premiums seem systematically too high or low | Default parameters don't match target market | Adjust expense ratio and profit margin constants in `pricing_engine.py` |

---

## 12. CAS Exam Relevance

| Step | CAS Exam | Topic |
|---|---|---|
| Exposure adjustment, pure premium, expense loading | **Exam 5** | Ratemaking — Indicated Rate Level Changes |
| Large loss capping, IBNR (assumed fully developed here) | **Exam 6** | Reserving — Loss Development |
| Bühlmann credibility blend | **Exam 6 / 7** | Credibility Theory |
| ILF calculation, GLM theory, Gini, off-balance | **Exam 8** | Advanced Ratemaking |
| 70/30 split, RMSE/MAE, calibration, Pearson residuals | **Exam PA** | Model Validation & Diagnostics |
| Full pipeline — feature engineering, validation, deployment | **Exam PA** | Predictive Analytics |
