"""
End-to-End Predictive Insurance Pricing Engine
===============================================
French Motor MTPL dataset | Frequency-Severity GLM | Streamlit-ready pipeline

Actuarial concepts covered per step are documented inline.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS  (adjust to match business targets)
# ─────────────────────────────────────────────
LARGE_LOSS_PERCENTILE  = 0.95   # cap individual losses above this quantile
EXPENSE_RATIO          = 0.28   # industry typical: commissions + opex + taxes
PROFIT_MARGIN          = 0.05   # target underwriting profit margin
MINIMUM_PREMIUM        = 80.0   # absolute floor (USD / EUR) per policy-year
CREDIBILITY_THRESHOLD  = 30     # min. claim count for full GLM credibility
ILF_LIMIT              = 100_000  # basic limits definition for ILF calculation
RANDOM_SEED            = 42


# ─────────────────────────────────────────────
# STEP 0: LOAD & CLEAN
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Actuarial concept: Exposure base selection.
    Exposure (years) normalises raw claim counts so the model learns
    a rate per unit of risk, not per policy record.
    """
    print("[1/7] Fetching French Motor MTPL dataset …")
    raw = fetch_openml("freMTPL2freq", version="active", as_frame=True, parser="auto")
    df  = raw.frame.copy()
    df["ClaimNb"] = pd.to_numeric(df["ClaimNb"], errors="coerce").fillna(0).astype(float)

    sev_raw = fetch_openml("freMTPL2sev", version="active", as_frame=True, parser="auto")
    sev     = sev_raw.frame.copy()
    sev_agg = sev.groupby("IDpol")["ClaimAmount"].sum().reset_index()
    sev_agg.columns = ["IDpol", "TotalLoss"]
    df["IDpol"] = pd.to_numeric(df["IDpol"], errors="coerce").astype(int)
    df = df.merge(sev_agg, on="IDpol", how="left")
    df["TotalLoss"] = df["TotalLoss"].fillna(0.0)

    df["Exposure"]  = df["Exposure"].clip(lower=1/365, upper=1.0)
    df["VehAge"]    = pd.to_numeric(df["VehAge"],     errors="coerce").fillna(0).clip(0, 20)
    df["DrivAge"]   = pd.to_numeric(df["DrivAge"],    errors="coerce").fillna(18).clip(18, 90)
    df["BonusMalus"]= pd.to_numeric(df["BonusMalus"], errors="coerce").fillna(100).clip(50, 230)
    df["Density"]   = np.log1p(pd.to_numeric(df["Density"], errors="coerce").fillna(1))
    df["VehPower"]  = pd.to_numeric(df["VehPower"],   errors="coerce").fillna(6).clip(4, 15)

    print(f"    Loaded {len(df):,} policies | {int(df['ClaimNb'].sum()):,} claims")
    return df


# ─────────────────────────────────────────────
# STEP 1: ACTUARIAL PRE-PROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Actuarial concepts:
      - Pure Premium = Total Loss / Exposure
      - Large Loss Capping: cap extreme losses so random catastrophic events
        do not permanently distort systemic rating factors.
      - Loss Development: we assume losses are fully developed for this dataset.
    """
    print("[2/7] Pre-processing losses …")

    df["PurePremium"] = df["TotalLoss"] / df["Exposure"]

    # ── Large Loss Capping ──────────────────────────────────────────────
    has_loss = df["PurePremium"] > 0
    cap_threshold = df.loc[has_loss, "PurePremium"].quantile(LARGE_LOSS_PERCENTILE)
    df["PurePremium_Capped"] = df["PurePremium"].clip(upper=cap_threshold)
    excess_loss = (df["PurePremium"] - df["PurePremium_Capped"]).clip(lower=0)
    mean_capped = df.loc[df["PurePremium_Capped"] > 0, "PurePremium_Capped"].mean()
    df["LargeLossLoad"] = excess_loss.sum() / (mean_capped * len(df))
    global LARGE_LOSS_LOAD
    LARGE_LOSS_LOAD = df["LargeLossLoad"].iloc[0]
    print(f"    Large loss cap @ {cap_threshold:,.0f} | load factor = {LARGE_LOSS_LOAD:.4f}")

    # ── ILF ─────────────────────────────────────────────────────────────
    df["ILF"] = _compute_ilf_series(df["TotalLoss"], ILF_LIMIT)

    # ── Credibility flag ─────────────────────────────────────────────────
    df["HighCredibility"] = df["ClaimNb"] >= CREDIBILITY_THRESHOLD

    # ── Feature Engineering ─────────────────────────────────────────────
    df["DrivAge_band"] = pd.cut(
        df["DrivAge"], bins=[17, 25, 35, 45, 55, 65, 91],
        labels=["18-25","26-35","36-45","46-55","56-65","66+"]
    )
    df["VehAge_band"] = pd.cut(
        df["VehAge"], bins=[-1, 2, 5, 10, 20],
        labels=["0-2","3-5","6-10","11+"]
    )
    df["BM_band"] = pd.cut(
        df["BonusMalus"], bins=[49, 80, 100, 130, 230],
        labels=["<80","80-100","101-130","130+"]
    )
    df["YoungHighPower"] = (
        (df["DrivAge"] <= 25) & (df["VehPower"] >= 10)
    ).astype(int)

    return df


def _compute_ilf_series(loss_col: pd.Series, limit: float) -> pd.Series:
    basic_lev = loss_col.clip(upper=ILF_LIMIT).mean()
    if basic_lev == 0:
        return pd.Series(np.ones(len(loss_col)), index=loss_col.index)
    return pd.Series(
        loss_col.clip(upper=limit) / (basic_lev + 1e-9),
        index=loss_col.index
    )


# ─────────────────────────────────────────────
# STEP 2: 70 / 30 TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """
    Actuarial concept: Out-of-sample validation.

    We split BEFORE fitting any model so the test set is never seen during
    training. This prevents overfitting and gives an honest picture of how
    the model will perform on new business written next quarter.

    70% train  — used to fit both GLMs.
    30% test   — held out; all validation metrics (lift chart, Gini, RMSE,
                 calibration, etc.) are computed on this set only.

    Stratification: we stratify on whether a claim occurred (ClaimNb > 0)
    so both splits have a similar claim rate. Without this, random chance
    could concentrate all claimants in one split.

    Why not cross-validation?
    Insurance GLMs are typically validated on a chronological hold-out
    (future quarters). Stratified random split is a reasonable proxy when
    time-stamped data is unavailable, as here.
    """
    print("[3/7] Splitting data 70 / 30 (stratified on claim indicator) …")
    df["_had_claim"] = (df["ClaimNb"] > 0).astype(int)
    train, test = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED,
        stratify=df["_had_claim"]
    )
    train = train.drop(columns="_had_claim").copy()
    test  = test.drop(columns="_had_claim").copy()
    print(f"    Train: {len(train):,} policies | Test: {len(test):,} policies")
    print(f"    Train claim rate: {(train['ClaimNb']>0).mean():.4f} | "
          f"Test claim rate: {(test['ClaimNb']>0).mean():.4f}")
    return train, test


# ─────────────────────────────────────────────
# STEP 3: FREQUENCY MODEL  (Poisson GLM)
# ─────────────────────────────────────────────
def build_freq_model(train: pd.DataFrame):
    """
    Actuarial concepts:
      - Quasi-Poisson GLM with log-link: standard Poisson coefficients but standard
        errors scaled by sqrt(phi) to correct for overdispersion (phi=2.66 confirmed
        by model_diagnostics.py).  scale='X2' estimates phi as Pearson X²/df and
        inflates all SEs accordingly — no new parameters, identical point estimates.
      - Offset = log(Exposure): ensures the model estimates a rate per year.
    Fitted on the TRAINING set only.
    """
    print("[4/7] Fitting Frequency GLM (Quasi-Poisson, log-offset) on training set …")
    features   = _get_glm_features(train)
    norm_stats = _compute_norm_stats(train, features)
    X      = _design_matrix(train, features, norm_stats)
    y      = train["ClaimNb"]
    offset = np.log(train["Exposure"])

    model = sm.GLM(
        y, X,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=offset
    ).fit(maxiter=200, tol=1e-8, scale="X2")

    phi = model.scale
    print(f"    Quasi-Poisson GLM  |  phi={phi:.3f}  |  SE multiplier=x{phi**0.5:.3f}")
    print(f"    Deviance={model.deviance:,.1f}  |  Pseudo-R²={1 - model.deviance/model.null_deviance:.4f}")
    return model, features, norm_stats


# ─────────────────────────────────────────────
# STEP 4: SEVERITY MODEL  (Gamma GLM)
# ─────────────────────────────────────────────
def build_sev_model(train: pd.DataFrame, freq_norm_stats: dict):
    """
    Actuarial concepts:
      - Filter to claims-only: severity is a conditional distribution.
      - Gamma GLM with log-link: strictly positive, right-skewed costs.
    Fitted on the TRAINING set only.
    """
    print("[5/7] Fitting Severity GLM (Gamma, claims-only training records) …")
    sev_train = train[train["ClaimNb"] > 0].copy()
    sev_train["AvgCost"] = sev_train["TotalLoss"] / sev_train["ClaimNb"]
    sev_train = sev_train[sev_train["AvgCost"] > 0]

    features  = _get_glm_features(sev_train)
    norm_stats = _compute_norm_stats(sev_train, features)
    X = _design_matrix(sev_train, features, norm_stats)
    y = sev_train["AvgCost"]

    model = sm.GLM(
        y, X,
        family=sm.families.Gamma(link=sm.families.links.Log())
    ).fit(maxiter=200, tol=1e-8)

    print(f"    Gamma GLM   |  Deviance={model.deviance:,.1f}  |  AIC={model.aic:,.1f}")
    print(f"    Pseudo-R²   = {1 - model.deviance/model.null_deviance:.4f}")
    return model, features, norm_stats


# ─────────────────────────────────────────────
# STEP 4b: ISOTONIC CORRECTIONS  (training set)
# ─────────────────────────────────────────────
def compute_isotonic_corrections(
    train: pd.DataFrame, freq_model, freq_features: list, freq_norm: dict
) -> dict:
    """
    Compute per-band multiplicative correction factors that enforce monotone
    relativities for DrivAge and VehAge.

    Why: model_diagnostics.py found 3 inversions in DrivAge and 2 in VehAge.
    Non-monotone adjacent bands are a noise artefact that is hard to defend
    to regulators.  Isotonic regression finds the closest non-increasing step
    function (expected direction: younger/newer = higher risk).

    Computed from the TRAINING SET only; the same factors are applied
    unchanged to both train and test in score_and_price().
    """
    print("    Computing isotonic corrections for DrivAge and VehAge …")

    X      = _design_matrix(train, freq_features, freq_norm)
    X      = X.reindex(columns=freq_model.model.exog_names, fill_value=0)
    pred   = freq_model.predict(X, offset=np.log(train["Exposure"]))

    t = train.copy()
    t["_pred"] = pred.values

    config = [
        ("DrivAge_band", ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"], False),
        ("VehAge_band",  ["0-2",   "3-5",   "6-10",  "11+"],                   False),
    ]

    iso_corr = {}
    for band_col, bands, increasing in config:
        # Exposure-weighted mean predicted frequency per band (in band order)
        agg = (
            t.groupby(band_col, observed=True)
             .apply(lambda g: (g["_pred"] * g["Exposure"]).sum() / (g["Exposure"].sum() + 1e-9))
             .reindex(bands)
        )
        raw_means = agg.values.astype(float)

        ir        = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        iso_means = ir.fit_transform(np.arange(len(bands), dtype=float), raw_means)

        corrections = {
            band: float(iso / (raw + 1e-9))
            for band, raw, iso in zip(bands, raw_means, iso_means)
        }
        iso_corr[band_col] = corrections

        base = raw_means.min() + 1e-9
        print(f"\n    {band_col}:")
        print(f"    {'Band':<10} {'GLM rel':>9} {'ISO rel':>9} {'Factor':>8}")
        for band, raw, iso in zip(bands, raw_means, iso_means):
            print(f"    {band:<10} {raw/base:>9.4f} {iso/base:>9.4f} {iso/(raw+1e-9):>8.4f}")

    print()
    return iso_corr


# ─────────────────────────────────────────────
# STEP 5: SCORE & BUILD TARIFF
# ─────────────────────────────────────────────
def score_and_price(df, freq_model, sev_model,
                    freq_features, sev_features,
                    freq_norm_stats, sev_norm_stats,
                    iso_corr=None):
    """
    Actuarial concept: Pure Premium Decomposition + credibility blend.
    Can be called on either the full dataset (for tariff building) or the
    test set (for out-of-sample validation).
    """
    X_freq = _design_matrix(df, freq_features, freq_norm_stats)
    X_sev  = _design_matrix(df, sev_features,  sev_norm_stats)

    X_freq = X_freq.reindex(columns=freq_model.model.exog_names, fill_value=0)
    X_sev  = X_sev.reindex(columns=sev_model.model.exog_names,   fill_value=0)

    df = df.copy()
    df["PredFreq"] = freq_model.predict(X_freq, offset=np.log(df["Exposure"]))

    # Apply isotonic corrections (computed from training set, applied identically to
    # train and test — no data leakage).
    if iso_corr is not None:
        for band_col, corrections in iso_corr.items():
            factor = df[band_col].astype(str).map(corrections).fillna(1.0)
            df["PredFreq"] = df["PredFreq"] * factor

    df["PredSev"]  = sev_model.predict(X_sev)
    df["PredPP"]   = df["PredFreq"] * df["PredSev"]

    # Bühlmann credibility blend
    df["Z"] = df["ClaimNb"] / (df["ClaimNb"] + CREDIBILITY_THRESHOLD)
    df["PredPP_Cred"] = (
        df["Z"] * (df["TotalLoss"] / (df["Exposure"] + 1e-9))
        + (1 - df["Z"]) * df["PredPP"]
    )

    load_factor  = 1 + LARGE_LOSS_LOAD
    permissible  = 1 - EXPENSE_RATIO - PROFIT_MARGIN
    df["IndicatedPremium"] = (
        df["PredPP_Cred"] * load_factor / permissible
    ).clip(lower=MINIMUM_PREMIUM)

    df["FreqFactor"] = df["PredFreq"] / df["PredFreq"].mean()
    df["SevFactor"]  = df["PredSev"]  / df["PredSev"].mean()
    df["LossRatio"]  = np.where(
        df["IndicatedPremium"] > 0,
        df["TotalLoss"] / df["IndicatedPremium"],
        np.nan
    )
    return df


# ─────────────────────────────────────────────
# STEP 6: VALIDATION SUITE  (test set only)
# ─────────────────────────────────────────────
def validate(test: pd.DataFrame, train: pd.DataFrame, output_dir: str = "."):
    """
    All metrics are computed on the HELD-OUT TEST SET (30% of data).
    The training set is passed in only to compute a naive benchmark.

    Tests performed:
      1. Lift chart & loss ratio by decile
      2. Double-lift (predicted vs actual pure premium per decile)
      3. Lorenz curve + Gini coefficient
      4. Root Mean Squared Error (RMSE) vs naive benchmark
      5. Mean Absolute Error (MAE) vs naive benchmark
      6. Calibration check (mean predicted PP vs mean actual PP by band)
      7. Pearson & Deviance residual diagnostics
      8. Premium adequacy — actual vs expected loss ratio at portfolio level
    """
    print("[6/7] Running validation suite on held-out TEST set …")
    t = test.dropna(subset=["IndicatedPremium", "TotalLoss"]).copy()
    t = t.sort_values("PredPP").reset_index(drop=True)
    t["Decile"] = pd.qcut(t["PredPP"], 10, labels=False, duplicates="drop") + 1

    decile_tbl = t.groupby("Decile").agg(
        ActualLoss  = ("TotalLoss",        "sum"),
        PredPremium = ("IndicatedPremium",  "sum"),
        Policies    = ("IDpol",             "count"),
        AvgPredPP   = ("PredPP",            "mean"),
        AvgActualPP = ("PurePremium",        "mean"),
    )
    decile_tbl["LossRatio"] = decile_tbl["ActualLoss"] / decile_tbl["PredPremium"]

    # ── 1 & 2: Lift + Double-lift ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Validation Suite — TEST SET (30%)", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(decile_tbl.index, decile_tbl["LossRatio"], color="#4C72B0", edgecolor="white")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Break-even LR=1")
    ax.set_title("1. Lift Chart: Loss Ratio by Decile")
    ax.set_xlabel("Predicted Risk Decile"); ax.set_ylabel("Actual Loss Ratio")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax2 = axes[0, 1]
    x = np.arange(len(decile_tbl)); w = 0.35
    ax2.bar(x-w/2, decile_tbl["AvgPredPP"],  w, label="Predicted PP", color="#4C72B0")
    ax2.bar(x+w/2, decile_tbl["AvgActualPP"], w, label="Actual PP",    color="#DD8452")
    ax2.set_xticks(x); ax2.set_xticklabels(decile_tbl.index)
    ax2.set_title("2. Double-Lift: Predicted vs Actual PP")
    ax2.set_xlabel("Decile"); ax2.set_ylabel("Pure Premium")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    # ── 3: Lorenz / Gini ─────────────────────────────────────────────────
    sorted_t  = t.sort_values("IndicatedPremium")
    pct       = np.linspace(0, 1, len(sorted_t))
    cum_loss  = (sorted_t["TotalLoss"].cumsum() /
                 (sorted_t["TotalLoss"].sum() + 1e-9)).values
    gini = float(1 - 2 * np.trapezoid(cum_loss, pct))

    ax3 = axes[0, 2]
    ax3.plot(pct, cum_loss, color="#4C72B0", lw=2, label=f"Model (Gini={gini:.3f})")
    ax3.plot([0,1],[0,1], "k--", lw=1, label="Random (Gini=0)")
    ax3.fill_between(pct, cum_loss, pct, alpha=0.15, color="#4C72B0")
    ax3.set_title("3. Lorenz Curve (test set)")
    ax3.set_xlabel("Cumulative % Policies"); ax3.set_ylabel("Cumulative % Losses")
    ax3.legend(); ax3.grid(alpha=0.3)

    # ── 4 & 5: RMSE and MAE vs naive mean benchmark ───────────────────────
    # Naive benchmark: predict the training mean pure premium for every policy
    naive_pp   = train["PurePremium"].mean()
    actual_pp  = t["PurePremium"].values
    pred_pp    = t["PredPP"].values

    rmse_model = np.sqrt(np.mean((pred_pp  - actual_pp) ** 2))
    rmse_naive = np.sqrt(np.mean((naive_pp - actual_pp) ** 2))
    mae_model  = np.mean(np.abs(pred_pp  - actual_pp))
    mae_naive  = np.mean(np.abs(naive_pp - actual_pp))

    ax4 = axes[1, 0]
    metrics = ["RMSE", "MAE"]
    model_vals = [rmse_model, mae_model]
    naive_vals = [rmse_naive, mae_naive]
    x4 = np.arange(2); w4 = 0.35
    ax4.bar(x4-w4/2, model_vals, w4, label="GLM model",      color="#4C72B0")
    ax4.bar(x4+w4/2, naive_vals, w4, label="Naive benchmark", color="#DD8452")
    ax4.set_xticks(x4); ax4.set_xticklabels(metrics)
    ax4.set_title("4 & 5. RMSE and MAE vs Naive Benchmark")
    ax4.set_ylabel("Error (€)"); ax4.legend(); ax4.grid(axis="y", alpha=0.3)
    for i, (mv, nv) in enumerate(zip(model_vals, naive_vals)):
        ax4.text(i-w4/2, mv*1.02, f"{mv:,.0f}", ha="center", fontsize=8)
        ax4.text(i+w4/2, nv*1.02, f"{nv:,.0f}", ha="center", fontsize=8)

    # ── 6: Calibration check ──────────────────────────────────────────────
    # Is the model systematically over- or under-predicting within bands?
    t["DrivAge_band"] = pd.cut(
        t["DrivAge"], bins=[17,25,35,45,55,65,91],
        labels=["18-25","26-35","36-45","46-55","56-65","66+"]
    )
    calib = t.groupby("DrivAge_band", observed=True).agg(
        MeanPredPP   = ("PredPP",      "mean"),
        MeanActualPP = ("PurePremium", "mean"),
        Count        = ("IDpol",       "count"),
    ).reset_index()
    calib["Ratio"] = calib["MeanPredPP"] / (calib["MeanActualPP"] + 1e-9)

    ax5 = axes[1, 1]
    x5 = np.arange(len(calib)); w5 = 0.35
    ax5.bar(x5-w5/2, calib["MeanPredPP"],   w5, label="Mean Predicted PP", color="#4C72B0")
    ax5.bar(x5+w5/2, calib["MeanActualPP"],  w5, label="Mean Actual PP",    color="#DD8452")
    ax5.set_xticks(x5); ax5.set_xticklabels(calib["DrivAge_band"], rotation=30)
    ax5.set_title("6. Calibration: Predicted vs Actual PP by Driver Age Band")
    ax5.set_ylabel("Mean Pure Premium (€)"); ax5.legend(); ax5.grid(axis="y", alpha=0.3)
    for i, r in enumerate(calib["Ratio"]):
        ax5.text(i, max(calib["MeanPredPP"].iloc[i], calib["MeanActualPP"].iloc[i])*1.03,
                 f"{r:.2f}x", ha="center", fontsize=8,
                 color="green" if 0.85 < r < 1.15 else "red")

    # ── 7: Pearson residuals ──────────────────────────────────────────────
    # For a Poisson GLM: residual = (actual - predicted) / sqrt(predicted)
    # Should be roughly N(0,1) if the model is well-specified
    t_nonzero = t[t["PredFreq"] > 0].copy()
    pearson_resid = (t_nonzero["ClaimNb"] - t_nonzero["PredFreq"]) / np.sqrt(t_nonzero["PredFreq"] + 1e-9)
    pearson_resid_clipped = pearson_resid.clip(-10, 10)

    ax6 = axes[1, 2]
    ax6.hist(pearson_resid_clipped, bins=80, color="#4C72B0", edgecolor="white", density=True)
    xr = np.linspace(-10, 10, 300)
    ax6.plot(xr, stats.norm.pdf(xr), "r--", lw=1.5, label="N(0,1) reference")
    ax6.set_title("7. Pearson Residuals (Frequency model, test set)")
    ax6.set_xlabel("Pearson Residual (clipped ±10)"); ax6.set_ylabel("Density")
    ax6.legend(); ax6.grid(alpha=0.3)

    fig.tight_layout()
    path = f"{output_dir}/validation_charts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 8: Premium adequacy (print only) ─────────────────────────────────
    portfolio_lr = t["TotalLoss"].sum() / t["IndicatedPremium"].sum()
    expected_lr  = 1 - EXPENSE_RATIO - PROFIT_MARGIN

    print(f"    Gini (test set)            = {gini:.4f}")
    print(f"    RMSE — model / naive       = {rmse_model:,.1f} / {rmse_naive:,.1f}  "
          f"({(1-rmse_model/rmse_naive)*100:.1f}% improvement)")
    print(f"    MAE  — model / naive       = {mae_model:,.1f}  / {mae_naive:,.1f}  "
          f"({(1-mae_model/mae_naive)*100:.1f}% improvement)")
    print(f"    Portfolio loss ratio (test)= {portfolio_lr:.4f}  "
          f"(target permissible = {expected_lr:.4f})")
    print(f"    Saved → {path}")

    diagnostics = {
        "gini":         gini,
        "rmse_model":   rmse_model,
        "rmse_naive":   rmse_naive,
        "mae_model":    mae_model,
        "mae_naive":    mae_naive,
        "portfolio_lr": portfolio_lr,
        "expected_lr":  expected_lr,
        "calibration":  calib,
    }
    return decile_tbl, diagnostics


# ─────────────────────────────────────────────
# TRAIN vs TEST COMPARISON
# ─────────────────────────────────────────────
def compare_train_test(train_scored: pd.DataFrame,
                       test_scored:  pd.DataFrame) -> pd.DataFrame:
    """
    Actuarial concept: Out-of-sample generalisation check.

    Computes the same set of metrics on both the training set and the
    held-out test set, then places them side by side. Large gaps between
    train and test indicate overfitting — the model has learnt noise
    specific to the training data that does not generalise to new business.

    Metrics compared:
      - Claim frequency (actual)
      - Mean pure premium (actual vs predicted)
      - Predicted/actual pure premium ratio  (calibration: should be ~1.0)
      - Mean indicated premium
      - Portfolio loss ratio
      - RMSE of pure premium predictions
      - MAE of pure premium predictions
      - Gini coefficient (discrimination power)
      - % policies priced at minimum premium floor
    """
    print("    Computing train vs test comparison metrics …")

    def _gini(df):
        s = df.sort_values("IndicatedPremium")
        pct = np.linspace(0, 1, len(s))
        cum = (s["TotalLoss"].cumsum() / (s["TotalLoss"].sum() + 1e-9)).values
        return float(1 - 2 * np.trapezoid(cum, pct))

    def _metrics(df, label):
        actual_pp = df["PurePremium"].values
        pred_pp   = df["PredPP"].values
        return {
            "Split":                    label,
            "Policies":                 len(df),
            "Claim rate":               round((df["ClaimNb"] > 0).mean(), 4),
            "Mean actual PP (€)":       round(float(actual_pp.mean()), 2),
            "Mean predicted PP (€)":    round(float(pred_pp.mean()),   2),
            "Pred/Actual PP ratio":     round(float(pred_pp.mean() / (actual_pp.mean() + 1e-9)), 4),
            "Mean indicated premium (€)": round(df["IndicatedPremium"].mean(), 2),
            "Portfolio loss ratio":     round(df["TotalLoss"].sum() / df["IndicatedPremium"].sum(), 4),
            "RMSE (€)":                 round(float(np.sqrt(np.mean((pred_pp - actual_pp)**2))), 2),
            "MAE (€)":                  round(float(np.mean(np.abs(pred_pp - actual_pp))),          2),
            "Gini":                     round(_gini(df), 4),
            "% at minimum premium":     round((df["IndicatedPremium"] <= MINIMUM_PREMIUM).mean() * 100, 2),
        }

    rows = [_metrics(train_scored, "Train (70%)"),
            _metrics(test_scored,  "Test  (30%)")]
    tbl = pd.DataFrame(rows).set_index("Split").T
    tbl["Train vs Test gap"] = (
        pd.to_numeric(tbl["Test  (30%)"], errors="coerce") -
        pd.to_numeric(tbl["Train (70%)"], errors="coerce")
    ).round(4)

    print("\n── Train vs Test Comparison ──────────────────────────────────")
    print(tbl.to_string())
    print()
    return tbl


# ─────────────────────────────────────────────
# MODEL EVALUATION  (3-metric train vs test)
# ─────────────────────────────────────────────
def evaluate_model(train_scored: pd.DataFrame, test_scored: pd.DataFrame,
                   out_dir: str = ".") -> dict:
    """
    Three complementary metrics that together give a complete picture of
    whether the model generalises from training to test.

    Metric 1 — Gini coefficient (discrimination / ranking)
      Measures whether the model correctly ranks policies from lowest to
      highest risk.  A good Gini means cheaper premiums go to genuinely
      safer risks.  A model can be Gini-good but still mis-priced in level.
      Train Gini >> Test Gini signals overfitting (learnt noise in the train
      ranks that don't transfer).

    Metric 2 — Normalized Poisson deviance (distributional fit)
      D_model / D_null where D = 2*sum(y*log(y/mu)-(y-mu)).  This is the
      natural scoring rule for Poisson count data — derived directly from the
      log-likelihood.  Scale-free (0% = null model = every policy gets the
      same rate; 100% = perfect).  Robust to portfolio size.  A large train/
      test gap here means the model is overfitting the distributional shape,
      not just the rankings.

    Metric 3 — E/A frequency ratio (calibration / level)
      sum(PredFreq) / sum(ClaimNb).  Should be 1.00 on both splits.
      On the training set a Poisson GLM with an intercept always achieves
      E/A = 1.000 exactly (score equations force sum(mu)=sum(y)).
      On the test set, E/A != 1.00 means the aggregate rate the model
      learned on training data does not transfer — the model is systematically
      over- or under-predicting.  Gini can look perfect while E/A is 1.20;
      they measure different failure modes.
    """
    print("\n── Model Evaluation: Train vs Test (3 Metrics) ────────────────────")

    def _gini(df):
        s   = df.sort_values("PredFreq")
        pct = np.linspace(0, 1, len(s))
        cum = (s["ClaimNb"].cumsum() / (s["ClaimNb"].sum() + 1e-9)).values
        return float(1 - 2 * np.trapezoid(cum, pct)), pct, cum

    def _norm_deviance(df):
        y   = df["ClaimNb"].values.astype(float)
        mu  = df["PredFreq"].values.astype(float)
        exp = df["Exposure"].values.astype(float)
        eps = 1e-10
        dev      = 2 * np.where(y > 0, y * np.log((y + eps) / (mu + eps)) - (y - mu), mu).sum()
        rate_null = y.sum() / (exp.sum() + eps)
        mu_null   = rate_null * exp
        dev_null  = 2 * np.where(y > 0,
                                  y * np.log((y + eps) / (mu_null + eps)) - (y - mu_null),
                                  mu_null).sum()
        return float(1.0 - dev / (dev_null + eps))

    def _ea(df):
        return float(df["PredFreq"].sum() / (df["ClaimNb"].sum() + 1e-9))

    # ── Compute metrics ──────────────────────────────────────────────────────
    tr_gini, tr_pct, tr_cum   = _gini(train_scored)
    te_gini, te_pct, te_cum   = _gini(test_scored)
    tr_nd  = _norm_deviance(train_scored)
    te_nd  = _norm_deviance(test_scored)
    tr_ea  = _ea(train_scored)
    te_ea  = _ea(test_scored)

    # ── Print comparison table ───────────────────────────────────────────────
    hdr = f"  {'Metric':<28} {'Train (70%)':>13} {'Test (30%)':>12} {'Gap':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = [
        ("Gini (discrimination)",    f"{tr_gini:.4f}",  f"{te_gini:.4f}", f"{te_gini-tr_gini:+.4f}"),
        ("Norm. Poisson deviance",   f"{tr_nd*100:.2f}%", f"{te_nd*100:.2f}%", f"{(te_nd-tr_nd)*100:+.2f}pp"),
        ("E/A frequency ratio",      f"{tr_ea:.4f}",   f"{te_ea:.4f}",  f"{te_ea-tr_ea:+.4f}"),
    ]
    for name, trv, tev, gap in rows:
        print(f"  {name:<28} {trv:>13} {tev:>12} {gap:>8}")

    print()
    gini_gap = abs(te_gini - tr_gini)
    ea_gap   = abs(te_ea - 1.0)
    print(f"  Interpretation:")
    print(f"    Gini gap = {gini_gap:.4f}  {'[OK — no overfitting signal]' if gini_gap < 0.03 else '[WARNING — possible overfitting]'}")
    print(f"    E/A test = {te_ea:.4f}  {'[OK — well-calibrated]' if ea_gap < 0.02 else '[WARNING — aggregate level bias]'}")
    print(f"    Deviance gap = {abs(te_nd-tr_nd)*100:.2f}pp  {'[OK]' if abs(te_nd-tr_nd) < 0.01 else '[WARNING — distributional overfitting]'}")

    # ── E/A by predicted-frequency decile (test set) ──────────────────────────
    ts = test_scored.copy()
    ts["_decile"] = pd.qcut(ts["PredFreq"], 10, labels=False, duplicates="drop") + 1
    ea_decile = ts.groupby("_decile").agg(
        Expected=("PredFreq", "sum"),
        Actual  =("ClaimNb",  "sum"),
    ).assign(EA=lambda d: d["Expected"] / (d["Actual"] + 1e-9))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Evaluation: Train vs Test — 3 Key Metrics", fontsize=12, fontweight="bold")

    # Panel 1: Lorenz curves
    ax = axes[0]
    ax.plot(tr_pct, tr_cum, color="#4C72B0", lw=2, label=f"Train (Gini={tr_gini:.3f})")
    ax.plot(te_pct, te_cum, color="#DD8452", lw=2, label=f"Test  (Gini={te_gini:.3f})", linestyle="--")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (Gini=0)")
    ax.fill_between(te_pct, te_cum, te_pct, alpha=0.10, color="#DD8452")
    ax.set_title("Metric 1: Lorenz Curve — Discrimination (Gini)")
    ax.set_xlabel("Cumulative % Policies (sorted by PredFreq)")
    ax.set_ylabel("Cumulative % Claims")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: E/A by decile
    ax2 = axes[1]
    colours = ["#C00000" if abs(r - 1.0) > 0.15 else "#4C72B0" for r in ea_decile["EA"]]
    ax2.bar(ea_decile.index, ea_decile["EA"], color=colours, edgecolor="white")
    ax2.axhline(1.0, color="red", linestyle="--", lw=1.5, label="Perfectly calibrated E/A=1")
    ax2.axhline(1.0 + 0.10, color="orange", linestyle=":", lw=1, label="+/-10% band")
    ax2.axhline(1.0 - 0.10, color="orange", linestyle=":", lw=1)
    ax2.set_title("Metric 3: E/A Ratio by Decile — Calibration (Test set)")
    ax2.set_xlabel("Predicted Frequency Decile"); ax2.set_ylabel("Expected / Actual claims")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Summary bar chart (Gini + norm deviance, train vs test)
    ax3 = axes[2]
    metric_labels = ["Gini", "Norm. Deviance (%)"]
    tr_vals = [tr_gini,     tr_nd * 100]
    te_vals = [te_gini,     te_nd * 100]
    x3 = np.arange(2); w3 = 0.35
    b1 = ax3.bar(x3 - w3/2, tr_vals, w3, label="Train (70%)", color="#4C72B0")
    b2 = ax3.bar(x3 + w3/2, te_vals, w3, label="Test  (30%)", color="#DD8452")
    ax3.set_xticks(x3); ax3.set_xticklabels(metric_labels)
    ax3.set_title("Metric 1 & 2: Train vs Test Summary")
    ax3.legend(); ax3.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h * 1.02,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = f"{out_dir}/evaluation_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")

    return {"tr_gini": tr_gini, "te_gini": te_gini,
            "tr_nd": tr_nd,    "te_nd": te_nd,
            "tr_ea": tr_ea,    "te_ea": te_ea}


# ─────────────────────────────────────────────
# MODEL PARAMETERS  (fixed from training set)
# ─────────────────────────────────────────────
def print_model_parameters(freq_model, sev_model, iso_corr: dict) -> None:
    """
    Display GLM coefficients and isotonic corrections.

    The key actuarial point: ALL parameters below were estimated from the 70%
    training set and applied unchanged to the test set.  No test-set information
    was used in any of these estimates.  This is what 'out-of-sample validation'
    means — if the test metrics are close to train metrics, the training
    parameters generalise to new business.
    """
    print("\n── Frequency GLM Parameters (Quasi-Poisson, trained on 70%) ─────────")
    phi = freq_model.scale
    print(f"    Pearson dispersion phi = {phi:.4f}  |  SE inflation = x{phi**0.5:.4f}")
    print(f"\n    {'Parameter':<35} {'Coeff':>8} {'Relativity':>11} {'Quasi-SE':>10} {'p-value':>9} {'Sig?':>5}")
    print("    " + "-" * 80)
    for name, coef, se, pval in zip(
        freq_model.model.exog_names,
        freq_model.params.values,
        freq_model.bse.values,
        freq_model.pvalues.values,
    ):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    {name:<35} {coef:>8.4f} {np.exp(coef):>11.4f} {se:>10.4f} {pval:>9.4f} {sig:>5}")

    print("\n── Severity GLM Parameters (Gamma, top 10 by |coeff|) ───────────────")
    sev_params = pd.DataFrame({
        "Parameter": sev_model.model.exog_names,
        "Coeff":     sev_model.params.values,
        "Relativity":np.exp(sev_model.params.values),
        "SE":        sev_model.bse.values,
        "p-value":   sev_model.pvalues.values,
    }).sort_values("Coeff", key=abs, ascending=False).head(10)
    print(f"\n    {'Parameter':<35} {'Coeff':>8} {'Relativity':>11} {'SE':>10} {'p-value':>9}")
    print("    " + "-" * 75)
    for _, row in sev_params.iterrows():
        print(f"    {row['Parameter']:<35} {row['Coeff']:>8.4f} {row['Relativity']:>11.4f} "
              f"{row['SE']:>10.4f} {row['p-value']:>9.4f}")

    print("\n── Isotonic Corrections (DrivAge + VehAge, computed from training set) ")
    for band_col, corrections in iso_corr.items():
        print(f"\n    {band_col}:")
        print(f"    {'Band':<10} {'Applied multiplier':>20}")
        for band, factor in corrections.items():
            direction = "UP" if factor > 1.001 else ("DOWN" if factor < 0.999 else "unchanged")
            print(f"    {band:<10} {factor:>20.4f}   ({direction})")

    print("\n    NOTE: All parameters above are fixed from the 70% training set.")
    print("    They are applied unchanged to the test set and to new business.")
    print("    Any closeness between train and test metrics confirms generalisation,")
    print("    not luck — no test information ever entered the fitting process.\n")


# ─────────────────────────────────────────────
# STEP 7: TARIFF MANUAL
# ─────────────────────────────────────────────
def build_tariff_manual(df, freq_model, sev_model, freq_features):
    """
    Actuarial concept: Relativities & Tariff Manual.
    Off-balance correction ensures portfolio premiums sum to predicted losses.
    Fitted on the full dataset (not just train) so the tariff reflects all data.
    """
    print("[7/7] Building tariff manual …")
    coef = pd.DataFrame({
        "Parameter":  freq_model.model.exog_names,
        "Coefficient": freq_model.params.values,
        "StdErr":      freq_model.bse.values,
        "PValue":      freq_model.pvalues.values,
    })
    coef["Relativity"]       = np.exp(coef["Coefficient"])
    coef["Significant_5pct"] = coef["PValue"] < 0.05

    total_pred_loss = df["PredPP"].sum() * df["Exposure"].sum()
    total_ind_prem  = df["IndicatedPremium"].sum()
    off_balance_factor = total_pred_loss / (total_ind_prem + 1e-9)
    return coef, off_balance_factor


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _get_glm_features(df):
    return ["DrivAge_band", "VehAge_band", "BM_band", "VehPower",
            "Density", "Region", "VehBrand", "YoungHighPower"]


def _compute_norm_stats(df: pd.DataFrame, features: list) -> dict:
    """
    Compute mean and std for numeric columns from a given dataset.
    Always call this on the TRAINING set; pass the result to _design_matrix
    for both train and test to prevent data leakage.
    """
    cat_cols = [c for c in features
                if (df[c].dtype.name == "category"
                    or pd.api.types.is_string_dtype(df[c])
                    or pd.api.types.is_object_dtype(df[c]))]
    num_cols = [c for c in features if c not in cat_cols]
    stats_dict = {}
    for col in num_cols:
        s = df[col].astype(float)
        stats_dict[col] = {"mean": float(s.mean()), "std": float(s.std() + 1e-9)}
    return stats_dict


def _design_matrix(df: pd.DataFrame, features: list, norm_stats: dict) -> pd.DataFrame:
    """
    One-hot encode categoricals; standardise numerics using pre-computed stats.
    Passing norm_stats computed from the training set prevents data leakage
    when this function is called on the test set.
    """
    def _is_cat(series):
        """True for category, object, or Arrow-backed string dtypes."""
        return (series.dtype.name == "category"
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(series))

    cat_cols = [c for c in features if _is_cat(df[c])]
    num_cols = [c for c in features if c not in cat_cols]

    parts = []
    for col in cat_cols:
        dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float)
        parts.append(dummies)
    for col in num_cols:
        s = df[col].astype(float)
        mean = norm_stats[col]["mean"]
        std  = norm_stats[col]["std"]
        parts.append(((s - mean) / std).rename(col))

    X = pd.concat(parts, axis=1).fillna(0)
    X.insert(0, "const", 1.0)
    return X


# ─────────────────────────────────────────────
# QUOTE FUNCTION  (used by Streamlit app)
# ─────────────────────────────────────────────
def quote_single_policy(
    freq_model, sev_model, freq_features, sev_features,
    freq_norm_stats, sev_norm_stats,
    driver_age: int, veh_age: int, bonus_malus: int,
    veh_power: int, density_raw: float, region: str,
    veh_brand: str, exposure: float = 1.0,
    expense_ratio: float = EXPENSE_RATIO,
    profit_margin: float = PROFIT_MARGIN,
    min_premium: float = MINIMUM_PREMIUM,
    large_loss_load: float = None,
    iso_corr: dict = None,
) -> dict:
    """
    Price a single hypothetical policy profile.
    norm_stats must come from the training set (passed in from the app cache).
    iso_corr  is the dict returned by compute_isotonic_corrections(); pass it
              so that the same isotonic adjustments applied during training are
              also applied here.
    """
    if large_loss_load is None:
        large_loss_load = LARGE_LOSS_LOAD

    single = pd.DataFrame([{
        "DrivAge": driver_age, "VehAge": veh_age, "BonusMalus": bonus_malus,
        "VehPower": veh_power, "Density": np.log1p(density_raw),
        "Region": region, "VehBrand": veh_brand,
        "YoungHighPower": int(driver_age <= 25 and veh_power >= 10),
        "Exposure": exposure, "ClaimNb": 0, "TotalLoss": 0,
    }])
    single["DrivAge_band"] = pd.cut(single["DrivAge"], bins=[17,25,35,45,55,65,91],
                                     labels=["18-25","26-35","36-45","46-55","56-65","66+"])
    single["VehAge_band"]  = pd.cut(single["VehAge"],  bins=[-1,2,5,10,20],
                                     labels=["0-2","3-5","6-10","11+"])
    single["BM_band"]      = pd.cut(single["BonusMalus"], bins=[49,80,100,130,230],
                                     labels=["<80","80-100","101-130","130+"])

    X_freq = _design_matrix(single, freq_features, freq_norm_stats)
    X_sev  = _design_matrix(single, sev_features,  sev_norm_stats)
    X_freq = X_freq.reindex(columns=freq_model.model.exog_names, fill_value=0)
    X_sev  = X_sev.reindex(columns=sev_model.model.exog_names,   fill_value=0)

    pred_freq = float(freq_model.predict(X_freq, offset=np.log([exposure]))[0])

    if iso_corr is not None:
        da = str(single["DrivAge_band"].iloc[0])
        va = str(single["VehAge_band"].iloc[0])
        pred_freq *= iso_corr.get("DrivAge_band", {}).get(da, 1.0)
        pred_freq *= iso_corr.get("VehAge_band",  {}).get(va, 1.0)

    pred_sev  = float(sev_model.predict(X_sev)[0])
    pred_pp   = pred_freq * pred_sev

    permissible = 1 - expense_ratio - profit_margin
    indicated   = pred_pp * (1 + large_loss_load) / permissible
    indicated   = max(indicated, min_premium)

    return {
        "PredFreq":          round(pred_freq, 6),
        "PredSev":           round(pred_sev, 2),
        "PurePremium":       round(pred_pp, 2),
        "LargeLossLoad":     round(large_loss_load, 4),
        "ExpenseRatio":      round(expense_ratio, 4),
        "ProfitMargin":      round(profit_margin, 4),
        "IndicatedPremium":  round(indicated, 2),
        "MinPremiumApplied": indicated == min_premium,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))

    df    = load_data()
    df    = preprocess(df)
    train, test = split_data(df)

    freq_model, freq_features, freq_norm = build_freq_model(train)
    sev_model,  sev_features,  sev_norm  = build_sev_model(train, freq_norm)

    # Isotonic corrections — computed from training set only
    iso_corr = compute_isotonic_corrections(train, freq_model, freq_features, freq_norm)

    # Score train and test sets with isotonic corrections applied
    train_scored = score_and_price(train, freq_model, sev_model,
                                   freq_features, sev_features,
                                   freq_norm, sev_norm, iso_corr=iso_corr)
    test_scored  = score_and_price(test, freq_model, sev_model,
                                   freq_features, sev_features,
                                   freq_norm, sev_norm, iso_corr=iso_corr)

    decile_tbl, diagnostics = validate(test_scored, train, output_dir=out_dir)
    comparison_tbl = compare_train_test(train_scored, test_scored)
    evaluate_model(train_scored, test_scored, out_dir=out_dir)
    print_model_parameters(freq_model, sev_model, iso_corr)

    # Score full dataset for tariff building (same isotonic corrections)
    df_scored = score_and_price(df, freq_model, sev_model,
                                freq_features, sev_features,
                                freq_norm, sev_norm, iso_corr=iso_corr)
    coef_tbl, ob_factor = build_tariff_manual(df_scored, freq_model, sev_model,
                                              freq_features)

    print("\n── Tariff Manual (Frequency GLM Relativities, significant only) ──")
    print(coef_tbl[coef_tbl["Significant_5pct"]].to_string(index=False))
    print(f"\nOff-balance factor: {ob_factor:.4f}")

    df_scored.to_parquet(f"{out_dir}/scored_policies.parquet", index=False)
    coef_tbl.to_csv(f"{out_dir}/tariff_relativities.csv", index=False)
    decile_tbl.to_csv(f"{out_dir}/lift_chart_data.csv")
    comparison_tbl.to_csv(f"{out_dir}/train_test_comparison.csv")
    print(f"\nAll outputs saved to: {out_dir}")

    sample = quote_single_policy(
        freq_model, sev_model, freq_features, sev_features,
        freq_norm, sev_norm,
        driver_age=23, veh_age=4, bonus_malus=120,
        veh_power=9, density_raw=2000, region="R11", veh_brand="B12",
    )
    print("\n── Sample Quote (young driver, 23yo, BM=120) ──")
    for k, v in sample.items():
        print(f"  {k:<22}: {v}")


if __name__ == "__main__":
    main()
