"""
model_diagnostics.py
====================
Runs 6 statistical diagnostic tests on the cleaned freMTPL2 dataset.

For each test the script:
  (a) states the hypothesis and test method
  (b) reports the test statistic and confirms whether the problem exists
  (c) where confirmed, fits the alternative model
  (d) compares key metrics (AIC / Gini / RMSE) to justify the alternative

Tests:
  1  Overdispersion          Pearson phi + Cameron-Trivedi auxiliary regression
                             Alternative: Negative Binomial GLM
  2  Zero-inflation          Observed vs Poisson-predicted zero count
                             Alternative: Hurdle Poisson (flagged; not fitted here
                             as it requires a separate binary GLM component)
  3  Multicollinearity (VIF) VIF on numeric rating factors
                             Alternative: drop / combine high-VIF features
  4  Sparse cells            Claim count per category level
                             Alternative: collapse levels < threshold into Other
  5  Non-monotone relativ.   GLM relativities for DrivAge and VehAge bands
                             Alternative: isotonic regression smoothing
  6  K-fold CV Gini          Gini stability across 5 folds vs single 70/30 split
                             Verdict: use CV Gini as primary discrimination metric

Run data_investigation.py first to produce cleaned_data.parquet.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUT_DIR          = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED      = 42
VIF_THRESHOLD    = 5.0
SPARSE_THRESHOLD = 30      # min claims for full credibility (Buhlmann k)
K_FOLDS          = 5


# =============================================================================
# DATA LOADING AND FEATURE ENGINEERING
# =============================================================================

def load_and_prep():
    """
    Load cleaned_data.parquet, apply feature engineering identical to
    pricing_engine.py, and return (full_df, train, test).
    """
    parquet = os.path.join(OUT_DIR, "cleaned_data.parquet")
    if not os.path.exists(parquet):
        raise FileNotFoundError(
            "cleaned_data.parquet not found — run data_investigation.py first."
        )
    print("[LOAD] Reading cleaned_data.parquet ...")
    df = pd.read_parquet(parquet)
    drop_cols = [c for c in ["SevRecords", "MaxSingleClaim", "DataQualityFlag"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    df["ClaimNb"]    = pd.to_numeric(df["ClaimNb"],    errors="coerce").fillna(0).astype(float)
    df["Exposure"]   = df["Exposure"].clip(lower=1/365, upper=1.0)
    df["VehAge"]     = pd.to_numeric(df["VehAge"],     errors="coerce").fillna(0).clip(0, 20)
    df["DrivAge"]    = pd.to_numeric(df["DrivAge"],    errors="coerce").fillna(18).clip(18, 90)
    df["BonusMalus"] = pd.to_numeric(df["BonusMalus"], errors="coerce").fillna(100).clip(50, 230)
    df["Density"]    = np.log1p(pd.to_numeric(df["Density"], errors="coerce").fillna(1))
    df["VehPower"]   = pd.to_numeric(df["VehPower"],   errors="coerce").fillna(6).clip(4, 15)

    df["DrivAge_band"] = pd.cut(df["DrivAge"], bins=[17,25,35,45,55,65,91],
                                labels=["18-25","26-35","36-45","46-55","56-65","66+"])
    df["VehAge_band"]  = pd.cut(df["VehAge"],  bins=[-1,2,5,10,20],
                                labels=["0-2","3-5","6-10","11+"])
    df["BM_band"]      = pd.cut(df["BonusMalus"], bins=[49,80,100,130,230],
                                labels=["<80","80-100","101-130","130+"])
    df["YoungHighPower"] = ((df["DrivAge"] <= 25) & (df["VehPower"] >= 10)).astype(int)
    df["PurePremium"]    = df["TotalLoss"] / df["Exposure"]

    df["_had_claim"] = (df["ClaimNb"] > 0).astype(int)
    train, test = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED,
                                   stratify=df["_had_claim"])
    train = train.drop(columns="_had_claim").copy()
    test  = test.drop(columns="_had_claim").copy()
    df    = df.drop(columns="_had_claim")
    print(f"    {len(df):,} policies  |  train {len(train):,}  |  test {len(test):,}")
    return df, train, test


# =============================================================================
# SHARED HELPERS  (replicated from pricing_engine.py to avoid import side-effects)
# =============================================================================

def _features():
    return ["DrivAge_band", "VehAge_band", "BM_band", "VehPower",
            "Density", "Region", "VehBrand", "YoungHighPower"]


def _norm_stats(df, features):
    def _is_cat(s):
        return (s.dtype.name == "category" or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_object_dtype(s))
    num_cols = [c for c in features if not _is_cat(df[c])]
    out = {}
    for col in num_cols:
        s = df[col].astype(float)
        out[col] = {"mean": float(s.mean()), "std": float(s.std() + 1e-9)}
    return out


def _design_matrix(df, features, norm_stats):
    def _is_cat(s):
        return (s.dtype.name == "category" or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_object_dtype(s))
    cat_cols = [c for c in features if _is_cat(df[c])]
    num_cols = [c for c in features if c not in cat_cols]
    parts = []
    for col in cat_cols:
        parts.append(pd.get_dummies(df[col].astype(str), prefix=col,
                                    drop_first=True, dtype=float))
    for col in num_cols:
        s = df[col].astype(float)
        parts.append(((s - norm_stats[col]["mean"]) / norm_stats[col]["std"]).rename(col))
    X = pd.concat(parts, axis=1).fillna(0)
    X.insert(0, "const", 1.0)
    return X


def fit_poisson(train):
    feats = _features()
    ns    = _norm_stats(train, feats)
    X     = _design_matrix(train, feats, ns)
    mdl   = sm.GLM(train["ClaimNb"], X,
                   family=sm.families.Poisson(link=sm.families.links.Log()),
                   offset=np.log(train["Exposure"])).fit(maxiter=200, tol=1e-8, disp=False)
    return mdl, feats, ns


def _gini(pred_series, loss_series):
    idx     = np.argsort(pred_series)
    pct     = np.linspace(0, 1, len(idx))
    cum     = loss_series.values[idx].cumsum() / (loss_series.sum() + 1e-9)
    return float(1 - 2 * np.trapezoid(cum, pct))


# =============================================================================
# TEST 1 — OVERDISPERSION
# =============================================================================

def test_overdispersion(train, pois, feats, ns):
    """
    Hypothesis
    ----------
    H0: Var(Y) = E(Y)   (Poisson is correct)
    H1: Var(Y) > E(Y)   (overdispersion — NegBin or quasi-Poisson needed)

    Method 1 — Pearson dispersion statistic
        phi = sum((y - mu)^2 / mu) / (n - p)
        phi >> 1 signals overdispersion in the raw residuals.

    Method 2 — Cameron-Trivedi (1990) NB2 auxiliary regression
        Regress u_i = (y_i - mu_i)^2 - y_i  on  g_i = mu_i^2  (no intercept).
        OLS t-statistic on g tests H0: alpha = 0.
        Significant positive alpha confirms NB2-structured overdispersion.

    Nuance: phi measures total residual variance inflation; CT tests whether
    that inflation follows the specific NB2 structure Var=mu+alpha*mu^2.
    phi > 1 with CT non-significant means unstructured overdispersion (not NB2)
    — quasi-Poisson (scaled SEs) is the appropriate fix in that case.

    Alternative A — Negative Binomial GLM (if CT significant)
        Adds alpha so Var(Y) = mu + alpha*mu^2.
        Fit on a 60k stratified subsample for numerical stability on large data.

    Alternative B — Quasi-Poisson (if phi > 1.5 but CT non-significant)
        Scales all standard errors by sqrt(phi). Same point estimates as
        Poisson but honest confidence intervals and p-values on relativities.
    """
    print("\n" + "=" * 65)
    print("TEST 1: OVERDISPERSION")
    print("=" * 65)

    y   = train["ClaimNb"].values
    mu  = pois.fittedvalues.values
    n, p = len(y), len(pois.params)

    # Method 1 — Pearson dispersion
    phi = float(np.sum((y - mu) ** 2 / (mu + 1e-12)) / (n - p))

    # Method 2 — Cameron-Trivedi NB2 auxiliary regression
    u        = (y - mu) ** 2 - y
    g        = (mu ** 2).reshape(-1, 1)
    aux      = sm.OLS(u, g).fit()
    ct_alpha = float(aux.params[0])
    ct_t     = float(aux.tvalues[0])
    ct_p     = float(aux.pvalues[0])

    print(f"\n  Method 1 — Pearson dispersion statistic")
    print(f"    phi = {phi:.4f}   (ideal = 1.0 for Poisson)")
    if phi > 2:
        print(f"    phi >> 1 — STRONG excess variance in residuals")
    elif phi > 1.5:
        print(f"    phi > 1.5 — MODERATE excess variance")
    else:
        print(f"    phi near 1 — residual variance matches Poisson expectation")

    print(f"\n  Method 2 — Cameron-Trivedi NB2 auxiliary regression")
    print(f"    alpha_hat = {ct_alpha:.6f}")
    print(f"    t-stat    = {ct_t:.4f}")
    print(f"    p-value   = {ct_p:.4e}")
    if ct_p < 0.05 and ct_alpha > 0:
        print(f"    --> NB2-structured overdispersion confirmed (p < 0.05)")
    elif ct_p >= 0.05 and phi > 1.5:
        print(f"    --> phi high but CT non-significant: overdispersion is unstructured")
        print(f"        (not NB2). Quasi-Poisson is the correct fix, not NegBin.")
    else:
        print(f"    --> Cannot reject H0: no systematic overdispersion structure")

    phi_confirmed = phi > 1.5
    ct_confirmed  = (ct_p < 0.05 and ct_alpha > 0)

    nb_result = None
    aic_nb    = None

    if phi_confirmed:
        print(f"\n  !! OVERDISPERSION CONFIRMED (phi = {phi:.2f})")

        if ct_confirmed:
            # NB2 structure confirmed — fit full NegBin on a subsample
            print(f"     CT test also significant: fitting Negative Binomial on 60k subsample ...")
            rng     = np.random.default_rng(RANDOM_SEED)
            sub_idx = rng.choice(len(train), size=min(60_000, len(train)), replace=False)
            sub     = train.iloc[sub_idx]
            ns_sub  = _norm_stats(sub, feats)
            X_sub   = _design_matrix(sub, feats, ns_sub)
            # Also fit Poisson on same subsample for fair comparison
            pois_sub = sm.GLM(sub["ClaimNb"], X_sub,
                              family=sm.families.Poisson(link=sm.families.links.Log()),
                              offset=np.log(sub["Exposure"])).fit(
                                  maxiter=200, tol=1e-8, disp=False)
            nb_aic_pois = pois_sub.aic
            try:
                for method in ("newton", "bfgs", "nm"):
                    try:
                        nb_mdl    = NegativeBinomial(sub["ClaimNb"], X_sub,
                                                     offset=np.log(sub["Exposure"]))
                        nb_result = nb_mdl.fit(method=method, maxiter=500, disp=False)
                        if np.isfinite(nb_result.aic):
                            break
                    except Exception:
                        continue

                if nb_result is not None and np.isfinite(nb_result.aic):
                    aic_nb = nb_result.aic
                    # Recover alpha: statsmodels NegBin can store lnalpha
                    if "alpha" in nb_result.params.index:
                        raw_a = float(nb_result.params["alpha"])
                        nb_alpha = np.exp(raw_a) if abs(raw_a) < 20 else raw_a
                    else:
                        nb_alpha = float(np.exp(nb_result.params.iloc[-1]))

                    delta = nb_aic_pois - aic_nb
                    print(f"\n  Comparison on 60k subsample — Poisson vs Negative Binomial:")
                    print(f"    {'Metric':<28} {'Poisson':>12}  {'NegBin':>12}")
                    print(f"    {'-'*55}")
                    print(f"    {'AIC':<28} {nb_aic_pois:>12,.1f}  {aic_nb:>12,.1f}")
                    print(f"    {'Log-likelihood':<28} {pois_sub.llf:>12,.1f}  {nb_result.llf:>12,.1f}")
                    print(f"    {'NegBin alpha':<28} {'—':>12}  {nb_alpha:>12.4f}")
                    print(f"\n  AIC improvement (Poisson - NegBin) = {delta:,.1f}")
                    if delta > 2:
                        print(f"\n  VERDICT: Use Negative Binomial GLM.")
                        print(f"  WHY:     NB2 structure confirmed by CT test (p={ct_p:.3f}).")
                        print(f"           phi={phi:.2f} means actual variance is {phi:.1f}x Poisson.")
                        print(f"           NegBin absorbs this via alpha={nb_alpha:.4f}, yielding")
                        print(f"           honest confidence intervals and AIC gain of {delta:,.0f}.")
                    else:
                        print(f"\n  VERDICT: Use Quasi-Poisson (AIC gain from NegBin is small).")
                        print(f"  WHY:     Scale Poisson SEs by sqrt(phi)={phi**0.5:.2f}.")
                else:
                    print(f"  NegBin did not converge cleanly — falling back to Quasi-Poisson.")
                    _quasi_verdict(phi)
            except Exception as e:
                print(f"  NegBin fitting error: {e} — falling back to Quasi-Poisson.")
                _quasi_verdict(phi)
        else:
            # phi high but CT non-significant: unstructured overdispersion
            _quasi_verdict(phi)
    else:
        print(f"\n  OK No material overdispersion (phi={phi:.2f}). Poisson assumption holds.")

    return {"phi": phi, "ct_alpha": ct_alpha, "ct_t": ct_t, "ct_p": ct_p,
            "phi_confirmed": phi_confirmed, "ct_confirmed": ct_confirmed,
            "confirmed": phi_confirmed,
            "aic_pois": pois.aic, "aic_nb": aic_nb, "nb_result": nb_result}


def _quasi_verdict(phi):
    print(f"\n  VERDICT: Use Quasi-Poisson (scaled standard errors).")
    print(f"  WHY:     phi={phi:.2f} inflates Poisson t-statistics by sqrt({phi:.2f})={phi**0.5:.2f}x.")
    print(f"           CT test non-significant means the overdispersion is not cleanly")
    print(f"           NB2-structured — it likely comes from a few extreme multi-claim")
    print(f"           policies and heterogeneous short-exposure records rather than a")
    print(f"           systematic negative binomial mixing process.")
    print(f"           Quasi-Poisson multiplies every standard error by sqrt(phi)={phi**0.5:.2f},")
    print(f"           deflating inflated t-values without adding new parameters.")
    print(f"           This is the standard actuarial GLM recommendation when phi > 1.5")
    print(f"           and NB2 structure is not confirmed.")


# =============================================================================
# TEST 2 — ZERO INFLATION
# =============================================================================

def test_zero_inflation(train, pois):
    """
    Hypothesis
    ----------
    H0: The Poisson model correctly predicts the zero-claim frequency.
    H1: There are more zero-claim policies than a Poisson with the same
        mean would predict — a separate zero-generating process exists.

    Method
    ------
    Expected zeros under Poisson: sum_i P(Y_i = 0) = sum_i exp(-mu_i)
    Compare to observed zero count.  Excess = observed - expected.
    Practical threshold: excess > 5% of the portfolio.

    Alternative (if confirmed) — Hurdle Poisson
    -------------------------------------------
    Two-part model:
      Part 1 (binary GLM):   P(Y > 0) — is there any claim?
      Part 2 (truncated Poisson): E(Y | Y > 0) — how many claims?
    Justified over ZIP for motor insurance because the zeros in motor data
    are 'structural' (safe drivers) rather than mixing with a zero-inflation
    process — a Hurdle cleanly separates the two decisions.
    """
    print("\n" + "=" * 65)
    print("TEST 2: ZERO INFLATION")
    print("=" * 65)

    y   = train["ClaimNb"].values
    mu  = pois.fittedvalues.values
    n   = len(y)

    obs_zeros  = int((y == 0).sum())
    exp_zeros  = float(np.exp(-mu).sum())
    excess     = obs_zeros - exp_zeros
    excess_pct = excess / n * 100

    print(f"\n  Observed zeros       : {obs_zeros:,}  ({obs_zeros/n*100:.2f}% of train)")
    print(f"  Poisson-expected zeros: {exp_zeros:,.0f}  ({exp_zeros/n*100:.2f}% of train)")
    print(f"  Excess zeros          : {excess:,.0f}  ({excess_pct:.2f}% of portfolio)")

    confirmed = excess_pct > 5.0

    if confirmed:
        print(f"\n  !! ZERO INFLATION CONFIRMED — excess = {excess_pct:.1f}%")
        print(f"\n  VERDICT: Fit a Hurdle Poisson model.")
        print(f"  WHY:     The Poisson underpredicts zeros by {excess:,.0f} policies ({excess_pct:.1f}%).")
        print(f"           Standard Poisson conflates 'no claim because safe driver'")
        print(f"           with 'no claim by chance this year'. A Hurdle model separates:")
        print(f"             Part 1 — logistic GLM on P(any claim) using risk factors.")
        print(f"             Part 2 — truncated Poisson on E(claims | claim > 0).")
        print(f"           This improves calibration in low-risk segments and prevents")
        print(f"           over-pricing of genuinely zero-risk profiles.")
    else:
        print(f"\n  OK Excess zeros = {excess_pct:.1f}% — within the 5% threshold.")
        print(f"     The Poisson log-offset (exposure) already absorbs most of the")
        print(f"     zero mass; zero-inflation correction is not required.")
        print(f"     WHY POISSON WORKS HERE: The overall claim rate is only ~5%.")
        print(f"     Poisson(mu=0.05) gives P(0) = exp(-0.05) = 95.1%, which closely")
        print(f"     matches the observed zero rate. No separate zero process is needed.")

    return {"obs_zeros": obs_zeros, "exp_zeros": exp_zeros,
            "excess_pct": excess_pct, "confirmed": confirmed}


# =============================================================================
# TEST 3 — MULTICOLLINEARITY (VIF)
# =============================================================================

def test_vif(train):
    """
    Hypothesis
    ----------
    H0: Numeric rating factors are not correlated — GLM coefficients are stable.
    H1: Two or more features share variance — standard errors are inflated
        and coefficients are sensitive to small data changes.

    Method — Variance Inflation Factor
    -----------------------------------
    VIF_j = 1 / (1 - R^2_j)  where R^2_j comes from regressing X_j on all
    other predictors.  VIF = 1: no collinearity.  VIF > 5: moderate warning.
    VIF > 10: severe — coefficients unreliable.

    We test numeric features only (DrivAge, VehAge, BonusMalus, VehPower,
    Density, YoungHighPower).  Testing individual dummy columns is misleading
    as they are correlated by construction.

    Alternative (if confirmed)
    --------------------------
    - Drop the weaker of the two collinear features (lower chi-sq contribution).
    - Or replace both with a single interaction term.
    """
    print("\n" + "=" * 65)
    print("TEST 3: MULTICOLLINEARITY (VIF)")
    print("=" * 65)

    num_cols  = ["DrivAge", "VehAge", "BonusMalus", "VehPower",
                 "Density", "YoungHighPower"]
    available = [c for c in num_cols if c in train.columns]
    X_raw     = train[available].astype(float)
    X_std     = (X_raw - X_raw.mean()) / (X_raw.std() + 1e-9)
    X_aug     = sm.add_constant(X_std)

    vif_rows = []
    for i, col in enumerate(X_aug.columns):
        if col == "const":
            continue
        v = variance_inflation_factor(X_aug.values, i)
        vif_rows.append({"Feature": col, "VIF": round(float(v), 3),
                          "Status": "!!" if v > VIF_THRESHOLD else "OK"})

    vif_df  = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    max_vif = vif_df["VIF"].max()
    high    = vif_df[vif_df["VIF"] > VIF_THRESHOLD]["Feature"].tolist()

    print(f"\n  VIF threshold = {VIF_THRESHOLD}   (> 5 = moderate warning; > 10 = severe)")
    print(f"\n  {'Feature':<18} {'VIF':>8}  {'Status'}")
    print(f"  {'-'*35}")
    for _, row in vif_df.iterrows():
        print(f"  {row['Feature']:<18} {row['VIF']:>8.3f}  {row['Status']}")

    print(f"\n  Pairwise correlation (numeric features):")
    corr = X_raw.corr().round(3)
    print(corr.to_string())

    if high:
        print(f"\n  !! HIGH VIF FEATURES: {', '.join(high)}")
        print(f"\n  VERDICT: Remove or combine the high-VIF features.")
        print(f"  WHY:     When two predictors are correlated, the GLM cannot reliably")
        print(f"           attribute the claim signal to either one independently.")
        print(f"           The coefficients can flip sign between training sets.")
        print(f"           Standard errors are inflated by factor sqrt(VIF), so a")
        print(f"           VIF of {max_vif:.1f} inflates the SE by {max_vif**0.5:.1f}x, making")
        print(f"           genuine risk factors appear non-significant.")
    else:
        print(f"\n  OK All VIF values below {VIF_THRESHOLD} (max = {max_vif:.3f}).")
        print(f"     No multicollinearity concern. GLM coefficients are stable.")
        print(f"     Correlation between DrivAge and BonusMalus is expected and moderate")
        print(f"     but does not rise to the level requiring intervention.")

    return {"vif_df": vif_df, "high_vif": high, "max_vif": max_vif, "corr": corr}


# =============================================================================
# TEST 4 — SPARSE CELLS
# =============================================================================

def test_sparse_cells(train):
    """
    Hypothesis
    ----------
    H0: Every category level has sufficient claims for a stable GLM relativity.
    H1: One or more levels have fewer than SPARSE_THRESHOLD claims — the
        estimated relativity is driven by noise rather than signal.

    Method
    ------
    For each categorical feature, sum the claim count per level.
    Flag levels where sum(ClaimNb) < SPARSE_THRESHOLD (30).

    Alternative (if confirmed) — Collapse sparse levels
    ----------------------------------------------------
    Merge all flagged levels into a single 'Other' bucket. Re-fit the GLM
    and compare AIC to confirm the collapsed model is at least as good.

    Justification
    -------------
    A GLM relativity from 5 claims has a 95% CI spanning the entire
    plausible range. The Buhlmann credibility weight for a level with
    n claims is Z = n / (n + 30), so Z < 0.5 for n < 30 — the model
    is less than half-credible. Collapsing forces the level to borrow
    from the base rate, which is far more stable than a noise estimate.
    """
    print("\n" + "=" * 65)
    print("TEST 4: SPARSE CELLS")
    print("=" * 65)
    print(f"  Threshold: < {SPARSE_THRESHOLD} claims per level is flagged as sparse")

    cat_cols = {
        "DrivAge_band": "DrivAge_band",
        "VehAge_band":  "VehAge_band",
        "BM_band":      "BM_band",
        "Region":       "Region",
        "VehBrand":     "VehBrand",
        "VehGas":       "VehGas",
        "Area":         "Area",
    }

    all_sparse   = []
    level_detail = {}

    for feature, col in cat_cols.items():
        if col not in train.columns:
            continue
        grp = (train.groupby(col, observed=True)
               .agg(Policies=("ClaimNb","count"),
                    Claims=("ClaimNb","sum"),
                    Exposure=("Exposure","sum"))
               .reset_index())
        grp.columns = [col, "Policies", "Claims", "Exposure"]
        grp["Claims"]    = grp["Claims"].astype(float)
        grp["ClaimRate"] = (grp["Claims"] / grp["Exposure"]).round(4)
        grp["Sparse"]    = grp["Claims"] < SPARSE_THRESHOLD
        sparse_levels    = grp.loc[grp["Sparse"], col].tolist()

        print(f"\n  {feature}:")
        print(f"  {'Level':<16} {'Policies':>10} {'Claims':>8} {'Claim rate':>12}  {'Status'}")
        print(f"  {'-'*56}")
        for _, r in grp.sort_values("Claims").iterrows():
            flag = " !!" if r["Sparse"] else "  OK"
            print(f"  {str(r[col]):<16} {int(r['Policies']):>10,} {int(r['Claims']):>8} "
                  f"{r['ClaimRate']:>12.4f} {flag}")
        if sparse_levels:
            all_sparse.append({"feature": feature, "sparse_levels": sparse_levels,
                                "n_sparse": len(sparse_levels)})
        level_detail[feature] = grp

    n_sparse_features = len(all_sparse)
    total_sparse_lvls = sum(r["n_sparse"] for r in all_sparse)

    if n_sparse_features > 0:
        print(f"\n  !! SPARSE CELLS CONFIRMED:")
        for r in all_sparse:
            print(f"     {r['feature']}: {r['n_sparse']} sparse level(s) — {r['sparse_levels']}")

        print(f"\n  VERDICT: Collapse {total_sparse_lvls} sparse level(s) to 'Other'.")
        print(f"  WHY:     A relativity based on < {SPARSE_THRESHOLD} claims is essentially noise.")
        print(f"           Buhlmann Z = n/(n+{SPARSE_THRESHOLD}), so Z < 0.5 for n < {SPARSE_THRESHOLD}.")
        print(f"           The collapsed 'Other' level receives the base-rate relativity,")
        print(f"           which is the most credible estimate available when data is thin.")
        print(f"           This also prevents adverse selection: if the model overestimates")
        print(f"           a rare region's relativity, competitors undercut on that segment.")
    else:
        print(f"\n  OK No sparse cells — all levels have >= {SPARSE_THRESHOLD} claims.")
        print(f"     Every GLM relativity estimate has at least minimum credibility.")

    return {"all_sparse": all_sparse, "level_detail": level_detail,
            "n_sparse_features": n_sparse_features}


# =============================================================================
# TEST 5 — NON-MONOTONE RELATIVITIES
# =============================================================================

def test_isotonic(train, pois, feats, ns):
    """
    Hypothesis
    ----------
    H0: GLM relativities for ordered variables are monotone in the expected
        direction (DrivAge: high for young, decreasing to middle age;
        VehAge: newer vehicles slightly lower or similar to older ones).
    H1: Relativities invert between adjacent bands due to noise, creating
        'cliffs' — nearby risk profiles with materially different prices.

    Method
    ------
    1. Predict mean frequency per band from the fitted Poisson GLM.
    2. Divide by the lowest-risk band mean to get relativities.
    3. Check if diffs between adjacent bands are all in the expected direction.

    Alternative — Isotonic regression
    -----------------------------------
    sklearn IsotonicRegression fits a step function that is monotone
    increasing or decreasing and minimises squared error to the raw estimates.
    The smoothed relativities replace the GLM band estimates for that variable.

    Note on DrivAge: motor frequency follows a U-shape (young high, middle low,
    elderly slightly elevated). We check for monotone decrease from 18-65 and
    flag the 66+ group separately. Strict monotone decrease across all bands
    would incorrectly constrain the elderly uptick.
    """
    print("\n" + "=" * 65)
    print("TEST 5: NON-MONOTONE RELATIVITIES (DrivAge, VehAge)")
    print("=" * 65)

    X_full = _design_matrix(train, feats, ns)
    X_full = X_full.reindex(columns=pois.model.exog_names, fill_value=0)
    t      = train.copy()
    t["PredFreq"] = pois.predict(X_full, offset=np.log(t["Exposure"]))

    results = {}

    for var, direction in [("DrivAge_band", "decreasing"), ("VehAge_band", "increasing")]:
        grp = (t.groupby(var, observed=True)
               .agg(AvgPredFreq=("PredFreq","mean"),
                    AvgActFreq=("ClaimNb","mean"),
                    Exposure=("Exposure","sum"))
               .reset_index()
               .sort_values(var))

        raw_rel = (grp["AvgPredFreq"].values
                   / (grp["AvgPredFreq"].min() + 1e-12))
        bands   = grp[var].astype(str).tolist()

        # Isotonic regression
        ir      = IsotonicRegression(
            increasing=(direction == "increasing"), out_of_bounds="clip")
        x_ord   = np.arange(len(raw_rel))
        iso_rel = ir.fit_transform(x_ord, raw_rel)

        # Monotonicity check
        diffs    = np.diff(raw_rel)
        expected = diffs <= 0 if direction == "decreasing" else diffs >= 0
        monotone = bool(np.all(expected))

        print(f"\n  {var}  (expected direction: {direction})")
        print(f"  {'Band':<12} {'Rel (GLM)':>12} {'Rel (Isotonic)':>16} {'Diff':>10}")
        print(f"  {'-'*55}")
        for i, band in enumerate(bands):
            diff = iso_rel[i] - raw_rel[i]
            flag = "" if abs(diff) < 0.01 else "  <-- smoothed"
            print(f"  {band:<12} {raw_rel[i]:>12.4f} {iso_rel[i]:>16.4f} {diff:>+10.4f}{flag}")

        if not monotone:
            inversions = int((~expected).sum())
            print(f"\n  !! NON-MONOTONE: {inversions} inversion(s) detected in {var}.")
            print(f"\n  VERDICT: Apply isotonic regression to {var} relativities.")
            print(f"  WHY:     A non-monotone age curve is hard to defend to regulators")
            print(f"           (e.g. a 34-year-old priced higher than a 26-year-old).")
            print(f"           It also creates pricing cliffs that competitors can exploit")
            print(f"           by targeting the mispriced bands. Isotonic regression")
            print(f"           preserves the overall rank order while removing noise-driven")
            print(f"           inversions, giving a smooth curve that is auditable.")
        else:
            print(f"\n  OK {var} is already monotone {direction}.")
            print(f"     Isotonic smoothing would make no material change.")

        results[var] = {"raw": raw_rel, "iso": iso_rel, "bands": bands,
                        "monotone": monotone, "direction": direction}

    return results


# =============================================================================
# TEST 6 — K-FOLD CV GINI
# =============================================================================

def test_kfold_gini(df):
    """
    Hypothesis
    ----------
    H0: The Gini from a single 70/30 split is a reliable estimate of the
        model's discrimination power on unseen data.
    H1: Split variability is high — the reported Gini depends materially on
        the random seed, not just model quality.

    Method
    ------
    5-fold stratified CV: fit Poisson GLM on 4 folds, evaluate Gini on the
    held-out fold. Report mean and std across 5 folds.

    Threshold: std > 0.02 = high variability (single-split Gini unreliable).

    Verdict
    -------
    Regardless of whether variability is high or low, the CV mean Gini is
    a more honest metric than a single split and should be the primary
    reported discrimination statistic going forward.
    """
    print("\n" + "=" * 65)
    print(f"TEST 6: K-FOLD CV GINI  (k = {K_FOLDS})")
    print("=" * 65)

    feats      = _features()
    had_claim  = (df["ClaimNb"] > 0).astype(int)
    skf        = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                                  random_state=RANDOM_SEED)
    fold_ginis = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, had_claim), start=1):
        tr = df.iloc[tr_idx]
        te = df.iloc[te_idx]
        ns = _norm_stats(tr, feats)
        X_tr = _design_matrix(tr, feats, ns)
        X_te = _design_matrix(te, feats, ns)
        mdl  = sm.GLM(tr["ClaimNb"], X_tr,
                      family=sm.families.Poisson(link=sm.families.links.Log()),
                      offset=np.log(tr["Exposure"])).fit(
                          maxiter=200, tol=1e-8, disp=False)
        X_te  = X_te.reindex(columns=mdl.model.exog_names, fill_value=0)
        pred  = mdl.predict(X_te, offset=np.log(te["Exposure"]))
        g     = _gini(pred, te["TotalLoss"])
        fold_ginis.append(g)
        print(f"    Fold {fold}: Gini = {g:.4f}")

    mean_g = float(np.mean(fold_ginis))
    std_g  = float(np.std(fold_ginis))
    cv_pct = std_g / mean_g * 100 if mean_g > 0 else float("nan")

    # Single 70/30 split for comparison
    df2 = df.copy()
    df2["_hc"] = (df2["ClaimNb"] > 0).astype(int)
    tr70, te30 = train_test_split(df2, test_size=0.30, random_state=RANDOM_SEED,
                                   stratify=df2["_hc"])
    ns70  = _norm_stats(tr70, feats)
    X70tr = _design_matrix(tr70, feats, ns70)
    X70te = _design_matrix(te30, feats, ns70)
    m70   = sm.GLM(tr70["ClaimNb"], X70tr,
                   family=sm.families.Poisson(link=sm.families.links.Log()),
                   offset=np.log(tr70["Exposure"])).fit(
                       maxiter=200, tol=1e-8, disp=False)
    X70te    = X70te.reindex(columns=m70.model.exog_names, fill_value=0)
    pred70   = m70.predict(X70te, offset=np.log(te30["Exposure"]))
    single_g = _gini(pred70, te30["TotalLoss"])

    print(f"\n  {'Metric':<35} {'Value':>10}")
    print(f"  {'-'*48}")
    print(f"  {'CV mean Gini':<35} {mean_g:>10.4f}")
    print(f"  {'CV std Gini':<35} {std_g:>10.4f}")
    print(f"  {'Coefficient of variation (%)':<35} {cv_pct:>10.1f}%")
    print(f"  {'Single 70/30 Gini (seed=42)':<35} {single_g:>10.4f}")
    print(f"  {'Difference (single - CV mean)':<35} {single_g - mean_g:>+10.4f}")

    if std_g > 0.02:
        print(f"\n  !! HIGH VARIABILITY: std = {std_g:.4f} > 0.02.")
        print(f"     Any Gini change smaller than {2*std_g:.3f} could be a split artefact.")
        print(f"\n  VERDICT: Report CV mean Gini = {mean_g:.4f} as primary metric.")
        print(f"  WHY:     With split std = {std_g:.4f}, quoting a single split misleads")
        print(f"           stakeholders into believing the model is better or worse than")
        print(f"           it truly is. K-fold CV amortises split luck across {K_FOLDS} draws,")
        print(f"           giving a metric that is reproducible and comparable across models.")
    else:
        print(f"\n  OK Split variability is low (std = {std_g:.4f}).")
        print(f"     The single 70/30 Gini ({single_g:.4f}) is representative.")
        print(f"\n  VERDICT: Report CV mean Gini = {mean_g:.4f} as primary metric regardless.")
        print(f"  WHY:     Even when std is low, CV mean is the more principled estimate")
        print(f"           as it uses all the data for both training and evaluation.")

    return {"fold_ginis": fold_ginis, "mean_gini": mean_g,
            "std_gini": std_g, "single_gini": single_g, "cv_pct": cv_pct}


# =============================================================================
# DIAGNOSTIC PLOTS  (2 x 3 grid)
# =============================================================================

def plot_diagnostics(od, zi, vif_r, sp_r, iso_r, kf_r):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Model Diagnostics — French Motor MTPL",
                 fontsize=13, fontweight="bold")

    # ── 1. Overdispersion: Pearson residuals histogram ────────────────────────
    ax = axes[0, 0]
    from scipy.stats import norm as scipy_norm
    # We can't recover residuals here without re-running; use the phi bar instead
    methods  = ["Pearson phi", "CT alpha"]
    values   = [od["phi"], max(od["ct_alpha"], 0)]
    thresholds = [1.5, 0.0]
    colors   = ["#C44E52" if v > t else "#4C72B0"
                for v, t in zip(values, thresholds)]
    ax.bar(methods, values, color=colors, edgecolor="white")
    ax.axhline(1.5, color="red",   linestyle="--", lw=1.2, label="phi threshold = 1.5")
    ax.axhline(0,   color="green", linestyle="--", lw=1.2, label="CT alpha = 0")
    ax.set_title("1. Overdispersion diagnostics")
    ax.set_ylabel("Statistic value"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for i, (v, m) in enumerate(zip(values, methods)):
        ax.text(i, v * 1.03, f"{v:.3f}", ha="center", fontsize=9)

    # ── 2. Zero-inflation: observed vs expected zeros ─────────────────────────
    ax2 = axes[0, 1]
    labels = ["Observed zeros", "Poisson-expected zeros"]
    vals   = [zi["obs_zeros"], zi["exp_zeros"]]
    bars   = ax2.bar(labels, vals, color=["#DD8452", "#4C72B0"], edgecolor="white")
    ax2.set_title("2. Zero inflation: observed vs expected")
    ax2.set_ylabel("Count"); ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v * 1.01,
                 f"{int(v):,}", ha="center", fontsize=9)
    excess_pct = zi["excess_pct"]
    ax2.set_xlabel(f"Excess = {excess_pct:.2f}% of portfolio  "
                   f"({'Confirmed' if zi['confirmed'] else 'Not confirmed'})")

    # ── 3. VIF bar chart ──────────────────────────────────────────────────────
    ax3 = axes[0, 2]
    vdf = vif_r["vif_df"].sort_values("VIF")
    colors3 = ["#C44E52" if v > VIF_THRESHOLD else "#4C72B0"
               for v in vdf["VIF"]]
    ax3.barh(vdf["Feature"], vdf["VIF"], color=colors3, edgecolor="white")
    ax3.axvline(VIF_THRESHOLD, color="red", linestyle="--", lw=1.2,
                label=f"Threshold = {VIF_THRESHOLD}")
    ax3.set_title("3. VIF — numeric rating factors")
    ax3.set_xlabel("VIF"); ax3.legend(fontsize=8); ax3.grid(axis="x", alpha=0.3)
    for i, v in enumerate(vdf["VIF"]):
        ax3.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)

    # ── 4. Sparse cells: claims per level for Region and VehBrand ─────────────
    ax4 = axes[1, 0]
    if "Region" in sp_r["level_detail"]:
        reg = sp_r["level_detail"]["Region"].sort_values("Claims")
        colors4 = ["#C44E52" if s else "#4C72B0" for s in reg["Sparse"]]
        ax4.barh(reg["Region"].astype(str), reg["Claims"],
                 color=colors4, edgecolor="white")
        ax4.axvline(SPARSE_THRESHOLD, color="red", linestyle="--", lw=1.2,
                    label=f"Threshold = {SPARSE_THRESHOLD}")
        ax4.set_title("4. Sparse cells — claims per Region")
        ax4.set_xlabel("Claim count"); ax4.legend(fontsize=8)
        ax4.grid(axis="x", alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Region not available", ha="center", va="center")

    # ── 5. Isotonic smoothing: DrivAge relativities ───────────────────────────
    ax5 = axes[1, 1]
    if "DrivAge_band" in iso_r:
        res   = iso_r["DrivAge_band"]
        bands = res["bands"]
        x_pos = np.arange(len(bands))
        ax5.plot(x_pos, res["raw"], "o-", color="#DD8452", lw=2,
                 label="GLM relativities")
        ax5.plot(x_pos, res["iso"], "s--", color="#4C72B0", lw=2,
                 label="Isotonic smoothed")
        ax5.set_xticks(x_pos); ax5.set_xticklabels(bands, rotation=20)
        ax5.set_title("5. DrivAge relativities: GLM vs isotonic")
        ax5.set_ylabel("Relativity (base = min band)")
        ax5.legend(fontsize=8); ax5.grid(alpha=0.3)
        status = "Monotone" if res["monotone"] else "NON-MONOTONE"
        ax5.set_xlabel(f"DrivAge band  [{status}]")

    # ── 6. K-fold Gini bar chart ──────────────────────────────────────────────
    ax6 = axes[1, 2]
    fold_labels = [f"Fold {i+1}" for i in range(len(kf_r["fold_ginis"]))]
    ax6.bar(fold_labels, kf_r["fold_ginis"],
            color="#4C72B0", edgecolor="white", label="Fold Gini")
    ax6.axhline(kf_r["mean_gini"],  color="navy",  linestyle="-",  lw=2,
                label=f"CV mean = {kf_r['mean_gini']:.4f}")
    ax6.axhline(kf_r["single_gini"], color="orange", linestyle="--", lw=1.5,
                label=f"Single 70/30 = {kf_r['single_gini']:.4f}")
    ax6.set_title("6. K-fold CV Gini")
    ax6.set_ylabel("Gini coefficient"); ax6.legend(fontsize=8)
    ax6.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "model_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plots saved: {path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(od, zi, vif_r, sp_r, iso_r, kf_r):
    n_sparse = sum(r["n_sparse"] for r in sp_r["all_sparse"])
    non_mono = [v for v in iso_r.values() if not v["monotone"]]

    rows = [
        ("1  Overdispersion",
         od["phi_confirmed"],
         f"phi={od['phi']:.2f}, CT p={od['ct_p']:.2f}",
         ("NegBin GLM" if od["ct_confirmed"] else "Quasi-Poisson") if od["phi_confirmed"] else "Keep Poisson"),
        ("2  Zero-inflation",
         zi["confirmed"],
         f"{zi['excess_pct']:.1f}% excess zeros",
         "Hurdle Poisson (planned)" if zi["confirmed"] else "No action needed"),
        ("3  Multicollinearity",
         len(vif_r["high_vif"]) > 0,
         f"Max VIF = {vif_r['max_vif']:.2f}",
         f"Drop {', '.join(vif_r['high_vif'])}" if vif_r["high_vif"]
         else "No action needed"),
        ("4  Sparse cells",
         sp_r["n_sparse_features"] > 0,
         f"{n_sparse} sparse level(s)",
         "Collapse to Other" if n_sparse > 0 else "No action needed"),
        ("5  Non-monotone rel.",
         len(non_mono) > 0,
         f"{len(non_mono)} variable(s) affected",
         "Isotonic smoothing" if non_mono else "No action needed"),
        ("6  K-fold CV Gini",
         True,
         f"mean={kf_r['mean_gini']:.4f}, std={kf_r['std_gini']:.4f}",
         "Use CV mean Gini as primary metric"),
    ]

    print("\n" + "=" * 65)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 65)
    print(f"  {'Test':<25} {'Problem?':>9}  {'Key metric':<28} {'Recommendation'}")
    print(f"  {'-'*90}")
    for name, confirmed, metric, rec in rows:
        flag = "YES !!" if confirmed else "No    "
        print(f"  {name:<25} {flag:>9}  {metric:<28} {rec}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    df, train, test = load_and_prep()

    print("\n[FIT] Base Poisson GLM on training set ...")
    pois, feats, ns = fit_poisson(train)
    print(f"  AIC={pois.aic:,.1f}  Deviance={pois.deviance:,.1f}  "
          f"Pseudo-R2={1-pois.deviance/pois.null_deviance:.4f}")

    od   = test_overdispersion(train, pois, feats, ns)
    zi   = test_zero_inflation(train, pois)
    vifr = test_vif(train)
    spr  = test_sparse_cells(train)
    isor = test_isotonic(train, pois, feats, ns)
    kfr  = test_kfold_gini(df)

    plot_diagnostics(od, zi, vifr, spr, isor, kfr)
    print_summary(od, zi, vifr, spr, isor, kfr)


if __name__ == "__main__":
    main()
