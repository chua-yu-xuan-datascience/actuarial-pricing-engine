"""
data_investigation.py
======================
Standalone data investigation and cleaning script for the French Motor MTPL
pricing dataset.  Run this BEFORE pricing_engine.py.

Three data-quality problems are diagnosed and fixed:
  1. Incomplete claims / IBNR  — claims counted but loss not yet settled
  2. Data entry errors          — impossible or implausible field values
  3. Inconsistent exposure      — exposures outside valid [1/365, 1.0] range

For each problem the script:
  (a) diagnoses whether the problem is present and how severe it is
  (b) applies a fix only when the problem is confirmed
  (c) writes an audit trail via a 'DataQualityFlag' column

Outputs:
  cleaned_data.parquet    — use this as input to pricing_engine.py
  flagged_records.csv     — audit log of every modified record
  data_quality_report.png — diagnostic plots for all three problems
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Thresholds (adjust to match your book) ────────────────────────────────────
IBNR_THRESHOLD     = 0.05   # warn if > 5% of claimant policies have zero settled loss
EXPOSURE_MIN       = 1/365  # minimum valid exposure (1 day)
EXPOSURE_MAX       = 1.0    # maximum valid exposure (1 policy-year)
SEVERITY_MIN       = 50     # minimum plausible avg cost per claim (€)
MAX_CLAIMS_POLICY  = 5      # flag policies with more claims than this

FIELD_VALID_RANGES = {
    "DrivAge":    (18,  99),
    "VehAge":     (0,   50),
    "BonusMalus": (50, 350),
    "VehPower":   (4,   15),
}


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_raw():
    """
    Load frequency and severity tables and join on IDpol.
    Severity is aggregated per policy (sum, count, max) so we can cross-check
    claim counts between the two tables.
    """
    print("\n[LOAD] Fetching French Motor MTPL dataset …")
    freq_raw = fetch_openml("freMTPL2freq", version="active", as_frame=True, parser="auto")
    df = freq_raw.frame.copy()
    df["ClaimNb"] = pd.to_numeric(df["ClaimNb"], errors="coerce").fillna(0).astype(float)
    df["IDpol"]   = pd.to_numeric(df["IDpol"], errors="coerce").astype(int)

    sev_raw = fetch_openml("freMTPL2sev", version="active", as_frame=True, parser="auto")
    sev     = sev_raw.frame.copy()

    # Aggregate: one severity row per policy
    # Note: freMTPL2sev can have multiple payment records per claim,
    # so ClaimRecords (row count) >= true claim count. We use it only
    # to detect policies with a payment but zero ClaimNb, not to correct ClaimNb.
    sev_agg = sev.groupby("IDpol")["ClaimAmount"].agg(
        TotalLoss      = "sum",
        SevRecords     = "count",
        MaxSingleClaim = "max",
    ).reset_index()

    df = df.merge(sev_agg, on="IDpol", how="left")
    df["TotalLoss"]      = df["TotalLoss"].fillna(0.0)
    df["SevRecords"]     = df["SevRecords"].fillna(0).astype(int)
    df["MaxSingleClaim"] = df["MaxSingleClaim"].fillna(0.0)

    print(f"    Raw freq table : {len(df):,} policies  |  {int(df['ClaimNb'].sum()):,} claims")
    print(f"    Raw sev table  : {len(sev):,} payment records  |  "
          f"{sev['IDpol'].nunique():,} unique policies")
    return df, sev


# ─────────────────────────────────────────────
# PROBLEM 1 — INCOMPLETE CLAIMS / IBNR
# ─────────────────────────────────────────────
def check_ibnr(df: pd.DataFrame) -> dict:
    """
    Three sub-cases are tested:

    Case A — IBNR proxy
        ClaimNb > 0 but TotalLoss = 0.
        The claim has been counted (reserve set) but no payment has been made yet.
        In a live book these would need chain-ladder development; here they are
        excluded from the severity model but kept in the frequency model.

    Case B — Reversed error
        ClaimNb = 0 but TotalLoss > 0.
        A payment exists in the severity table for a policy the frequency table
        says never had a claim.  Likely a booking or policy-matching error.
        Fix: set ClaimNb = 1 (minimum plausible) and flag.

    Case C — Zero or negative loss amounts in the severity table
        Individual ClaimAmount <= 0.  Can indicate recoveries (subrogation /
        salvage) entered against the wrong policy, or data entry errors.
        These are flagged in the severity table for review; no automatic fix.
    """
    print("\n" + "=" * 65)
    print("PROBLEM 1: INCOMPLETE CLAIMS / IBNR")
    print("=" * 65)

    claims_mask = df["ClaimNb"] > 0

    # Case A
    ibnr_mask   = claims_mask & (df["TotalLoss"] == 0)
    n_claims    = int(claims_mask.sum())
    n_ibnr      = int(ibnr_mask.sum())
    ibnr_rate   = n_ibnr / max(n_claims, 1)

    # Case B
    reversed_mask = (~claims_mask) & (df["TotalLoss"] > 0)
    n_reversed    = int(reversed_mask.sum())

    # Settled severity stats
    settled_mask = claims_mask & (df["TotalLoss"] > 0)
    avg_sev = float(
        (df.loc[settled_mask, "TotalLoss"] / df.loc[settled_mask, "ClaimNb"]).mean()
    ) if settled_mask.sum() > 0 else 0.0

    # Low per-claim severity (potential partial settlement / below deductible)
    avg_cost_series = pd.Series(
        np.where(df["ClaimNb"] > 0, df["TotalLoss"] / df["ClaimNb"], np.nan),
        index=df.index,
    )
    low_sev_mask = settled_mask & (avg_cost_series < SEVERITY_MIN)
    n_low_sev    = int(low_sev_mask.sum())

    print(f"\n  Claimant policies (ClaimNb > 0)          : {n_claims:,}  "
          f"({n_claims / len(df) * 100:.1f}% of portfolio)")
    print(f"  Case A — claim counted, zero loss        : {n_ibnr:,}  "
          f"({ibnr_rate * 100:.1f}% of claimants)")
    print(f"  Case B — loss recorded, ClaimNb = 0      : {n_reversed:,}")
    print(f"  Policies with avg cost < €{SEVERITY_MIN}          "
          f": {n_low_sev:,}")
    print(f"  Average settled severity (excl. IBNR)    : €{avg_sev:,.0f}")

    problem_detected = (ibnr_rate > IBNR_THRESHOLD) or (n_reversed > 0)

    if ibnr_rate > IBNR_THRESHOLD:
        print(f"\n  !! IBNR rate {ibnr_rate * 100:.1f}% exceeds threshold "
              f"{IBNR_THRESHOLD * 100:.0f}% — fix will be applied")
    else:
        print(f"\n  OK IBNR rate {ibnr_rate * 100:.1f}% is within threshold")

    if n_reversed > 0:
        print(f"  !! {n_reversed:,} reversed-error policies detected — ClaimNb will be set to 1")

    return {
        "ibnr_mask":        ibnr_mask,
        "reversed_mask":    reversed_mask,
        "low_sev_mask":     low_sev_mask,
        "n_ibnr":           n_ibnr,
        "n_reversed":       n_reversed,
        "n_low_sev":        n_low_sev,
        "ibnr_rate":        ibnr_rate,
        "avg_sev":          avg_sev,
        "problem_detected": problem_detected,
    }


# ─────────────────────────────────────────────
# PROBLEM 2 — DATA ENTRY ERRORS
# ─────────────────────────────────────────────
def check_data_errors(df: pd.DataFrame) -> dict:
    """
    Checks:
      - Each numeric rating factor against hard valid-range limits
      - Missing values in key modelling columns
      - Duplicate policy IDs (same IDpol appearing more than once)
      - Policies with more than MAX_CLAIMS_POLICY claims in one year

    Fixes applied:
      - NaN values → column median (imputed and flagged)
      - Out-of-range values → clipped to valid bounds and flagged
      - Duplicate records → keep first occurrence, drop the rest
    """
    print("\n" + "=" * 65)
    print("PROBLEM 2: DATA ENTRY ERRORS & IMPOSSIBLE VALUES")
    print("=" * 65)

    flags = {}
    summary_rows = []

    for col, (lo, hi) in FIELD_VALID_RANGES.items():
        series     = pd.to_numeric(df[col], errors="coerce")
        nan_mask   = series.isna()
        below_mask = series < lo
        above_mask = series > hi
        bad_mask   = nan_mask | below_mask | above_mask

        flags[col] = bad_mask
        n_bad = int(bad_mask.sum())
        summary_rows.append({
            "Field":          col,
            "Valid range":    f"[{lo}, {hi}]",
            "N missing":      int(nan_mask.sum()),
            "N below floor":  int(below_mask.sum()),
            "N above cap":    int(above_mask.sum()),
            "Total flagged":  n_bad,
            "% of portfolio": round(n_bad / len(df) * 100, 3),
        })

    # Density: must be > 0 (log1p transform is applied later in pricing engine)
    density    = pd.to_numeric(df["Density"], errors="coerce")
    flags["Density"] = density.isna() | (density <= 0)
    n_bad_den  = int(flags["Density"].sum())
    summary_rows.append({
        "Field":          "Density",
        "Valid range":    "(0, ∞)",
        "N missing":      int(density.isna().sum()),
        "N below floor":  int((density <= 0).sum()),
        "N above cap":    0,
        "Total flagged":  n_bad_den,
        "% of portfolio": round(n_bad_den / len(df) * 100, 3),
    })

    # Duplicate IDpol
    dup_mask    = df.duplicated(subset="IDpol", keep=False)
    n_dup       = int(dup_mask.sum())

    # Extreme claim counts per policy
    high_claims = df["ClaimNb"] > MAX_CLAIMS_POLICY
    n_high      = int(high_claims.sum())

    # Missing values in all key modelling columns
    key_cols = ["DrivAge", "VehAge", "BonusMalus", "VehPower",
                "Density", "Region", "VehBrand", "VehGas", "Area"]
    missing_tbl = {c: int(df[c].isna().sum()) for c in key_cols if c in df.columns}

    summary_df = pd.DataFrame(summary_rows)
    total_field_flags = int(sum(m.sum() for m in flags.values()))

    print("\n  Field range audit:")
    print(summary_df.to_string(index=False))
    print(f"\n  Duplicate IDpol entries           : {n_dup:,}")
    print(f"  Policies with > {MAX_CLAIMS_POLICY} claims         : {n_high:,}")
    print(f"\n  Missing values by column:")
    for col, n in missing_tbl.items():
        marker = "  !!" if n > 0 else "  OK"
        print(f"    {marker} {col:<15}: {n:,}")
    print(f"\n  Total field-level anomalies       : {total_field_flags:,}")

    flags["dup_mask"]         = dup_mask
    flags["high_claims_mask"] = high_claims

    return {
        "flags":             flags,
        "summary_df":        summary_df,
        "missing_tbl":       missing_tbl,
        "total_flagged":     total_field_flags,
        "n_dup":             n_dup,
        "problem_detected":  (total_field_flags > 0) or (n_dup > 0),
    }


# ─────────────────────────────────────────────
# PROBLEM 3 — INCONSISTENT EXPOSURE
# ─────────────────────────────────────────────
def check_exposure(df: pd.DataFrame) -> dict:
    """
    Valid exposure window: [1/365, 1.0] policy-years.

    Sub-cases:
      - Zero or negative    → impossible; cannot compute an offset for GLM
      - Below 1 day         → likely a system artefact; annualised frequency
                              would be artificially inflated
      - Above 1 policy-year → impossible in an annual motor book
      - Short exposure + claim — especially distorting for frequency estimation

    Fixes:
      - Zero/negative → row removed (cannot be modelled)
      - Below min / above max → clipped and flagged
    """
    print("\n" + "=" * 65)
    print("PROBLEM 3: INCONSISTENT / INVALID EXPOSURE")
    print("=" * 65)

    exp = pd.to_numeric(df["Exposure"], errors="coerce")

    missing_mask   = exp.isna()
    zero_neg_mask  = (exp <= 0) & ~missing_mask
    below_min_mask = (exp > 0) & (exp < EXPOSURE_MIN)
    above_max_mask = exp > EXPOSURE_MAX
    short_claim    = below_min_mask & (df["ClaimNb"] > 0)

    total_invalid  = int((missing_mask | zero_neg_mask).sum())
    total_clipped  = int((below_min_mask | above_max_mask).sum())

    print(f"\n  Missing exposure                  : {int(missing_mask.sum()):,}")
    print(f"  Zero or negative                  : {int(zero_neg_mask.sum()):,}  "
          f"[will be DROPPED]")
    print(f"  Below 1 day (< {EXPOSURE_MIN:.5f})      : {int(below_min_mask.sum()):,}  "
          f"[will be CLIPPED]")
    print(f"  Above 1 year (> 1.0)              : {int(above_max_mask.sum()):,}  "
          f"[will be CLIPPED]")
    print(f"  Short exposure with claim         : {int(short_claim.sum()):,}")

    print(f"\n  Exposure quantiles (raw):")
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.00]
    for q in quantiles:
        print(f"    p{q * 100:5.1f} = {exp.quantile(q):.6f}")

    problem_detected = (total_invalid + total_clipped) > 0
    if not problem_detected:
        print("\n  OK All exposures within valid range")
    else:
        print(f"\n  !! {total_invalid + total_clipped:,} exposure records require correction")

    return {
        "missing_mask":     missing_mask,
        "zero_neg_mask":    zero_neg_mask,
        "below_min_mask":   below_min_mask,
        "above_max_mask":   above_max_mask,
        "short_claim":      short_claim,
        "total_invalid":    total_invalid,
        "total_clipped":    total_clipped,
        "problem_detected": problem_detected,
    }


# ─────────────────────────────────────────────
# APPLY FIXES
# ─────────────────────────────────────────────
def apply_fixes(
    df_raw: pd.DataFrame,
    ibnr:   dict,
    errors: dict,
    exp:    dict,
) -> pd.DataFrame:
    """
    Applies all confirmed fixes in a single pass.

    Order of operations:
      1. Add DataQualityFlag column to track every change
      2. Fix Problem 1 (IBNR, reversed errors)
      3. Fix Problem 2 (range clips, median imputation, duplicate removal)
      4. Fix Problem 3 (drop zero-exposure rows, clip remainder)

    Rows are dropped LAST so that flag information is captured first.
    All masks are aligned to df_raw's index, which is preserved in df.copy().
    """
    print("\n" + "=" * 65)
    print("APPLYING FIXES")
    print("=" * 65)

    df = df_raw.copy()
    df["DataQualityFlag"] = ""

    # ── Problem 1A: IBNR ─────────────────────────────────────────────────────
    if ibnr["n_ibnr"] > 0:
        mask = ibnr["ibnr_mask"]
        df.loc[mask, "DataQualityFlag"] += "IBNR|"
        print(f"  1A Flagged {mask.sum():,} IBNR policies "
              "(kept for frequency; exclude from severity downstream)")

    # ── Problem 1B: Reversed error ────────────────────────────────────────────
    if ibnr["n_reversed"] > 0:
        mask = ibnr["reversed_mask"]
        df.loc[mask, "ClaimNb"]          = 1
        df.loc[mask, "DataQualityFlag"] += "REV_ERR|"
        print(f"  1B Set ClaimNb=1 for {mask.sum():,} reversed-error policies")

    # ── Problem 2: Range clips and NaN imputation ─────────────────────────────
    for col, (lo, hi) in FIELD_VALID_RANGES.items():
        series    = pd.to_numeric(df[col], errors="coerce")
        nan_mask  = series.isna()

        if nan_mask.sum() > 0:
            median_val = series.median()
            df.loc[nan_mask, col]               = median_val
            df.loc[nan_mask, "DataQualityFlag"] += f"FILL_{col}|"
            print(f"  2  Imputed median ({median_val:.1f}) for "
                  f"{nan_mask.sum():,} missing {col} values")

        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        out_of_range = (df[col] < lo) | (df[col] > hi)
        if out_of_range.sum() > 0:
            df.loc[out_of_range, "DataQualityFlag"] += f"CLIP_{col}|"
            df[col] = df[col].clip(lower=lo, upper=hi)
            print(f"  2  Clipped {out_of_range.sum():,} {col} values to [{lo}, {hi}]")

    # Density: fill negatives/NaN with 1 (smallest valid raw density before log1p)
    dens = pd.to_numeric(df["Density"], errors="coerce")
    bad_dens = dens.isna() | (dens <= 0)
    if bad_dens.sum() > 0:
        df.loc[bad_dens, "Density"]          = 1
        df.loc[bad_dens, "DataQualityFlag"] += "FILL_Density|"
        print(f"  2  Fixed {bad_dens.sum():,} invalid Density values → 1")

    # ── Problem 2: Duplicates ─────────────────────────────────────────────────
    dup_mask = errors["flags"]["dup_mask"]
    if dup_mask.sum() > 0:
        df.loc[dup_mask, "DataQualityFlag"] += "DUP|"
        n_before = len(df)
        df = df[~df.duplicated(subset="IDpol", keep="first")].copy()
        print(f"  2  Removed {n_before - len(df):,} duplicate IDpol records")

    # ── Problem 3: Zero / negative exposure — drop rows ──────────────────────
    zero_neg = exp["zero_neg_mask"]
    if zero_neg.sum() > 0:
        # Flag before dropping (mask still aligned after duplicate removal
        # because we used loc-based flagging above)
        surviving = zero_neg.index.isin(df.index)
        drop_mask = zero_neg & pd.Series(True, index=df_raw.index).reindex(df.index, fill_value=False)
        df.loc[zero_neg[zero_neg].index.intersection(df.index), "DataQualityFlag"] += "EXP_ZERO|"
        n_before = len(df)
        df = df[df["Exposure"].astype(float) > 0].copy()
        print(f"  3  Dropped {n_before - len(df):,} policies with zero/negative exposure")

    # ── Problem 3: Clip remaining exposure ────────────────────────────────────
    exp_series   = df["Exposure"].astype(float)
    too_low      = exp_series < EXPOSURE_MIN
    too_high     = exp_series > EXPOSURE_MAX
    if too_low.sum() > 0:
        df.loc[too_low,  "DataQualityFlag"] += "EXP_LOW|"
        print(f"  3  Flagged+clipped {too_low.sum():,} exposures below 1 day → {EXPOSURE_MIN:.5f}")
    if too_high.sum() > 0:
        df.loc[too_high, "DataQualityFlag"] += "EXP_HIGH|"
        print(f"  3  Flagged+clipped {too_high.sum():,} exposures above 1 year → 1.0")
    df["Exposure"] = df["Exposure"].astype(float).clip(lower=EXPOSURE_MIN, upper=EXPOSURE_MAX)

    # ── Finalise flag column ──────────────────────────────────────────────────
    df["DataQualityFlag"] = df["DataQualityFlag"].str.strip("|").replace("", "CLEAN")

    n_clean   = int((df["DataQualityFlag"] == "CLEAN").sum())
    n_flagged = len(df) - n_clean
    print(f"\n  Result: {len(df):,} policies  "
          f"|  {n_clean:,} clean  |  {n_flagged:,} flagged  "
          f"({n_flagged / len(df) * 100:.1f}%)")
    return df


# ─────────────────────────────────────────────
# DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────
def plot_diagnostics(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """
    3 × 3 plot grid:
      Row 1: IBNR / settlement completeness
      Row 2: Data entry error distributions (raw vs cleaned)
      Row 3: Exposure distribution and flag breakdown
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        "Data Quality Investigation — French Motor MTPL",
        fontsize=13, fontweight="bold",
    )

    # ── Row 1: IBNR ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    claimant   = df_raw[df_raw["ClaimNb"] > 0]
    n_ibnr     = int((claimant["TotalLoss"] == 0).sum())
    n_settled  = int(len(claimant) - n_ibnr)
    n_reversed = int(((df_raw["ClaimNb"] == 0) & (df_raw["TotalLoss"] > 0)).sum())
    bars       = ax.bar(
        ["Settled claims", "IBNR\n(no loss yet)", "Reversed error\n(loss, 0 claims)"],
        [n_settled, n_ibnr, n_reversed],
        color=["#4C72B0", "#DD8452", "#C44E52"], edgecolor="white",
    )
    ax.set_title("1a. Claim settlement status")
    ax.set_ylabel("Policy count"); ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, [n_settled, n_ibnr, n_reversed]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:,}", ha="center", fontsize=9)

    ax2 = axes[0, 1]
    settled = df_raw[(df_raw["ClaimNb"] > 0) & (df_raw["TotalLoss"] > 0)].copy()
    settled["AvgCost"] = settled["TotalLoss"] / settled["ClaimNb"]
    cap99 = settled["AvgCost"].quantile(0.99)
    settled["AvgCost"].clip(upper=cap99).hist(
        ax=ax2, bins=60, color="#4C72B0", edgecolor="white", density=True,
    )
    ax2.axvline(SEVERITY_MIN, color="red", linestyle="--",
                label=f"Min plausible (€{SEVERITY_MIN})")
    ax2.set_title("1b. Avg cost per claim (99th pct cap)")
    ax2.set_xlabel("€"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    ax3 = axes[0, 2]
    claim_dist = (
        df_raw["ClaimNb"].clip(0, MAX_CLAIMS_POLICY + 1)
        .astype(int).value_counts().sort_index()
    )
    ax3.bar(claim_dist.index, claim_dist.values,
            color="#4C72B0", edgecolor="white")
    ax3.axvline(MAX_CLAIMS_POLICY + 0.5, color="red", linestyle="--",
                label=f"Flag threshold ({MAX_CLAIMS_POLICY})")
    ax3.set_title("1c. Claims per policy")
    ax3.set_xlabel("Number of claims"); ax3.set_ylabel("Policies")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    # ── Row 2: Data entry errors ──────────────────────────────────────────────
    plot_fields = ["DrivAge", "BonusMalus", "VehAge"]
    for i, col in enumerate(plot_fields):
        lo, hi = FIELD_VALID_RANGES[col]
        ax = axes[1, i]
        raw_s   = pd.to_numeric(df_raw[col],   errors="coerce").dropna()
        clean_s = pd.to_numeric(df_clean[col], errors="coerce").dropna()
        lo_q    = raw_s.quantile(0.001)
        hi_q    = raw_s.quantile(0.999)
        bins    = np.linspace(min(lo_q, lo - 1), max(hi_q, hi + 1), 60)
        ax.hist(raw_s,   bins=bins, alpha=0.45, color="#DD8452",
                label="Raw",   density=True)
        ax.hist(clean_s, bins=bins, alpha=0.70, color="#4C72B0",
                label="Clean", density=True)
        ax.axvline(lo, color="red",   linestyle="--", lw=1.2, label=f"Floor={lo}")
        ax.axvline(hi, color="green", linestyle="--", lw=1.2, label=f"Cap={hi}")
        ax.set_title(f"2. {col}: raw vs cleaned")
        ax.set_xlabel(col); ax.set_ylabel("Density")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Row 3: Exposure ───────────────────────────────────────────────────────
    ax7 = axes[2, 0]
    exp_raw = pd.to_numeric(df_raw["Exposure"], errors="coerce").dropna()
    ax7.hist(exp_raw, bins=80, color="#4C72B0", edgecolor="white", density=True)
    ax7.axvline(EXPOSURE_MIN, color="red",   linestyle="--", lw=1.5,
                label=f"Min (1/365 ≈ {EXPOSURE_MIN:.4f})")
    ax7.axvline(EXPOSURE_MAX, color="green", linestyle="--", lw=1.5,
                label="Max (1.0)")
    ax7.set_title("3a. Raw exposure distribution")
    ax7.set_xlabel("Exposure (years)"); ax7.set_ylabel("Density")
    ax7.legend(fontsize=8); ax7.grid(alpha=0.3)

    ax8 = axes[2, 1]
    exp_clean = pd.to_numeric(df_clean["Exposure"], errors="coerce").dropna()
    ax8.hist(exp_clean, bins=80, color="#4C72B0", edgecolor="white", density=True)
    ax8.set_title("3b. Cleaned exposure distribution")
    ax8.set_xlabel("Exposure (years)"); ax8.set_ylabel("Density")
    ax8.grid(alpha=0.3)

    ax9 = axes[2, 2]
    flag_counts = (
        df_clean["DataQualityFlag"]
        .str.split("|").explode()
        .value_counts()
        .drop("CLEAN", errors="ignore")
        .head(10)
    )
    if len(flag_counts) > 0:
        flag_counts.sort_values().plot(kind="barh", ax=ax9,
                                       color="#4C72B0", edgecolor="white")
    ax9.set_title("3c. Flag type distribution (excl. CLEAN)")
    ax9.set_xlabel("Occurrence count"); ax9.grid(axis="x", alpha=0.3)
    ax9.invert_yaxis()

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "data_quality_report.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plots saved → {path}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_summary(
    df_raw:   pd.DataFrame,
    df_clean: pd.DataFrame,
    ibnr:     dict,
    errors:   dict,
    exp_diag: dict,
):
    print("\n" + "=" * 65)
    print("DATA QUALITY SUMMARY")
    print("=" * 65)

    rows = [
        ("Raw policies",                       len(df_raw),                         ""),
        ("Cleaned policies",                    len(df_clean),                       ""),
        ("Policies removed",                    len(df_raw) - len(df_clean),         ""),
        ("",                                    "",                                  ""),
        ("── Problem 1: IBNR / Completeness ──","",                                 ""),
        ("  IBNR (claim counted, zero loss)",   ibnr["n_ibnr"],
         f"{ibnr['ibnr_rate'] * 100:.1f}% of claimants"),
        ("  Reversed error (loss, 0 claims)",   ibnr["n_reversed"],                 ""),
        ("  Low severity (< €50 avg)",          ibnr["n_low_sev"],                  ""),
        ("",                                    "",                                  ""),
        ("── Problem 2: Data Entry Errors ────","",                                 ""),
        ("  Field-level anomalies",             errors["total_flagged"],             ""),
        ("  Duplicate IDpol records",           errors["n_dup"],                     ""),
        ("",                                    "",                                  ""),
        ("── Problem 3: Exposure ─────────────","",                                 ""),
        ("  Zero / negative (dropped)",         int(exp_diag["zero_neg_mask"].sum()),""),
        ("  Below 1 day (clipped)",             int(exp_diag["below_min_mask"].sum()),""),
        ("  Above 1 year (clipped)",            int(exp_diag["above_max_mask"].sum()),""),
        ("  Short exposure with claim",         int(exp_diag["short_claim"].sum()),  ""),
        ("",                                    "",                                  ""),
        ("── Cleaned dataset ─────────────────","",                                 ""),
        ("  CLEAN records",
         int((df_clean["DataQualityFlag"] == "CLEAN").sum()),
         f"{(df_clean['DataQualityFlag'] == 'CLEAN').mean() * 100:.1f}%"),
        ("  Flagged records",
         int((df_clean["DataQualityFlag"] != "CLEAN").sum()),
         f"{(df_clean['DataQualityFlag'] != 'CLEAN').mean() * 100:.1f}%"),
    ]

    for label, value, note in rows:
        if not label:
            print()
        elif label.startswith("──"):
            print(f"\n  {label}")
        else:
            note_str = f"  {note}" if note else ""
            print(f"    {label:<42} {str(value):<10}{note_str}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df_raw, sev = load_raw()

    ibnr_diag  = check_ibnr(df_raw)
    error_diag = check_data_errors(df_raw)
    exp_diag   = check_exposure(df_raw)

    df_clean = apply_fixes(df_raw, ibnr_diag, error_diag, exp_diag)

    plot_diagnostics(df_raw, df_clean)
    print_summary(df_raw, df_clean, ibnr_diag, error_diag, exp_diag)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_parquet = os.path.join(OUT_DIR, "cleaned_data.parquet")
    out_flags   = os.path.join(OUT_DIR, "flagged_records.csv")

    df_clean.to_parquet(out_parquet, index=False)

    df_clean[df_clean["DataQualityFlag"] != "CLEAN"][[
        "IDpol", "ClaimNb", "TotalLoss", "Exposure",
        "DrivAge", "VehAge", "BonusMalus", "VehPower", "DataQualityFlag",
    ]].to_csv(out_flags, index=False)

    print(f"\n  cleaned_data.parquet → {out_parquet}")
    print(f"  flagged_records.csv  → {out_flags}")
    print("\n  Next step: run pricing_engine.py using cleaned_data.parquet as input.")


if __name__ == "__main__":
    main()
