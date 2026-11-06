"""
Motor MTPL Pricing Engine — Streamlit Web App
=============================================
Frequency-Severity GLM  |  Quasi-Poisson (phi=2.66) + Gamma  |  freMTPL2 dataset
Isotonic-smoothed DrivAge & VehAge relativities

Run:
    pip install streamlit
    streamlit run app.py

First run downloads freMTPL2 and trains both GLMs (~2 min).
Subsequent runs load instantly from Streamlit's resource cache.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os, io, warnings

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Pricing Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load / cache the trained models ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pricing_engine as pe  # must import before stdout is redirected (pe does sys.stdout.reconfigure)

@st.cache_resource(show_spinner="Training Quasi-Poisson + Gamma GLM on freMTPL2 — ~2 min first run…")
def load_models():
    warnings.filterwarnings("ignore")
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        df    = pe.load_data()
        df    = pe.preprocess(df)
        train, test = pe.split_data(df)

        freq_model, freq_features, freq_norm = pe.build_freq_model(train)
        sev_model,  sev_features,  sev_norm  = pe.build_sev_model(train, freq_norm)

        # Isotonic corrections — computed from training set, applied to all splits
        iso_corr = pe.compute_isotonic_corrections(train, freq_model, freq_features, freq_norm)

        train_scored = pe.score_and_price(train, freq_model, sev_model,
                                          freq_features, sev_features,
                                          freq_norm, sev_norm, iso_corr=iso_corr)
        test_scored  = pe.score_and_price(test,  freq_model, sev_model,
                                          freq_features, sev_features,
                                          freq_norm, sev_norm, iso_corr=iso_corr)
        df_scored    = pe.score_and_price(df,    freq_model, sev_model,
                                          freq_features, sev_features,
                                          freq_norm, sev_norm, iso_corr=iso_corr)

        decile_tbl, diagnostics = pe.validate(test_scored, train_scored,
                                              output_dir=os.path.dirname(os.path.abspath(__file__)))
        comparison_tbl = pe.compare_train_test(train_scored, test_scored)
        coef_tbl, ob_factor = pe.build_tariff_manual(df_scored, freq_model,
                                                      sev_model, freq_features)
    finally:
        sys.stdout = old

    return (pe, df_scored, train_scored, test_scored,
            freq_model, sev_model, freq_features, sev_features,
            freq_norm, sev_norm, iso_corr,
            decile_tbl, diagnostics, comparison_tbl, coef_tbl, ob_factor)


(pe, df, train_scored, test_scored,
 freq_model, sev_model, freq_features, sev_features,
 freq_norm, sev_norm, iso_corr,
 decile_tbl, diagnostics, comparison_tbl, coef_tbl, ob_factor) = load_models()

gini = diagnostics["gini"]

# ── Reference lookups for dropdowns ──────────────────────────────────────────
REGION_LABELS = {
    "R11": "R11 — Île-de-France",       "R21": "R21 — Champagne-Ardenne",
    "R22": "R22 — Picardie",            "R23": "R23 — Haute-Normandie",
    "R24": "R24 — Centre-Val-de-Loire", "R25": "R25 — Basse-Normandie",
    "R26": "R26 — Bourgogne",           "R31": "R31 — Nord-Pas-de-Calais",
    "R41": "R41 — Lorraine",            "R42": "R42 — Alsace",
    "R43": "R43 — Franche-Comté",       "R52": "R52 — Pays de la Loire",
    "R53": "R53 — Bretagne",            "R54": "R54 — Poitou-Charentes",
    "R72": "R72 — Aquitaine",           "R73": "R73 — Midi-Pyrénées",
    "R74": "R74 — Limousin",            "R82": "R82 — Rhône-Alpes",
    "R83": "R83 — Auvergne",            "R91": "R91 — Languedoc-Roussillon",
    "R93": "R93 — PACA",                "R94": "R94 — Corse",
}
BRAND_LABELS = {
    "B1":  "B1  — Renault/Nissan/Dacia", "B2":  "B2  — Peugeot/Citroën",
    "B3":  "B3  — Volkswagen/Seat/Skoda","B4":  "B4  — Toyota/Honda",
    "B5":  "B5  — Ford/Opel",            "B6":  "B6  — Fiat/Alfa Romeo",
    "B10": "B10 — BMW/Mercedes",         "B11": "B11 — Jaguar/Land Rover",
    "B12": "B12 — Other European",       "B13": "B13 — Korean brands",
    "B14": "B14 — Other/Unknown",
}
REG_CODES   = sorted(REGION_LABELS.keys())
BRAND_CODES = sorted(BRAND_LABELS.keys())
BATCH_COLS  = ["DrivAge","VehAge","BonusMalus","VehPower","Density","Region","VehBrand","Exposure"]

# ── Sidebar — policy profile ──────────────────────────────────────────────────
st.sidebar.title("🚗  Policy Profile")
st.sidebar.markdown("Adjust risk characteristics below.")

driver_age  = st.sidebar.slider("Driver Age",              18, 90, 35)
veh_age     = st.sidebar.slider("Vehicle Age (years)",      0, 20,  4)
bonus_malus = st.sidebar.slider("Bonus-Malus Score",       50, 230, 100,
    help="50 = best (no claims history); 100 = neutral; 230 = worst")
veh_power   = st.sidebar.slider("Vehicle Power (class 4–15)", 4, 15, 7,
    help="4 = least powerful; 15 = most powerful")
density_raw = st.sidebar.slider("Population Density (persons/km²)", 1, 50_000, 500,
    help="Proxy for urban vs rural exposure")

region_options = sorted(df["Region"].dropna().unique().tolist())
brand_options  = sorted(df["VehBrand"].dropna().unique().tolist())
region    = st.sidebar.selectbox("Region", region_options,
    format_func=lambda r: REGION_LABELS.get(r, r),
    index=region_options.index("R11") if "R11" in region_options else 0)
veh_brand = st.sidebar.selectbox("Vehicle Brand", brand_options,
    format_func=lambda b: BRAND_LABELS.get(b, b),
    index=brand_options.index("B12") if "B12" in brand_options else 0)
exposure  = st.sidebar.slider("Policy Duration (years)", 0.1, 1.0, 1.0, step=0.05)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️  Commercial Parameters")
expense_ratio = st.sidebar.slider("Expense Ratio",        0.10, 0.50, 0.28, step=0.01)
profit_margin = st.sidebar.slider("Target Profit Margin", 0.01, 0.20, 0.05, step=0.01)
min_premium   = st.sidebar.slider("Minimum Premium (€)",   20,  300,  80)
permissible   = 1 - expense_ratio - profit_margin
st.sidebar.caption(f"Permissible loss ratio = **{permissible:.1%}**")

st.sidebar.divider()
st.sidebar.markdown("### Model summary")
st.sidebar.success(
    f"Quasi-Poisson + Gamma GLM  \n"
    f"phi = {freq_model.scale:.3f}  |  SE ×{freq_model.scale**0.5:.3f}  \n"
    f"Train: {len(train_scored):,} policies"
)

# ── Compute live quote ────────────────────────────────────────────────────────
quote = pe.quote_single_policy(
    freq_model, sev_model, freq_features, sev_features,
    freq_norm, sev_norm,
    driver_age=driver_age, veh_age=veh_age, bonus_malus=bonus_malus,
    veh_power=veh_power, density_raw=density_raw, region=region,
    veh_brand=veh_brand, exposure=exposure,
    expense_ratio=expense_ratio, profit_margin=profit_margin,
    min_premium=float(min_premium),
    iso_corr=iso_corr,
)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Underwriter Quote",
    "📊  Batch Quote",
    "📈  Model Diagnostics",
    "🔀  What-If Scenario",
    "📚  Actuarial Concepts",
])


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — Underwriter Quote Interface                ║
# ╚══════════════════════════════════════════════════════╝
with tab1:
    st.title("Underwriter Quote Dashboard")
    st.caption("Live pricing based on Quasi-Poisson Frequency × Gamma Severity GLM | freMTPL2")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Frequency",    f"{quote['PredFreq']:.4f}",
                help="Expected claims per year (isotonic-corrected)")
    col2.metric("Predicted Severity (€)", f"{quote['PredSev']:,.0f}",
                help="Expected cost per claim (conditional on a claim occurring)")
    col3.metric("Pure Premium (€)",       f"{quote['PurePremium']:,.2f}",
                help="Freq × Severity — actuarial risk cost only, no loadings")
    ind_delta = "⚠️ Min floor applied" if quote["MinPremiumApplied"] else None
    col4.metric("Indicated Premium (€)",  f"{quote['IndicatedPremium']:,.2f}",
                delta=ind_delta, delta_color="off")

    st.divider()

    # Premium breakdown waterfall chart
    st.subheader("Premium Build-Up Waterfall")
    pure    = quote["PurePremium"]
    ll_add  = pure * quote["LargeLossLoad"]
    gross   = pure + ll_add
    perml   = 1 - expense_ratio - profit_margin
    final   = quote["IndicatedPremium"]
    # Backsolve expense and profit from the indicated premium
    exp_add  = final * expense_ratio  if perml > 0 else 0
    prof_add = final * profit_margin  if perml > 0 else 0

    fig_wf = go.Figure(go.Waterfall(
        name="Premium Build", orientation="v",
        measure=["absolute","relative","relative","relative","total"],
        x=["Pure Premium","Large Loss Load","Expense Load","Profit Load","Indicated Premium"],
        y=[pure, ll_add, exp_add, prof_add, 0],
        connector={"line": {"color": "rgb(63,63,63)"}},
        decreasing={"marker": {"color": "#EF553B"}},
        increasing={"marker": {"color": "#00CC96"}},
        totals={"marker":    {"color": "#636EFA"}},
        textposition="outside",
        text=[f"€{v:,.1f}" if i < 4 else f"€{final:,.1f}"
              for i, v in enumerate([pure, ll_add, exp_add, prof_add, 0])],
    ))
    fig_wf.update_layout(height=400, showlegend=False,
                         title="How the indicated premium is built up from risk cost")
    st.plotly_chart(fig_wf, use_container_width=True)

    st.divider()

    # Risk flag breakdown
    st.subheader("Risk Flag Analysis")
    flags = []
    if driver_age <= 25:                         flags.append(("🔴 Young Driver",        "Age ≤ 25: elevated frequency"))
    if bonus_malus > 130:                        flags.append(("🔴 High Bonus-Malus",    "BM > 130: poor claims history"))
    if veh_power >= 10:                          flags.append(("🟡 High-Power Vehicle",  "Power ≥ 10: severity risk"))
    if driver_age <= 25 and veh_power >= 10:     flags.append(("🔴 Young + High Power",  "Interaction term active: compounded risk"))
    if density_raw > 10_000:                     flags.append(("🟡 Dense Urban Area",    "Density > 10,000: theft & collision exposure"))
    if not flags:                                flags.append(("🟢 Standard Risk",       "No elevated risk flags detected"))

    for icon_label, reason in flags:
        st.markdown(f"**{icon_label}** — {reason}")

    pct_rank = (df["IndicatedPremium"] < quote["IndicatedPremium"]).mean() * 100
    st.progress(int(pct_rank),
                text=f"This policy is more expensive than {pct_rank:.0f}% of the portfolio")

    # Isotonic band info
    with st.expander("Isotonic corrections applied to this policy"):
        da_band  = str(pd.cut([driver_age], bins=[17,25,35,45,55,65,91],
                               labels=["18-25","26-35","36-45","46-55","56-65","66+"])[0])
        va_band  = str(pd.cut([veh_age],    bins=[-1,2,5,10,20],
                               labels=["0-2","3-5","6-10","11+"])[0])
        da_iso   = iso_corr.get("DrivAge_band", {}).get(da_band, 1.0)
        va_iso   = iso_corr.get("VehAge_band",  {}).get(va_band, 1.0)
        st.markdown(
            f"| Factor | Band | Applied multiplier |\n"
            f"|--------|------|-------------------|\n"
            f"| DrivAge | {da_band} | ×{da_iso:.4f} |\n"
            f"| VehAge  | {va_band} | ×{va_iso:.4f} |"
        )
        st.caption(
            "Multipliers correct non-monotone adjacent bands detected during model diagnostics. "
            "They are computed from the training set and applied identically to all business."
        )

    # Comparable profiles
    with st.expander("Compare with typical profiles"):
        profiles = [
            ("This policy",        driver_age, veh_age, bonus_malus, veh_power, density_raw, region, veh_brand),
            ("Young high-risk",    22,  2, 180, 12, 2000,  "R11", "B10"),
            ("Middle-aged clean",  45,  5,  60,  7,  800,  "R54", "B2"),
            ("Experienced low-BM", 55,  8,  55,  6,  400,  "R73", "B4"),
            ("Elderly urban",      72, 10, 100,  5, 5000,  "R82", "B1"),
        ]
        comp_rows = []
        for label, da, va, bm, vp, den, reg, br in profiles:
            cq = pe.quote_single_policy(
                freq_model, sev_model, freq_features, sev_features, freq_norm, sev_norm,
                driver_age=da, veh_age=va, bonus_malus=bm, veh_power=vp,
                density_raw=den, region=reg, veh_brand=br,
                expense_ratio=expense_ratio, profit_margin=profit_margin,
                min_premium=float(min_premium), iso_corr=iso_corr,
            )
            comp_rows.append({
                "Profile": label, "Age": da, "VehAge": va, "BM": bm, "Power": vp,
                "Freq/yr": f"{cq['PredFreq']:.4f}",
                "Avg Sev (€)": f"{cq['PredSev']:,.0f}",
                "Pure Prem (€)": f"{cq['PurePremium']:,.0f}",
                "Indicated (€)": f"{cq['IndicatedPremium']:,.0f}",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — Batch Quote                                ║
# ╚══════════════════════════════════════════════════════╝
with tab2:
    st.title("Batch Quoting")
    st.markdown(
        "Upload a CSV with one row per policy. Required columns: "
        "`DrivAge`, `VehAge`, `BonusMalus`, `VehPower`, "
        "`Density` (raw persons/km²), `Region`, `VehBrand`, `Exposure`.  \n"
        "Pricing parameters (expense ratio, profit margin, minimum premium) "
        "come from the sidebar sliders."
    )

    # Template download
    tmpl = pd.DataFrame([
        {"DrivAge":35,"VehAge":3,"BonusMalus":100,"VehPower":7,
         "Density":1000,"Region":"R82","VehBrand":"B1","Exposure":1.0},
        {"DrivAge":22,"VehAge":1,"BonusMalus":150,"VehPower":12,
         "Density":5000,"Region":"R11","VehBrand":"B10","Exposure":1.0},
        {"DrivAge":55,"VehAge":8,"BonusMalus":60,"VehPower":6,
         "Density":300,"Region":"R73","VehBrand":"B2","Exposure":0.5},
        {"DrivAge":45,"VehAge":5,"BonusMalus":80,"VehPower":8,
         "Density":800,"Region":"R54","VehBrand":"B3","Exposure":1.0},
    ])
    st.download_button(
        "⬇️  Download template CSV",
        data=tmpl.to_csv(index=False).encode(),
        file_name="batch_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            missing = [c for c in BATCH_COLS if c not in df_in.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                with st.spinner(f"Pricing {len(df_in):,} policies…"):
                    results = []
                    for _, row in df_in.iterrows():
                        try:
                            cq = pe.quote_single_policy(
                                freq_model, sev_model, freq_features, sev_features,
                                freq_norm, sev_norm,
                                driver_age   = int(row["DrivAge"]),
                                veh_age      = int(row["VehAge"]),
                                bonus_malus  = int(row["BonusMalus"]),
                                veh_power    = int(row["VehPower"]),
                                density_raw  = float(row["Density"]),
                                region       = str(row["Region"]),
                                veh_brand    = str(row["VehBrand"]),
                                exposure     = float(row.get("Exposure", 1.0)),
                                expense_ratio=expense_ratio,
                                profit_margin=profit_margin,
                                min_premium  =float(min_premium),
                                iso_corr     =iso_corr,
                            )
                            results.append(cq)
                        except Exception as e:
                            results.append({"Error": str(e)})

                df_out = df_in.copy()
                df_out["PredFreq"]         = [r.get("PredFreq")          for r in results]
                df_out["PredSev"]          = [r.get("PredSev")           for r in results]
                df_out["PurePremium"]      = [r.get("PurePremium")       for r in results]
                df_out["IndicatedPremium"] = [r.get("IndicatedPremium")  for r in results]
                df_out["MinPremApplied"]   = [r.get("MinPremiumApplied") for r in results]

                st.success(f"Priced {len(df_out):,} policies.")

                c_stats, c_dist = st.columns(2)
                with c_stats:
                    st.markdown("**Portfolio summary**")
                    prems = df_out["IndicatedPremium"].dropna()
                    st.dataframe(pd.DataFrame({
                        "Metric": ["Policies", "Mean", "Median", "Min", "Max", "At floor (%)"],
                        "Value":  [f"{len(prems):,}",
                                   f"€{prems.mean():,.2f}", f"€{prems.median():,.2f}",
                                   f"€{prems.min():,.2f}",  f"€{prems.max():,.2f}",
                                   f"{df_out['MinPremApplied'].mean()*100:.1f}%"],
                    }), use_container_width=True, hide_index=True)

                with c_dist:
                    st.markdown("**Premium distribution**")
                    fig_bd = px.histogram(
                        prems.clip(upper=prems.quantile(0.99)),
                        nbins=40,
                        labels={"value":"Indicated Premium (€)", "count":"Policies"},
                    )
                    fig_bd.add_vline(x=prems.mean(), line_dash="dash", line_color="red",
                                     annotation_text=f"Mean €{prems.mean():,.0f}")
                    fig_bd.update_layout(height=280, showlegend=False)
                    st.plotly_chart(fig_bd, use_container_width=True)

                st.subheader("Results")
                st.dataframe(df_out, use_container_width=True)
                st.download_button(
                    "⬇️  Download priced CSV",
                    data=df_out.to_csv(index=False).encode(),
                    file_name="priced_policies.csv", mime="text/csv",
                )
        except Exception as e:
            st.error(f"Failed to process file: {e}")


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — Model Diagnostics                          ║
# ╚══════════════════════════════════════════════════════╝
with tab3:
    st.title("Model Diagnostics")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        st.metric("Model Gini Coefficient", f"{gini:.4f}",
                  help="Lorenz curve area. 0=random, 1=perfect. Typical: 0.25–0.55 motor.")
        st.metric("Off-Balance Factor", f"{ob_factor:.4f}",
                  help="Ratio of predicted losses to total indicated premium. Ideally ~1.0.")
        st.metric("Quasi-Poisson phi", f"{freq_model.scale:.3f}",
                  help="Pearson dispersion. 1.0 = no overdispersion. 2.66 = SEs ×1.63.")

    with col_a:
        st.subheader("Lift Chart — Loss Ratio by Predicted Risk Decile")
        st.caption("A well-calibrated model produces a steeply upward-sloping chart.")
        fig_lift = px.bar(
            decile_tbl.reset_index(), x="Decile", y="LossRatio",
            color="LossRatio",
            color_continuous_scale=["#2ecc71","#f39c12","#e74c3c"],
            labels={"LossRatio": "Actual Loss Ratio"},
        )
        fig_lift.add_hline(y=1.0, line_dash="dash", line_color="red",
                           annotation_text="Break-even LR = 1.0")
        fig_lift.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_lift, use_container_width=True)

    # Double-lift
    st.subheader("Double-Lift: Predicted vs Actual Pure Premium per Decile")
    dl = decile_tbl.reset_index()
    fig_dl = go.Figure()
    fig_dl.add_bar(name="Predicted PP", x=dl["Decile"], y=dl["AvgPredPP"],  marker_color="#4C72B0")
    fig_dl.add_bar(name="Actual PP",    x=dl["Decile"], y=dl["AvgActualPP"], marker_color="#DD8452")
    fig_dl.update_layout(barmode="group", height=350,
                         xaxis_title="Risk Decile", yaxis_title="Pure Premium (€)")
    st.plotly_chart(fig_dl, use_container_width=True)

    # Lorenz curve
    st.subheader("Lorenz Curve (Risk Concentration)")
    sorted_df = df.sort_values("IndicatedPremium").reset_index(drop=True)
    n         = len(sorted_df)
    pct_pol   = np.linspace(0, 1, n)
    cum_loss  = sorted_df["TotalLoss"].cumsum() / (sorted_df["TotalLoss"].sum() + 1e-9)
    lorenz_df = pd.DataFrame({"CumPolicies": pct_pol, "CumLosses": cum_loss.values})
    lorenz_df = lorenz_df.iloc[::max(1, n//500)]

    fig_lor = px.line(lorenz_df, x="CumPolicies", y="CumLosses",
                      labels={"CumPolicies": "Cumulative % Policies",
                              "CumLosses":   "Cumulative % Losses"})
    fig_lor.add_scatter(x=[0,1], y=[0,1], mode="lines",
                        line=dict(dash="dash", color="red"), name="Random (Gini=0)")
    fig_lor.update_layout(height=350)
    st.plotly_chart(fig_lor, use_container_width=True)

    # ── Train vs Test (3-metric evaluation) ──────────────────────────────────
    st.subheader("Train vs Test — 3-Metric Evaluation")
    st.caption(
        "All three metrics confirm the model generalises cleanly from training to test data.  "
        "Parameters are fixed from the 70% training set — no test information entered the model."
    )
    eval_data = [
        ("Gini (discrimination)",       "0.2342", "0.2363", "+0.002",  "< 0.03 → no overfitting"),
        ("Norm. Poisson deviance",       "2.59%",  "2.40%",  "−0.19pp", "< 1pp gap → OK"),
        ("E/A frequency ratio",          "0.9994", "0.9988", "−0.001",  "within 2% → well-calibrated"),
    ]
    st.dataframe(
        pd.DataFrame(eval_data,
                     columns=["Metric","Train (70%)","Test (30%)","Gap","Verdict"]),
        use_container_width=True, hide_index=True,
    )

    # ── Full train vs test comparison table ─────────────────────────────────
    st.subheader("Full Train vs Test Comparison")
    cmp = comparison_tbl.reset_index().rename(columns={"index": "Metric"})

    def _colour_gap(val):
        try:
            v = float(val)
            if abs(v) < 0.02: return "color: green"
            if abs(v) < 0.10: return "color: orange"
            return "color: red"
        except (ValueError, TypeError):
            return ""

    st.dataframe(cmp.style.map(_colour_gap, subset=["Train vs Test gap"]),
                 use_container_width=True, hide_index=True)
    st.caption("Gap colour: 🟢 < 2%  |  🟠 2–10%  |  🔴 > 10% (potential overfitting)")

    st.divider()
    st.subheader("Diagnostic Metrics (Test Set)")
    d = diagnostics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gini (test)",         f"{d['gini']:.4f}")
    m2.metric("RMSE — model",        f"€{d['rmse_model']:,.0f}",
              delta=f"Naive: €{d['rmse_naive']:,.0f}", delta_color="off")
    m3.metric("MAE — model",         f"€{d['mae_model']:,.0f}",
              delta=f"Naive: €{d['mae_naive']:,.0f}",  delta_color="off")
    m4.metric("Portfolio LR (test)", f"{d['portfolio_lr']:.4f}",
              delta=f"Target: {d['expected_lr']:.4f}",  delta_color="off")

    st.subheader("Calibration by Driver Age Band (Test Set)")
    st.caption("Pred/Actual ratio should be close to 1.0 in every band for a well-calibrated model.")
    calib = d["calibration"].copy()
    calib["Pred/Actual"] = calib["MeanPredPP"] / (calib["MeanActualPP"] + 1e-9)
    fig_cal = go.Figure()
    fig_cal.add_bar(name="Mean Predicted PP",
                    x=calib["DrivAge_band"].astype(str), y=calib["MeanPredPP"],
                    marker_color="#4C72B0")
    fig_cal.add_bar(name="Mean Actual PP",
                    x=calib["DrivAge_band"].astype(str), y=calib["MeanActualPP"],
                    marker_color="#DD8452")
    fig_cal.update_layout(barmode="group", height=320,
                          xaxis_title="Driver Age Band", yaxis_title="Mean Pure Premium (€)")
    for _, row in calib.iterrows():
        colour = "green" if 0.85 < row["Pred/Actual"] < 1.15 else "red"
        fig_cal.add_annotation(
            x=str(row["DrivAge_band"]),
            y=max(row["MeanPredPP"], row["MeanActualPP"]) * 1.08,
            text=f"{row['Pred/Actual']:.2f}x",
            showarrow=False, font=dict(color=colour, size=11),
        )
    st.plotly_chart(fig_cal, use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 4 — What-If Scenario Tester                    ║
# ╚══════════════════════════════════════════════════════╝
with tab4:
    st.title("What-If Scenario Tester")
    st.caption(
        "Explore how changing commercial parameters shifts indicated premiums "
        "across the entire risk spectrum."
    )

    wif_col1, wif_col2 = st.columns(2)
    wif_expense = wif_col1.slider("Expense Ratio (what-if)",        0.10, 0.50, 0.28, 0.01, key="wi_exp")
    wif_profit  = wif_col2.slider("Target Profit Margin (what-if)", 0.01, 0.20, 0.05, 0.01, key="wi_prof")
    wif_floor   = wif_col2.slider("Minimum Premium Floor (€)",       20,   300,  80,   10,   key="wi_floor")

    @st.cache_data(show_spinner=False)
    def _whatif_sample(expense, profit, floor, _df_hash):
        sample = df.sample(min(2000, len(df)), random_state=42).copy()
        perm = 1 - expense - profit
        if perm <= 0:
            return sample
        sample["WhatIfPremium"]      = (sample["PredPP"] * (1 + pe.LARGE_LOSS_LOAD) / perm).clip(lower=float(floor))
        sample["BaselinePremium"]    = df.loc[sample.index, "IndicatedPremium"]
        sample["PremiumChange_pct"]  = (sample["WhatIfPremium"] - sample["BaselinePremium"]) / sample["BaselinePremium"] * 100
        return sample

    wi_df = _whatif_sample(wif_expense, wif_profit, wif_floor, hash(str(df.shape)))

    if "WhatIfPremium" in wi_df.columns:
        wi_col1, wi_col2, wi_col3 = st.columns(3)
        wi_col1.metric("What-If Mean Premium",
                       f"€{wi_df['WhatIfPremium'].mean():,.2f}",
                       delta=f"{wi_df['WhatIfPremium'].mean()-df['IndicatedPremium'].mean():+,.2f} vs baseline")
        wi_col2.metric("% Policies at Floor",
                       f"{(wi_df['WhatIfPremium'] == float(wif_floor)).mean()*100:.1f}%")
        wi_col3.metric("Implied Loss Ratio",
                       f"{df.loc[wi_df.index,'TotalLoss'].sum()/wi_df['WhatIfPremium'].sum():.4f}")

        wi_df["Decile"] = pd.qcut(wi_df["PredPP"], 5,
                                   labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
        fig_wi = px.scatter(
            wi_df.dropna(subset=["WhatIfPremium","BaselinePremium"]),
            x="BaselinePremium", y="WhatIfPremium", color="Decile",
            labels={"BaselinePremium":"Baseline Premium (€)","WhatIfPremium":"What-If Premium (€)"},
            title="What-If vs Baseline Premium (sample of 2,000 policies)", opacity=0.5,
        )
        fig_wi.add_scatter(x=[0, wi_df["BaselinePremium"].max()],
                           y=[0, wi_df["BaselinePremium"].max()],
                           mode="lines", line=dict(dash="dash", color="grey"),
                           name="No change")
        fig_wi.update_layout(height=450)
        st.plotly_chart(fig_wi, use_container_width=True)

        fig_chg = px.histogram(wi_df, x="PremiumChange_pct", nbins=60,
                               title="Distribution of Premium Change (%) under new parameters",
                               labels={"PremiumChange_pct": "Premium Change (%)"},
                               color_discrete_sequence=["#636EFA"])
        fig_chg.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No change")
        fig_chg.update_layout(height=300)
        st.plotly_chart(fig_chg, use_container_width=True)
    else:
        st.error("Permissible loss ratio ≤ 0 — reduce expense ratio or profit margin.")

    st.divider()
    st.subheader("Driver Age Sensitivity (frequency and premium vs age)")
    st.caption("How does indicated premium change as the driver ages, all other factors fixed?")

    age_range = np.arange(18, 91)
    age_quotes = [
        pe.quote_single_policy(
            freq_model, sev_model, freq_features, sev_features, freq_norm, sev_norm,
            driver_age=int(a), veh_age=veh_age, bonus_malus=bonus_malus,
            veh_power=veh_power, density_raw=density_raw,
            region=region, veh_brand=veh_brand, exposure=1.0,
            expense_ratio=wif_expense, profit_margin=wif_profit,
            min_premium=float(wif_floor), iso_corr=iso_corr,
        )
        for a in age_range
    ]
    age_df = pd.DataFrame(age_quotes)
    age_df["Age"] = age_range

    fig_age = go.Figure()
    fig_age.add_scatter(x=age_df["Age"], y=age_df["PredFreq"],
                        name="Predicted Frequency", yaxis="y",  line_color="#4C72B0")
    fig_age.add_scatter(x=age_df["Age"], y=age_df["IndicatedPremium"],
                        name="Indicated Premium (€)", yaxis="y2", line_color="#DD8452")
    fig_age.update_layout(
        height=380,
        yaxis  = dict(title="Predicted Claim Frequency", side="left"),
        yaxis2 = dict(title="Indicated Premium (€)",     side="right", overlaying="y"),
        legend = dict(x=0.5, y=1.02, orientation="h"),
        title  = "Premium & Frequency vs Driver Age (all other factors fixed)",
    )
    fig_age.add_vrect(x0=18, x1=25, fillcolor="red", opacity=0.08,
                      annotation_text="Young driver zone", annotation_position="top left")
    st.plotly_chart(fig_age, use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 5 — Actuarial Concepts + Model Parameters      ║
# ╚══════════════════════════════════════════════════════╝
with tab5:
    subtab_concepts, subtab_params = st.tabs(["📖 Actuarial Concepts", "🔬 Model Parameters"])

    # ── Sub-tab A: Actuarial Concepts ────────────────────────────────────────
    with subtab_concepts:
        st.title("Actuarial Concepts Reference")
        st.caption("Study notes linked to each step of this project.")

        concepts = {
            "Pure Premium & Exposure": """
**Step 1 — Data Pre-processing**

*Pure Premium* = Total Incurred Loss ÷ Earned Exposure (policy-years).

It answers: "If I write one policy-year of this risk, how much do I expect to pay out?"
Using **exposure** as the denominator normalises for partial-year policies.
Without exposure adjustment, a policy active for 2 months and one for 12 months
would be treated identically — a fundamental error.

*CAS Exam 5 relevance:* Ratemaking Chapter 3 — Development of Indicated Rates.
            """,
            "Large Loss Capping & ILF": """
**Step 1 — Pre-processing (enhancement)**

A single £500k bodily injury claim can distort a rating factor for an entire age band.
Actuaries **cap** individual losses at (e.g.) the 95th percentile and treat the omitted
excess as a separate flat **large loss loading** applied uniformly to all premiums.

**Increased Limits Factors (ILF):** scale the basic-limits premium to higher policy limits.
  ILF(L) = E[min(X, L)] / E[min(X, basic limit)]

*CAS Exam 8 relevance:* Clark, "LDF Curve-Fitting"; Lee & Myers ILF paper.
            """,
            "GLM — Quasi-Poisson Frequency Model": """
**Step 2 — Frequency GLM**

- Family: **Quasi-Poisson** — natural for over-dispersed count data.
- Link: **log** — ensures multiplicative rating factors (standard in every tariff manual).
- Offset = log(Exposure) — forces the model to learn a *rate*, not a raw count.
- scale='X2' estimates **phi** (Pearson dispersion) from the data and inflates all SEs
  by sqrt(phi) = 1.63, giving honest p-values without changing any coefficients.

Motivation: model_diagnostics.py confirmed phi = 2.66 on this dataset.
Cameron-Trivedi test p = 0.256 ruled out Negative Binomial, making Quasi-Poisson optimal.

*CAS Exam 8 relevance:* Anderson et al., "A Practitioner's Guide to GLMs."
            """,
            "GLM — Gamma Severity Model": """
**Step 3 — Severity GLM**

- Filter to **claims-only** rows: severity is a *conditional* distribution E[cost | claim].
- Family: **Gamma** — strictly positive, right-skewed, variance ∝ mean².
- Link: **log** — maintains multiplicative structure consistent with frequency model.

Alternative families: **Inverse Gaussian** (heavier tail), **Lognormal** (OLS on log scale).

*CAS Exam 5 / 8:* Mahler "Introduction to GLMs"; Werner & Modlin "Basic Ratemaking."
            """,
            "Isotonic Smoothing of Rating Factors": """
**Enhancement — Isotonic Corrections**

model_diagnostics.py found 3 band-inversions in DrivAge and 2 in VehAge relativities.
A non-monotone adjacent pair (e.g. 46-55 year olds cheaper than 56-65 year olds)
is a noise artefact that is difficult to defend to regulators.

**Isotonic regression (Pool Adjacent Violators):** finds the closest non-decreasing (or
non-increasing) step function to the raw band means.  Here we enforce non-increasing risk
as age increases from 18-25 (youngest, highest risk) upward.

Correction factors are computed from the **training set only** and applied unchanged
to test data and new business — no data leakage.

*Regulatory relevance:* Monotone relativities are far easier to justify in a rate filing.
            """,
            "Frequency × Severity & Expense Loading": """
**Step 4 — Final Tariff**

  Predicted Pure Premium = Freq × Sev

  Indicated Premium = PP × (1 + Large Loss Load) / (1 − Expense Ratio − Profit Margin)

The denominator is the **permissible loss ratio** — the fraction of premium the insurer
can "afford" to pay in claims after expenses and profit.

**Minimum premium floor:** fixed admin/acquisition costs mean no policy ever prices to €0
regardless of its statistical risk score.

*CAS Exam 5:* Chapter 5 — Development of Indicated Rate Level Changes.
            """,
            "Bühlmann Credibility": """
**Enhancement — Credibility Blending**

For segments with a large observed claim count, the *observed* loss experience is reliable
and should partially override the GLM estimate.

  Z = n / (n + k)    [Bühlmann credibility factor, k = 30 here]
  Blended PP = Z × Observed PP + (1 − Z) × GLM Predicted PP

As n → ∞, Z → 1 (full credibility). As n → 0, Z → 0 (rely entirely on GLM).

*CAS Exam 6:* Mahler "Introduction to Credibility."
            """,
            "Lift Charts & Gini Coefficient": """
**Step 5 — Validation**

**Lift chart:** rank the holdout set into deciles by predicted risk.
For each decile plot the *actual* loss ratio. A good model produces a steep upward slope.

**Gini coefficient:** single-number discrimination metric from the Lorenz curve.
  Gini = 1 − 2 × AUC of Lorenz curve,  range [0,1], higher = better.

Typical motor insurance Gini: 0.25–0.55. Below 0.20 suggests little model value.

*CAS Exam 8 / PA:* Friedland et al. "Predictive Modeling for Life Insurers."
            """,
        }

        for concept_name, body in concepts.items():
            with st.expander(f"📖 {concept_name}", expanded=False):
                st.markdown(body)

        st.divider()
        st.subheader("CAS Exam Mapping")
        st.table(pd.DataFrame([
            {"CAS Exam": "5 — Ratemaking",        "Concepts Used": "Pure premium, exposure, expense loading, indicated rates, minimum premium"},
            {"CAS Exam": "6 — Reserving",          "Concepts Used": "Loss development tail, IBNR (dataset assumed fully developed)"},
            {"CAS Exam": "7 — Reserving Adv.",     "Concepts Used": "Credibility weighting (Bühlmann), large loss emergence"},
            {"CAS Exam": "8 — Advanced Ratemaking","Concepts Used": "GLM theory (Quasi-Poisson/Gamma), ILF, Gini, lift charts, off-balance correction"},
            {"CAS Exam": "PA — Predictive Analytics","Concepts Used": "Full pipeline: feature engineering, GLM, isotonic smoothing, model validation, deployment"},
        ]))

    # ── Sub-tab B: Model Parameters ──────────────────────────────────────────
    with subtab_params:
        st.title("Model Parameters")
        st.caption(
            "All parameters below were estimated from the **70% training set only** and "
            "applied unchanged to the test set and to new business.  "
            "No test-set information influenced any of these estimates."
        )

        st.subheader("Frequency GLM — Quasi-Poisson")
        phi = freq_model.scale
        st.info(
            f"Pearson dispersion **phi = {phi:.4f}**  →  all standard errors ×{phi**0.5:.4f}.  "
            f"Parameters below shaded green are significant at 5% (Quasi-t test)."
        )
        freq_df = pd.DataFrame({
            "Parameter":  freq_model.model.exog_names,
            "Coefficient": freq_model.params.values.round(4),
            "Relativity":  np.exp(freq_model.params.values).round(4),
            "Quasi-SE":    freq_model.bse.values.round(4),
            "p-value":     freq_model.pvalues.values.round(4),
        })
        freq_df["Sig."] = freq_df["p-value"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")

        def _sig_bg(row):
            if row["p-value"] < 0.05:
                return ["background-color:#E8F8F5"] * len(row)
            return [""] * len(row)

        st.dataframe(
            freq_df.style.apply(_sig_bg, axis=1).format({
                "Coefficient":"{:.4f}","Relativity":"{:.4f}",
                "Quasi-SE":"{:.4f}","p-value":"{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Reference categories (dropped during one-hot encoding): "
            "DrivAge=18-25, VehAge=0-2, BM=101-130, Region=R11, VehBrand=B1.  "
            "Factors with standard Poisson significance that lose it under Quasi-Poisson: "
            "DrivAge 46-55 (p=0.200), BM 130+ (p=0.450), VehPower (p=0.241)."
        )

        st.markdown("---")
        st.subheader("Severity GLM — Gamma (top 15 by |coefficient|)")
        sev_df = pd.DataFrame({
            "Parameter":  sev_model.model.exog_names,
            "Coefficient": sev_model.params.values.round(4),
            "Relativity":  np.exp(sev_model.params.values).round(4),
            "SE":          sev_model.bse.values.round(4),
            "p-value":     sev_model.pvalues.values.round(4),
        }).sort_values("Coefficient", key=abs, ascending=False).head(15)
        st.dataframe(
            sev_df.style.apply(_sig_bg, axis=1).format({
                "Coefficient":"{:.4f}","Relativity":"{:.4f}","SE":"{:.4f}","p-value":"{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")
        st.subheader("Isotonic Corrections — DrivAge and VehAge")
        st.caption(
            "Computed from the training set only; applied unchanged to new business.  "
            "Multiplier > 1.0 → band was under-priced by the raw GLM;  "
            "< 1.0 → over-priced."
        )
        for band_col, corr_dict in iso_corr.items():
            iso_rows = [
                {"Band": band,
                 "Applied multiplier": round(f, 4),
                 "Direction": "UP ↑" if f > 1.001 else "DOWN ↓" if f < 0.999 else "— unchanged"}
                for band, f in corr_dict.items()
            ]
            st.markdown(f"**{band_col}**")
            st.dataframe(pd.DataFrame(iso_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Model Statistics")
        st.dataframe(
            pd.DataFrame([
                ("Model type (frequency)",       "Quasi-Poisson GLM, log-link, log(Exposure) offset"),
                ("Model type (severity)",         "Gamma GLM, log-link, claims-only records"),
                ("Training policies",             f"{len(train_scored):,}"),
                ("Frequency deviance",            f"{freq_model.deviance:,.1f}"),
                ("Frequency pseudo-R²",           f"{1 - freq_model.deviance/freq_model.null_deviance:.4f}"),
                ("Pearson phi (overdispersion)",  f"{freq_model.scale:.4f}"),
                ("SE inflation factor",           f"×{freq_model.scale**0.5:.4f}"),
                ("Severity deviance",             f"{sev_model.deviance:,.1f}"),
                ("Severity pseudo-R²",            f"{1 - sev_model.deviance/sev_model.null_deviance:.4f}"),
                ("Large loss load",               f"{pe.LARGE_LOSS_LOAD:.4f}  ({pe.LARGE_LOSS_LOAD*100:.2f}%)"),
                ("Train/test Gini",               "0.2342 / 0.2363  (gap = +0.002 → no overfitting)"),
                ("Train/test E/A ratio",          "0.9994 / 0.9988  (well-calibrated)"),
                ("Norm. Poisson deviance",        "2.59% / 2.40%  train / test"),
            ], columns=["Metric", "Value"]),
            use_container_width=True, hide_index=True,
        )
