#!/usr/bin/env python3
"""
Analysis script for MLMI16 CW3 — Personalisation & Inspiration Study
=====================================================================
Between-subjects IV : condition (personalised vs. generic)
Within-subjects IV  : scenario  (travel vs. cooking)

DVs (per scenario):
  - Inspired-by  composite  (2 items, 5-pt Likert)
  - Inspired-to  composite  (2 items, 5-pt Likert)
  - Save choice             (binary behavioural measure)

Manipulation check:
  - Perceived personalisation composite (2 items, 5-pt Likert)

Hypotheses:
  H1: Personalised > Generic on Inspired-by   (one-tailed Welch t-test)
  H2: Personalised > Generic on Inspired-to   (one-tailed Welch t-test)
  H2 (behav.): Personalised > Generic on save (chi-squared, logistic regression)
  Manipulation check: Personalised > Generic on perceived personalisation

Covariates:
  - Trait inspiration (4-item composite)
  - Domain interest  (travel / cooking, single item each)
  - LLM experience   (single item)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, fisher_exact, levene, shapiro, ttest_ind

matplotlib.use("Agg")

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except ImportError:
    HAS_SM = False

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "user_study_MLMI16.csv"
PLOT_DIR  = Path(__file__).resolve().parent / "plots"

# Column definitions
LIKERT_COLS_TRAVEL  = ["inspired_by_1",  "inspired_by_2",
                        "inspired_to_1",  "inspired_to_2",
                        "manip_check_1",  "manip_check_2"]
LIKERT_COLS_COOKING = ["inspired_by_1c", "inspired_by_2c",
                        "inspired_to_1c", "inspired_to_2c",
                        "manip_check_1c", "manip_check_2c"]
TRAIT_COLS          = ["trait_insp_1", "trait_insp_2", "trait_insp_3", "trait_insp_4"]

PAL        = {"personalised": "#9467bd", "generic": "#DD8452"}
COND_ORDER = ["personalised", "generic"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def section(title):
    width = 72
    print(f"\n{'=' * width}\n  {title}\n{'=' * width}\n")


def subsection(title):
    print(f"\n--- {title} ---\n")


def cohens_d(g1, g2):
    """Cohen's d (pooled SD) for two independent groups."""
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else np.nan


def cronbach_alpha(df_items):
    """Cronbach's alpha for a DataFrame of item scores."""
    k = df_items.shape[1]
    if k < 2:
        return np.nan
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def welch_df(g1, g2):
    """Welch–Satterthwaite approximation for degrees of freedom."""
    n1, n2 = len(g1), len(g2)
    s1_sq, s2_sq = g1.var(ddof=1), g2.var(ddof=1)
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    den = (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
    return num / den if den > 0 else np.nan


def welch_ttest_one_tailed(pers, gen, label, alt="greater"):
    """
    One-tailed Welch t-test (personalised > generic) with Cohen's d.
    Reports: t, df, one-tailed p, Cohen's d.
    """
    g_pers = pers[label].dropna() if isinstance(label, str) else pers.dropna()
    g_gen  = gen[label].dropna()  if isinstance(label, str) else gen.dropna()
    if isinstance(label, str):
        g_pers = pers[label].dropna()
        g_gen  = gen[label].dropna()
    if len(g_pers) < 2 or len(g_gen) < 2:
        print(f"  Not enough data (n_pers={len(g_pers)}, n_gen={len(g_gen)}).")
        return
    t_stat, t_p = ttest_ind(g_pers, g_gen, equal_var=False, alternative=alt)
    df_w = welch_df(g_pers, g_gen)
    d    = cohens_d(g_pers, g_gen)
    print(f"  Welch t({df_w:.2f}) = {t_stat:+.3f},  p = {t_p:.4f} (one-tailed),  d = {d:+.3f}")


def run_welch(df, pers, gen, var, label):
    """Convenience wrapper: print label then run welch_ttest_one_tailed."""
    print(f"  {label}")
    g_pers = pers[var].dropna()
    g_gen  = gen[var].dropna()
    welch_ttest_one_tailed(g_pers, g_gen, label=None)


# ── 1. Data loading & cleaning ───────────────────────────────────────────────

def load_and_clean():
    section("1  DATA LOADING & CLEANING")

    df = pd.read_csv(DATA_PATH)
    print(f"Raw rows: {len(df)}")

    df = df.dropna(subset=["session_id"]).copy()
    df = df[df["session_id"].str.strip() != ""].copy()
    print(f"Rows after dropping empty session_id: {len(df)}")

    all_numeric = (
        TRAIT_COLS + ["travel_interest", "cooking_interest", "llm_experience"]
        + LIKERT_COLS_TRAVEL + LIKERT_COLS_COOKING
    )
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Backward-compatibility: cooking_interest added in later data collection.
    if "cooking_interest" not in df.columns:
        df["cooking_interest"] = np.nan

    df["condition"] = df["condition"].str.strip().str.lower()
    valid_conditions = {"personalised", "generic"}
    n_bad = (~df["condition"].isin(valid_conditions)).sum()
    if n_bad:
        print(f"  WARNING: {n_bad} rows with unrecognised condition — dropping.")
        df = df[df["condition"].isin(valid_conditions)].copy()

    # Exclude implausibly fast or slow respondents (symmetric IQR fence: Q±3×IQR).
    if "total_duration_s" in df.columns and df["total_duration_s"].notna().sum() > 3:
        dur = df["total_duration_s"]
        q1, q3 = dur.quantile(0.25), dur.quantile(0.75)
        iqr = q3 - q1
        lo_cut = q1 - 3 * iqr
        hi_cut = q3 + 3 * iqr
        fast = dur < lo_cut
        slow = dur > hi_cut
        if fast.any():
            print(f"  Excluding {fast.sum()} implausibly fast respondent(s) "
                  f"(< {lo_cut:.0f} s, Q1 − 3×IQR).")
        if slow.any():
            print(f"  Excluding {slow.sum()} implausibly slow respondent(s) "
                  f"(> {hi_cut:.0f} s, Q3 + 3×IQR).")
        df = df[~(fast | slow)].copy()

    print(f"Final analytic sample: N = {len(df)}")
    print(f"  personalised: n = {(df['condition'] == 'personalised').sum()}")
    print(f"  generic:      n = {(df['condition'] == 'generic').sum()}")
    return df


# ── 2. Composite scores ──────────────────────────────────────────────────────

def build_composites(df):
    section("2  COMPOSITE SCORES & INTERNAL CONSISTENCY")

    # Travel
    df["inspired_by"]    = df[["inspired_by_1",  "inspired_by_2"]].mean(axis=1)
    df["inspired_to"]    = df[["inspired_to_1",  "inspired_to_2"]].mean(axis=1)
    df["perceived_pers"] = df[["manip_check_1",  "manip_check_2"]].mean(axis=1)
    df["trait_insp"]     = df[TRAIT_COLS].mean(axis=1)
    df["save_binary"]    = (df["save_choice"].str.strip().str.lower() == "yes").astype(int)

    # Cooking
    df["inspired_by_c"]    = df[["inspired_by_1c", "inspired_by_2c"]].mean(axis=1)
    df["inspired_to_c"]    = df[["inspired_to_1c", "inspired_to_2c"]].mean(axis=1)
    df["perceived_pers_c"] = df[["manip_check_1c", "manip_check_2c"]].mean(axis=1)
    df["save_binary_c"]    = (df["save_choice_c"].str.strip().str.lower() == "yes").astype(int)

    # Overall (averaged across scenarios)
    df["inspired_by_overall"]    = df[["inspired_by",    "inspired_by_c"]].mean(axis=1)
    df["inspired_to_overall"]    = df[["inspired_to",    "inspired_to_c"]].mean(axis=1)
    df["perceived_pers_overall"] = df[["perceived_pers", "perceived_pers_c"]].mean(axis=1)
    df["save_overall"]           = df[["save_binary",    "save_binary_c"]].mean(axis=1)

    print("Cronbach's alpha (>= .70 acceptable):")
    for label, cols in [
        ("Trait inspiration",          TRAIT_COLS),
        ("Inspired-by (travel)",       ["inspired_by_1",  "inspired_by_2"]),
        ("Inspired-to (travel)",       ["inspired_to_1",  "inspired_to_2"]),
        ("Perceived pers. (travel)",   ["manip_check_1",  "manip_check_2"]),
        ("Inspired-by (cooking)",      ["inspired_by_1c", "inspired_by_2c"]),
        ("Inspired-to (cooking)",      ["inspired_to_1c", "inspired_to_2c"]),
        ("Perceived pers. (cooking)",  ["manip_check_1c", "manip_check_2c"]),
    ]:
        sub = df[cols].dropna()
        a   = cronbach_alpha(sub)
        print(f"  {label:35s}  α = {a:.3f}  (n = {len(sub)})")

    return df


# ── 3. Sample descriptives ───────────────────────────────────────────────────

def sample_descriptives(df):
    section("3  SAMPLE DESCRIPTIVES")

    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        print(f"Age:  M = {age.mean():.1f}, SD = {age.std():.1f}, "
              f"range = {age.min():.0f}–{age.max():.0f}, n = {age.notna().sum()}")

    if "gender" in df.columns:
        print("\nGender:")
        print(df["gender"].value_counts().to_string())

    if "education" in df.columns:
        print("\nEducation:")
        print(df["education"].value_counts().to_string())

    if "llm_experience" in df.columns:
        llm = df["llm_experience"].dropna()
        print(f"\nLLM experience:  M = {llm.mean():.2f}, SD = {llm.std(ddof=1):.2f}")

    print("\nPre-scenario measures (full analytic sample, 1–5 scales):")
    for var, label in [
        ("trait_insp",      "Trait inspiration (composite)"),
        ("travel_interest",  "Travel interest"),
        ("cooking_interest", "Cooking interest"),
    ]:
        if var not in df.columns:
            continue
        s = df[var].dropna()
        if len(s) == 0:
            continue
        print(f"  {label:35s}  M = {s.mean():.2f}, SD = {s.std(ddof=1):.2f}, n = {len(s)}")

    desc_vars = [
        ("inspired_by",       "Inspired-by (travel)"),
        ("inspired_to",       "Inspired-to (travel)"),
        ("perceived_pers",    "Perceived pers. (travel)"),
        ("inspired_by_c",     "Inspired-by (cooking)"),
        ("inspired_to_c",     "Inspired-to (cooking)"),
        ("perceived_pers_c",  "Perceived pers. (cooking)"),
    ]
    print("\nDescriptives by condition:")
    print(f"{'Variable':32s} {'Condition':14s} {'n':>4s} {'M':>6s} {'SD':>6s} {'Mdn':>5s}")
    print("-" * 68)
    for var, label in desc_vars:
        if var not in df.columns:
            continue
        for cond in COND_ORDER:
            vals = df.loc[df["condition"] == cond, var].dropna()
            print(f"{label:32s} {cond:14s} {len(vals):4d} "
                  f"{vals.mean():6.2f} {vals.std():6.2f} {vals.median():5.1f}")


# ── 4. Assumption checks ─────────────────────────────────────────────────────

def assumption_checks(df, pers, gen):
    """
    Shapiro–Wilk normality + Levene's test for all primary outcome variables.
    Returns a summary list of (label, status, notes) for reporting.
    """
    section("4  ASSUMPTION CHECKS (Shapiro–Wilk normality, Levene's variance equality)")

    summary = []
    for var, label in [
        ("inspired_by",     "Inspired-by (travel)"),
        ("inspired_to",     "Inspired-to (travel)"),
        ("perceived_pers",  "Perceived pers. (travel)"),
        ("inspired_by_c",   "Inspired-by (cooking)"),
        ("inspired_to_c",   "Inspired-to (cooking)"),
        ("perceived_pers_c","Perceived pers. (cooking)"),
    ]:
        subsection(label)
        g1, g2 = pers[var].dropna(), gen[var].dropna()
        flags  = []

        if len(g1) >= 3:
            w1, p1 = shapiro(g1)
            print(f"  Shapiro–Wilk (personalised): W = {w1:.3f}, p = {p1:.4f}")
            if p1 < 0.05:
                flags.append("non-normal (personalised)")
        if len(g2) >= 3:
            w2, p2 = shapiro(g2)
            print(f"  Shapiro–Wilk (generic):      W = {w2:.3f}, p = {p2:.4f}")
            if p2 < 0.05:
                flags.append("non-normal (generic)")
        if len(g1) >= 2 and len(g2) >= 2:
            lev_stat, lev_p = levene(g1, g2)
            print(f"  Levene's test:               F = {lev_stat:.3f}, p = {lev_p:.4f}")
            if lev_p < 0.05:
                flags.append("unequal variances")

        status = "CAUTION" if flags else "OK"
        notes  = ", ".join(flags) if flags else "no major flags"
        summary.append((f"Primary / {label}", status, notes))

    return summary


# ── 5. Manipulation check ────────────────────────────────────────────────────

def manipulation_check(pers, gen):
    section("5  MANIPULATION CHECK — Perceived Personalisation")
    print("  Hypothesis (one-tailed): personalised > generic\n")

    for var, label in [
        ("perceived_pers",   "Travel scenario"),
        ("perceived_pers_c", "Cooking scenario"),
    ]:
        print(f"  {label}")
        g_pers = pers[var].dropna()
        g_gen  = gen[var].dropna()
        t_stat, t_p = ttest_ind(g_pers, g_gen, equal_var=False, alternative="greater")
        df_w = welch_df(g_pers, g_gen)
        d    = cohens_d(g_pers, g_gen)
        print(f"    Welch t({df_w:.2f}) = {t_stat:+.3f},  "
              f"p = {t_p:.4f} (one-tailed),  d = {d:+.3f}")


# ── 6. Primary analyses — H1 & H2 (composites averaged across scenarios) ─────

def primary_analyses(df, pers, gen):
    section("6  PRIMARY ANALYSES — H1 & H2 (composites averaged across scenarios)")
    print("  One-tailed Welch t-tests, α = .05 (personalised > generic)\n")

    for var, label in [
        ("inspired_by_overall", "H1  Inspired-by (overall)"),
        ("inspired_to_overall", "H2  Inspired-to (overall)"),
    ]:
        print(f"  {label}")
        g_pers = pers[var].dropna()
        g_gen  = gen[var].dropna()
        t_stat, t_p = ttest_ind(g_pers, g_gen, equal_var=False, alternative="greater")
        df_w = welch_df(g_pers, g_gen)
        d    = cohens_d(g_pers, g_gen)
        print(f"    Welch t({df_w:.2f}) = {t_stat:+.3f},  "
              f"p = {t_p:.4f} (one-tailed),  d = {d:+.3f}")


# ── 7. Behavioural analysis — Save choice ────────────────────────────────────

def behavioural_analysis(df, pers, gen):
    section("7  BEHAVIOURAL ANALYSIS — Save Choice (chi-squared + logistic regression)")

    def run_save(save_var, label):
        subsection(label)
        ct = pd.crosstab(df["condition"], df[save_var])
        print("Contingency table (condition × save):")
        print(ct.to_string(), "\n")

        if ct.shape == (2, 2):
            chi2, chi_p, dof, expected = chi2_contingency(ct, correction=False)
            print(f"  χ²({dof}) = {chi2:.3f}, p = {chi_p:.4f}")
            if (expected < 5).any():
                odds, fisher_p = fisher_exact(ct, alternative="greater")
                print(f"  Fisher's exact (one-tailed): OR = {odds:.3f}, p = {fisher_p:.4f}")
                print("  (Fisher's preferred: expected cell count < 5)")
        else:
            print("  Cannot run χ² — table is not 2×2.")

        if HAS_SM:
            sub = df[["condition", save_var]].dropna().copy()
            sub["pers_dummy"] = (sub["condition"] == "personalised").astype(int)
            if sub[save_var].nunique() > 1:
                try:
                    model = smf.logit(f"{save_var} ~ pers_dummy", data=sub).fit(disp=0)
                    b   = model.params["pers_dummy"]
                    se  = model.bse["pers_dummy"]
                    z   = model.tvalues["pers_dummy"]
                    p   = model.pvalues["pers_dummy"]
                    or_ = np.exp(b)
                    ci  = np.exp(model.conf_int().loc["pers_dummy"])
                    print(f"\n  Logistic regression ({save_var} ~ personalised):")
                    print(f"    β = {b:+.3f},  SE = {se:.3f},  z = {z:+.3f},  p = {p:.4f}")
                    print(f"    OR = {or_:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
                except Exception as e:
                    print(f"  Logistic regression failed: {e}")
            else:
                print("  Logistic regression skipped — no variance in outcome.")
        else:
            print("  (install statsmodels for logistic regression)")

    run_save("save_binary",   "Travel — Save choice")
    run_save("save_binary_c", "Cooking — Save choice")

    subsection("Save choice averaged across scenarios (Welch t-test)")
    g_pers = pers["save_overall"].dropna()
    g_gen  = gen["save_overall"].dropna()
    t_stat, t_p = ttest_ind(g_pers, g_gen, equal_var=False, alternative="greater")
    df_w = welch_df(g_pers, g_gen)
    d    = cohens_d(g_pers, g_gen)
    print(f"  Welch t({df_w:.2f}) = {t_stat:+.3f},  "
          f"p = {t_p:.4f} (one-tailed),  d = {d:+.3f}")


# ── 8. Exploratory — Per-scenario tests of H1 & H2 ──────────────────────────

def exploratory_per_scenario(pers, gen):
    section("8  EXPLORATORY — H1 & H2 per scenario (one-tailed Welch t-tests)")

    for var, label in [
        ("inspired_by",   "H1  Inspired-by (travel)"),
        ("inspired_by_c", "H1  Inspired-by (cooking)"),
        ("inspired_to",   "H2  Inspired-to (travel)"),
        ("inspired_to_c", "H2  Inspired-to (cooking)"),
    ]:
        print(f"  {label}")
        g_pers = pers[var].dropna()
        g_gen  = gen[var].dropna()
        t_stat, t_p = ttest_ind(g_pers, g_gen, equal_var=False, alternative="greater")
        df_w = welch_df(g_pers, g_gen)
        d    = cohens_d(g_pers, g_gen)
        print(f"    Welch t({df_w:.2f}) = {t_stat:+.3f},  "
              f"p = {t_p:.4f} (one-tailed),  d = {d:+.3f}")


# ── 9. Exploratory — ANCOVA ──────────────────────────────────────────────────

def exploratory_ancova(df, assumption_summary):
    section("9  EXPLORATORY — ANCOVA (controlling for covariates)")
    print("  Covariates: trait inspiration, domain interest, LLM experience\n")

    if not HAS_SM:
        print("  (install statsmodels for ANCOVA)")
        return

    ancova_specs = [
        ("inspired_by",   "Inspired-by (travel)",  "travel_interest"),
        ("inspired_to",   "Inspired-to (travel)",  "travel_interest"),
        ("inspired_by_c", "Inspired-by (cooking)", "cooking_interest"),
        ("inspired_to_c", "Inspired-to (cooking)", "cooking_interest"),
    ]
    for dv, label, domain_interest in ancova_specs:
        cov_cols = ["condition", dv, "trait_insp", domain_interest, "llm_experience"]
        sub = df[cov_cols].dropna().copy()
        sub["pers_dummy"] = (sub["condition"] == "personalised").astype(int)
        if len(sub) < 6:
            print(f"  [{label}] Not enough data for ANCOVA (n={len(sub)}).")
            continue
        try:
            covariates = f"trait_insp + {domain_interest} + llm_experience"
            model  = smf.ols(f"{dv} ~ pers_dummy + {covariates}", data=sub).fit()
            b      = model.params["pers_dummy"]
            se     = model.bse["pers_dummy"]
            t      = model.tvalues["pers_dummy"]
            p      = model.pvalues["pers_dummy"]
            p_one  = p / 2 if t > 0 else 1 - p / 2
            print(f"  {label}:")
            print(f"    β(pers) = {b:+.3f},  SE = {se:.3f},  t = {t:+.3f},  "
                  f"p(two-tailed) = {p:.4f},  p(one-tailed) = {p_one:.4f}")

            ancova_flags = []

            # Residual normality
            if len(model.resid) >= 3:
                w_res, p_res = shapiro(model.resid)
                print(f"    Residual normality (Shapiro–Wilk): W = {w_res:.3f}, p = {p_res:.4f}")
                if p_res < 0.05:
                    ancova_flags.append("non-normal residuals")

            # Homogeneity of regression slopes
            slope_model = smf.ols(f"{dv} ~ pers_dummy * ({covariates})", data=sub).fit()
            slope_parts = []
            for term in ["pers_dummy:trait_insp",
                         f"pers_dummy:{domain_interest}",
                         "pers_dummy:llm_experience"]:
                p_term = slope_model.pvalues.get(term, np.nan)
                slope_parts.append(f"{term} p = {p_term:.4f}")
                if not np.isnan(p_term) and p_term < 0.05:
                    ancova_flags.append(f"slope heterogeneity: {term}")
            print("    Homogeneity of slopes: " + ";  ".join(slope_parts))

            # Multicollinearity (VIF)
            x = sub[["trait_insp", domain_interest, "llm_experience"]].copy()
            x = sm.add_constant(x, has_constant="add")
            vif_parts = []
            for i, col in enumerate(x.columns):
                if col == "const":
                    continue
                vif = variance_inflation_factor(x.values, i)
                vif_parts.append(f"{col} = {vif:.2f}")
                if vif > 5:
                    ancova_flags.append(f"high VIF: {col} ({vif:.2f})")
            print("    Multicollinearity (VIF): " + ",  ".join(vif_parts))

            status = "CAUTION" if ancova_flags else "OK"
            notes  = ", ".join(ancova_flags) if ancova_flags else "no major flags"
            assumption_summary.append((f"ANCOVA / {label}", status, notes))
        except Exception as e:
            print(f"  [{label}] ANCOVA failed: {e}")
            assumption_summary.append((f"ANCOVA / {label}", "CAUTION",
                                       f"model/diagnostics failed: {e}"))


# ── 10. Exploratory — Mixed ANOVA (condition × scenario) ────────────────────

def exploratory_mixed_anova(df, assumption_summary):
    section("10  EXPLORATORY — Mixed ANOVA (condition [between] × scenario [within])")

    if not HAS_PINGOUIN:
        print("  (install pingouin for mixed ANOVA: pip install pingouin)")
        return

    for dv_travel, dv_cook, label in [
        ("inspired_by",    "inspired_by_c",    "Inspired-by"),
        ("inspired_to",    "inspired_to_c",    "Inspired-to"),
        ("perceived_pers", "perceived_pers_c", "Perceived personalisation"),
    ]:
        subsection(label)
        long = pd.DataFrame({
            "subject":   np.concatenate([df.index, df.index]),
            "condition": np.concatenate([df["condition"].values,
                                         df["condition"].values]),
            "scenario":  ["travel"] * len(df) + ["cooking"] * len(df),
            "score":     np.concatenate([df[dv_travel].values, df[dv_cook].values]),
        }).dropna()

        if long["score"].notna().sum() < 6:
            print("  Not enough data.")
            continue
        try:
            aov = pg.mixed_anova(data=long, dv="score",
                                  within="scenario", between="condition",
                                  subject="subject")
            print(aov.to_string(index=False))

            # Residual normality via OLS approximation
            if HAS_SM and len(long) >= 8:
                resid_model = smf.ols(
                    "score ~ C(condition) * C(scenario) + C(subject)", data=long
                ).fit()
                if len(resid_model.resid) >= 3:
                    w_mx, p_mx = shapiro(resid_model.resid)
                    print(f"Residual normality (Shapiro–Wilk on OLS-approx): "
                          f"W = {w_mx:.3f}, p = {p_mx:.4f}")
                    status = "OK" if p_mx >= 0.05 else "CAUTION"
                    note   = ("normal residuals (approx)" if p_mx >= 0.05
                              else "non-normal residuals (approx)")
                    assumption_summary.append((f"Mixed ANOVA / {label}", status, note))
        except Exception as e:
            print(f"  Mixed ANOVA failed: {e}")
            assumption_summary.append((f"Mixed ANOVA / {label}", "CAUTION",
                                       f"model failed: {e}"))


# ── 11. Plots ────────────────────────────────────────────────────────────────

def make_plots(df):
    section("12  PLOTS")
    PLOT_DIR.mkdir(exist_ok=True)

    # Figure 1: DVs by condition × scenario (violin + strip + mean diamond)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)

    dv_pairs = [
        ("inspired_by", "inspired_by_c", "Inspired-by"),
        ("inspired_to", "inspired_to_c", "Inspired-to"),
    ]
    for ax, (tv, cv, title) in zip(axes, dv_pairs):
        x_ticks, x_labels = [], []
        idx = 0
        for scen_label, var in [("Travel", tv), ("Cooking", cv)]:
            for j, cond in enumerate(COND_ORDER):
                pos  = idx + j * 0.6
                vals = df.loc[df["condition"] == cond, var].dropna().values
                vp   = ax.violinplot(vals, positions=[pos], widths=0.5, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(PAL[cond])
                    body.set_alpha(0.35)
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
                ax.scatter(np.full_like(vals, pos) + jitter, vals,
                           color=PAL[cond], s=14, alpha=0.7,
                           edgecolors="white", linewidths=0.3, zorder=3)
                ax.scatter([pos], [vals.mean()], color=PAL[cond], s=50, marker="D",
                           edgecolors="black", linewidths=0.6, zorder=4)
            x_ticks.append(idx + 0.3)
            x_labels.append(scen_label)
            idx += 2

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Rating (1–5)" if ax == axes[0] else "")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])

    legend_handles = [
        Patch(facecolor=PAL[c], alpha=0.5, label=c.capitalize()) for c in COND_ORDER
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out1 = PLOT_DIR / "fig1_dvs_by_condition_scenario.pdf"
    fig.savefig(out1, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out1}")

    # Figure 2: Manipulation check — perceived personalisation
    fig, axes2 = plt.subplots(1, 2, figsize=(5, 3.2), sharey=True)

    for ax, (var, scen_label) in zip(
        axes2, [("perceived_pers", "Travel"), ("perceived_pers_c", "Cooking")]
    ):
        for j, cond in enumerate(COND_ORDER):
            vals = df.loc[df["condition"] == cond, var].dropna().values
            vp   = ax.violinplot(vals, positions=[j], widths=0.6, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(PAL[cond])
                body.set_alpha(0.35)
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full_like(vals, j) + jitter, vals,
                       color=PAL[cond], s=14, alpha=0.7,
                       edgecolors="white", linewidths=0.3, zorder=3)
            ax.scatter([j], [vals.mean()], color=PAL[cond], s=50, marker="D",
                       edgecolors="black", linewidths=0.6, zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Personalised", "Generic"])
        ax.set_ylabel("Perceived personalisation (1–5)" if ax == axes2[0] else "")
        ax.set_title(scen_label, fontsize=11)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])

    fig.tight_layout()
    out2 = PLOT_DIR / "fig2_manipulation_check.pdf"
    fig.savefig(out2, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out2}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    df = load_and_clean()
    df = build_composites(df)

    sample_descriptives(df)

    pers = df[df["condition"] == "personalised"]
    gen  = df[df["condition"] == "generic"]

    assumption_summary = assumption_checks(df, pers, gen)

    manipulation_check(pers, gen)
    primary_analyses(df, pers, gen)
    behavioural_analysis(df, pers, gen)
    exploratory_per_scenario(pers, gen)
    exploratory_ancova(df, assumption_summary)
    exploratory_mixed_anova(df, assumption_summary)

    make_plots(df)

    out = Path(__file__).resolve().parent / "processed_data.csv"
    df.to_csv(out, index=False)
    print(f"\nProcessed data saved to: {out}")


if __name__ == "__main__":
    main()
