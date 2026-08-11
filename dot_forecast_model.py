"""
DOT Budget Forecast Model -- standalone script.

Packages the validated linear-trend forecasting approach from
dot_forecast.ipynb (DOT-filtered, expense-scope-aligned budget data; a
simple OLS trend per series with classic prediction intervals) into a
reusable, importable module instead of a notebook.

CHANGE from the notebook's backtest: models here train on FY2017-2024 ONLY
(8 annual observations), holding out FY2025, FY2026, AND FY2027 -- three
full years, not two -- so backtest() has a longer, stricter validation
window than the notebook's FY2017-2025 -> FY2026-2027 backtest.

Usage:
    python dot_forecast_model.py

Functions:
    load_data()      -- corrected DOT year-by-year table (adopted, modified,
                         actual), cached to data/dot_yearly.csv
    train_models()    -- fit + save per-series linear trend models (FY2017-2024)
    predict(year)     -- forecast a given year for all three series, with
                         95% prediction intervals
    backtest()        -- train on FY2017-2024, evaluate FY2025-2027 against
                         the real values already in the data
"""

import os

import joblib
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 130)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# --- Paths ---------------------------------------------------------------
DATA_DIR = "raw datasets"
BUDGET_PATH = f"{DATA_DIR}/DOT_Budget_2017_2027.csv"
SPENDING_PATH = f"{DATA_DIR}/DOT_Spending_Merged.csv"

YEARLY_CSV_PATH = "data/dot_yearly.csv"
MODELS_PATH = "models/dot_forecast_models.pkl"

SERIES_COLS = ["adopted", "modified", "actual"]
TRAIN_END_YEAR = 2024          # train on FY2017-2024 only
HOLDOUT_YEARS = [2025, 2026, 2027]


# ---------------------------------------------------------------------------
# 1. load_data()
# ---------------------------------------------------------------------------
def load_data(force_rebuild: bool = False) -> pd.DataFrame:
    """Return the corrected DOT year-by-year table (fiscal_year, adopted,
    modified, actual), FY2017-2027.

    Applies the same two corrections as dot_analysis.ipynb / dot_forecast.ipynb:
      1. Agency-scope filter: budget rows restricted to
         Agency == "Department of Transportation" (fixes FY2022 multi-agency
         contamination in the raw budget file).
      2. Expense-scope filter: spending rows restricted to those whose
         Budget Code (leading token) exists in DOT's own operating budget
         codes (fixes the capital-vs-expense scope mismatch).

    Cached to data/dot_yearly.csv -- if that file already exists and
    force_rebuild is False, it's loaded directly instead of re-reading and
    re-filtering the much larger raw CSVs.
    """
    if os.path.exists(YEARLY_CSV_PATH) and not force_rebuild:
        print(f"[load_data] Found cached table at {YEARLY_CSV_PATH} -- loading directly.")
        df = pd.read_csv(YEARLY_CSV_PATH)
        print(f"[load_data] Loaded {len(df)} rows, FY{df['fiscal_year'].min()}-{df['fiscal_year'].max()}.")
        return df

    print("[load_data] No cached table found (or force_rebuild=True) -- rebuilding from raw CSVs.")

    # --- Budget: DOT-only ---
    budget_raw = pd.read_csv(BUDGET_PATH)
    budget = budget_raw[budget_raw["Agency"] == "Department of Transportation"].copy()
    print(f"[load_data] Budget: {len(budget_raw):,} raw rows -> {len(budget):,} DOT-only rows")

    # --- Spending: expense-scope only ---
    spending_raw = pd.read_csv(SPENDING_PATH)
    dot_budget_codes = set(budget["Budget Code"].astype(str).str.strip())
    spending_code_lead = spending_raw["Budget Code"].astype(str).str.extract(r"^([^\s(]+)")[0]
    is_expense = spending_code_lead.isin(dot_budget_codes)
    spending = spending_raw[is_expense].copy()
    print(f"[load_data] Spending: {len(spending_raw):,} raw rows -> {len(spending):,} expense-scope rows")

    # --- Aggregate to fiscal year, FY2017-2027 ---
    budget_by_year = (
        budget[budget["Year"].between(2017, 2027)]
        .groupby("Year")[["Adopted", "Modified"]]
        .sum()
        .rename(columns={"Adopted": "adopted", "Modified": "modified"})
    )
    spending_by_year = (
        spending[spending["Fiscal year"].between(2017, 2027)]
        .groupby("Fiscal year")["Check Amount"]
        .sum()
        .rename("actual")
    )

    df = (
        budget_by_year.join(spending_by_year, how="outer")
        .rename_axis("fiscal_year")
        .reset_index()
        .sort_values("fiscal_year")
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(YEARLY_CSV_PATH), exist_ok=True)
    df.to_csv(YEARLY_CSV_PATH, index=False)
    print(f"[load_data] Saved corrected yearly table to {YEARLY_CSV_PATH} ({len(df)} rows).")

    return df


# ---------------------------------------------------------------------------
# Shared OLS fit + prediction-interval math (used by train_models/predict)
# ---------------------------------------------------------------------------
def _fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple linear OLS fit (via np.polyfit) plus everything needed to
    build a 95%-style prediction interval later: slope, intercept, R²,
    residual standard error, n, degrees of freedom, x-mean, and Sxx.
    """
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    resid = y - y_pred

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    dof = n - 2
    resid_se = np.sqrt(ss_res / dof)
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)

    return {
        "slope": float(slope), "intercept": float(intercept), "r2": float(r2),
        "n": n, "dof": dof, "residual_se": float(resid_se),
        "xbar": float(xbar), "sxx": float(sxx),
    }


# Two-tailed t critical values, alpha=0.05, by degrees of freedom -- a
# standard statistics-textbook table. Used instead of scipy.stats.t.ppf:
# scipy's import has been observed to hang for minutes at a time in this
# environment (same instability seen with sklearn during earlier work),
# and this project's degrees of freedom is always small and known (dof=6
# for the FY2017-2024, n=8 training window used throughout this script).
# For dof beyond the table, falls back to the normal-approximation z=1.96,
# which is what a t-distribution converges to as dof -> infinity anyway.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t_critical_95(dof: int) -> float:
    """Two-tailed 95% t critical value for the given degrees of freedom."""
    if dof in _T_CRIT_95:
        return _T_CRIT_95[dof]
    if dof > 30:
        return 1.960  # normal approximation, standard practice for large dof
    raise ValueError(f"No t-critical value available for dof={dof} (must be >= 1).")


def _predict_one(model: dict, year: int, alpha: float = 0.05) -> dict:
    """Point forecast + prediction interval for one series/model at one year."""
    if alpha != 0.05:
        raise NotImplementedError(
            "Only alpha=0.05 (95% intervals) is supported by the t-critical "
            "lookup table used here -- see _T_CRIT_95."
        )

    slope, intercept = model["slope"], model["intercept"]
    yhat = slope * year + intercept

    se_pred = model["residual_se"] * np.sqrt(
        1 + 1 / model["n"] + (year - model["xbar"]) ** 2 / model["sxx"]
    )
    t_crit = _t_critical_95(model["dof"])
    margin = t_crit * se_pred

    return {
        "fiscal_year": year,
        "point_forecast": yhat,
        "lower_95": yhat - margin,
        "upper_95": yhat + margin,
        "margin": margin,
    }


# ---------------------------------------------------------------------------
# 2. train_models()
# ---------------------------------------------------------------------------
def train_models() -> dict:
    """Fit a linear trend for each series on FY2017-2024 ONLY (holding out
    FY2025, FY2026, FY2027), save the fitted models to
    models/dot_forecast_models.pkl via joblib, and print slope/intercept/R²
    for each series.
    """
    df = load_data()
    train_df = df[df["fiscal_year"] <= TRAIN_END_YEAR].reset_index(drop=True)

    print(f"\n[train_models] Training window: FY2017-{TRAIN_END_YEAR} "
          f"({len(train_df)} observations). Held out: FY{HOLDOUT_YEARS[0]}-{HOLDOUT_YEARS[-1]}.")

    models = {}
    x = train_df["fiscal_year"].values.astype(float)
    print(f"\n{'series':>10}  {'slope ($/yr)':>16}  {'intercept':>18}  {'r2':>6}")
    for col in SERIES_COLS:
        y = train_df[col].values.astype(float)
        model = _fit_ols(x, y)
        models[col] = model
        print(f"{col:>10}  {model['slope']:>16,.2f}  {model['intercept']:>18,.2f}  {model['r2']:>6.4f}")

    metadata = {
        "train_start_year": int(train_df["fiscal_year"].min()),
        "train_end_year": TRAIN_END_YEAR,
        "holdout_years": HOLDOUT_YEARS,
        "series": SERIES_COLS,
    }

    os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
    joblib.dump({"models": models, "metadata": metadata}, MODELS_PATH)
    print(f"\n[train_models] Saved models to {MODELS_PATH}.")

    return models


# ---------------------------------------------------------------------------
# 3. predict(year)
# ---------------------------------------------------------------------------
def predict(year: int, alpha: float = 0.05) -> pd.DataFrame:
    """Load the saved models and return a forecast for `year` for all three
    series, each with a 95% (or 1-alpha) prediction interval.

    Returns a DataFrame indexed by series with columns:
    point_forecast, lower_95, upper_95, margin.
    """
    if not os.path.exists(MODELS_PATH):
        raise FileNotFoundError(
            f"{MODELS_PATH} not found -- run train_models() first."
        )

    saved = joblib.load(MODELS_PATH)
    models = saved["models"]

    rows = []
    for col in SERIES_COLS:
        forecast = _predict_one(models[col], year, alpha=alpha)
        forecast["series"] = col
        rows.append(forecast)

    result = pd.DataFrame(rows).set_index("series")[
        ["fiscal_year", "point_forecast", "lower_95", "upper_95", "margin"]
    ]
    return result


# ---------------------------------------------------------------------------
# 4. backtest()
# ---------------------------------------------------------------------------
def backtest() -> pd.DataFrame:
    """Train on FY2017-2024, predict FY2025/2026/2027 for all three series,
    and print predicted vs. actual vs. % error against the real values
    already sitting in the data.

    FY2027 Actual is flagged explicitly as an unfair comparison -- it's a
    partial fiscal year (only a few weeks elapsed as of this analysis), so
    its real value is nowhere near a comparable full-year total. FY2025 and
    FY2026 are the real validation years.
    """
    print("=" * 70)
    print("BACKTEST: train FY2017-2024, hold out FY2025-2027")
    print("=" * 70)

    train_models()  # (re)fit + save, so backtest() is correct standalone

    df = load_data()

    rows = []
    for col in SERIES_COLS:
        for year in HOLDOUT_YEARS:
            forecast = predict(year).loc[col]
            real = df.loc[df["fiscal_year"] == year, col].iloc[0]
            pct_error = (forecast["point_forecast"] - real) / real * 100
            within = forecast["lower_95"] <= real <= forecast["upper_95"]
            rows.append({
                "series": col,
                "fiscal_year": year,
                "predicted": forecast["point_forecast"],
                "actual": real,
                "pct_error": pct_error,
                "lower_95": forecast["lower_95"],
                "upper_95": forecast["upper_95"],
                "within_95_PI": within,
            })

    results = pd.DataFrame(rows)

    print("\nPredicted vs. actual vs. % error, all series x FY2025-2027:\n")
    print(results.to_string(index=False))

    print(
        "\nNOTE: FY2027 Actual is a partial fiscal year (a few weeks elapsed "
        "as of this analysis) -- its real value is nowhere near a comparable "
        "full-year total, so that one comparison is NOT a fair test of the "
        "model. FY2025 and FY2026 are the real validation years; every "
        "other row (including FY2027 Adopted/Modified, which ARE known "
        "full-year figures set before the fiscal year starts) is fair."
    )

    is_unfair = (results["series"] == "actual") & (results["fiscal_year"] == 2027)
    fair = results[~is_unfair]
    print(f"\nFair-comparison summary ({len(fair)} of {len(results)} rows, excludes FY2027 Actual):")
    print(f"  Within 95% PI  : {fair['within_95_PI'].sum()} / {len(fair)}")
    print(f"  Mean |% error| : {fair['pct_error'].abs().mean():.2f}%")
    print(f"  Max  |% error| : {fair['pct_error'].abs().max():.2f}%")

    return results


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# STEP 1: Train models (FY2017-2024)")
    print("#" * 70)
    train_models()

    print("\n" + "#" * 70)
    print("# STEP 2: Backtest (FY2025-2027 held out)")
    print("#" * 70)
    backtest()

    print("\n" + "#" * 70)
    print("# STEP 3: Demo forecast, FY2028-2032")
    print("#" * 70)
    for yr in range(2028, 2033):
        forecast = predict(yr)
        print(f"\nFY{yr}:")
        print(forecast.to_string())
